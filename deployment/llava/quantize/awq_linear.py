"""Self-contained AWQ 4-bit GEMM linear layer (vendored from AutoAWQ 0.2.9).

Why vendored: `import awq` eagerly runs `from awq.models import *`, which imports
recent transformers model classes (qwen3, deepseek_v3, ...) that DO NOT exist in
the `transformers==4.37.2` pinned by this LLaVA repo, so the import crashes.
We only need the `WQLinear_GEMM` kernel module for inference, so we copy it here
verbatim (packing layout is byte-for-byte identical to AutoAWQ, so the optional
`awq_ext` CUDA kernels from the `autoawq-kernels` package work unchanged).

Runtime acceleration:
  * If the `awq_ext` extension (pip package `autoawq-kernels`) is importable AND
    the input is on CUDA, the fused INT4 kernels are used (fast).
  * Otherwise we fall back to a pure-PyTorch dequantize+matmul path (correct,
    works on CPU/GPU, slower). The stored weights are still real 4-bit.

The packing matches AutoAWQ GEMM exactly:
    W[out, in] = scales[in // group_size, out] * (q[out, in] - zeros[...])
with q an unsigned 4-bit code in [0, 15].
"""

from __future__ import annotations

import importlib
import warnings

import torch
import torch.nn as nn

# --- optional fused CUDA kernels (pip install autoawq-kernels) ---------------
try:
    awq_ext = importlib.import_module("awq_ext")
except Exception:  # noqa: BLE001 - any import failure means "no kernels"
    awq_ext = None

# Set False the first time an awq_ext call fails (e.g. a kernel wheel whose API
# does not match this packing). After that we silently use the pytorch path, so
# an incompatible kernel can never crash a long eval run mid-way.
_AWQ_EXT_OK = awq_ext is not None
_NAIVE_WARNED = False

AWQ_ORDER = [0, 2, 4, 6, 1, 3, 5, 7]
AWQ_REVERSE_ORDER = [0, 4, 1, 5, 2, 6, 3, 7]


def get_best_device() -> str:
    if torch.cuda.is_available():
        return "cuda:0"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


# --- pure-pytorch unpack / dequant (verbatim from awq.utils.packing_utils) ----

def unpack_awq(qweight: torch.Tensor, qzeros: torch.Tensor, bits: int):
    shifts = torch.arange(0, 32, bits, device=qzeros.device)
    iweights = torch.bitwise_right_shift(qweight[:, :, None], shifts[None, None, :]).to(torch.int8)
    iweights = iweights.view(iweights.shape[0], -1)
    if qzeros is not None:
        izeros = torch.bitwise_right_shift(qzeros[:, :, None], shifts[None, None, :]).to(torch.int8)
        izeros = izeros.view(izeros.shape[0], -1)
    else:
        izeros = qzeros
    return iweights, izeros


def reverse_awq_order(iweights: torch.Tensor, izeros: torch.Tensor, bits: int):
    reverse_order_tensor = torch.arange(
        iweights.shape[-1], dtype=torch.int32, device=iweights.device
    )
    reverse_order_tensor = reverse_order_tensor.view(-1, 32 // bits)
    reverse_order_tensor = reverse_order_tensor[:, AWQ_REVERSE_ORDER]
    reverse_order_tensor = reverse_order_tensor.view(-1)
    if izeros is not None:
        izeros = izeros[:, reverse_order_tensor]
    iweights = iweights[:, reverse_order_tensor]
    return iweights, izeros


def dequantize_gemm(qweight, qzeros, scales, bits, group_size):
    iweight, izeros = unpack_awq(qweight, qzeros, bits)
    iweight, izeros = reverse_awq_order(iweight, izeros, bits)
    iweight = torch.bitwise_and(iweight, (2 ** bits) - 1)
    izeros = torch.bitwise_and(izeros, (2 ** bits) - 1)
    scales = scales.repeat_interleave(group_size, dim=0)
    izeros = izeros.repeat_interleave(group_size, dim=0)
    iweight = (iweight - izeros) * scales
    return iweight  # (in_features, out_features)


class WQLinear_GEMM(nn.Module):
    """4-bit AWQ GEMM linear. Inference-only port of AutoAWQ's class."""

    def __init__(self, w_bit, group_size, in_features, out_features, bias, dev):
        super().__init__()
        if w_bit != 4:
            raise NotImplementedError("Only 4-bit is supported.")
        self.in_features = in_features
        self.out_features = out_features
        self.w_bit = w_bit
        self.group_size = group_size if group_size != -1 else in_features
        assert self.in_features % self.group_size == 0
        assert out_features % (32 // self.w_bit) == 0

        self.register_buffer(
            "qweight",
            torch.zeros((in_features, out_features // (32 // self.w_bit)), dtype=torch.int32, device=dev),
        )
        self.register_buffer(
            "qzeros",
            torch.zeros((in_features // self.group_size, out_features // (32 // self.w_bit)), dtype=torch.int32, device=dev),
        )
        self.register_buffer(
            "scales",
            torch.zeros((in_features // self.group_size, out_features), dtype=torch.float16, device=dev),
        )
        if bias:
            self.register_buffer("bias", torch.zeros((out_features), dtype=torch.float16, device=dev))
        else:
            self.bias = None

    @classmethod
    def from_linear(cls, linear, w_bit, group_size, init_only=False, scales=None, zeros=None):
        awq_linear = cls(
            w_bit, group_size, linear.in_features, linear.out_features,
            linear.bias is not None, linear.weight.device,
        )
        if init_only:  # skeleton for loading a saved state_dict
            return awq_linear

        assert scales is not None and zeros is not None
        scale_zeros = zeros * scales
        awq_linear.scales = scales.clone().half()
        if linear.bias is not None:
            awq_linear.bias = linear.bias.clone().half()

        pack_num = 32 // awq_linear.w_bit
        intweight = []
        for idx in range(awq_linear.in_features):
            intweight.append(
                torch.round(
                    (linear.weight.data[:, idx] + scale_zeros[idx // group_size])
                    / awq_linear.scales[idx // group_size]
                ).to(torch.int)[:, None]
            )
        intweight = torch.cat(intweight, dim=1).t().contiguous().to(torch.int32)

        # int4 codes must fit in [0, 15]; guard the (rare) fp boundary case.
        intweight = intweight.clamp_(0, (1 << w_bit) - 1)

        dev = intweight.device
        if "mps" in str(dev):  # MPS lacks __lshift__
            intweight = intweight.to("cpu")

        qweight = torch.zeros(
            (intweight.shape[0], intweight.shape[1] // 32 * awq_linear.w_bit),
            dtype=torch.int32, device=intweight.device,
        )
        for col in range(intweight.shape[1] // pack_num):
            for i in range(pack_num):
                qweight[:, col] |= intweight[:, col * pack_num + AWQ_ORDER[i]] << (i * awq_linear.w_bit)
        awq_linear.qweight = qweight.to(dev)

        zeros = zeros.to(dtype=torch.int32, device=dev)
        if "mps" in str(dev):
            zeros = zeros.to("cpu")
        qzeros = torch.zeros(
            (zeros.shape[0], zeros.shape[1] // 32 * awq_linear.w_bit),
            dtype=torch.int32, device=zeros.device,
        )
        for col in range(zeros.shape[1] // pack_num):
            for i in range(pack_num):
                qzeros[:, col] |= zeros[:, col * pack_num + AWQ_ORDER[i]] << (i * awq_linear.w_bit)
        awq_linear.qzeros = qzeros.to(dev)
        return awq_linear

    def _naive(self, x2d):
        w = dequantize_gemm(self.qweight, self.qzeros, self.scales, self.w_bit, self.group_size)
        return torch.matmul(x2d, w.to(x2d.dtype))

    @torch.no_grad()
    def forward(self, x):
        global _AWQ_EXT_OK, _NAIVE_WARNED
        out_shape = x.shape[:-1] + (self.out_features,)
        input_dtype = x.dtype
        x = x.to(torch.float16)
        x2d = x.reshape(-1, x.shape[-1])

        if x2d.shape[0] == 0:
            out = torch.zeros((0, self.out_features), dtype=torch.float16, device=x.device)
        elif _AWQ_EXT_OK and x.is_cuda:
            try:
                if x2d.shape[0] >= 1024:  # large batch: dequantize then dense matmul
                    w = awq_ext.dequantize_weights_cuda(self.qweight, self.scales, self.qzeros, 0, 0, 0, False)
                    out = torch.matmul(x2d, w)
                else:  # small batch: fused GEMM
                    out = awq_ext.gemm_forward_cuda(x2d, self.qweight, self.scales, self.qzeros, 8)
            except Exception as e:  # noqa: BLE001 - mismatched kernel API/ABI
                _AWQ_EXT_OK = False
                warnings.warn(f"awq_ext call failed ({e}); permanently falling back to the "
                              f"PyTorch dequant path. Results are unaffected, throughput is lower.")
                out = self._naive(x2d)
        else:
            if not _NAIVE_WARNED:
                msg = ("input on CPU" if not x.is_cuda else
                       "awq_ext not installed (pip install autoawq-kernels)")
                warnings.warn(f"{msg}; using the PyTorch dequant path (correct, slower).")
                _NAIVE_WARNED = True
            out = self._naive(x2d)

        if self.bias is not None:
            out = out + self.bias
        out = out.reshape(out_shape)
        return out.to(input_dtype)

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"bias={self.bias is not None}, w_bit={self.w_bit}, group_size={self.group_size}"
        )
