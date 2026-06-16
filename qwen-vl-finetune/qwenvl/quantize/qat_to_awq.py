"""Convert this repo's group-wise symmetric INT4 LSQ-QAT Qwen3-VL checkpoint into
real AutoAWQ-packed 4-bit weights (AWQ GEMM layout).

QAT (see qwenvl/train/qat_modules.py) is *symmetric signed*:
    code q in [-8, 7], per-group scale s, NO zero point
    W_dq = q * s,  groups laid out along the input dim (group_size cols / group)

AWQ GEMM is *asymmetric unsigned*:
    W = scales * (q_awq - zeros),  q_awq in [0, 15], zeros an int zero-point

The symmetric model maps onto AWQ EXACTLY by choosing a constant zero-point:
    zeros  = 8           (= 2^(bits-1))
    scales = s
    q_awq  = q + 8  in [0, 15]
=>  scales * (q_awq - 8) = s * q = W_dq   (no extra error beyond fp16 scales)

`qat_quantized_weights.bin` (the training sidecar produced by
qat_modules.materialize_quantized_state_dict) stores, per quantized linear:
    <name>.weight     fake-quantized weight W_dq (on the int grid)
    <name>.log_scale  per-group log-scale (length out_features * in_groups)
    <name>.bias       (optional)
We use it as the single source of truth for which layers were quantized. For
Qwen3-VL those are the language-model linears (model.language_model.layers.*
self_attn.{q,k,v,o}_proj and mlp.{gate,up,down}_proj); the vision tower
(model.visual), the merger, lm_head, embeddings and norms stay bf16.
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional

import torch
import torch.nn as nn

from .awq_linear import WQLinear_GEMM, unpack_awq, reverse_awq_order

DEFAULT_BITS = 4
DEFAULT_GROUP_SIZE = 128


# ---------------------------------------------------------------------------
def set_module(model: nn.Module, name: str, new_module: nn.Module) -> None:
    """Replace the submodule at dotted `name` (e.g. model.language_model.layers.0.mlp.up_proj)."""
    parent_name, _, child = name.rpartition(".")
    parent = model.get_submodule(parent_name) if parent_name else model
    setattr(parent, child, new_module)


def _scales_zeros_from_qat(log_scale: torch.Tensor, out_features: int, in_features: int,
                           bits: int, group_size: int):
    """Build AWQ (scales, zeros) from a QAT log_scale vector.

    QAT views the weight flat as (out_features * in_features / group_size, group_size),
    so the group index for weight element (o, i) is  o * n_in_groups + (i // group_size),
    i.e. log_scale.reshape(out_features, n_in_groups)[o, j] is the scale of output o,
    input-group j. AWQ wants scales/zeros shaped (n_in_groups, out_features).
    """
    n_in_groups = in_features // group_size
    assert log_scale.numel() == out_features * n_in_groups, (
        f"log_scale has {log_scale.numel()} entries, expected "
        f"{out_features} * {n_in_groups} = {out_features * n_in_groups}"
    )
    s = log_scale.reshape(out_features, n_in_groups).to(torch.float32).exp()  # (out, n_ig)
    scales = s.t().contiguous()                                              # (n_ig, out)
    zero_point = float(1 << (bits - 1))                                      # 8 for 4-bit
    zeros = torch.full((n_in_groups, out_features), zero_point, dtype=torch.float32)
    return scales, zeros


def pack_qat_linear(weight: torch.Tensor, log_scale: torch.Tensor,
                    bias: Optional[torch.Tensor], bits: int, group_size: int) -> WQLinear_GEMM:
    """Pack one QAT fake-quant linear (weight=W_dq) into a WQLinear_GEMM."""
    out_features, in_features = weight.shape
    scales, zeros = _scales_zeros_from_qat(log_scale, out_features, in_features, bits, group_size)
    tmp = nn.Linear(in_features, out_features, bias=bias is not None)
    tmp.weight.data = weight.detach().to(torch.float32)
    if bias is not None:
        tmp.bias.data = bias.detach().to(torch.float32)
    return WQLinear_GEMM.from_linear(tmp, bits, group_size, scales=scales, zeros=zeros)


@torch.no_grad()
def verify_pack(wq: WQLinear_GEMM, weight: torch.Tensor, log_scale: torch.Tensor,
                bits: int, group_size: int) -> int:
    """Return the max integer-code mismatch between the packed module and the
    QAT fake-quant codes. 0 == bit-exact reproduction of the trained weights."""
    out_features, in_features = weight.shape
    n_in_groups = in_features // group_size
    s = log_scale.reshape(out_features, n_in_groups).to(torch.float32).exp()
    s_full = s.repeat_interleave(group_size, dim=1)                          # (out, in)
    qmin, qmax = -(1 << (bits - 1)), (1 << (bits - 1)) - 1
    q_ref = torch.round(weight.to(torch.float32) / s_full).clamp(qmin, qmax)  # (out, in)

    iw, iz = unpack_awq(wq.qweight.cpu(), wq.qzeros.cpu(), bits)
    iw, iz = reverse_awq_order(iw, iz, bits)
    iw = torch.bitwise_and(iw, (1 << bits) - 1)                              # (in, out), 0..15
    q_got = iw.t().to(torch.float32) - float(1 << (bits - 1))                # (out, in), -8..7
    return int((q_got - q_ref).abs().max().item())


def quantized_layer_names(qat_state: Dict[str, torch.Tensor]) -> List[str]:
    return sorted({k[: -len(".log_scale")] for k in qat_state if k.endswith(".log_scale")})


@torch.no_grad()
def convert_to_awq(model: nn.Module, qat_bin_path: str,
                   bits: int = DEFAULT_BITS, group_size: int = DEFAULT_GROUP_SIZE,
                   verify: bool = True, verbose: bool = True) -> List[str]:
    """Swap every QAT-quantized nn.Linear in `model` for a WQLinear_GEMM, in place.

    Returns the list of converted module names. Weights/scales come from the
    `qat_quantized_weights.bin` sidecar, so it does not matter whether the
    safetensors were baked (qat_modules.bake_qat_weights_inplace) or not.
    """
    qat_state = torch.load(qat_bin_path, map_location="cpu", weights_only=False)
    names = quantized_layer_names(qat_state)
    if verbose:
        print(f"[awq] converting {len(names)} linear layers (bits={bits}, group_size={group_size})")

    max_err = 0
    missing: List[str] = []
    for i, name in enumerate(names):
        weight = qat_state[f"{name}.weight"]
        log_scale = qat_state[f"{name}.log_scale"]
        bias = qat_state.get(f"{name}.bias")

        try:
            target = model.get_submodule(name)
        except AttributeError:
            missing.append(name)
            continue
        if not isinstance(target, nn.Linear):
            raise TypeError(f"{name} in model is {type(target)}, expected nn.Linear")
        dev = target.weight.device

        wq = pack_qat_linear(weight, log_scale, bias, bits, group_size)
        if verify:
            err = verify_pack(wq, weight, log_scale, bits, group_size)
            max_err = max(max_err, err)
            if err != 0:
                print(f"[awq][WARN] {name}: {err} mismatched int codes (expected 0)")
        set_module(model, name, wq.to(dev))
        if verbose and (i + 1) % 50 == 0:
            print(f"[awq]   {i + 1}/{len(names)} packed")

    if missing:
        raise RuntimeError(
            f"{len(missing)} sidecar modules were not found in the model, e.g. "
            f"{missing[:3]}. Does the loaded architecture match the QAT checkpoint?")
    if verbose:
        print(f"[awq] done. max int-code error across all layers = {max_err} (0 = bit-exact)")

    # Record packing metadata under a BENIGN key (not `quantization_config`):
    # a HF `quantization_config` with quant_method="awq" makes transformers'
    # from_pretrained try to import the autoawq package. We load the AWQ weights
    # ourselves, so we must not trigger that path.
    if hasattr(model, "config"):
        model.config.awq_packing = {
            "bits": bits,
            "group_size": group_size,
            "zero_point": 1 << (bits - 1),
            "version": "gemm",
        }
    return names


def maybe_convert_to_awq(model: nn.Module, model_path: str, enable: bool,
                         qat_bin: Optional[str] = None,
                         bits: int = DEFAULT_BITS, group_size: int = DEFAULT_GROUP_SIZE,
                         verify: bool = True) -> bool:
    """One-liner hook for eval scripts.

    If `enable` is False, leaves the (bf16 fake-quant baseline) model untouched and
    returns False. If True, packs the language-model linears into real AWQ 4-bit
    using the training sidecar (defaults to `<model_path>/qat_quantized_weights.bin`)
    and returns True. Call it right after from_pretrained().
    """
    if not enable:
        return False
    if qat_bin is None:
        qat_bin = os.path.join(os.path.expanduser(model_path), "qat_quantized_weights.bin")
    if not os.path.isfile(qat_bin):
        raise FileNotFoundError(
            f"--awq was set but the QAT sidecar was not found: {qat_bin}\n"
            f"Pass --awq-qat-bin /path/to/qat_quantized_weights.bin explicitly."
        )
    print(f"[eval] REAL AWQ 4-bit inference — packing weights from {qat_bin}")
    convert_to_awq(model, qat_bin, bits=bits, group_size=group_size, verify=verify)
    model.eval()
    return True


@torch.no_grad()
def build_awq_skeleton(model: nn.Module, names: List[str],
                       bits: int = DEFAULT_BITS, group_size: int = DEFAULT_GROUP_SIZE,
                       device=None) -> None:
    """Replace the named nn.Linear modules with empty WQLinear_GEMM modules so a
    previously-saved packed state_dict can be loaded with load_state_dict().

    `device` forces where the empty buffers live; pass a real device when the
    incoming nn.Linear weights may be on `meta` (e.g. straight after a
    low_cpu_mem_usage from_pretrained that left missing keys unmaterialized).
    """
    for name in names:
        lin = model.get_submodule(name)
        dev = device if device is not None else lin.weight.device
        wq = WQLinear_GEMM(bits, group_size, lin.in_features, lin.out_features,
                           lin.bias is not None, dev)
        set_module(model, name, wq)
