"""Group-wise weight-only LSQ QAT for LLaVA.

This implementation WRAPS each nn.Linear (sharing its weight Parameter), so it
plays nicely with DeepSpeed ZeRO partitioning, bf16 mixed precision, and HF
Trainer. The previous version subclassed nn.Linear and copied the weight,
which produced 0-length shards under ZeRO-3 and required eager scale init
before DeepSpeed had fully wrapped the model — both of which led to NaN
training. This file ports the working architecture from 蒸馏/llava/train/qat.py
while preserving the public API used by llava/train/train_qat.py.

Public API:
  - QuantizedLinear
  - replace_linears_with_qat
  - collect_qat_param_groups
  - materialize_quantized_state_dict
  - bake_qat_weights_inplace
  - strip_log_scale_from_checkpoint_dir
"""

from __future__ import annotations

import math
import os
from typing import Iterable, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------- core ops ----------

class _STERound(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return torch.round(x)

    @staticmethod
    def backward(ctx, g):
        return g


def _ste_round(x: torch.Tensor) -> torch.Tensor:
    return _STERound.apply(x)


def _grad_scale(x: torch.Tensor, scale: float) -> torch.Tensor:
    """Keep the forward value while scaling its backward gradient."""
    return x.detach() + (x - x.detach()) * scale


# ---------- module ----------

class QuantizedLinear(nn.Module):
    """Wraps an nn.Linear and applies group-wise LSQ fake quantization on the
    weight at forward time. The original weight Parameter is *shared* (not
    copied), so the optimizer/DeepSpeed sees the same parameter object.

    Group layout is FLAT: weight is viewed as (num_groups, group_size) where
    num_groups = (out_features * in_features) // group_size. log_scale is
    a 1D Parameter of length num_groups (fp32 for stability).
    """

    # MSE-optimal scale init: 20 candidate scales per group, sampled as
    # ratios of (max_abs / qmax). The optimal ratio for 4-bit usually lies
    # below 1.0 because outliers push max-init off the MSE optimum; we let
    # the lower bound dip to 0.3 to give the grid search room.
    _INIT_NUM_CANDIDATES: int = 20
    _INIT_SEARCH_LO: float = 0.3
    _INIT_SEARCH_HI: float = 1.2
    _MIN_SCALE: float = 1e-8
    _MAX_SCALE: float = 1e4
    _MAX_LOG_SCALE_GRAD: float = 1.0

    def __init__(self, lin: nn.Linear, bits: int = 4, group_size: int = 128):
        super().__init__()
        assert bits in (4, 8), f"QuantizedLinear supports bits in {{4,8}}, got {bits}"
        self.in_features = lin.in_features
        self.out_features = lin.out_features
        self.bits = bits
        self.qmin = -(1 << (bits - 1))         # -8  / -128
        self.qmax = (1 << (bits - 1)) - 1      #  7  /  127

        # Share weight + bias (no copy). This is the key fix vs. the previous
        # implementation: DS-managed params stay managed, no double bookkeeping.
        self.weight = lin.weight
        self.bias = lin.bias

        if group_size <= 0 or self.in_features % group_size != 0:
            self.group_size = self.in_features
        else:
            self.group_size = group_size
        self.num_groups = (self.out_features * self.in_features) // self.group_size
        self._scale_grad_factor = 1.0 / math.sqrt(self.group_size * self.qmax)

        self.log_scale = nn.Parameter(
            torch.zeros(self.num_groups, dtype=torch.float32)
        )
        # Non-persistent buffer so checkpoint roundtrip is silent.
        self.register_buffer(
            "_scale_initialized",
            torch.zeros((), dtype=torch.bool),
            persistent=False,
        )
        self.register_buffer(
            "_nonfinite_scale_grad_count",
            torch.zeros((), dtype=torch.long),
            persistent=False,
        )
        self.register_buffer(
            "_nonfinite_weight_grad_count",
            torch.zeros((), dtype=torch.long),
            persistent=False,
        )
        self.log_scale.register_hook(self._stabilize_log_scale_grad)
        if self.weight.requires_grad:
            self.weight.register_hook(self._record_weight_grad)

    def _stabilize_log_scale_grad(self, grad: torch.Tensor) -> torch.Tensor:
        """Prevent one bad scale gradient from poisoning every QAT weight."""
        nonfinite = ~torch.isfinite(grad)
        if nonfinite.any():
            self._nonfinite_scale_grad_count.add_(nonfinite.sum())
        return torch.nan_to_num(
            grad,
            nan=0.0,
            posinf=self._MAX_LOG_SCALE_GRAD,
            neginf=-self._MAX_LOG_SCALE_GRAD,
        ).clamp_(-self._MAX_LOG_SCALE_GRAD, self._MAX_LOG_SCALE_GRAD)

    def _record_weight_grad(self, grad: torch.Tensor) -> torch.Tensor:
        """Record bad QAT weight gradients without changing training values."""
        nonfinite = ~torch.isfinite(grad)
        if nonfinite.any():
            self._nonfinite_weight_grad_count.add_(nonfinite.sum())
        return grad

    @torch.no_grad()
    def init_scales_from_weight(self) -> None:
        """Per-group scale init via MSE-optimal grid search.

        For each group, evaluate `_INIT_NUM_CANDIDATES` scale candidates
        spanning `_INIT_SEARCH_LO .. _INIT_SEARCH_HI` (as a multiplier of
        max_abs / qmax) and pick the one minimizing
        ||W - dequant(quant(W, s))||² on that group. Compared to a fixed
        max- or 99th-percentile heuristic, this lets each group trade
        outlier clipping against grid-point density on its own merits,
        which matters most at 4-bit where the grid is coarse and the
        optimal trade-off is far from `max/qmax`. Runs once at init
        (and on the first forward for ZeRO-3 partitioned shards).
        """
        if bool(self._scale_initialized.item()):
            return
        W = self.weight.detach()
        if W.numel() == 0:
            # ZeRO-3 partitioned shard — defer to first forward after DS init.
            return
        Wf = W.float().view(-1, self.group_size)            # (G, gs)
        num_groups = Wf.shape[0]
        if not torch.isfinite(Wf).all():
            raise FloatingPointError("Cannot initialize QAT scales from non-finite weights.")

        max_abs = Wf.abs().amax(dim=-1).clamp(min=self._MIN_SCALE)  # (G,)
        ratios = torch.linspace(
            self._INIT_SEARCH_LO, self._INIT_SEARCH_HI,
            self._INIT_NUM_CANDIDATES,
            device=Wf.device, dtype=Wf.dtype,
        )                                                    # (C,)
        # Per-group candidate scales: (G, C)
        candidates = (max_abs / self.qmax).unsqueeze(-1) * ratios

        # Chunked search to cap peak memory at ~chunk * C * gs floats.
        # With chunk=4096, C=20, gs=128, fp32 this is ~40 MB / step.
        best_scale = torch.empty(num_groups, device=Wf.device, dtype=Wf.dtype)
        chunk = 4096
        for i in range(0, num_groups, chunk):
            Wc = Wf[i:i + chunk]                            # (g, gs)
            Sc = candidates[i:i + chunk]                    # (g, C)
            # Quantize each group under each candidate scale.
            wq = (Wc.unsqueeze(1) / Sc.unsqueeze(-1))       # (g, C, gs)
            wq = wq.round().clamp(self.qmin, self.qmax)
            wdq = wq * Sc.unsqueeze(-1)                     # (g, C, gs)
            mse = (wdq - Wc.unsqueeze(1)).pow(2).mean(dim=-1)  # (g, C)
            best_idx = mse.argmin(dim=-1)                   # (g,)
            best_scale[i:i + chunk] = Sc.gather(
                1, best_idx.unsqueeze(-1)
            ).squeeze(-1)

        best_scale = best_scale.clamp(min=self._MIN_SCALE, max=self._MAX_SCALE)
        self.log_scale.data.copy_(torch.log(best_scale))
        self._scale_initialized.fill_(True)

    @classmethod
    def from_linear(cls, lin: nn.Linear, bits: int = 4, group_size: int = 128) -> "QuantizedLinear":
        q = cls(lin, bits=bits, group_size=group_size)
        # Attempt eager init. If running under DeepSpeed ZeRO-3, the weight is
        # currently partitioned (shard size 0) and init will be deferred to the
        # first forward (which runs after DS engine init re-gathers params).
        q.init_scales_from_weight()
        return q

    def _quantize_weight(self) -> torch.Tensor:
        """Return fake-quantized weight in the same dtype as self.weight."""
        # LSQ scales their gradients by 1 / sqrt(group_size * qmax). Without
        # this, scale updates are much larger than weight updates and W4 QAT
        # can diverge within the first few optimizer steps.
        log_scale = self.log_scale.float().clamp(
            min=math.log(self._MIN_SCALE),
            max=math.log(self._MAX_SCALE),
        )
        scale = _grad_scale(
            torch.exp(log_scale),
            self._scale_grad_factor,
        ).unsqueeze(-1)                                      # (num_groups, 1)
        w = self.weight
        w_groups = w.view(-1, self.group_size).float()
        w_q = _ste_round(w_groups / scale).clamp(self.qmin, self.qmax)
        w_dq = (w_q * scale).view_as(w)
        return w_dq.to(w.dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Lazy init on first real forward — guarantees scales are initialized
        # AFTER DS has set up parameters (avoids partition / dtype races).
        if not bool(self._scale_initialized.item()):
            self.init_scales_from_weight()
        w = self._quantize_weight()
        return F.linear(x, w, self.bias)

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"bias={self.bias is not None}, bits={self.bits}, group_size={self.group_size}"
        )


# ---------- replacement / utilities ----------

DEFAULT_QAT_SKIP_KEYWORDS: Tuple[str, ...] = (
    "mm_projector",
    "vision_tower",
    "vision_resampler",
    "lm_head",
    "embed_tokens",
)


def _module_should_skip(name: str, skip_keywords: Iterable[str]) -> bool:
    return any(k in name for k in skip_keywords)


def replace_linears_with_qat(
    model: nn.Module,
    bits: int = 4,
    group_size: int = 128,
    skip_keywords: Iterable[str] = DEFAULT_QAT_SKIP_KEYWORDS,
    verbose: bool = True,
) -> List[str]:
    """Replace eligible nn.Linear submodules with QuantizedLinear in-place.

    QuantizedLinear shares the underlying weight Parameter, so this is safe
    under DeepSpeed ZeRO-2 / ZeRO-3 and HF Trainer.
    """
    replaced: List[str] = []
    for parent_name, parent in list(model.named_modules()):
        for child_name, child in list(parent.named_children()):
            full_name = f"{parent_name}.{child_name}" if parent_name else child_name
            if not isinstance(child, nn.Linear):
                continue
            if _module_should_skip(full_name, skip_keywords):
                continue
            qlin = QuantizedLinear.from_linear(child, bits=bits, group_size=group_size)
            setattr(parent, child_name, qlin)
            replaced.append(full_name)

    if verbose:
        try:
            rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
        except Exception:
            rank = 0
        if rank == 0:
            print(f"[QAT] Replaced {len(replaced)} Linear -> QuantizedLinear "
                  f"(bits={bits}, group_size={group_size}).")
    return replaced


def collect_qat_param_groups(
    model: nn.Module,
    base_lr: float,
    scale_lr: Optional[float] = None,
    weight_decay: float = 0.0,
) -> list:
    """Return optimizer param groups separating QAT scale params from others."""
    if scale_lr is None:
        scale_lr = base_lr
    scale_params, other_params = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if n.endswith(".log_scale") or n.endswith("log_scale"):
            scale_params.append(p)
        else:
            other_params.append(p)
    groups = [
        {"params": other_params, "lr": base_lr, "weight_decay": weight_decay},
    ]
    if scale_params:
        groups.append({"params": scale_params, "lr": scale_lr, "weight_decay": 0.0})
    return groups


@torch.no_grad()
def materialize_quantized_state_dict(model: nn.Module) -> dict:
    """Produce a state dict with fake-quantized weights for downstream low-bit
    deployment. Includes log_scale for AWQ conversion."""
    sd = {}
    for name, module in model.named_modules():
        if isinstance(module, QuantizedLinear):
            sd[name + ".weight"] = module._quantize_weight().detach().cpu()
            if module.bias is not None:
                sd[name + ".bias"] = module.bias.detach().cpu()
            sd[name + ".log_scale"] = module.log_scale.detach().cpu()
    return sd


def bake_qat_weights_inplace(model: nn.Module, is_zero3: bool = False) -> int:
    """Overwrite each QuantizedLinear's weight.data with its fake-quantized
    value, so a vanilla nn.Linear forward gives the same outputs as QAT
    inference. After baking, the checkpoint is loadable by stock LLaVA.

    Under ZeRO-3, gather the partitioned weight + log_scale and write back on
    modifier_rank=0 so the change replicates to all shards.
    """
    count = 0
    if is_zero3:
        import deepspeed
        try:
            rank = torch.distributed.get_rank()
        except Exception:
            rank = 0
        for _, module in model.named_modules():
            if not isinstance(module, QuantizedLinear):
                continue
            params = [module.weight, module.log_scale]
            with deepspeed.zero.GatheredParameters(params, modifier_rank=0):
                if rank == 0:
                    wq = module._quantize_weight().detach()
                    module.weight.data.copy_(wq.to(module.weight.dtype))
            count += 1
    else:
        for _, module in model.named_modules():
            if not isinstance(module, QuantizedLinear):
                continue
            wq = module._quantize_weight().detach()
            module.weight.data.copy_(wq.to(module.weight.dtype))
            count += 1
    return count


def _is_weight_shard(path: str) -> bool:
    """Restrict log_scale stripping to actual weight files. HF checkpoints
    also drop training_args.bin / optimizer.pt / scheduler.pt / rng_state.pt
    in the same dir; those contain pickled non-tensor objects that torch.load
    with weights_only=True (PyTorch 2.6 default) refuses to deserialize."""
    base = os.path.basename(path)
    if base.endswith(".safetensors"):
        return base.startswith("model") or base.startswith("pytorch_model")
    if base.endswith(".bin"):
        return base.startswith("pytorch_model")
    return False


def _strip_log_scale_from_file(path: str) -> int:
    if path.endswith(".safetensors"):
        try:
            from safetensors.torch import load_file, save_file
        except ImportError:
            return 0
        sd = load_file(path)
        keep = {k: v for k, v in sd.items() if "log_scale" not in k}
        removed = len(sd) - len(keep)
        if removed:
            save_file(keep, path, metadata={"format": "pt"})
        return removed
    if path.endswith(".bin"):
        # weights_only=False: we trust our own checkpoint files.
        sd = torch.load(path, map_location="cpu", weights_only=False)
        if not isinstance(sd, dict):
            return 0
        keep = {k: v for k, v in sd.items() if "log_scale" not in k}
        removed = len(sd) - len(keep)
        if removed:
            torch.save(keep, path)
        return removed
    return 0


def strip_log_scale_from_checkpoint_dir(output_dir: str) -> int:
    """Walk a saved HF checkpoint dir on rank 0 and strip log_scale keys from
    weight files (handles sharded saves and the index file)."""
    import json
    import glob
    total = 0
    candidates = glob.glob(os.path.join(output_dir, "*.bin")) + \
                 glob.glob(os.path.join(output_dir, "*.safetensors"))
    for path in candidates:
        if not _is_weight_shard(path):
            continue
        total += _strip_log_scale_from_file(path)
    for idx_name in ("pytorch_model.bin.index.json", "model.safetensors.index.json"):
        idx_path = os.path.join(output_dir, idx_name)
        if not os.path.isfile(idx_path):
            continue
        with open(idx_path) as f:
            idx = json.load(f)
        wmap = idx.get("weight_map", {})
        new_wmap = {k: v for k, v in wmap.items() if "log_scale" not in k}
        if len(new_wmap) != len(wmap):
            idx["weight_map"] = new_wmap
            if "metadata" in idx and "total_size" in idx["metadata"]:
                idx["metadata"].pop("total_size", None)
            with open(idx_path, "w") as f:
                json.dump(idx, f, indent=2)
    return total
