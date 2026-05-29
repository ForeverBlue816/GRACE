"""GRACE distillation losses for Qwen3-VL QAT.

Implements the three GRACE components (Chen et al., "Gated Relational
Alignment via Confidence-based Distillation for Efficient VLMs"):

  1. Confidence-Gated Decoupled Knowledge Distillation (GDKD)
       decoupled KD (TCKD + NCKD) with per-token entropy gating.
  2. Relational Centered Kernel Alignment (RCKA)
       CKA between per-image visual-token Gram matrices at the
       penultimate LLM layer.
  3. Adaptive Information-Bottleneck controller
       projected dual-ascent on the distillation weight beta.

Total objective:  L_total = L_CE + beta * L_GDKD + omega * L_RCKA

QAT itself is untouched (see qat_modules.py); GRACE only adds loss terms
on top of the quantized student's forward pass.

Notes / deviations from the paper, with rationale:
  * DKD is evaluated only on valid (non-ignored, label-bearing) positions
    rather than every token. This matches the supervised-token convention
    of the trainer and slashes the KL memory footprint (V ~= 1.5e5).
  * gated_dkd_loss processes the valid-token axis in chunks so the
    transient (chunk, vocab) tensors stay bounded even for long packed
    sequences.
  * RCKA is computed per image (split via image_grid_thw) and averaged,
    matching the paper's intra-sample formulation under data-flattening
    where one micro-batch holds several images in a single sequence.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn.functional as F


# ======================================================================
# 1. Confidence-Gated Decoupled Knowledge Distillation
# ======================================================================

def decoupled_kd_per_token(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    target: torch.Tensor,
    temperature: float,
    alpha: float,
    beta_dkd: float,
) -> torch.Tensor:
    """Per-token Decoupled KD loss (Zhao et al., 2022).

    Args:
        student_logits, teacher_logits: (N, V) logits at N valid positions.
        target: (N,) ground-truth class index per position.
        temperature: softmax temperature T.
        alpha: TCKD (target-class) weight.
        beta_dkd: NCKD (non-target "dark knowledge") weight.

    Returns:
        (N,) per-token loss = alpha * TCKD + beta_dkd * NCKD, scaled by T^2.
    """
    T = temperature
    s = student_logits.float() / T
    t = teacher_logits.float() / T
    idx = target.view(-1, 1)  # (N, 1)

    log_p_s = F.log_softmax(s, dim=-1)
    log_p_t = F.log_softmax(t, dim=-1)

    # --- TCKD: KL of the binary {target, non-target} distributions ---
    pt_t = log_p_t.gather(1, idx).exp()           # teacher P(target)  (N,1)
    pt_s = log_p_s.gather(1, idx).exp()           # student P(target)  (N,1)
    bin_t = torch.cat([pt_t, 1.0 - pt_t], dim=1).clamp_min(1e-8)
    bin_s = torch.cat([pt_s, 1.0 - pt_s], dim=1).clamp_min(1e-8)
    tckd = (bin_t * (bin_t.log() - bin_s.log())).sum(dim=1)        # (N,)

    # --- NCKD: KL over the renormalized non-target classes ---
    # Mask the target logit to -inf so softmax renormalizes over the rest.
    neg = torch.finfo(s.dtype).min
    log_p_s_nt = F.log_softmax(s.scatter(1, idx, neg), dim=-1)
    log_p_t_nt = F.log_softmax(t.scatter(1, idx, neg), dim=-1)
    term = log_p_t_nt.exp() * (log_p_t_nt - log_p_s_nt)
    term = term.scatter(1, idx, 0.0)              # target class contributes 0
    nckd = term.sum(dim=1)                                        # (N,)

    return (alpha * tckd + beta_dkd * nckd) * (T * T)


def confidence_weights(teacher_logits: torch.Tensor) -> torch.Tensor:
    """Per-token confidence gate g_i = exp(-H_i / log|V|).

    teacher_logits: (N, V) raw (temperature-1) teacher logits — the gate
    uses the teacher's *output* distribution entropy, not the temperatured
    one used inside DKD.
    """
    p = F.softmax(teacher_logits.float(), dim=-1)
    H = -(p * torch.log(p.clamp_min(1e-12))).sum(dim=-1)           # (N,)
    h_norm = H / math.log(teacher_logits.shape[-1])
    return torch.exp(-h_norm)                                     # (N,)


def gated_dkd_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    labels: torch.Tensor,
    temperature: float = 2.0,
    alpha: float = 1.0,
    beta_dkd: float = 4.0,
    ignore_index: int = -100,
    chunk_size: int = 2048,
) -> torch.Tensor:
    """Confidence-gated DKD over valid, next-token-shifted positions.

    student_logits, teacher_logits: (B, S, V)
    labels: (B, S)  — token id at supervised positions, ignore_index else.

    L_GDKD = sum_i g_i * L_i / sum_i g_i   (Eq. 6 of the paper)
    """
    # Next-token shift, matching HuggingFace's CE convention.
    s = student_logits[:, :-1, :]
    t = teacher_logits[:, :-1, :]
    tgt = labels[:, 1:]

    valid = tgt != ignore_index
    if valid.sum() == 0:
        return student_logits.new_zeros(())

    s = s[valid]            # (N, V)
    t = t[valid]            # (N, V)
    tgt = tgt[valid]        # (N,)

    num = student_logits.new_zeros(())
    den = student_logits.new_zeros(())
    for i in range(0, s.shape[0], chunk_size):
        sc = s[i : i + chunk_size]
        tc = t[i : i + chunk_size]
        gc = tgt[i : i + chunk_size]
        per_tok = decoupled_kd_per_token(sc, tc, gc, temperature, alpha, beta_dkd)
        g = confidence_weights(tc)
        num = num + (g * per_tok).sum()
        den = den + g.sum()

    return num / den.clamp_min(1e-8)


# ======================================================================
# 2. Relational Centered Kernel Alignment
# ======================================================================

def _gram(feats: torch.Tensor) -> torch.Tensor:
    """Row-l2-normalized linear Gram matrix. feats (n, d) -> (n, n)."""
    fn = F.normalize(feats.float(), p=2, dim=-1)
    return fn @ fn.t()


def _center(K: torch.Tensor) -> torch.Tensor:
    """Double-centering, equivalent to H K H with H = I - 1/n."""
    m0 = K.mean(dim=0, keepdim=True)
    m1 = K.mean(dim=1, keepdim=True)
    return K - m0 - m1 + K.mean()


def _cka(K_t: torch.Tensor, K_s: torch.Tensor) -> torch.Tensor:
    """Centered Kernel Alignment in [0, 1] between two Gram matrices."""
    kt = _center(K_t)
    ks = _center(K_s)
    hsic_ts = (kt * ks).sum()
    hsic_tt = (kt * kt).sum()
    hsic_ss = (ks * ks).sum()
    return hsic_ts / torch.sqrt((hsic_tt * hsic_ss).clamp_min(1e-12))


def _cka_safe(
    K_t: torch.Tensor, K_s: torch.Tensor, eps: float = 1e-8
) -> Optional[torch.Tensor]:
    """CKA, returning None when either Gram has near-zero post-centering norm.

    A degenerate Gram (all tokens collapsed onto the same direction) makes the
    HSIC denominator vanish; the clamped sqrt then yields an arbitrary value
    with a high-magnitude gradient. Skipping that image is more honest than
    emitting noise.
    """
    kt = _center(K_t)
    ks = _center(K_s)
    hsic_tt = (kt * kt).sum()
    hsic_ss = (ks * ks).sum()
    if float(hsic_tt) < eps or float(hsic_ss) < eps:
        return None
    hsic_ts = (kt * ks).sum()
    return hsic_ts / torch.sqrt(hsic_tt * hsic_ss)


def rcka_loss(
    student_hidden: torch.Tensor,
    teacher_hidden: torch.Tensor,
    image_mask: torch.Tensor,
    video_mask: Optional[torch.Tensor],
    image_grid_thw: Optional[torch.Tensor],
    video_grid_thw: Optional[torch.Tensor],
    merge_size: int,
    min_tokens: int = 4,
    degenerate_eps: float = 1e-8,
) -> torch.Tensor:
    """1 - CKA over per-item visual-token Gram matrices, averaged.

    student_hidden / teacher_hidden : (B, S, d_*) penultimate-layer hidden states.
    image_mask  / video_mask        : (B, S) bool — token-id mask per modality.
    image_grid_thw / video_grid_thw : (num_*, 3) — split flattened tokens per item.

    Each modality is aligned strictly to its OWN grid. The earlier single-mask
    form mis-aligned mixed image+video samples whenever a video preceded an
    image: the loop walked `hidden[image_mask|video_mask]` from offset 0 using
    image-only `per_img`, so every Gram was sliced from the wrong rows.

    CKA is scale- and dimension-invariant, so d_s != d_t is fine; only the
    per-item token count must match between teacher and student (it does,
    since both consume the same grid_thw).
    """
    zero = student_hidden.new_zeros(())
    losses: list = []

    for mask, grid in (
        (image_mask, image_grid_thw),
        (video_mask, video_grid_thw),
    ):
        if grid is None or mask is None:
            continue
        if mask.sum() == 0:
            continue
        sh = student_hidden[mask]    # (n_total_modality, d_s)
        th = teacher_hidden[mask]    # (n_total_modality, d_t)
        if sh.shape[0] == 0:
            continue

        per_item = (
            grid[:, 0] * grid[:, 1] * grid[:, 2]
        ) // (merge_size * merge_size)

        off = 0
        for cnt in per_item.tolist():
            cnt = int(cnt)
            if cnt >= min_tokens and off + cnt <= sh.shape[0]:
                cka = _cka_safe(
                    _gram(th[off : off + cnt]),
                    _gram(sh[off : off + cnt]),
                    eps=degenerate_eps,
                )
                if cka is not None:
                    losses.append(1.0 - cka)
            off += cnt

    if not losses:
        return zero
    return torch.stack(losses).mean()


# ======================================================================
# 3. Adaptive Information-Bottleneck controller
# ======================================================================

class AdaptiveIBController:
    """Projected dual-ascent controller for the distillation weight beta.

        beta <- clip( beta + eta * (EMA[L_GDKD] - tau),  [beta_min, beta_max] )

    When the EMA-smoothed gated loss exceeds the information-preservation
    budget tau, beta rises to strengthen distillation; once satisfied beta
    decays so cross-entropy can drive fine-grained task adaptation.

    `update` must be fed an L_GDKD value that is already averaged across
    ranks, so beta stays identical on every process.
    """

    def __init__(
        self,
        tau: float = 0.35,
        eta: float = 0.0015,
        beta_init: float = 1.0,
        beta_min: float = 0.1,
        beta_max: float = 5.0,
        ema_decay: float = 0.99,
    ):
        self.tau = tau
        self.eta = eta
        self.beta = float(beta_init)
        self.beta_min = beta_min
        self.beta_max = beta_max
        self.ema_decay = ema_decay
        self.ema: Optional[float] = None

    def update(self, gdkd_value: float) -> float:
        if self.ema is None:
            self.ema = gdkd_value
        else:
            d = self.ema_decay
            self.ema = d * self.ema + (1.0 - d) * gdkd_value
        self.beta += self.eta * (self.ema - self.tau)
        self.beta = max(self.beta_min, min(self.beta_max, self.beta))
        return self.beta

    def state_dict(self) -> dict:
        return {"beta": self.beta, "ema": self.ema}

    def load_state_dict(self, sd: dict) -> None:
        self.beta = sd.get("beta", self.beta)
        self.ema = sd.get("ema", self.ema)
