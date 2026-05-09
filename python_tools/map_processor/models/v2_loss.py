"""v2 loss stack.

- Tile classification: label-smoothed CE (eps=0.05).
- Blend `present`: BCE-Dice (focal-BCE + soft Dice, balanced for sparse positives).
- Blend `secondary`, `direction`: logit-adjusted CE (Menon et al., ICLR 2021,
  arXiv:2007.07314), masked to present=1.
- Focal Frequency Loss (Jiang et al., ICCV 2021, arXiv:2012.12821) on the
  rendered RGB of (predicted softmax) vs (ground-truth class indices).

Per architecture_research_2026-05.md items 3, 4, 8.
"""
from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .texture_losses import FocalFrequencyLoss
from .v2_render import PaletteRenderer


# ---------------------- per-loss helpers ----------------------

def label_smoothed_ce(logits: torch.Tensor, target: torch.Tensor,
                      eps: float = 0.05, ignore_index: int = -100) -> torch.Tensor:
    """logits: (B, V, H, W). target: (B, H, W) long. Returns scalar."""
    return F.cross_entropy(logits, target, label_smoothing=eps, ignore_index=ignore_index)


def focal_bce(logits: torch.Tensor, target: torch.Tensor, gamma: float = 2.0) -> torch.Tensor:
    """Focal binary cross-entropy. logits: (B, 1, H, W). target: (B, H, W) {0,1}."""
    target = target.float().unsqueeze(1)
    p = torch.sigmoid(logits)
    pt = p * target + (1 - p) * (1 - target)
    focal = (1 - pt).pow(gamma)
    bce = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    return (focal * bce).mean()


def soft_dice(logits: torch.Tensor, target: torch.Tensor, eps: float = 1.0) -> torch.Tensor:
    target = target.float().unsqueeze(1)
    p = torch.sigmoid(logits)
    inter = (p * target).flatten(2).sum(-1)
    denom = p.flatten(2).sum(-1) + target.flatten(2).sum(-1) + eps
    return (1.0 - 2.0 * inter / denom).mean()


def bce_dice(logits, target, gamma: float = 2.0) -> torch.Tensor:
    return 0.5 * focal_bce(logits, target, gamma) + 0.5 * soft_dice(logits, target)


def logit_adjusted_ce(logits: torch.Tensor, target: torch.Tensor,
                      class_freqs: torch.Tensor, *, mask: Optional[torch.Tensor] = None,
                      tau: float = 1.0) -> torch.Tensor:
    """Menon et al. ICLR 2021. Subtract tau * log(prior) from logits before CE.

    logits: (B, C, H, W). target: (B, H, W) long. mask: (B, H, W) bool, True
    where the loss is valid. class_freqs: (C,) prior probabilities (sum 1).
    """
    log_prior = torch.log(class_freqs.clamp_min(1e-8))
    adjusted = logits - tau * log_prior.view(1, -1, 1, 1)
    losses = F.cross_entropy(adjusted, target.clamp_min(0), reduction="none")
    if mask is not None:
        losses = losses * mask.float()
        denom = mask.float().sum().clamp_min(1.0)
        return losses.sum() / denom
    return losses.mean()


# ---------------------- combined module ----------------------


class V2LossModule(nn.Module):
    """Combines all v2 losses behind one forward(...).

    Args:
        palette_renderer: shared PaletteRenderer instance (used by FFL).
        weights: dict of per-loss scalars; missing keys use defaults.
    """

    DEFAULTS = {
        "tile": 1.0,
        "blend_present": 0.3,
        "blend_secondary": 0.2,
        "blend_direction": 0.1,
        "single_present": 0.3,
        "single_secondary": 0.2,
        "single_direction": 0.1,
        "ffl": 0.1,
    }

    def __init__(self, palette_renderer: PaletteRenderer,
                 n_directions: int, palette_size: int,
                 weights: Optional[Dict[str, float]] = None,
                 label_smooth: float = 0.05, gumbel_tau: float = 0.5):
        super().__init__()
        self.renderer = palette_renderer
        self.ffl = FocalFrequencyLoss(alpha=1.0)
        self.n_directions = n_directions
        self.palette_size = palette_size
        self.label_smooth = label_smooth
        self.gumbel_tau = gumbel_tau
        self.weights = {**self.DEFAULTS, **(weights or {})}

    def _blend_losses(self, head_out: Dict[str, torch.Tensor],
                      target: Dict[str, torch.Tensor],
                      dir_freqs: torch.Tensor, sec_freqs: torch.Tensor,
                      prefix: str) -> Dict[str, torch.Tensor]:
        present = target["present"]              # (B, H, W) {0,1}
        secondary = target["secondary"]          # (B, H, W) long, -1 where absent
        direction = target["direction"]          # (B, H, W) long, -1 where absent
        l_pres = bce_dice(head_out["present"], present)
        mask = present > 0
        l_sec = logit_adjusted_ce(head_out["secondary"], secondary, sec_freqs, mask=mask)
        l_dir = logit_adjusted_ce(head_out["direction"], direction, dir_freqs, mask=mask)
        return {
            f"{prefix}_present": l_pres,
            f"{prefix}_secondary": l_sec,
            f"{prefix}_direction": l_dir,
        }

    def forward(self, out: Dict[str, torch.Tensor],
                target_tiles: torch.Tensor,
                target_blend: Dict[str, torch.Tensor],
                target_single: Dict[str, torch.Tensor],
                dir_freqs: torch.Tensor, sec_freqs: torch.Tensor,
                ) -> Dict[str, torch.Tensor]:
        comp: Dict[str, torch.Tensor] = {}
        # Tile CE.
        comp["tile"] = label_smoothed_ce(out["tiles"], target_tiles.long(),
                                         eps=self.label_smooth, ignore_index=-100)
        # Blend layers.
        comp.update(self._blend_losses(out["blend"], target_blend,
                                        dir_freqs, sec_freqs, prefix="blend"))
        comp.update(self._blend_losses(out["single_edge"], target_single,
                                        dir_freqs, sec_freqs, prefix="single"))
        # FFL on rendered RGB. Uses straight-through Gumbel-softmax so the
        # tile head receives gradient signal that punishes blurry/uniform
        # outputs in the frequency domain.
        if self.weights.get("ffl", 0.0) > 0:
            pred_rgb = self.renderer.soft_render(out["tiles"], tau=self.gumbel_tau)
            gt_rgb = self.renderer.hard_render(target_tiles.long())
            comp["ffl"] = self.ffl(pred_rgb, gt_rgb)
        else:
            comp["ffl"] = out["tiles"].sum() * 0.0
        # Weighted total.
        total = sum(self.weights.get(k, 0.0) * v for k, v in comp.items())
        comp["total"] = total
        return comp


# ---------------------- sampler ----------------------

def cluster_balanced_weights(cluster_ids: list[int],
                              power: float = 1.0) -> torch.Tensor:
    """Per-sample weights = 1/(cluster_size**power).
    v3 default power=1.0 (linear) up from 0.5 (sqrt) — diagnostic showed
    sqrt was too gentle for clusters of size 1-2 (rare snowy / Iceland
    biomes were sampled ~5x less than they should have been).
    """
    from collections import Counter
    sizes = Counter(cluster_ids)
    w = torch.tensor([1.0 / max(sizes[c], 1) ** power for c in cluster_ids],
                     dtype=torch.float64)
    return w * (len(w) / w.sum())
