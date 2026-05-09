"""
Finetune a pretrained Hugging Face vision model (ViT) for RA3 blend prediction.

Key ideas
---------
- Convert each 5x5 neighborhood into a tiny 3-channel "image":
  - Texture IDs are categorical. We map each global texture ID -> a fixed RGB color (palette lookup).
  - Elevation is injected by modulating brightness with a normalized elevation channel.
  - Then upsample to 224x224 to match ViT pretraining.

- Multi-head / masked loss (single model predicts BOTH layers):
  - blend_present: binary
  - blend_neighbor: which of the 8 neighbors is the secondary texture (8 classes, not 363!)
  - blend_direction: multiclass (only trained when blend_present==1)
  - se_present: binary
  - se_neighbor: which of the 8 neighbors is the secondary texture
  - se_direction: multiclass (only trained when se_present==1)

- Key insight: The secondary texture is ALWAYS one of the 8 neighbors (99.9% of cases).
  So we predict WHICH neighbor, not which of 363 global textures.
  This dramatically simplifies the problem and provides cleaner gradients.

- Direction is almost fully determined by which neighbors have the secondary texture.
  With explicit 8-bit neighbor difference features, direction should train easily.

Prereqs
-------
pip install -r requirements-ml.txt
Then first convert the dataset:
  python scripts/prepare_blend_dataset_memmap.py --npz ... --out-dir ...

Example
-------
python scripts/train_blend_model_hf.py \
  --data-dir "../blendinfo dataset/_generated/prepared_w5_elev_full" \
  --out-dir "../blendinfo dataset/_generated/hf_vit_blend" \
  --model "google/vit-base-patch16-224-in21k" \
  --max-train-samples 2000000 --max-eval-samples 200000 \
  --epochs 1 --batch-size 64 --lr 3e-5
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np


def _require_torch_and_hf():
    try:
        import torch  # noqa: F401
        import transformers  # noqa: F401
    except Exception as e:
        raise SystemExit(
            "Missing ML deps. Install with:\n"
            "  pip install -r requirements-ml.txt\n"
            f"Original error: {e}"
        )


@dataclass
class PreparedMeta:
    n_samples: int
    window: int
    num_textures: int  # for palette (still need this for RGB mapping)
    num_neighbor_classes: int  # 8 (TL,T,TR,L,R,BL,B,BR)
    dir_values: List[int]
    dir_num_classes: int
    elev_mean: float
    elev_std: float
    include_elevation: bool = True
    extra_dim: int = 0
    extra_feature_names: List[str] = None  # type: ignore[assignment]
    map_style_dim: int = 0
    map_style_names: List[str] = None  # type: ignore[assignment]
    mask_ignore_value: int = 255


def _load_prepared_meta(data_dir: Path) -> PreparedMeta:
    meta_path = data_dir / "prepared_meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    # num_neighbor_classes: 8 for TL,T,TR,L,R,BL,B,BR
    neighbor_info = meta.get("neighbor", {})
    num_neighbor = int(neighbor_info.get("num_classes", 8))
    mask_info = meta.get("mask", {})
    return PreparedMeta(
        n_samples=int(meta["n_samples"]),
        window=int(meta["window"]),
        num_textures=int(meta["vocab"]["num_textures"]),
        num_neighbor_classes=num_neighbor,
        dir_values=list(meta["direction"]["values"]),
        dir_num_classes=int(meta["direction"]["num_classes"]),
        elev_mean=float(meta["elevation_norm"]["mean"]),
        elev_std=float(meta["elevation_norm"]["std"]),
        include_elevation=bool(meta.get("include_elevation", True)),
        extra_dim=int(meta.get("extra_dim", 0)),
        extra_feature_names=list(meta.get("extra_feature_names") or []),
        map_style_dim=int(meta.get("map_style_dim", 0)),
        map_style_names=list(meta.get("map_style_names") or []),
        mask_ignore_value=int(mask_info.get("ignore_value", 255)),
    )


def _memmap_load(data_dir: Path, name: str, dtype: np.dtype) -> np.ndarray:
    p = data_dir / name
    arr = np.load(p, mmap_mode="r")
    if arr.dtype != dtype:
        # allow compatible ints, but we want to be explicit for training
        arr = arr.astype(dtype, copy=False)
    return arr


def _split_maps(num_maps: int, val_frac: float, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    ids = np.arange(num_maps, dtype=np.int32)
    rng.shuffle(ids)
    n_val = int(round(float(num_maps) * float(val_frac)))
    n_val = max(1, min(n_val, num_maps - 1))
    val = ids[:n_val]
    train = ids[n_val:]
    return train, val


def _sample_indices_for_map_ids(map_id: np.ndarray, allowed: np.ndarray, n: int, seed: int) -> np.ndarray:
    """
    Sample `n` indices uniformly from the dataset, restricted to map_ids in `allowed`.
    Works with memmap arrays without materializing all indices.
    """
    return _sample_indices_for_map_ids_filtered(map_id=map_id, allowed=allowed, n=n, seed=seed, extra_mask_fn=None)


def _sample_indices_for_map_ids_filtered(
    *,
    map_id: np.ndarray,
    allowed: np.ndarray,
    n: int,
    seed: int,
    extra_mask_fn,
) -> np.ndarray:
    """
    Like _sample_indices_for_map_ids, but supports an additional boolean mask function:
      extra_mask_fn(cand_indices: np.ndarray) -> np.ndarray[bool]
    This is used for positive-aware sampling (e.g., y_present==1).
    """
    rng = np.random.default_rng(seed)
    total = int(map_id.shape[0])

    picked: List[int] = []
    # grow in batches to reduce Python overhead
    batch = min(2_000_000, max(200_000, n * 2))
    while len(picked) < n:
        cand = rng.integers(0, total, size=batch, dtype=np.int64)
        mids = map_id[cand].astype(np.int32, copy=False)
        ok = np.isin(mids, allowed, assume_unique=False)
        if extra_mask_fn is not None:
            ok2 = extra_mask_fn(cand)
            ok = ok & ok2
        keep = cand[ok]
        if keep.size == 0:
            continue
        need = n - len(picked)
        picked.extend(keep[:need].tolist())
    return np.asarray(picked, dtype=np.int64)


class BlendDataset:
    """
    Indexable dataset backed by memmap .npy arrays.
    Returns dicts compatible with HF Trainer collate.
    """

    def __init__(
        self,
        tex: np.ndarray,
        tex_local_norm: Optional[np.ndarray],
        elev: np.ndarray,
        extra: Optional[np.ndarray],
        y_blend_present: np.ndarray,
        y_blend_mask: Optional[np.ndarray],
        y_blend_sec_legacy: np.ndarray,
        y_blend_dir: np.ndarray,
        y_se_present: np.ndarray,
        y_se_mask: Optional[np.ndarray],
        y_se_sec_legacy: np.ndarray,
        y_se_dir: np.ndarray,
        map_style: Optional[np.ndarray],
        indices: np.ndarray,
    ):
        self.tex = tex
        self.tex_local_norm = tex_local_norm
        self.elev = elev
        self.extra = extra
        self.y_blend_present = y_blend_present
        self.y_blend_mask = y_blend_mask
        self.y_blend_sec = y_blend_sec_legacy
        self.y_blend_dir = y_blend_dir
        self.y_se_present = y_se_present
        self.y_se_mask = y_se_mask
        self.y_se_sec = y_se_sec_legacy
        self.y_se_dir = y_se_dir
        self.map_style = map_style
        self.indices = indices

    def __len__(self) -> int:
        return int(self.indices.shape[0])

    def __getitem__(self, i: int) -> Dict[str, object]:
        idx = int(self.indices[i])
        out: Dict[str, object] = {
            "tex": np.asarray(self.tex[idx], dtype=np.int64),
            "elev": np.asarray(self.elev[idx], dtype=np.float32),
            "labels_blend_present": int(self.y_blend_present[idx]),
            # legacy (still useful for debugging)
            "labels_blend_sec": int(self.y_blend_sec[idx]),
            "labels_blend_dir": int(self.y_blend_dir[idx]),
            "labels_se_present": int(self.y_se_present[idx]),
            "labels_se_sec": int(self.y_se_sec[idx]),
            "labels_se_dir": int(self.y_se_dir[idx]),
        }
        if self.y_blend_mask is not None:
            out["labels_blend_mask"] = int(self.y_blend_mask[idx])
        if self.y_se_mask is not None:
            out["labels_se_mask"] = int(self.y_se_mask[idx])
        if self.map_style is not None:
            out["map_style"] = np.asarray(self.map_style[idx], dtype=np.float32)
        if self.tex_local_norm is not None:
            out["tex_local_norm"] = np.asarray(self.tex_local_norm[idx], dtype=np.float32)
        if self.extra is not None:
            out["extra"] = np.asarray(self.extra[idx], dtype=np.float32)
        return out


def _make_token_collator(
    *,
    window: int,
    elev_mean: float,
    elev_std: float,
    mask_ignore_value: int,
    map_style_dim: int,
    dist_boundary_extra_idx: int = -1,
):
    """
    Collator for the token-transformer architecture:
    - Inputs: tex ids [B,25], elev [B,25], optional extra_features [B,K], optional map_style [B,S]
    - Labels: packed into labels [B,6]:
        [blend_present, blend_mask_u8_or_255, blend_dir, se_present, se_mask_u8_or_255, se_dir]
      Directions are -100 when ignored (same masking semantics as before).
    """
    import torch

    w = int(window)
    if w * w != 25:
        raise ValueError(f"Expected window 5 (25 cells), got {w}")
    ignore_mask = int(mask_ignore_value)

    def collate(batch: List[Dict[str, object]]) -> Dict[str, torch.Tensor]:
        tex = torch.stack([torch.from_numpy(b["tex"]) for b in batch], dim=0).long()  # [B,25]
        elev = torch.stack([torch.from_numpy(b["elev"]) for b in batch], dim=0).float()  # [B,25]
        tex_local_norm = None
        if "tex_local_norm" in batch[0]:
            tex_local_norm = torch.stack([torch.from_numpy(b["tex_local_norm"]) for b in batch], dim=0).float()

        # normalize elevation (match training statistics)
        z = (elev - float(elev_mean)) / float(elev_std)

        extra = None
        if "extra" in batch[0]:
            extra = torch.stack([torch.from_numpy(b["extra"]) for b in batch], dim=0).float()

        ms = None
        if map_style_dim > 0 and "map_style" in batch[0]:
            ms = torch.stack([torch.from_numpy(b["map_style"]) for b in batch], dim=0).float()

        b_present = torch.tensor([b["labels_blend_present"] for b in batch], dtype=torch.long)
        se_present = torch.tensor([b["labels_se_present"] for b in batch], dtype=torch.long)

        # mask labels (uint8) if present, else ignore
        if "labels_blend_mask" in batch[0]:
            b_mask = torch.tensor([int(b["labels_blend_mask"]) for b in batch], dtype=torch.long)
        else:
            b_mask = torch.full_like(b_present, ignore_mask)
        if "labels_se_mask" in batch[0]:
            se_mask = torch.tensor([int(b["labels_se_mask"]) for b in batch], dtype=torch.long)
        else:
            se_mask = torch.full_like(se_present, ignore_mask)

        b_dir = torch.tensor([b["labels_blend_dir"] for b in batch], dtype=torch.long)
        se_dir = torch.tensor([b["labels_se_dir"] for b in batch], dtype=torch.long)

        # ignore direction when not present or invalid
        ignore = torch.full_like(b_dir, -100)
        b_dir = torch.where((b_present > 0) & (b_dir >= 0), b_dir, ignore)
        se_dir = torch.where((se_present > 0) & (se_dir >= 0), se_dir, ignore)

        # CRITICAL: train mask on negatives too.
        # - negatives => mask must be 0 (all bits off), NOT ignored
        # - positives with unknown secondary-neighbor pattern => mask=255 (ignore_value) from dataset
        zeros = torch.zeros_like(b_mask)
        b_mask = torch.where((b_present > 0), b_mask, zeros)
        se_mask = torch.where((se_present > 0), se_mask, torch.zeros_like(se_mask))

        labels = torch.stack([b_present, b_mask, b_dir, se_present, se_mask, se_dir], dim=1).long()

        out: Dict[str, torch.Tensor] = {
            "tex": tex,
            "elev_z": z,
            "labels": labels,
        }
        if tex_local_norm is not None:
            out["tex_local_norm"] = tex_local_norm
        if extra is not None:
            out["extra_features"] = extra
        if ms is not None:
            out["map_style"] = ms

        # Phase 1: Extract dist_to_boundary from extra features if index is known
        if dist_boundary_extra_idx >= 0 and extra is not None and extra.shape[1] > dist_boundary_extra_idx:
            out["dist_to_boundary"] = extra[:, dist_boundary_extra_idx]

        return out

    return collate


class AsymmetricLoss:
    """
    Asymmetric Loss (ASL) for multi-label classification.
    Better than BCE for imbalanced multi-label problems (like neighbor mask prediction).
    Ref: https://arxiv.org/abs/2009.14119

    Instantiated as a plain class to avoid issues with nested nn.Module inside
    the model factory. The actual torch import happens at call time.
    """

    def __init__(self, gamma_neg=4, gamma_pos=0, clip=0.05, disable_torch_grad_focal_loss=True):
        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.disable_torch_grad_focal_loss = disable_torch_grad_focal_loss

    def __call__(self, x, y):
        import torch
        # x: logits, y: targets (0 or 1)
        xs_pos = torch.sigmoid(x)
        xs_neg = 1 - xs_pos
        # Asymmetric clipping (hard threshold for easy negatives)
        if self.clip is not None and self.clip > 0:
            xs_neg = (xs_neg + self.clip).clamp(max=1)
        # Basic CE
        los_pos = y * torch.log(xs_pos.clamp(min=1e-8))
        los_neg = (1 - y) * torch.log(xs_neg.clamp(min=1e-8))
        loss = los_pos + los_neg
        # Asymmetric focusing
        if self.gamma_neg > 0 or self.gamma_pos > 0:
            if self.disable_torch_grad_focal_loss:
                torch.set_grad_enabled(False)
            pt0 = xs_pos * y
            pt1 = xs_neg * (1 - y)
            pt = pt0 + pt1
            one_sided_gamma = self.gamma_pos * y + self.gamma_neg * (1 - y)
            one_sided_w = torch.pow(1 - pt, one_sided_gamma)
            if self.disable_torch_grad_focal_loss:
                torch.set_grad_enabled(True)
            loss *= one_sided_w
        return -loss.mean()


def _build_token_model(
    *,
    num_textures: int,
    dir_num_classes: int,
    extra_dim: int,
    map_style_dim: int,
    mask_ignore_value: int,
    hidden: int,
    n_layers: int,
    n_heads: int,
    dropout: float,
    # Phase 1 improvements
    use_asl: bool = False,
    asl_gamma_neg: float = 4.0,
    asl_gamma_pos: float = 0.0,
    asl_clip: float = 0.05,
    use_logit_adj: bool = False,
    dir_class_prior: object = None,  # Optional tensor
    logit_adj_tau: float = 1.0,
    use_cascaded_heads: bool = False,
    mixup_alpha: float = 0.0,
    use_mt_cp: bool = False,
    mt_cp_period: int = 100,
    use_feature_gate: bool = False,
    feature_gate_l1: float = 0.001,
    use_dist_boundary_weight: bool = False,
    dist_boundary_scale: float = 8.0,
):
    """
    Token transformer that operates directly on discrete texture IDs (5x5 window).
    This is far better suited than ViT-on-random-RGB for learning pairwise dominance and local rules.
    """
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    seq_len = 25
    center_idx = 12  # center of 5x5 flattened row-major (0..24)
    ignore_mask = int(mask_ignore_value)

    class TokenBlendModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.num_textures = int(num_textures)
            self.hidden = int(hidden)
            self.ignore_mask = int(ignore_mask)

            self.tex_emb = nn.Embedding(self.num_textures, self.hidden)
            self.elev_proj = nn.Linear(1, self.hidden, bias=True)
            # local palette-order signal (0..1); critical for dominance rules
            self.local_proj = nn.Linear(1, self.hidden, bias=True)
            self.pos_emb = nn.Embedding(seq_len, self.hidden)

            # Boundary complexity weighting for direction loss
            # Weight direction loss higher for cells with more unique neighbor textures
            # Rationale: 100% of direction errors occur at cells with 8 different neighbors
            self.use_boundary_complexity_weighting = False  # Enabled via --boundary-complexity-weighting CLI flag
            # Weights for different complexity levels:
            # [1-2 unique, 3-4 unique, 5-6 unique, 7-8 unique]
            self.register_buffer('complexity_weights', torch.tensor([1.0, 1.5, 2.0, 3.0], dtype=torch.float32))

            enc_layer = nn.TransformerEncoderLayer(
                d_model=self.hidden,
                nhead=int(n_heads),
                dim_feedforward=int(self.hidden * 4),
                dropout=float(dropout),
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.encoder = nn.TransformerEncoder(enc_layer, num_layers=int(n_layers))

            # --- Phase 1: Feature Gate with L1 Regularization ---
            self._use_feature_gate = bool(use_feature_gate)
            self._feature_gate_l1 = float(feature_gate_l1)
            _eff_extra_dim = int(extra_dim)
            if _eff_extra_dim > 0 and self._use_feature_gate:
                self.feature_gate = nn.Parameter(torch.ones(_eff_extra_dim))
                self.extra_proj = nn.Linear(_eff_extra_dim, self.hidden)
            elif _eff_extra_dim > 0:
                self.feature_gate = None
                self.extra_proj = nn.Linear(_eff_extra_dim, self.hidden)
            else:
                self.feature_gate = None
                self.extra_proj = None
            self.map_style_proj = nn.Linear(int(map_style_dim), self.hidden) if int(map_style_dim) > 0 else None

            # Heads - blend_present and blend_mask use center token
            self.blend_present = nn.Linear(self.hidden, 1)
            self.blend_mask = nn.Linear(self.hidden, 8)  # multi-label neighbor mask

            # Hierarchical direction: row (top/mid/bottom) + column (left/mid/right)
            # Plus type discriminator for ambiguous cases (e.g., TopLeft vs ExceptTopLeft)
            # Input: pooled features from 8 neighbor positions (position-aware)
            self.neighbor_idxs = [6, 7, 8, 11, 13, 16, 17, 18]  # 8 neighbor positions in 5x5
            neighbor_pooled_dim = self.hidden  # We'll use attention-pooled neighbors

            # Position-aware direction: pool neighbor embeddings with attention
            self.dir_query = nn.Parameter(torch.randn(1, 1, self.hidden) * 0.02)
            self.dir_attn = nn.MultiheadAttention(self.hidden, num_heads=4, batch_first=True)

            # Hierarchical direction heads
            self.blend_dir_row = nn.Linear(self.hidden, 3)   # top=0, mid=1, bottom=2
            self.blend_dir_col = nn.Linear(self.hidden, 3)   # left=0, mid=1, right=2
            self.blend_dir_type = nn.Linear(self.hidden, 3)  # straight=0, diagonal=1, except=2

            # Legacy: keep original direction head for compatibility
            self.blend_dir = nn.Linear(self.hidden, int(dir_num_classes))

            self.se_present = nn.Linear(self.hidden, 1)
            self.se_mask = nn.Linear(self.hidden, 8)
            self.se_dir_row = nn.Linear(self.hidden, 3)
            self.se_dir_col = nn.Linear(self.hidden, 3)
            self.se_dir_type = nn.Linear(self.hidden, 3)
            self.se_dir = nn.Linear(self.hidden, int(dir_num_classes))

            # loss config (set from CLI)
            self.use_focal_loss = False
            self.focal_gamma = 2.0
            self.focal_alpha = 0.10  # default: punish false positives more
            self.loss_weight_present = 1.0
            self.loss_weight_mask = 1.0
            self.loss_weight_dir = 1.0
            self.loss_weight_consistency = 0.5

            # Direction class weights (inverse frequency)
            # Computed from training data - see analysis
            self.register_buffer('dir_class_weights', torch.tensor([
                0.0, 0.5, 0.5, 0.9, 0.9, 0.5, 0.5, 0.9, 0.9,
                10.0, 10.0, 1.0, 1.0, 10.0, 10.0, 1.0, 1.0
            ], dtype=torch.float32))

            # --- Phase 1: ASL for neighbor mask ---
            self._use_asl = bool(use_asl)
            if self._use_asl:
                self._asl_fn = AsymmetricLoss(
                    gamma_neg=float(asl_gamma_neg),
                    gamma_pos=float(asl_gamma_pos),
                    clip=float(asl_clip),
                )
            else:
                self._asl_fn = None

            # --- Phase 1: Logit Adjustment for direction head ---
            self._use_logit_adj = bool(use_logit_adj)
            if self._use_logit_adj and dir_class_prior is not None:
                _prior = dir_class_prior if isinstance(dir_class_prior, torch.Tensor) else torch.tensor(dir_class_prior, dtype=torch.float32)
                _log_prior = float(logit_adj_tau) * torch.log(_prior + 1e-12)
                self.register_buffer('dir_log_prior', _log_prior)
            else:
                self._use_logit_adj = False
                self.register_buffer('dir_log_prior', torch.zeros(int(dir_num_classes)))

            # --- Phase 1: Cascaded output heads ---
            self._use_cascaded_heads = bool(use_cascaded_heads)

            # --- Phase 1: Embedding-space MixUp ---
            self.mixup_alpha = float(mixup_alpha)

            # --- Phase 1: MT-CP Loss Prioritization ---
            self._use_mt_cp = bool(use_mt_cp)
            self._mt_cp_period = int(mt_cp_period)
            self._mt_cp_step = 0
            self._task_loss_history = {'present': [], 'mask': [], 'dir': []}
            self._task_initial_loss = {}
            # Dynamic weights start at 1.0 (uniform); updated every mt_cp_period steps
            self._mt_cp_weights = {'present': 1.0, 'mask': 1.0, 'dir': 1.0}

            # --- Phase 1: Distance-to-boundary weighting ---
            self._use_dist_boundary_weight = bool(use_dist_boundary_weight)
            self._dist_boundary_scale = float(dist_boundary_scale)

        def _compute_boundary_complexity(self, tex: torch.Tensor) -> torch.Tensor:
            """
            Compute the number of unique textures among the 8 neighbors for each sample.
            tex: [B, 25] texture IDs (5x5 flattened row-major)
            Returns: [B] int tensor with values 1-8 (number of unique neighbor textures)
            """
            # Neighbor indices in flattened 5x5: TL=6, T=7, TR=8, L=11, R=13, BL=16, B=17, BR=18
            neighbor_tex = tex[:, self.neighbor_idxs]  # [B, 8]
            # Count unique values per row - use a loop since torch.unique doesn't support batch mode
            B = neighbor_tex.shape[0]
            unique_counts = torch.zeros(B, dtype=torch.long, device=tex.device)
            for i in range(B):
                unique_counts[i] = torch.unique(neighbor_tex[i]).numel()
            return unique_counts

        def _compute_boundary_complexity_fast(self, tex: torch.Tensor) -> torch.Tensor:
            """
            Fast vectorized computation of boundary complexity.
            tex: [B, 25] texture IDs (5x5 flattened row-major)
            Returns: [B] int tensor with values 1-8 (number of unique neighbor textures)

            Uses sorting + diff to count uniques without per-sample loops.
            """
            neighbor_tex = tex[:, self.neighbor_idxs]  # [B, 8]
            # Sort each row
            sorted_tex, _ = torch.sort(neighbor_tex, dim=1)
            # Count transitions (where sorted[i] != sorted[i-1])
            # Pad with -1 at start to count first element as a unique
            padded = torch.cat([torch.full((sorted_tex.shape[0], 1), -1, device=tex.device, dtype=tex.dtype), sorted_tex], dim=1)
            transitions = (padded[:, 1:] != padded[:, :-1]).sum(dim=1)  # [B]
            return transitions

        def _get_complexity_weight(self, unique_counts: torch.Tensor) -> torch.Tensor:
            """
            Map unique neighbor counts (1-8) to complexity weight multipliers.
            unique_counts: [B] int tensor with values 1-8
            Returns: [B] float tensor with weights
            """
            # Map 1-2 -> index 0 (weight 1.0)
            # Map 3-4 -> index 1 (weight 1.5)
            # Map 5-6 -> index 2 (weight 2.0)
            # Map 7-8 -> index 3 (weight 3.0)
            weight_idx = torch.clamp((unique_counts - 1) // 2, 0, 3)
            return self.complexity_weights[weight_idx]

        @staticmethod
        def _focal_loss(logits: torch.Tensor, targets: torch.Tensor, gamma: float, alpha: float) -> torch.Tensor:
            probs = torch.sigmoid(logits)
            p_t = probs * targets + (1 - probs) * (1 - targets)
            alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
            focal_weight = alpha_t * ((1 - p_t) ** gamma)
            bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
            return (focal_weight * bce).mean()

        @staticmethod
        def _masked_ce(logits: torch.Tensor, target: torch.Tensor, weight: torch.Tensor = None) -> torch.Tensor:
            mask = target != -100
            if not torch.any(mask):
                return logits.new_zeros(())
            return F.cross_entropy(logits[mask], target[mask], weight=weight, reduction="mean")

        @staticmethod
        def _masked_ce_weighted(
            logits: torch.Tensor,
            target: torch.Tensor,
            sample_weights: torch.Tensor,
            class_weight: torch.Tensor = None,
        ) -> torch.Tensor:
            """
            Cross-entropy with per-sample weights (for complexity weighting).
            logits: [B, C]
            target: [B] with -100 for ignored samples
            sample_weights: [B] per-sample importance weights
            class_weight: [C] optional class weights
            Returns: weighted mean loss
            """
            mask = target != -100
            if not torch.any(mask):
                return logits.new_zeros(())
            # Compute per-sample CE loss
            per_sample_loss = F.cross_entropy(
                logits[mask], target[mask], weight=class_weight, reduction="none"
            )
            # Apply sample weights
            weights_masked = sample_weights[mask]
            weighted_loss = per_sample_loss * weights_masked
            # Normalize by sum of weights (weighted mean)
            return weighted_loss.sum() / weights_masked.sum().clamp(min=1e-8)

        def _masked_bce_mask8(self, logits8: torch.Tensor, mask_u8: torch.Tensor) -> torch.Tensor:
            """
            logits8: [B,8]
            mask_u8: [B] uint8 stored in labels as int64; 255 => ignore
            Uses ASL if enabled, otherwise standard BCE.
            """
            valid = mask_u8 != int(self.ignore_mask)
            if not torch.any(valid):
                return logits8.new_zeros(())
            m = mask_u8[valid].to(torch.int64)
            # unpack bits into targets [Bv,8]
            bits = torch.stack([(m >> i) & 1 for i in range(8)], dim=1).to(dtype=logits8.dtype)
            if self._use_asl and self._asl_fn is not None:
                return self._asl_fn(logits8[valid], bits)
            return F.binary_cross_entropy_with_logits(logits8[valid], bits, reduction="mean")

        @staticmethod
        def _mask_any_prob_from_logits8(logits8: torch.Tensor) -> torch.Tensor:
            """
            Convert per-bit logits to P(any bit is 1), assuming independent bits:
              p_any = 1 - Π_i (1 - sigmoid(li))
            """
            p = torch.sigmoid(logits8)
            return 1.0 - torch.prod(1.0 - p, dim=1)

        def _get_dir_pooled(self, x: torch.Tensor) -> torch.Tensor:
            """
            Position-aware pooling for direction prediction.
            Uses attention over neighbor positions to get spatially-informed representation.
            x: [B, 25, H] encoder output
            Returns: [B, H] direction-specific pooled features
            """
            B = x.shape[0]
            # Extract neighbor embeddings [B, 8, H]
            neighbor_emb = x[:, self.neighbor_idxs, :]
            
            # Query with learnable direction query
            query = self.dir_query.expand(B, -1, -1)  # [B, 1, H]
            
            # Cross-attention: query attends to neighbor embeddings
            attn_out, _ = self.dir_attn(query, neighbor_emb, neighbor_emb)  # [B, 1, H]
            
            # Combine with center token
            center = x[:, center_idx, :]  # [B, H]
            return center + attn_out.squeeze(1)

        def forward(
            self,
            tex: torch.Tensor,
            elev_z: torch.Tensor,
            labels: Optional[torch.Tensor] = None,
            tex_local_norm: Optional[torch.Tensor] = None,
            extra_features: Optional[torch.Tensor] = None,
            map_style: Optional[torch.Tensor] = None,
            dist_to_boundary: Optional[torch.Tensor] = None,
        ):
            # tex: [B,25] global ids
            # elev_z: [B,25] normalized
            B = tex.shape[0]
            pos = torch.arange(seq_len, device=tex.device, dtype=torch.long).unsqueeze(0).expand(B, -1)

            x = self.tex_emb(torch.clamp(tex, 0, self.num_textures - 1))  # [B,25,H]
            x = x + self.elev_proj(elev_z.unsqueeze(-1)) + self.pos_emb(pos)
            if tex_local_norm is not None:
                x = x + self.local_proj(tex_local_norm.unsqueeze(-1))

            # --- Phase 1: Embedding-space MixUp ---
            mixup_perm = None
            mixup_lam = 1.0
            if self.training and self.mixup_alpha > 0:
                mixup_lam = float(torch.distributions.Beta(self.mixup_alpha, self.mixup_alpha).sample())
                mixup_perm = torch.randperm(B, device=x.device)
                x = mixup_lam * x + (1.0 - mixup_lam) * x[mixup_perm]

            x = self.encoder(x)  # [B,25,H]

            # Center pooled for present/mask
            pooled = x[:, center_idx, :]  # center cell representation
            if self.extra_proj is not None and extra_features is not None:
                # --- Phase 1: Feature Gate with L1 ---
                ef = extra_features
                if self._use_feature_gate and self.feature_gate is not None:
                    ef = ef * self.feature_gate
                pooled = pooled + self.extra_proj(ef)
            if self.map_style_proj is not None and map_style is not None:
                pooled = pooled + self.map_style_proj(map_style)

            # Position-aware pooled for direction
            dir_pooled = self._get_dir_pooled(x)
            if self.extra_proj is not None and extra_features is not None:
                ef_dir = extra_features
                if self._use_feature_gate and self.feature_gate is not None:
                    ef_dir = ef_dir * self.feature_gate
                dir_pooled = dir_pooled + self.extra_proj(ef_dir)
            if self.map_style_proj is not None and map_style is not None:
                dir_pooled = dir_pooled + self.map_style_proj(map_style)

            b_present_logit = self.blend_present(pooled).squeeze(-1)  # [B]
            b_mask = self.blend_mask(pooled)  # [B,8]

            # Hierarchical direction using position-aware features
            b_dir_row = self.blend_dir_row(dir_pooled)   # [B, 3]
            b_dir_col = self.blend_dir_col(dir_pooled)   # [B, 3]
            b_dir_type = self.blend_dir_type(dir_pooled) # [B, 3]
            b_dir = self.blend_dir(dir_pooled)  # [B,D] legacy head

            se_present_logit = self.se_present(pooled).squeeze(-1)
            se_mask = self.se_mask(pooled)
            se_dir_row = self.se_dir_row(dir_pooled)
            se_dir_col = self.se_dir_col(dir_pooled)
            se_dir_type = self.se_dir_type(dir_pooled)
            se_dir = self.se_dir(dir_pooled)

            # --- Phase 1: Cascaded output heads ---
            if self._use_cascaded_heads:
                b_gate = torch.sigmoid(b_present_logit.detach()).unsqueeze(-1)  # [B,1]
                b_mask = b_mask * b_gate
                b_dir = b_dir * b_gate
                b_dir_row = b_dir_row * b_gate
                b_dir_col = b_dir_col * b_gate
                b_dir_type = b_dir_type * b_gate

                se_gate = torch.sigmoid(se_present_logit.detach()).unsqueeze(-1)  # [B,1]
                se_mask = se_mask * se_gate
                se_dir = se_dir * se_gate
                se_dir_row = se_dir_row * se_gate
                se_dir_col = se_dir_col * se_gate
                se_dir_type = se_dir_type * se_gate

            # --- Phase 1: Logit Adjustment for direction ---
            if self._use_logit_adj:
                b_dir = b_dir + self.dir_log_prior
                se_dir = se_dir + self.dir_log_prior

            # Alias for output compatibility
            b_present = b_present_logit
            se_present = se_present_logit

            loss = None
            if labels is not None:
                # labels: [B,6] = (b_present, b_mask_u8_or_255, b_dir, se_present, se_mask_u8_or_255, se_dir)
                b_present_y = labels[:, 0].float()
                b_mask_y = labels[:, 1].long()
                b_dir_y = labels[:, 2].long()
                se_present_y = labels[:, 3].float()
                se_mask_y = labels[:, 4].long()
                se_dir_y = labels[:, 5].long()

                # --- Phase 1: MixUp label mixing ---
                if mixup_perm is not None:
                    lam = mixup_lam
                    # Mix continuous targets (present)
                    b_present_y = lam * b_present_y + (1.0 - lam) * b_present_y[mixup_perm]
                    se_present_y = lam * se_present_y + (1.0 - lam) * se_present_y[mixup_perm]
                    # For mask labels (uint8 bit-packed): mix the unpacked bit targets later
                    # We can't meaningfully mix discrete mask/dir labels, so we mix the losses instead.
                    # Compute losses for original and permuted targets, then combine.
                    b_mask_y_perm = b_mask_y[mixup_perm]
                    b_dir_y_perm = b_dir_y[mixup_perm]
                    se_mask_y_perm = se_mask_y[mixup_perm]
                    se_dir_y_perm = se_dir_y[mixup_perm]

                if bool(getattr(self, "use_focal_loss", False)):
                    loss_b_present = self._focal_loss(b_present, b_present_y, float(self.focal_gamma), float(self.focal_alpha))
                    loss_se_present = self._focal_loss(se_present, se_present_y, float(self.focal_gamma), float(self.focal_alpha))
                else:
                    loss_b_present = F.binary_cross_entropy_with_logits(b_present, b_present_y, reduction="mean")
                    loss_se_present = F.binary_cross_entropy_with_logits(se_present, se_present_y, reduction="mean")

                # Mask loss (with MixUp interpolation if active)
                if mixup_perm is not None:
                    loss_b_mask = lam * self._masked_bce_mask8(b_mask, b_mask_y) + (1.0 - lam) * self._masked_bce_mask8(b_mask, b_mask_y_perm)
                    loss_se_mask = lam * self._masked_bce_mask8(se_mask, se_mask_y) + (1.0 - lam) * self._masked_bce_mask8(se_mask, se_mask_y_perm)
                else:
                    loss_b_mask = self._masked_bce_mask8(b_mask, b_mask_y)
                    loss_se_mask = self._masked_bce_mask8(se_mask, se_mask_y)

                # Direction loss with class weights and boundary complexity weighting
                # Complex boundaries (more unique neighbor textures) get higher weight
                dir_weights = getattr(self, 'dir_class_weights', None)

                if getattr(self, 'use_boundary_complexity_weighting', False):
                    # Compute boundary complexity: number of unique textures among 8 neighbors
                    unique_counts = self._compute_boundary_complexity_fast(tex)
                    complexity_sample_weights = self._get_complexity_weight(unique_counts)

                    # Use weighted CE for direction losses (with MixUp interpolation if active)
                    if mixup_perm is not None:
                        loss_b_dir = lam * self._masked_ce_weighted(b_dir, b_dir_y, complexity_sample_weights, class_weight=dir_weights) + (1.0 - lam) * self._masked_ce_weighted(b_dir, b_dir_y_perm, complexity_sample_weights, class_weight=dir_weights)
                        loss_se_dir = lam * self._masked_ce_weighted(se_dir, se_dir_y, complexity_sample_weights, class_weight=dir_weights) + (1.0 - lam) * self._masked_ce_weighted(se_dir, se_dir_y_perm, complexity_sample_weights, class_weight=dir_weights)
                    else:
                        loss_b_dir = self._masked_ce_weighted(b_dir, b_dir_y, complexity_sample_weights, class_weight=dir_weights)
                        loss_se_dir = self._masked_ce_weighted(se_dir, se_dir_y, complexity_sample_weights, class_weight=dir_weights)
                else:
                    if mixup_perm is not None:
                        loss_b_dir = lam * self._masked_ce(b_dir, b_dir_y, weight=dir_weights) + (1.0 - lam) * self._masked_ce(b_dir, b_dir_y_perm, weight=dir_weights)
                        loss_se_dir = lam * self._masked_ce(se_dir, se_dir_y, weight=dir_weights) + (1.0 - lam) * self._masked_ce(se_dir, se_dir_y_perm, weight=dir_weights)
                    else:
                        loss_b_dir = self._masked_ce(b_dir, b_dir_y, weight=dir_weights)
                        loss_se_dir = self._masked_ce(se_dir, se_dir_y, weight=dir_weights)

                # Hierarchical direction losses (convert dir class to row/col/type)
                # This mapping is computed on-the-fly from the direction class
                b_dir_row_y, b_dir_col_y, b_dir_type_y = self._dir_to_hier(b_dir_y)
                se_dir_row_y, se_dir_col_y, se_dir_type_y = self._dir_to_hier(se_dir_y)

                if mixup_perm is not None:
                    b_dir_row_y_p, b_dir_col_y_p, b_dir_type_y_p = self._dir_to_hier(b_dir_y_perm)
                    se_dir_row_y_p, se_dir_col_y_p, se_dir_type_y_p = self._dir_to_hier(se_dir_y_perm)

                if getattr(self, 'use_boundary_complexity_weighting', False):
                    if mixup_perm is not None:
                        loss_b_hier = lam * (
                            self._masked_ce_weighted(b_dir_row, b_dir_row_y, complexity_sample_weights) +
                            self._masked_ce_weighted(b_dir_col, b_dir_col_y, complexity_sample_weights) +
                            self._masked_ce_weighted(b_dir_type, b_dir_type_y, complexity_sample_weights)
                        ) / 3.0 + (1.0 - lam) * (
                            self._masked_ce_weighted(b_dir_row, b_dir_row_y_p, complexity_sample_weights) +
                            self._masked_ce_weighted(b_dir_col, b_dir_col_y_p, complexity_sample_weights) +
                            self._masked_ce_weighted(b_dir_type, b_dir_type_y_p, complexity_sample_weights)
                        ) / 3.0
                        loss_se_hier = lam * (
                            self._masked_ce_weighted(se_dir_row, se_dir_row_y, complexity_sample_weights) +
                            self._masked_ce_weighted(se_dir_col, se_dir_col_y, complexity_sample_weights) +
                            self._masked_ce_weighted(se_dir_type, se_dir_type_y, complexity_sample_weights)
                        ) / 3.0 + (1.0 - lam) * (
                            self._masked_ce_weighted(se_dir_row, se_dir_row_y_p, complexity_sample_weights) +
                            self._masked_ce_weighted(se_dir_col, se_dir_col_y_p, complexity_sample_weights) +
                            self._masked_ce_weighted(se_dir_type, se_dir_type_y_p, complexity_sample_weights)
                        ) / 3.0
                    else:
                        # Also apply complexity weighting to hierarchical direction losses
                        loss_b_hier = (
                            self._masked_ce_weighted(b_dir_row, b_dir_row_y, complexity_sample_weights) +
                            self._masked_ce_weighted(b_dir_col, b_dir_col_y, complexity_sample_weights) +
                            self._masked_ce_weighted(b_dir_type, b_dir_type_y, complexity_sample_weights)
                        ) / 3.0
                        loss_se_hier = (
                            self._masked_ce_weighted(se_dir_row, se_dir_row_y, complexity_sample_weights) +
                            self._masked_ce_weighted(se_dir_col, se_dir_col_y, complexity_sample_weights) +
                            self._masked_ce_weighted(se_dir_type, se_dir_type_y, complexity_sample_weights)
                        ) / 3.0
                else:
                    if mixup_perm is not None:
                        loss_b_hier = lam * (
                            self._masked_ce(b_dir_row, b_dir_row_y) +
                            self._masked_ce(b_dir_col, b_dir_col_y) +
                            self._masked_ce(b_dir_type, b_dir_type_y)
                        ) / 3.0 + (1.0 - lam) * (
                            self._masked_ce(b_dir_row, b_dir_row_y_p) +
                            self._masked_ce(b_dir_col, b_dir_col_y_p) +
                            self._masked_ce(b_dir_type, b_dir_type_y_p)
                        ) / 3.0
                        loss_se_hier = lam * (
                            self._masked_ce(se_dir_row, se_dir_row_y) +
                            self._masked_ce(se_dir_col, se_dir_col_y) +
                            self._masked_ce(se_dir_type, se_dir_type_y)
                        ) / 3.0 + (1.0 - lam) * (
                            self._masked_ce(se_dir_row, se_dir_row_y_p) +
                            self._masked_ce(se_dir_col, se_dir_col_y_p) +
                            self._masked_ce(se_dir_type, se_dir_type_y_p)
                        ) / 3.0
                    else:
                        loss_b_hier = (
                            self._masked_ce(b_dir_row, b_dir_row_y) +
                            self._masked_ce(b_dir_col, b_dir_col_y) +
                            self._masked_ce(b_dir_type, b_dir_type_y)
                        ) / 3.0
                        loss_se_hier = (
                            self._masked_ce(se_dir_row, se_dir_row_y) +
                            self._masked_ce(se_dir_col, se_dir_col_y) +
                            self._masked_ce(se_dir_type, se_dir_type_y)
                        ) / 3.0

                # Combined direction loss: original + hierarchical
                loss_dir_combined = (loss_b_dir + loss_b_hier) + (loss_se_dir + loss_se_hier)

                # Aggregate per-task losses (for MT-CP and weighting)
                loss_present_total = loss_b_present + loss_se_present
                loss_mask_total = loss_b_mask + loss_se_mask

                # --- Phase 1: MT-CP Loss Prioritization ---
                if self._use_mt_cp and self.training:
                    # Track per-task losses
                    self._task_loss_history['present'].append(float(loss_present_total.detach()))
                    self._task_loss_history['mask'].append(float(loss_mask_total.detach()))
                    self._task_loss_history['dir'].append(float(loss_dir_combined.detach()))
                    self._mt_cp_step += 1

                    # Record initial losses (first batch)
                    if not self._task_initial_loss:
                        self._task_initial_loss = {
                            'present': max(float(loss_present_total.detach()), 1e-8),
                            'mask': max(float(loss_mask_total.detach()), 1e-8),
                            'dir': max(float(loss_dir_combined.detach()), 1e-8),
                        }

                    # Update weights every mt_cp_period steps
                    if self._mt_cp_step % self._mt_cp_period == 0 and len(self._task_loss_history['present']) >= self._mt_cp_period:
                        # Compute recent average loss per task
                        recent_n = self._mt_cp_period
                        avg_present = sum(self._task_loss_history['present'][-recent_n:]) / recent_n
                        avg_mask = sum(self._task_loss_history['mask'][-recent_n:]) / recent_n
                        avg_dir = sum(self._task_loss_history['dir'][-recent_n:]) / recent_n

                        # Ratio of current to initial loss (lower ratio = task is easier/more progressed)
                        ratios = torch.tensor([
                            avg_present / self._task_initial_loss['present'],
                            avg_mask / self._task_initial_loss['mask'],
                            avg_dir / self._task_initial_loss['dir'],
                        ], dtype=torch.float32)

                        # Softmax over ratios: harder tasks (higher ratio) get higher weight
                        weights = torch.softmax(ratios, dim=0) * 3.0  # scale so they sum to 3 (average weight = 1)
                        self._mt_cp_weights = {
                            'present': float(weights[0]),
                            'mask': float(weights[1]),
                            'dir': float(weights[2]),
                        }

                        # Trim history to avoid unbounded memory growth
                        max_hist = self._mt_cp_period * 10
                        for k in self._task_loss_history:
                            if len(self._task_loss_history[k]) > max_hist:
                                self._task_loss_history[k] = self._task_loss_history[k][-max_hist:]

                    w_p = self._mt_cp_weights['present']
                    w_m = self._mt_cp_weights['mask']
                    w_d = self._mt_cp_weights['dir']
                else:
                    w_p = float(getattr(self, "loss_weight_present", 1.0))
                    w_m = float(getattr(self, "loss_weight_mask", 1.0))
                    w_d = float(getattr(self, "loss_weight_dir", 1.0))

                w_c = float(getattr(self, "loss_weight_consistency", 0.0))

                # Consistency loss (only where mask label is known; negatives are known with mask=0).
                valid_bm = (b_mask_y != int(self.ignore_mask)).float()
                valid_sm = (se_mask_y != int(self.ignore_mask)).float()
                p_any_b = self._mask_any_prob_from_logits8(b_mask)
                p_any_s = self._mask_any_prob_from_logits8(se_mask)
                eps = 1e-6
                pb = torch.clamp(p_any_b, eps, 1 - eps)
                ps = torch.clamp(p_any_s, eps, 1 - eps)
                loss_cons_b = (-(b_present_y * torch.log(pb) + (1 - b_present_y) * torch.log(1 - pb)) * valid_bm).sum() / torch.clamp(valid_bm.sum(), min=1.0)
                loss_cons_s = (-(se_present_y * torch.log(ps) + (1 - se_present_y) * torch.log(1 - ps)) * valid_sm).sum() / torch.clamp(valid_sm.sum(), min=1.0)

                loss = (
                    w_p * loss_present_total + w_m * loss_mask_total +
                    w_d * loss_dir_combined +
                    w_c * (loss_cons_b + loss_cons_s)
                )

                # --- Phase 1: Feature Gate L1 regularization ---
                if self._use_feature_gate and self.feature_gate is not None:
                    l1_loss = self._feature_gate_l1 * self.feature_gate.abs().mean()
                    loss = loss + l1_loss

                # --- Phase 1: Distance-to-boundary weighting ---
                if self._use_dist_boundary_weight and dist_to_boundary is not None:
                    boundary_weight = 1.0 + self._dist_boundary_scale * torch.exp(-dist_to_boundary.float())
                    loss = loss * boundary_weight.mean()

            logits = (b_present, b_mask, b_dir, se_present, se_mask, se_dir)
            return {
                "loss": loss,
                "logits": logits,
                "logits_blend_present": b_present,
                "logits_blend_mask": b_mask,
                "logits_blend_dir": b_dir,
                "logits_blend_dir_row": b_dir_row,
                "logits_blend_dir_col": b_dir_col,
                "logits_blend_dir_type": b_dir_type,
                "logits_se_present": se_present,
                "logits_se_mask": se_mask,
                "logits_se_dir": se_dir,
            }
        
        def _dir_to_hier(self, dir_class: torch.Tensor) -> tuple:
            """
            Convert direction class to hierarchical (row, col, type).
            Row: 0=top, 1=mid, 2=bottom
            Col: 0=left, 1=mid, 2=right
            Type: 0=straight, 1=diagonal, 2=except
            
            Returns: (row_labels, col_labels, type_labels) - all same shape as input
            """
            device = dir_class.device
            # Direction value to (row, col, type) mapping
            # dir_values: [-1, 1, 2, 4, 8, 17, 18, 20, 24, 33, 34, 36, 40, 49, 50, 52, 56]
            # Index:        0  1  2  3  4   5   6   7   8   9  10  11  12  13  14  15  16
            hier_map = {
                0: (-100, -100, -100),  # Invalid
                1: (1, 0, 0),   # Left: mid row, left col, straight
                2: (2, 1, 0),   # Bottom: bottom row, mid col, straight
                3: (0, 2, 2),   # ExceptTopRight: top row, right col, except
                4: (0, 0, 2),   # ExceptTopLeft: top row, left col, except
                5: (1, 2, 0),   # Right: mid row, right col, straight
                6: (0, 1, 0),   # Top: top row, mid col, straight
                7: (2, 2, 2),   # ExceptBottomRight: bottom row, right col, except
                8: (2, 0, 2),   # ExceptBottomLeft: bottom row, left col, except
                9: (-100, -100, -100),  # Unknown (33)
                10: (-100, -100, -100), # Unknown (34)
                11: (2, 0, 1),  # BottomLeft: bottom row, left col, diagonal
                12: (2, 2, 1),  # BottomRight: bottom row, right col, diagonal
                13: (-100, -100, -100), # Unknown (49)
                14: (-100, -100, -100), # Unknown (50)
                15: (0, 0, 1),  # TopLeft: top row, left col, diagonal
                16: (0, 2, 1),  # TopRight: top row, right col, diagonal
            }
            
            # Create lookup tensors
            row_lut = torch.tensor([hier_map.get(i, (-100, -100, -100))[0] for i in range(17)], 
                                   dtype=torch.long, device=device)
            col_lut = torch.tensor([hier_map.get(i, (-100, -100, -100))[1] for i in range(17)], 
                                   dtype=torch.long, device=device)
            type_lut = torch.tensor([hier_map.get(i, (-100, -100, -100))[2] for i in range(17)], 
                                    dtype=torch.long, device=device)
            
            # Handle -100 (ignored) in input
            valid = (dir_class >= 0) & (dir_class < 17)
            clamped = torch.clamp(dir_class, 0, 16)
            
            row = torch.where(valid, row_lut[clamped], torch.tensor(-100, device=device))
            col = torch.where(valid, col_lut[clamped], torch.tensor(-100, device=device))
            typ = torch.where(valid, type_lut[clamped], torch.tensor(-100, device=device))
            
            return row, col, typ

    return TokenBlendModel()


def _build_palette(num_textures: int, seed: int, semantic: bool = False) -> np.ndarray:
    """
    Build RGB palette for texture visualization.
    
    If semantic=False (legacy): Random colors, loses texture relationships.
    If semantic=True: Encode texture semantics in RGB:
      - R = texture INDEX / num_textures (priority signal)
      - G = varies based on index (visual distinction)
      - B = complementary to provide contrast
    """
    if not semantic:
        rng = np.random.default_rng(seed)
        pal = rng.random((max(1, num_textures), 3), dtype=np.float32)
        pal[0] = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        return pal
    
    # Semantic palette: encode texture INDEX as primary signal
    # This lets the model "see" which textures have priority (lower index = higher priority)
    n = max(1, num_textures)
    pal = np.zeros((n, 3), dtype=np.float32)
    for i in range(n):
        # R = priority (lower index = lower R = higher priority in blending)
        pal[i, 0] = float(i) / n
        # G,B = provide visual distinction using golden ratio for even spacing
        golden = 0.618033988749895
        pal[i, 1] = (float(i) * golden) % 1.0
        pal[i, 2] = (float(i) * golden * 2) % 1.0
    return pal


def _make_collator(
    window: int,
    palette: np.ndarray,
    elev_mean: float,
    elev_std: float,
    model_name: str,
):
    import torch
    import torch.nn.functional as F
    from transformers import AutoImageProcessor

    # Prefer fast processor when available (noticeably reduces CPU overhead)
    try:
        processor = AutoImageProcessor.from_pretrained(model_name, use_fast=True)
    except TypeError:
        processor = AutoImageProcessor.from_pretrained(model_name)
    mean = torch.tensor(processor.image_mean, dtype=torch.float32).view(1, 3, 1, 1)
    std = torch.tensor(processor.image_std, dtype=torch.float32).view(1, 3, 1, 1)

    pal = torch.tensor(palette, dtype=torch.float32)  # [V,3]
    w = int(window)
    if w * w != 25:
        # current pipeline is window=5; keep it explicit for now
        raise ValueError(f"Expected window 5 (25 cells), got {w}")

    def collate(batch: List[Dict[str, object]]) -> Dict[str, torch.Tensor]:
        tex = torch.stack([torch.from_numpy(b["tex"]) for b in batch], dim=0).long()  # [B,25]
        elev = torch.stack([torch.from_numpy(b["elev"]) for b in batch], dim=0).float()  # [B,25]
        extra = None
        if "extra" in batch[0]:
            extra = torch.stack([torch.from_numpy(b["extra"]) for b in batch], dim=0).float()

        # palette lookup => RGB patch
        tex_clamped = torch.clamp(tex, 0, pal.shape[0] - 1)
        rgb = pal[tex_clamped]  # [B,25,3]
        rgb = rgb.view(-1, 5, 5, 3).permute(0, 3, 1, 2).contiguous()  # [B,3,5,5]

        # elevation modulation (brightness)
        elev_img = elev.view(-1, 1, 5, 5)
        z = (elev_img - float(elev_mean)) / float(elev_std)
        bright = torch.sigmoid(z)  # [0..1]
        rgb = rgb * (0.65 + 0.35 * bright)

        # upsample to model input size
        rgb = F.interpolate(rgb, size=(224, 224), mode="bilinear", align_corners=False)
        rgb = (rgb - mean) / std

        b_present = torch.tensor([b["labels_blend_present"] for b in batch], dtype=torch.long)
        b_sec = torch.tensor([b["labels_blend_sec"] for b in batch], dtype=torch.long)
        b_dir = torch.tensor([b["labels_blend_dir"] for b in batch], dtype=torch.long)

        se_present = torch.tensor([b["labels_se_present"] for b in batch], dtype=torch.long)
        se_sec = torch.tensor([b["labels_se_sec"] for b in batch], dtype=torch.long)
        se_dir = torch.tensor([b["labels_se_dir"] for b in batch], dtype=torch.long)

        # ignore secondary/dir when not present
        ignore = torch.full_like(b_sec, -100)
        b_sec = torch.where((b_present > 0.5) & (b_sec >= 0), b_sec, ignore)
        b_dir = torch.where((b_present > 0.5) & (b_dir >= 0), b_dir, ignore)
        se_sec = torch.where((se_present > 0.5) & (se_sec >= 0), se_sec, ignore)
        se_dir = torch.where((se_present > 0.5) & (se_dir >= 0), se_dir, ignore)

        labels = torch.stack([b_present, b_sec, b_dir, se_present, se_sec, se_dir], dim=1).long()  # [B,6]
        out = {
            "pixel_values": rgb,
            "labels": labels,
        }
        if extra is not None:
            out["extra_features"] = extra
        return out

    return collate


def _build_model(model_name: str, num_neighbor_classes: int, dir_num_classes: int, extra_dim: int):
    """
    Build multi-head model.
    
    num_neighbor_classes: 8 (which of the 8 neighbors is the secondary texture)
    dir_num_classes: ~17 (blend direction values)
    """
    import torch
    import torch.nn as nn
    from transformers import AutoConfig, AutoModel

    cfg = AutoConfig.from_pretrained(model_name)
    backbone = AutoModel.from_pretrained(model_name, config=cfg)

    hidden = int(getattr(cfg, "hidden_size", getattr(cfg, "dim", 768)))

    class MultiHead(nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = backbone
            self.extra_dim = int(extra_dim)
            self.extra_proj = None
            if self.extra_dim > 0:
                self.extra_proj = nn.Linear(self.extra_dim, hidden)
            self.blend_present = nn.Linear(hidden, 1)
            # CHANGED: predict which of 8 neighbors, not which of 363 textures!
            self.blend_sec = nn.Linear(hidden, int(num_neighbor_classes))
            self.blend_dir = nn.Linear(hidden, int(dir_num_classes))

            self.se_present = nn.Linear(hidden, 1)
            self.se_sec = nn.Linear(hidden, int(num_neighbor_classes))
            self.se_dir = nn.Linear(hidden, int(dir_num_classes))

        @staticmethod
        def _masked_ce(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
            """
            CrossEntropy over only valid targets (target != -100).
            Avoids NaNs that can happen when an eval batch has zero valid items.
            """
            mask = target != -100
            if not torch.any(mask):
                return logits.new_zeros(())
            return nn.functional.cross_entropy(logits[mask], target[mask], reduction="mean")

        @staticmethod
        def _focal_loss(logits: torch.Tensor, targets: torch.Tensor, gamma: float = 2.0, alpha: float = 0.25) -> torch.Tensor:
            """
            Focal Loss for binary classification.
            FL(p_t) = -alpha_t * (1-p_t)^gamma * log(p_t)
            
            This focuses learning on hard examples and handles class imbalance better than pos_weight.
            - gamma: focusing parameter (2.0 is common). Higher = more focus on hard examples.
            - alpha: weight for positive class (0.25 means we weight negatives 3x more, improving precision)
            """
            probs = torch.sigmoid(logits)
            # p_t = p for y=1, (1-p) for y=0
            p_t = probs * targets + (1 - probs) * (1 - targets)
            # alpha_t = alpha for y=1, (1-alpha) for y=0
            alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
            # focal weight
            focal_weight = alpha_t * ((1 - p_t) ** gamma)
            # BCE loss per element
            bce = nn.functional.binary_cross_entropy_with_logits(logits, targets, reduction="none")
            return (focal_weight * bce).mean()

        def forward(
            self,
            pixel_values: torch.Tensor,
            extra_features: Optional[torch.Tensor] = None,
            labels: Optional[torch.Tensor] = None,
        ):
            out = self.backbone(pixel_values=pixel_values)
            pooled = getattr(out, "pooler_output", None)
            if pooled is None:
                pooled = out.last_hidden_state[:, 0]
            if self.extra_proj is not None and extra_features is not None:
                pooled = pooled + self.extra_proj(extra_features)
            b_present = self.blend_present(pooled).squeeze(-1)  # [B]
            b_sec = self.blend_sec(pooled)  # [B,V]
            b_dir = self.blend_dir(pooled)  # [B,D]

            se_present = self.se_present(pooled).squeeze(-1)  # [B]
            se_sec = self.se_sec(pooled)  # [B,V]
            se_dir = self.se_dir(pooled)  # [B,D]

            loss = None
            if labels is not None:
                # labels: [B,6] = (b_present, b_sec, b_dir, se_present, se_sec, se_dir)
                b_present_y = labels[:, 0].float()
                b_sec_y = labels[:, 1].long()
                b_dir_y = labels[:, 2].long()
                se_present_y = labels[:, 3].float()
                se_sec_y = labels[:, 4].long()
                se_dir_y = labels[:, 5].long()

                # Loss for blend_present / se_present
                # Options: BCE with pos_weight, Focal Loss, or BCE with separate FP/FN penalties
                use_focal = getattr(self, "use_focal_loss", False)
                focal_gamma = getattr(self, "focal_gamma", 2.0)
                focal_alpha = getattr(self, "focal_alpha", 0.25)  # Weight for positive class
                
                if use_focal:
                    # Focal Loss: FL(p_t) = -alpha_t * (1-p_t)^gamma * log(p_t)
                    # Better for class imbalance, focuses on hard examples
                    loss_b_present = self._focal_loss(b_present, b_present_y, focal_gamma, focal_alpha)
                    loss_se_present = self._focal_loss(se_present, se_present_y, focal_gamma, focal_alpha)
                else:
                    # Pos-weighted BCE (legacy, tends to over-predict)
                    pw_b = getattr(self, "pos_weight_blend_present", None)
                    pw_s = getattr(self, "pos_weight_se_present", None)
                    pw_b_t = None
                    pw_s_t = None
                    if pw_b is not None:
                        pw_b_t = torch.tensor(float(pw_b), device=b_present.device, dtype=b_present.dtype)
                    if pw_s is not None:
                        pw_s_t = torch.tensor(float(pw_s), device=se_present.device, dtype=se_present.dtype)

                    bce_b = nn.BCEWithLogitsLoss(pos_weight=pw_b_t) if pw_b_t is not None else nn.BCEWithLogitsLoss()
                    bce_s = nn.BCEWithLogitsLoss(pos_weight=pw_s_t) if pw_s_t is not None else nn.BCEWithLogitsLoss()
                    loss_b_present = bce_b(b_present, b_present_y)
                    loss_se_present = bce_s(se_present, se_present_y)

                loss_b_sec = self._masked_ce(b_sec, b_sec_y)
                loss_b_dir = self._masked_ce(b_dir, b_dir_y)

                loss_se_sec = self._masked_ce(se_sec, se_sec_y)
                loss_se_dir = self._masked_ce(se_dir, se_dir_y)

                # Weighted sum of losses - presence matters most, direction/neighbor secondary
                w_present = getattr(self, "loss_weight_present", 1.0)
                w_sec = getattr(self, "loss_weight_sec", 1.0)
                w_dir = getattr(self, "loss_weight_dir", 1.0)
                
                loss = (
                    w_present * loss_b_present + w_sec * loss_b_sec + w_dir * loss_b_dir +
                    w_present * loss_se_present + w_sec * loss_se_sec + w_dir * loss_se_dir
                )

            logits = (b_present, b_sec, b_dir, se_present, se_sec, se_dir)
            return {
                "loss": loss,
                "logits": logits,
                "logits_blend_present": b_present,
                "logits_blend_sec": b_sec,
                "logits_blend_dir": b_dir,
                "logits_se_present": se_present,
                "logits_se_sec": se_sec,
                "logits_se_dir": se_dir,
            }

    return MultiHead()


def main() -> int:
    _require_torch_and_hf()
    import torch
    from transformers import Trainer, TrainingArguments

    import inspect

    ap = argparse.ArgumentParser(description="Finetune pretrained ViT for RA3 blend prediction (multi-head).")
    ap.add_argument("--data-dir", required=True, help="Prepared dataset dir (output of prepare_blend_dataset_memmap.py)")
    ap.add_argument("--out-dir", required=True, help="Output dir for HF Trainer artifacts")
    ap.add_argument("--model", default="facebook/deit-tiny-patch16-224", help="HF model id (used only for --arch vit)")
    ap.add_argument(
        "--arch",
        default="token",
        choices=["token", "vit"],
        help=(
            "Model architecture. "
            "token (recommended): small Transformer over discrete 5x5 texture IDs (best for local rules/pair dominance). "
            "vit: ViT over RGB palette image (legacy)."
        ),
    )
    ap.add_argument("--token-hidden", type=int, default=256, help="Token transformer hidden size")
    ap.add_argument("--token-layers", type=int, default=4, help="Token transformer layer count")
    ap.add_argument("--token-heads", type=int, default=8, help="Token transformer attention heads")
    ap.add_argument("--token-dropout", type=float, default=0.10, help="Token transformer dropout")
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--val-frac", type=float, default=0.1, help="Fraction of maps reserved for validation")
    ap.add_argument("--max-train-samples", type=int, default=2_000_000, help="Subsample training examples")
    ap.add_argument("--max-eval-samples", type=int, default=200_000, help="Subsample eval examples")
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--grad-accum-steps", type=int, default=1, help="Gradient accumulation to simulate larger batch sizes.")
    ap.add_argument("--num-workers", type=int, default=0, help="Dataloader worker processes (Windows: start with 0 or 2).")
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--fp16", action="store_true")
    ap.add_argument("--require-cuda", action="store_true", help="Fail fast if CUDA is not available (recommended).")
    ap.add_argument("--max-steps", type=int, default=-1, help="Optional hard cap on steps (for smoke tests)")
    # Default cadence: keep evaluation/checkpoints relatively sparse (10x less frequent than the early prototype defaults).
    ap.add_argument("--eval-steps", type=int, default=2000)
    ap.add_argument("--save-steps", type=int, default=2000)
    ap.add_argument("--logging-steps", type=int, default=200)
    ap.add_argument(
        "--train-blend-pos-frac",
        type=float,
        default=0.00,
        help="Fraction of training indices forced to have blend_present=1 (stabilizes training on sparse positives).",
    )
    ap.add_argument(
        "--train-se-pos-frac",
        type=float,
        default=0.00,
        help="Fraction of training indices forced to have single_edge_present=1 (very sparse; prevents collapse).",
    )
    ap.add_argument(
        "--auto-pos-weight",
        action="store_true",
        help="Use pos-weighted BCE for present heads based on sampled training distribution.",
    )
    ap.add_argument(
        "--pos-weight-scale-blend",
        type=float,
        default=1.0,
        help="Scale factor applied to auto pos_weight for blend_present (lower => higher precision, higher => higher recall).",
    )
    ap.add_argument(
        "--pos-weight-scale-se",
        type=float,
        default=1.0,
        help="Scale factor applied to auto pos_weight for se_present.",
    )
    ap.add_argument(
        "--use-focal-loss",
        action="store_true",
        help="Use Focal Loss instead of pos-weighted BCE for blend_present/se_present. Better for precision.",
    )
    ap.add_argument(
        "--focal-gamma",
        type=float,
        default=2.0,
        help="Focal Loss gamma parameter. Higher = more focus on hard examples.",
    )
    ap.add_argument(
        "--focal-alpha",
        type=float,
        default=0.25,
        help="Focal Loss alpha (positive class weight). 0.25 optimal for ~8%% positive rate. Lower = prioritize precision over recall.",
    )
    ap.add_argument(
        "--loss-weight-present",
        type=float,
        default=1.0,
        help="Weight for blend_present/se_present loss terms.",
    )
    ap.add_argument(
        "--loss-weight-sec",
        type=float,
        default=1.0,
        help="Weight for secondary-structure loss terms (neighbor-mask in token arch).",
    )
    ap.add_argument(
        "--loss-weight-dir",
        type=float,
        default=1.0,
        help="Weight for blend_dir/se_dir (direction) loss terms.",
    )
    ap.add_argument(
        "--loss-weight-consistency",
        type=float,
        default=0.5,
        help="Aux loss weight tying present to predicted mask-any (token arch). Higher => fewer false positives.",
    )
    ap.add_argument(
        "--boundary-complexity-weighting",
        action="store_true",
        help=(
            "Weight direction loss by boundary complexity (number of unique neighbor textures). "
            "Cells with more different neighbors get higher weight. Addresses errors at complex junctions."
        ),
    )
    ap.add_argument(
        "--complexity-weight-1-2",
        type=float,
        default=1.0,
        help="Direction loss weight for cells with 1-2 unique neighbor textures.",
    )
    ap.add_argument(
        "--complexity-weight-3-4",
        type=float,
        default=1.5,
        help="Direction loss weight for cells with 3-4 unique neighbor textures.",
    )
    ap.add_argument(
        "--complexity-weight-5-6",
        type=float,
        default=2.0,
        help="Direction loss weight for cells with 5-6 unique neighbor textures.",
    )
    ap.add_argument(
        "--complexity-weight-7-8",
        type=float,
        default=3.0,
        help="Direction loss weight for cells with 7-8 unique neighbor textures (maximum complexity).",
    )
    ap.add_argument(
        "--semantic-palette",
        action="store_true",
        help="Use semantic RGB palette (encodes texture priority) instead of random colors.",
    )
    ap.add_argument(
        "--present-threshold",
        type=float,
        default=0.5,
        help="Decision threshold for present metrics (precision/recall/F1).",
    )
    ap.add_argument(
        "--report-best-threshold",
        action="store_true",
        help="Also compute best-F1 threshold over a small grid and report it (metrics only).",
    )
    ap.add_argument(
        "--resume-from",
        default="",
        help="Resume training from a checkpoint dir. If empty, will auto-resume from the latest checkpoint in --out-dir (if any).",
    )
    ap.add_argument(
        "--no-force-intervals-on-resume",
        action="store_true",
        help=(
            "If set, do NOT overwrite checkpoint-loaded eval/save/logging intervals. "
            "Default behavior is to force the CLI/default intervals even when resuming."
        ),
    )
    ap.add_argument(
        "--early-stop",
        action="store_true",
        help="Enable early stopping based on an eval metric plateau (step-based alternative to 'epochs without improvement').",
    )
    ap.add_argument(
        "--early-stop-metric",
        default="eval_blend_neighbor_acc",
        help="Which eval metric to monitor for early stopping (must exist in eval metrics dict).",
    )
    ap.add_argument(
        "--early-stop-greater-is-better",
        action="store_true",
        help="If set, higher metric is better. If not set, lower is better (e.g. loss).",
    )
    ap.add_argument(
        "--early-stop-patience-evals",
        type=int,
        default=5,
        help="Stop after this many consecutive evals without meaningful improvement.",
    )
    ap.add_argument(
        "--early-stop-min-delta",
        type=float,
        default=0.002,
        help="Minimum absolute improvement required to reset patience.",
    )
    # --- Phase 1 improvement flags ---
    ap.add_argument(
        "--use-asl",
        action="store_true",
        help="Use Asymmetric Loss (ASL) for the 8-bit neighbor mask instead of BCE. Better for imbalanced multi-label.",
    )
    ap.add_argument("--asl-gamma-neg", type=float, default=4.0, help="ASL gamma for negative samples (higher = more suppression of easy negatives).")
    ap.add_argument("--asl-gamma-pos", type=float, default=0.0, help="ASL gamma for positive samples.")
    ap.add_argument("--asl-clip", type=float, default=0.05, help="ASL probability clipping for negatives.")
    ap.add_argument(
        "--use-logit-adj",
        action="store_true",
        help="Apply logit adjustment to direction head using class priors (long-tail correction).",
    )
    ap.add_argument("--logit-adj-tau", type=float, default=1.0, help="Temperature for logit adjustment (tau * log(prior)).")
    ap.add_argument(
        "--use-cascaded-heads",
        action="store_true",
        help="Gate mask and direction logits by present probability (cascaded output heads).",
    )
    ap.add_argument(
        "--mixup-alpha",
        type=float,
        default=0.0,
        help="Alpha for Beta distribution in embedding-space MixUp. 0 disables. Recommended: 0.2.",
    )
    ap.add_argument(
        "--use-mt-cp",
        action="store_true",
        help="Use MT-CP automatic multi-task loss balancing (replaces manual loss weights).",
    )
    ap.add_argument("--mt-cp-period", type=int, default=100, help="Steps between MT-CP weight updates.")
    ap.add_argument(
        "--use-feature-gate",
        action="store_true",
        help="Add learnable per-feature gate with L1 regularization to extra features.",
    )
    ap.add_argument("--feature-gate-l1", type=float, default=0.001, help="L1 regularization coefficient for feature gate.")
    ap.add_argument(
        "--use-dist-boundary-weight",
        action="store_true",
        help="Weight loss by distance-to-boundary (cells near texture boundaries get higher weight).",
    )
    ap.add_argument("--dist-boundary-scale", type=float, default=8.0, help="Scale factor for boundary distance weighting: w = 1 + scale * exp(-dist).")

    args = ap.parse_args()

    cuda_ok = bool(torch.cuda.is_available())
    print(f"[device] torch={torch.__version__} cuda_available={cuda_ok} cuda_version={torch.version.cuda}")
    if cuda_ok:
        try:
            print(f"[device] cuda_device0={torch.cuda.get_device_name(0)}")
        except Exception:
            pass
        # Ampere+ perf tweaks
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass
        try:
            torch.backends.cudnn.benchmark = True
        except Exception:
            pass

    if args.fp16 and not cuda_ok:
        raise SystemExit(
            "You requested --fp16 but CUDA is not available.\n"
            "Your current torch build appears to be CPU-only.\n"
            "Install CUDA-enabled torch first (example):\n"
            "  pip install --upgrade --force-reinstall torch torchvision --index-url https://download.pytorch.org/whl/cu124\n"
        )
    if args.require_cuda and not cuda_ok:
        raise SystemExit(
            "CUDA is not available but --require-cuda was set.\n"
            "Install CUDA-enabled torch first (example):\n"
            "  pip install --upgrade --force-reinstall torch torchvision --index-url https://download.pytorch.org/whl/cu124\n"
        )

    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    meta = _load_prepared_meta(data_dir)
    if meta.window != 5:
        raise SystemExit(f"Currently only supports window=5, got window={meta.window}")

    # load memmaps
    tex = _memmap_load(data_dir, "tex.npy", np.int32)  # [N,25] global ids
    tex_local_norm = None
    tln_path = data_dir / "tex_local_norm.npy"
    if tln_path.exists():
        tex_local_norm = _memmap_load(data_dir, "tex_local_norm.npy", np.float32)
    elev = _memmap_load(data_dir, "elev.npy", np.float32)  # [N,25]
    extra = None
    extra_path = data_dir / "extra.npy"
    if extra_path.exists():
        extra = _memmap_load(data_dir, "extra.npy", np.float32)
    ybp = _memmap_load(data_dir, "y_blend_present.npy", np.uint8)
    ybm = None
    ybm_path = data_dir / "y_blend_mask.npy"
    if ybm_path.exists():
        ybm = _memmap_load(data_dir, "y_blend_mask.npy", np.uint8)
    ybs = _memmap_load(data_dir, "y_blend_sec.npy", np.int32)  # legacy
    ybd = _memmap_load(data_dir, "y_blend_dir.npy", np.int16)
    ysp = _memmap_load(data_dir, "y_se_present.npy", np.uint8)
    ysm = None
    ysm_path = data_dir / "y_se_mask.npy"
    if ysm_path.exists():
        ysm = _memmap_load(data_dir, "y_se_mask.npy", np.uint8)
    yss = _memmap_load(data_dir, "y_se_sec.npy", np.int32)  # legacy
    ysd = _memmap_load(data_dir, "y_se_dir.npy", np.int16)
    map_id = _memmap_load(data_dir, "map_id.npy", np.int32)
    map_style = None
    map_style_path = data_dir / "map_style.npy"
    if map_style_path.exists():
        map_style = _memmap_load(data_dir, "map_style.npy", np.float32)

    if str(args.arch).lower() == "token":
        if ybm is None or ysm is None:
            raise SystemExit(
                "Token architecture requires mask labels (y_blend_mask.npy and y_se_mask.npy).\n"
                "Re-run dataset generation + prepare steps using the updated scripts to produce these files."
            )

    num_maps = int(map_id.max()) + 1
    train_maps, val_maps = _split_maps(num_maps, val_frac=float(args.val_frac), seed=int(args.seed))
    # Positive-aware sampling for training so each batch sees positives (especially single-edge which is ~1%).
    n_train = int(args.max_train_samples)
    blend_pos = int(round(n_train * float(args.train_blend_pos_frac)))
    se_pos = int(round(n_train * float(args.train_se_pos_frac)))
    blend_pos = max(0, min(blend_pos, n_train))
    se_pos = max(0, min(se_pos, n_train - blend_pos))
    uni = n_train - blend_pos - se_pos

    def _blend_pos_mask(cand: np.ndarray) -> np.ndarray:
        return (ybp[cand] != 0)

    def _se_pos_mask(cand: np.ndarray) -> np.ndarray:
        return (ysp[cand] != 0)

    parts: List[np.ndarray] = []
    if uni > 0:
        parts.append(_sample_indices_for_map_ids(map_id, train_maps, uni, seed=int(args.seed)))
    if blend_pos > 0:
        parts.append(
            _sample_indices_for_map_ids_filtered(
                map_id=map_id,
                allowed=train_maps,
                n=blend_pos,
                seed=int(args.seed) + 101,
                extra_mask_fn=_blend_pos_mask,
            )
        )
    if se_pos > 0:
        parts.append(
            _sample_indices_for_map_ids_filtered(
                map_id=map_id,
                allowed=train_maps,
                n=se_pos,
                seed=int(args.seed) + 202,
                extra_mask_fn=_se_pos_mask,
            )
        )
    train_idx = np.concatenate(parts, axis=0) if len(parts) > 1 else parts[0]
    rng = np.random.default_rng(int(args.seed))
    rng.shuffle(train_idx)
    val_idx = _sample_indices_for_map_ids(map_id, val_maps, int(args.max_eval_samples), seed=int(args.seed) + 999)

    train_ds = BlendDataset(tex, tex_local_norm, elev, extra, ybp, ybm, ybs, ybd, ysp, ysm, yss, ysd, map_style, train_idx)
    val_ds = BlendDataset(tex, tex_local_norm, elev, extra, ybp, ybm, ybs, ybd, ysp, ysm, yss, ysd, map_style, val_idx)

    # --- Phase 1: Determine dist_to_boundary extra feature index ---
    dist_boundary_extra_idx = -1
    if meta.extra_feature_names:
        for i, name in enumerate(meta.extra_feature_names):
            if name == "dist_to_boundary":
                dist_boundary_extra_idx = i
                break
    if bool(args.use_dist_boundary_weight) and dist_boundary_extra_idx < 0:
        print("[warn] --use-dist-boundary-weight requested but 'dist_to_boundary' not found in extra features. Disabling.")
        args.use_dist_boundary_weight = False
    elif bool(args.use_dist_boundary_weight):
        print(f"[phase1] dist_to_boundary found at extra feature index {dist_boundary_extra_idx}")

    # --- Phase 1: Compute direction class priors for logit adjustment ---
    dir_class_prior_tensor = None
    if bool(args.use_logit_adj):
        # Compute class prior from training data direction labels
        import torch as _torch_tmp
        dir_counts = np.zeros(int(meta.dir_num_classes), dtype=np.float64)
        sampled_dirs = np.asarray(ybd[train_idx], dtype=np.int64)
        valid_dirs = sampled_dirs[sampled_dirs >= 0]
        valid_dirs = valid_dirs[valid_dirs < int(meta.dir_num_classes)]
        for d in valid_dirs:
            dir_counts[d] += 1.0
        total = max(dir_counts.sum(), 1.0)
        dir_class_prior_np = (dir_counts / total).astype(np.float32)
        dir_class_prior_tensor = _torch_tmp.tensor(dir_class_prior_np, dtype=_torch_tmp.float32)
        print(f"[phase1] Logit adjustment: tau={args.logit_adj_tau}, dir priors computed from {int(total)} samples")

    # Collator + model selection
    if str(args.arch).lower() == "token":
        print("[arch] Using token-transformer (recommended)")
        collate = _make_token_collator(
            window=meta.window,
            elev_mean=meta.elev_mean,
            elev_std=meta.elev_std,
            mask_ignore_value=int(meta.mask_ignore_value),
            map_style_dim=int(meta.map_style_dim),
            dist_boundary_extra_idx=dist_boundary_extra_idx if bool(args.use_dist_boundary_weight) else -1,
        )
        model = _build_token_model(
            num_textures=int(meta.num_textures),
            dir_num_classes=int(meta.dir_num_classes),
            extra_dim=(int(extra.shape[1]) if extra is not None and getattr(extra, "ndim", 0) == 2 else int(meta.extra_dim)),
            map_style_dim=int(meta.map_style_dim),
            mask_ignore_value=int(meta.mask_ignore_value),
            hidden=int(args.token_hidden),
            n_layers=int(args.token_layers),
            n_heads=int(args.token_heads),
            dropout=float(args.token_dropout),
            # Phase 1 improvements
            use_asl=bool(args.use_asl),
            asl_gamma_neg=float(args.asl_gamma_neg),
            asl_gamma_pos=float(args.asl_gamma_pos),
            asl_clip=float(args.asl_clip),
            use_logit_adj=bool(args.use_logit_adj),
            dir_class_prior=dir_class_prior_tensor,
            logit_adj_tau=float(args.logit_adj_tau),
            use_cascaded_heads=bool(args.use_cascaded_heads),
            mixup_alpha=float(args.mixup_alpha),
            use_mt_cp=bool(args.use_mt_cp),
            mt_cp_period=int(args.mt_cp_period),
            use_feature_gate=bool(args.use_feature_gate),
            feature_gate_l1=float(args.feature_gate_l1),
            use_dist_boundary_weight=bool(args.use_dist_boundary_weight),
            dist_boundary_scale=float(args.dist_boundary_scale),
        )
    else:
        print("[arch] Using ViT RGB palette (legacy)")
        palette = _build_palette(meta.num_textures, seed=int(args.seed), semantic=bool(args.semantic_palette))
        if bool(args.semantic_palette):
            print("[palette] Using semantic RGB palette (encodes texture priority)")
        collate = _make_collator(
            window=meta.window,
            palette=palette,
            elev_mean=meta.elev_mean,
            elev_std=meta.elev_std,
            model_name=str(args.model),
        )
        model = _build_model(
            str(args.model),
            num_neighbor_classes=meta.num_neighbor_classes,
            dir_num_classes=meta.dir_num_classes,
            extra_dim=(int(extra.shape[1]) if extra is not None and getattr(extra, "ndim", 0) == 2 else int(meta.extra_dim)),
        )
    # configure class-imbalance handling (pos_weight) on the model
    if bool(args.auto_pos_weight):
        # compute on sampled train distribution (cheap + matches training sampling)
        b = np.asarray(ybp[train_idx], dtype=np.int64)
        s = np.asarray(ysp[train_idx], dtype=np.int64)
        b_pos = float(np.sum(b != 0))
        s_pos = float(np.sum(s != 0))
        n = float(b.shape[0])
        b_neg = n - b_pos
        s_neg = n - s_pos
        # avoid div by zero
        b_w = float(b_neg / max(1.0, b_pos))
        s_w = float(s_neg / max(1.0, s_pos))
        b_w *= float(args.pos_weight_scale_blend)
        s_w *= float(args.pos_weight_scale_se)
        print(
            f"[loss] auto pos_weight blend_present={b_w:.3f} (pos_rate={b_pos/max(1.0,n):.4f}, scale={float(args.pos_weight_scale_blend):.3f})"
        )
        print(
            f"[loss] auto pos_weight se_present={s_w:.3f} (pos_rate={s_pos/max(1.0,n):.4f}, scale={float(args.pos_weight_scale_se):.3f})"
        )
        try:
            model.pos_weight_blend_present = float(b_w)  # type: ignore[attr-defined]
            model.pos_weight_se_present = float(s_w)  # type: ignore[attr-defined]
        except Exception:
            pass

    # Configure Focal Loss if requested (overrides pos_weight)
    if bool(args.use_focal_loss):
        print(f"[loss] Using Focal Loss with gamma={args.focal_gamma}, alpha={args.focal_alpha}")
        print(f"       (alpha={args.focal_alpha} means positive class weighted {args.focal_alpha:.2f}, negative class weighted {1-args.focal_alpha:.2f})")
        try:
            model.use_focal_loss = True  # type: ignore[attr-defined]
            model.focal_gamma = float(args.focal_gamma)  # type: ignore[attr-defined]
            model.focal_alpha = float(args.focal_alpha)  # type: ignore[attr-defined]
        except Exception:
            pass

    # Configure loss weights
    if args.loss_weight_present != 1.0 or args.loss_weight_sec != 1.0 or args.loss_weight_dir != 1.0:
        print(f"[loss] Loss weights: present={args.loss_weight_present}, sec={args.loss_weight_sec}, dir={args.loss_weight_dir}")
        try:
            model.loss_weight_present = float(args.loss_weight_present)  # type: ignore[attr-defined]
            model.loss_weight_sec = float(args.loss_weight_sec)  # type: ignore[attr-defined]
            # token arch uses loss_weight_mask instead of loss_weight_sec
            model.loss_weight_mask = float(args.loss_weight_sec)  # type: ignore[attr-defined]
            model.loss_weight_dir = float(args.loss_weight_dir)  # type: ignore[attr-defined]
        except Exception:
            pass

    # Configure present/mask consistency weight (token arch)
    if float(getattr(args, "loss_weight_consistency", 0.0)) != 0.5:
        print(f"[loss] Consistency weight: {float(args.loss_weight_consistency)}")
    try:
        model.loss_weight_consistency = float(args.loss_weight_consistency)  # type: ignore[attr-defined]
    except Exception:
        pass

    # Configure boundary complexity weighting for direction loss (token arch only)
    if bool(args.boundary_complexity_weighting) and str(args.arch).lower() == "token":
        print(f"[loss] Boundary complexity weighting ENABLED")
        print(f"       Weights: 1-2 neighbors={args.complexity_weight_1_2}, "
              f"3-4={args.complexity_weight_3_4}, 5-6={args.complexity_weight_5_6}, "
              f"7-8={args.complexity_weight_7_8}")
        try:
            model.use_boundary_complexity_weighting = True  # type: ignore[attr-defined]
            # Update the complexity weights buffer
            import torch
            new_weights = torch.tensor([
                float(args.complexity_weight_1_2),
                float(args.complexity_weight_3_4),
                float(args.complexity_weight_5_6),
                float(args.complexity_weight_7_8),
            ], dtype=torch.float32)
            model.register_buffer('complexity_weights', new_weights)
        except Exception as e:
            print(f"[warn] Failed to configure complexity weighting: {e}")
    elif bool(args.boundary_complexity_weighting) and str(args.arch).lower() != "token":
        print(f"[warn] Boundary complexity weighting only supported for token architecture, ignoring.")

    # --- Phase 1: Log enabled improvements ---
    if str(args.arch).lower() == "token":
        phase1_features = []
        if bool(args.use_asl):
            phase1_features.append(f"ASL(gamma_neg={args.asl_gamma_neg}, gamma_pos={args.asl_gamma_pos}, clip={args.asl_clip})")
        if bool(args.use_logit_adj):
            phase1_features.append(f"LogitAdj(tau={args.logit_adj_tau})")
        if bool(args.use_cascaded_heads):
            phase1_features.append("CascadedHeads")
        if float(args.mixup_alpha) > 0:
            phase1_features.append(f"MixUp(alpha={args.mixup_alpha})")
        if bool(args.use_mt_cp):
            phase1_features.append(f"MT-CP(period={args.mt_cp_period})")
        if bool(args.use_feature_gate):
            phase1_features.append(f"FeatureGate(L1={args.feature_gate_l1})")
        if bool(args.use_dist_boundary_weight):
            phase1_features.append(f"DistBoundaryWeight(scale={args.dist_boundary_scale})")
        if phase1_features:
            print(f"[phase1] Enabled improvements: {', '.join(phase1_features)}")
        else:
            print("[phase1] No Phase 1 improvements enabled (all disabled by default).")

    # TrainingArguments
    steps_per_epoch = math.ceil(len(train_ds) / max(1, int(args.batch_size)))
    warmup = min(1000, max(100, steps_per_epoch // 10))
    # Transformers has renamed some TrainingArguments fields across versions
    # (e.g., evaluation_strategy -> eval_strategy). Build args defensively.
    ta_kwargs = {
        "output_dir": str(out_dir),
        "per_device_train_batch_size": int(args.batch_size),
        "per_device_eval_batch_size": int(args.batch_size),
        "gradient_accumulation_steps": int(args.grad_accum_steps),
        "dataloader_num_workers": int(args.num_workers),
        "learning_rate": float(args.lr),
        "num_train_epochs": float(args.epochs),
        "warmup_steps": int(warmup),
        # prefer new name but keep old; we'll filter to signature below
        "evaluation_strategy": "steps",
        "eval_strategy": "steps",
        "eval_steps": int(args.eval_steps),
        "save_steps": int(args.save_steps),
        "logging_steps": int(args.logging_steps),
        "fp16": bool(args.fp16),
        "seed": int(args.seed),
        "report_to": [],
        "remove_unused_columns": False,
        "max_steps": int(args.max_steps),
        "disable_tqdm": False,
    }
    # Ensure Trainer packs multiple labels in a stable order (if supported by this transformers version)
    ta_sig = inspect.signature(TrainingArguments.__init__)
    ta_kwargs = {k: v for k, v in ta_kwargs.items() if k in ta_sig.parameters}
    targs = TrainingArguments(**ta_kwargs)

    def _safe_div(a: float, b: float) -> float:
        return float(a) / float(b) if float(b) != 0.0 else 0.0

    def _bin_metrics_from_logits(logits_1d: np.ndarray, y_true01: np.ndarray, thresh: float) -> Dict[str, float]:
        # logits_1d: [N]
        p = 1.0 / (1.0 + np.exp(-logits_1d.astype(np.float64)))
        y_pred = (p >= float(thresh)).astype(np.int64)
        y_true = y_true01.astype(np.int64)

        tp = int(np.sum((y_pred == 1) & (y_true == 1)))
        tn = int(np.sum((y_pred == 0) & (y_true == 0)))
        fp = int(np.sum((y_pred == 1) & (y_true == 0)))
        fn = int(np.sum((y_pred == 0) & (y_true == 1)))
        acc = _safe_div(tp + tn, tp + tn + fp + fn)
        prec = _safe_div(tp, tp + fp)
        rec = _safe_div(tp, tp + fn)
        f1 = _safe_div(2 * prec * rec, prec + rec)
        return {
            "acc": float(acc),
            "prec": float(prec),
            "rec": float(rec),
            "f1": float(f1),
            "pos_rate": float(np.mean(y_true)),
        }

    def _best_f1_threshold(logits_1d: np.ndarray, y_true01: np.ndarray) -> Tuple[float, float]:
        # small grid search; cheap and stable for reporting
        grid = np.linspace(0.1, 0.9, num=17, dtype=np.float64)
        best_t = 0.5
        best_f = -1.0
        for t in grid:
            f = _bin_metrics_from_logits(logits_1d, y_true01, float(t))["f1"]
            if f > best_f:
                best_f = float(f)
                best_t = float(t)
        return best_t, best_f

    def _masked_topk_acc(logits: np.ndarray, y: np.ndarray, k: int) -> float:
        # logits: [N,C], y: [N] with -100 masked
        mask = y != -100
        if not np.any(mask):
            return 0.0
        logits_m = logits[mask]
        y_m = y[mask].astype(np.int64)
        # top-k via partial sort (faster than full argsort)
        topk = np.argpartition(-logits_m, kth=min(k - 1, logits_m.shape[1] - 1), axis=1)[:, :k]
        hit = np.any(topk == y_m[:, None], axis=1)
        return float(np.mean(hit))

    def _masked_acc(logits: np.ndarray, y: np.ndarray) -> float:
        mask = y != -100
        if not np.any(mask):
            return 0.0
        pred = np.argmax(logits[mask], axis=1).astype(np.int64)
        y_m = y[mask].astype(np.int64)
        return float(np.mean(pred == y_m))

    def _mask8_metrics(logits8: np.ndarray, mask_u8: np.ndarray, ignore_value: int) -> Dict[str, float]:
        """
        logits8: [N,8], mask_u8: [N] uint8 (0..255), 255=ignore.
        Computes:
          - exact: exact match rate (all 8 bits correct)
          - bit_acc: mean per-bit accuracy
          - jaccard: mean IoU over bits (skip empty-union cases)
        """
        m = mask_u8.astype(np.int64, copy=False)
        valid = m != int(ignore_value)
        if not np.any(valid):
            return {"exact": 0.0, "bit_acc": 0.0, "jaccard": 0.0, "n": 0.0}
        m = m[valid]
        pred = (logits8[valid] >= 0.0).astype(np.int64)  # sigmoid>=0.5 threshold in logit space
        true = np.stack([(m >> i) & 1 for i in range(8)], axis=1).astype(np.int64)
        bit_acc = float(np.mean(pred == true))
        exact = float(np.mean(np.all(pred == true, axis=1)))
        inter = np.sum((pred == 1) & (true == 1), axis=1).astype(np.float64)
        union = np.sum((pred == 1) | (true == 1), axis=1).astype(np.float64)
        ok = union > 0
        jacc = float(np.mean(inter[ok] / union[ok])) if np.any(ok) else 1.0
        return {"exact": exact, "bit_acc": bit_acc, "jaccard": jacc, "n": float(np.sum(valid))}

    def compute_metrics(eval_pred):
        preds = eval_pred.predictions
        labels = eval_pred.label_ids

        def _split6(x):
            # Accept tuple/list, object arrays, or stacked arrays [N,6]
            if isinstance(x, (tuple, list)) and len(x) == 1 and isinstance(x[0], (tuple, list)) and len(x[0]) == 6:
                return tuple(x[0])
            # Some Transformers versions return a 7-tuple where items 1..6 are our logits heads.
            if isinstance(x, (tuple, list)) and len(x) == 7:
                return tuple(x[1:7])
            if isinstance(x, (tuple, list)) and len(x) == 6:
                return tuple(x)
            # With hierarchical direction heads, Trainer may return 10+ items.
            # The first item is the 6-tuple of core logits (b_present, b_mask, b_dir, se_present, se_mask, se_dir).
            if isinstance(x, (tuple, list)) and len(x) >= 10:
                first = x[0]
                if isinstance(first, (tuple, list)) and len(first) == 6:
                    return tuple(first)
                # Otherwise try items 1..6 (skip the nested tuple)
                if len(x) >= 7 and hasattr(x[1], 'shape'):
                    return tuple(x[1:7])
            if isinstance(x, np.ndarray) and x.dtype == object and x.shape and int(x.shape[0]) == 6:
                return tuple(list(x))
            if isinstance(x, np.ndarray) and x.ndim == 2 and int(x.shape[1]) == 6:
                return tuple(x[:, i] for i in range(6))
            return None

        preds6 = _split6(preds)
        labels6 = _split6(labels)
        if preds6 is None or labels6 is None:
            return {}

        b_present_log, b_mid, b_dir_log, se_present_log, se_mid, se_dir_log = preds6
        y_b_present, y_b_mid, y_b_dir, y_se_present, y_se_mid, y_se_dir = labels6

        metrics: Dict[str, float] = {}
        thresh = float(args.present_threshold)
        m = _bin_metrics_from_logits(np.asarray(b_present_log).reshape(-1), np.asarray(y_b_present).reshape(-1), thresh=thresh)
        metrics.update({f"blend_present_{k}": v for k, v in m.items()})
        m2 = _bin_metrics_from_logits(np.asarray(se_present_log).reshape(-1), np.asarray(y_se_present).reshape(-1), thresh=thresh)
        metrics.update({f"se_present_{k}": v for k, v in m2.items()})
        metrics["present_threshold"] = thresh

        if bool(args.report_best_threshold):
            bt, bf = _best_f1_threshold(np.asarray(b_present_log).reshape(-1), np.asarray(y_b_present).reshape(-1))
            st, sf = _best_f1_threshold(np.asarray(se_present_log).reshape(-1), np.asarray(y_se_present).reshape(-1))
            metrics["blend_present_best_thresh"] = float(bt)
            metrics["blend_present_best_f1"] = float(bf)
            metrics["se_present_best_thresh"] = float(st)
            metrics["se_present_best_f1"] = float(sf)

        y_b_dir = np.asarray(y_b_dir)
        y_se_dir = np.asarray(y_se_dir)

        # Secondary structure: either neighbor-index CE (legacy) or mask8 BCE (token arch)
        if isinstance(b_mid, np.ndarray) and b_mid.ndim == 2 and int(b_mid.shape[1]) == 8:
            # mask8
            # Report mask metrics on positives only (otherwise exact is dominated by mask=0 negatives).
            yb01 = np.asarray(y_b_present).astype(np.int64, copy=False) != 0
            ys01 = np.asarray(y_se_present).astype(np.int64, copy=False) != 0
            mm = _mask8_metrics(np.asarray(b_mid)[yb01], np.asarray(y_b_mid)[yb01], ignore_value=int(meta.mask_ignore_value))
            metrics.update({f"blend_mask_pos_{k}": float(v) for k, v in mm.items()})
            mm2 = _mask8_metrics(np.asarray(se_mid)[ys01], np.asarray(y_se_mid)[ys01], ignore_value=int(meta.mask_ignore_value))
            metrics.update({f"se_mask_pos_{k}": float(v) for k, v in mm2.items()})
        else:
            # neighbor index (legacy path)
            y_b_sec = np.asarray(y_b_mid)
            y_se_sec = np.asarray(y_se_mid)
            metrics["blend_neighbor_acc"] = _masked_acc(np.asarray(b_mid), y_b_sec)
            metrics["blend_neighbor_top3"] = _masked_topk_acc(np.asarray(b_mid), y_b_sec, k=3)
            metrics["se_neighbor_acc"] = _masked_acc(np.asarray(se_mid), y_se_sec)
            metrics["se_neighbor_top3"] = _masked_topk_acc(np.asarray(se_mid), y_se_sec, k=3)
            metrics["blend_valid_neighbor_n"] = float(int(np.sum(y_b_sec != -100)))
            metrics["se_valid_neighbor_n"] = float(int(np.sum(y_se_sec != -100)))

        metrics["blend_dir_acc"] = _masked_acc(np.asarray(b_dir_log), y_b_dir)
        metrics["se_dir_acc"] = _masked_acc(np.asarray(se_dir_log), y_se_dir)

        metrics["blend_valid_dir_n"] = float(int(np.sum(y_b_dir != -100)))
        metrics["se_valid_dir_n"] = float(int(np.sum(y_se_dir != -100)))

        return metrics

    trainer = Trainer(
        model=model,
        args=targs,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=collate,
        compute_metrics=compute_metrics,
    )

    # When resuming, some Transformers versions can effectively reuse checkpoint intervals.
    # To avoid surprise (e.g., evaluating every 200 steps from an older run), force the
    # current CLI/default cadence right after checkpoint load. We do this on the first
    # training step to ensure it runs after internal resume logic.
    if not bool(args.no_force_intervals_on_resume):
        from transformers import TrainerCallback

        _desired_eval_steps = int(args.eval_steps)
        _desired_save_steps = int(args.save_steps)
        _desired_logging_steps = int(args.logging_steps)

        class _ForceIntervalsOnResume(TrainerCallback):
            def __init__(self):
                self._did_force = False

            def on_step_begin(self, args, state, control, **kwargs):
                if self._did_force:
                    return control
                # Only force once training actually starts (after resume has loaded state).
                if getattr(state, "global_step", 0) is None:
                    return control
                try:
                    args.eval_steps = int(_desired_eval_steps)
                    args.save_steps = int(_desired_save_steps)
                    args.logging_steps = int(_desired_logging_steps)
                except Exception:
                    return control
                self._did_force = True
                return control

        trainer.add_callback(_ForceIntervalsOnResume())

    # Early stopping (version-proof; avoids relying on Trainer's built-in callback signatures).
    if bool(args.early_stop):
        from transformers import TrainerCallback

        class _PlateauStopper(TrainerCallback):
            def __init__(self, metric: str, greater_is_better: bool, patience: int, min_delta: float):
                self.metric = str(metric)
                self.greater_is_better = bool(greater_is_better)
                self.patience = int(patience)
                self.min_delta = float(min_delta)
                self.best = None
                self.bad = 0

            def on_evaluate(self, args, state, control, metrics=None, **kwargs):
                if not metrics:
                    return control
                if self.metric not in metrics:
                    # can't early-stop if metric isn't present
                    return control
                v = metrics[self.metric]
                try:
                    v = float(v)
                except Exception:
                    return control

                if self.best is None:
                    self.best = v
                    self.bad = 0
                    return control

                improved = False
                if self.greater_is_better:
                    improved = (v - self.best) > self.min_delta
                else:
                    improved = (self.best - v) > self.min_delta

                if improved:
                    self.best = v
                    self.bad = 0
                else:
                    self.bad += 1

                if self.bad >= self.patience:
                    print(
                        f"[early-stop] stopping: metric={self.metric} best={self.best:.6f} current={v:.6f} "
                        f"bad_evals={self.bad} patience={self.patience} min_delta={self.min_delta}"
                    )
                    control.should_training_stop = True
                return control

        trainer.add_callback(
            _PlateauStopper(
                metric=str(args.early_stop_metric),
                greater_is_better=bool(args.early_stop_greater_is_better),
                patience=int(args.early_stop_patience_evals),
                min_delta=float(args.early_stop_min_delta),
            )
        )

    # Resume handling
    resume_path: Optional[str] = None
    if str(args.resume_from).strip():
        resume_path = str(args.resume_from).strip()
    else:
        # auto-resume from latest checkpoint in out_dir, if any
        ckpts = sorted(out_dir.glob("checkpoint-*"), key=lambda p: int(p.name.split("-")[-1]) if "-" in p.name else -1)
        if ckpts:
            resume_path = str(ckpts[-1])
    if resume_path:
        print(f"[resume] Resuming from checkpoint: {resume_path}")
        trainer.train(resume_from_checkpoint=resume_path)
    else:
        trainer.train()
    trainer.save_model(str(out_dir / "final"))

    # also save palette + dir values for inference reproducibility
    if str(args.arch).lower() != "token":
        np.save(out_dir / "palette.npy", np.asarray(palette, dtype=np.float32))
    (out_dir / "dir_values.json").write_text(json.dumps(meta.dir_values, indent=2), encoding="utf-8")

    print(f"Done. Model saved to: {out_dir / 'final'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


