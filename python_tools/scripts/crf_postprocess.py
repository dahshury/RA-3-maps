"""
ConvCRF post-processing for RA3 blend predictions.

Applies Convolutional Conditional Random Field (ConvCRF) refinement to the
per-cell blend predictions produced by the base Token Transformer model.
The CRF operates on the full H x W spatial grid of a map to enforce local
consistency that the independent per-cell model cannot capture.

Two separate CRF heads are trained:
  - Presence CRF: 2-class (blend / no-blend), kernel_size=5
  - Direction CRF: 17-class (blend direction), kernel_size=3

Neighbor mask prediction is NOT refined by CRF because it is a per-cell
multi-label output (8-bit mask) rather than a spatially-structured label.

Modes:
  python crf_postprocess.py --mode train \
      --model_path <path_to_pretrained_checkpoint> \
      --data_dir <path_to_prepared_dataset_dir> \
      --maps_dir <path_to_map_files> \
      --out_dir <output_directory>

  python crf_postprocess.py --mode eval \
      --model_path <path_to_pretrained_checkpoint> \
      --crf_path <path_to_trained_crf_weights> \
      --data_dir <path_to_prepared_dataset_dir> \
      --maps_dir <path_to_map_files>
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# path bootstrap
# ---------------------------------------------------------------------------
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from map_processor.ra3map import Ra3Map  # noqa: E402
from map_processor.assets.terrain.blend_tile_data import BlendTileData  # noqa: E402
from map_processor.assets.terrain.height_map_data import HeightMapData  # noqa: E402

# ---------------------------------------------------------------------------
# Texture decoding (same as predict_blends.py)
# ---------------------------------------------------------------------------

def _get_texture_from_tile(x: int, y: int, tile_value: int) -> int:
    row_first = (y % 8) // 2 * 16 + (y % 2) * 2
    current = (x % 8) // 2 * 4 + (x % 2) + row_first
    return (int(tile_value) - current) // 64


def _decode_texture_grid(blend: BlendTileData) -> np.ndarray:
    tiles = np.asarray(blend.tiles, dtype=np.int32)
    w, h = tiles.shape
    tex = np.zeros((w, h), dtype=np.int32)
    for x in range(w):
        for y in range(h):
            tex[x, y] = _get_texture_from_tile(x, y, int(tiles[x, y]))
    return tex


# ---------------------------------------------------------------------------
# Ground truth extraction from a parsed map
# ---------------------------------------------------------------------------

_NEIGHBOR_OFFSETS = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]


def _extract_ground_truth(blend: BlendTileData, dir_values: List[int]) -> dict:
    """Extract per-cell ground truth grids from a parsed BlendTileData.

    Returns dict with:
        blend_present:  (W, H) uint8 0/1
        blend_dir:      (W, H) int32 direction class index (-1 if absent)
        se_present:     (W, H) uint8 0/1
        se_dir:         (W, H) int32 direction class index (-1 if absent)
        blend_mask_u8:  (W, H) uint8 neighbor bitmask (0 if absent)
        se_mask_u8:     (W, H) uint8 neighbor bitmask (0 if absent)
    """
    blends = np.asarray(blend.blends, dtype=np.int32)
    se = np.asarray(blend.single_edge_blends, dtype=np.int32)
    info = blend.blend_info or []
    tex_grid = _decode_texture_grid(blend)
    w, h = blends.shape

    # Build direction value -> class index lookup
    dir_val_to_cls = {v: i for i, v in enumerate(dir_values)}

    blend_present = (blends > 0).astype(np.uint8)
    se_present = (se > 0).astype(np.uint8)
    blend_dir = np.full((w, h), -1, dtype=np.int32)
    se_dir = np.full((w, h), -1, dtype=np.int32)
    blend_mask_u8 = np.zeros((w, h), dtype=np.uint8)
    se_mask_u8 = np.zeros((w, h), dtype=np.uint8)

    def _sec_tex_from_info(bi, x, y):
        """Decode secondary texture index from a BlendInfo entry."""
        tile = bi.secondary_texture_tile
        row_first = (y % 8) // 2 * 16 + (y % 2) * 2
        current = (x % 8) // 2 * 4 + (x % 2) + row_first
        return (int(tile) - current) // 64

    def _neighbor_mask(tex_grid, x, y, sec_tex, w, h):
        """Compute 8-bit mask: which neighbors have texture == sec_tex."""
        mask = 0
        for ni, (dx, dy) in enumerate(_NEIGHBOR_OFFSETS):
            nx, ny = x + dx, y + dy
            if 0 <= nx < w and 0 <= ny < h and tex_grid[nx, ny] == sec_tex:
                mask |= (1 << ni)
        return mask

    for x in range(w):
        for y in range(h):
            idx_b = int(blends[x, y])
            if idx_b > 0 and idx_b <= len(info):
                bi = info[idx_b - 1]
                dir_raw = int(bi.blend_direction.value) if bi.blend_direction else 0
                blend_dir[x, y] = dir_val_to_cls.get(dir_raw, 0)
                sec_tex = _sec_tex_from_info(bi, x, y)
                blend_mask_u8[x, y] = _neighbor_mask(tex_grid, x, y, sec_tex, w, h)

            idx_se = int(se[x, y])
            if idx_se > 0 and idx_se <= len(info):
                bi = info[idx_se - 1]
                dir_raw = int(bi.blend_direction.value) if bi.blend_direction else 0
                se_dir[x, y] = dir_val_to_cls.get(dir_raw, 0)
                sec_tex = _sec_tex_from_info(bi, x, y)
                se_mask_u8[x, y] = _neighbor_mask(tex_grid, x, y, sec_tex, w, h)

    return {
        "blend_present": blend_present,
        "blend_dir": blend_dir,
        "se_present": se_present,
        "se_dir": se_dir,
        "blend_mask_u8": blend_mask_u8,
        "se_mask_u8": se_mask_u8,
    }


# ---------------------------------------------------------------------------
# Base model loading (reuse from predict_blends.py)
# ---------------------------------------------------------------------------

def _load_base_model(checkpoint_dir: Path, meta: dict, hidden: int = 384, n_layers: int = 6):
    """Load pretrained Token Transformer and freeze all parameters."""
    import torch
    import torch.nn as nn

    num_textures = meta["vocab"]["num_textures"]
    dir_num_classes = meta["direction"]["num_classes"]
    extra_dim = meta.get("extra_dim", 0)
    map_style_dim = meta.get("map_style_dim", 0)

    seq_len = 25
    center_idx = 12
    n_heads = 8
    dropout = 0.1

    class TokenBlendModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.num_textures = int(num_textures)
            self.hidden = int(hidden)

            self.tex_emb = nn.Embedding(self.num_textures, self.hidden)
            self.elev_proj = nn.Linear(1, self.hidden, bias=True)
            self.pos_emb = nn.Embedding(seq_len, self.hidden)
            self.local_proj = nn.Linear(1, self.hidden, bias=True)

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

            self.extra_proj = nn.Linear(int(extra_dim), self.hidden) if int(extra_dim) > 0 else None
            self.map_style_proj = nn.Linear(int(map_style_dim), self.hidden) if int(map_style_dim) > 0 else None

            # Main heads
            self.blend_present = nn.Linear(self.hidden, 1)
            self.blend_mask = nn.Linear(self.hidden, 8)
            self.blend_dir = nn.Linear(self.hidden, int(dir_num_classes))

            # Hierarchical direction heads
            self.neighbor_idxs = [6, 7, 8, 11, 13, 16, 17, 18]
            self.dir_query = nn.Parameter(torch.randn(1, 1, self.hidden) * 0.02)
            self.dir_attn = nn.MultiheadAttention(self.hidden, num_heads=4, batch_first=True)
            self.blend_dir_row = nn.Linear(self.hidden, 3)
            self.blend_dir_col = nn.Linear(self.hidden, 3)
            self.blend_dir_type = nn.Linear(self.hidden, 3)

            # SE heads
            self.se_present = nn.Linear(self.hidden, 1)
            self.se_mask = nn.Linear(self.hidden, 8)
            self.se_dir = nn.Linear(self.hidden, int(dir_num_classes))
            self.se_dir_row = nn.Linear(self.hidden, 3)
            self.se_dir_col = nn.Linear(self.hidden, 3)
            self.se_dir_type = nn.Linear(self.hidden, 3)

            self.register_buffer("dir_class_weights", torch.ones(17, dtype=torch.float32))

        def _get_dir_pooled(self, x):
            B = x.shape[0]
            neighbor_emb = x[:, self.neighbor_idxs, :]
            query = self.dir_query.expand(B, -1, -1)
            attn_out, _ = self.dir_attn(query, neighbor_emb, neighbor_emb)
            center = x[:, center_idx, :]
            return center + attn_out.squeeze(1)

        def forward(self, tex, elev_z, tex_local_norm=None, extra_features=None, map_style=None, labels=None):
            B = tex.shape[0]
            pos = torch.arange(seq_len, device=tex.device, dtype=torch.long).unsqueeze(0).expand(B, -1)

            x = self.tex_emb(torch.clamp(tex, 0, self.num_textures - 1))
            x = x + self.elev_proj(elev_z.unsqueeze(-1)) + self.pos_emb(pos)
            if tex_local_norm is not None:
                x = x + self.local_proj(tex_local_norm.unsqueeze(-1))
            x = self.encoder(x)

            pooled = x[:, center_idx, :]
            if self.extra_proj is not None and extra_features is not None:
                pooled = pooled + self.extra_proj(extra_features)
            if self.map_style_proj is not None and map_style is not None:
                pooled = pooled + self.map_style_proj(map_style)

            dir_pooled = self._get_dir_pooled(x)
            if self.extra_proj is not None and extra_features is not None:
                dir_pooled = dir_pooled + self.extra_proj(extra_features)
            if self.map_style_proj is not None and map_style is not None:
                dir_pooled = dir_pooled + self.map_style_proj(map_style)

            return {
                "logits_blend_present": self.blend_present(pooled).squeeze(-1),
                "logits_blend_mask": self.blend_mask(pooled),
                "logits_blend_dir": self.blend_dir(dir_pooled),
                "logits_se_present": self.se_present(pooled).squeeze(-1),
                "logits_se_mask": self.se_mask(pooled),
                "logits_se_dir": self.se_dir(dir_pooled),
            }

    model = TokenBlendModel()

    # Load checkpoint weights
    from safetensors.torch import load_file
    weights_path = checkpoint_dir / "model.safetensors"
    if weights_path.exists():
        state_dict = load_file(str(weights_path))
        model.load_state_dict(state_dict, strict=True)
        _log(f"  Loaded base model weights from {weights_path}")
    else:
        raise FileNotFoundError(f"No model.safetensors in {checkpoint_dir}")

    # Freeze all base model parameters
    for p in model.parameters():
        p.requires_grad = False
    model.eval()
    return model


# ---------------------------------------------------------------------------
# ConvCRF Module
# ---------------------------------------------------------------------------

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvCRF(nn.Module):
    """Convolutional CRF for spatial smoothing of predictions on a 2D grid.

    Takes per-cell model logits reshaped to (B, C, H, W) and refines them
    via learned convolutional message-passing with a compatibility matrix.

    Parameters
    ----------
    num_classes : int
        Number of classes for this prediction head.
    kernel_size : int
        Spatial extent of the pairwise potential convolution.
    num_iterations : int
        Number of mean-field inference iterations.
    use_bilateral : bool
        If True, add a bilateral branch that uses spatial feature maps
        (e.g. texture IDs) as an additional gating signal.
    """

    def __init__(
        self,
        num_classes: int,
        kernel_size: int = 5,
        num_iterations: int = 5,
        use_bilateral: bool = False,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.num_iterations = num_iterations
        self.use_bilateral = use_bilateral

        # Pairwise (spatial) potential: learned convolution kernel
        # Each class sends messages that are aggregated by convolution.
        self.pairwise_conv = nn.Conv2d(
            num_classes,
            num_classes,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            bias=False,
        )
        # Initialize with a small identity-like kernel so initial messages
        # are close to the unary (warm start).
        nn.init.zeros_(self.pairwise_conv.weight)

        # Compatibility matrix: learns which class transitions are penalized.
        # Initialized as -I so that same-class messages reinforce, different
        # classes repel (standard Potts-like prior).
        self.compatibility = nn.Parameter(
            torch.eye(num_classes, dtype=torch.float32) * -1.0
        )

        # Optional bilateral branch: uses texture features to gate messages.
        if use_bilateral:
            self.bilateral_conv = nn.Conv2d(
                num_classes,
                num_classes,
                kernel_size=kernel_size,
                padding=kernel_size // 2,
                bias=False,
            )
            nn.init.zeros_(self.bilateral_conv.weight)
            # Texture embedding for bilateral gate (one channel in, one out)
            self.tex_gate = nn.Sequential(
                nn.Conv2d(1, 16, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(16, 1, kernel_size=3, padding=1),
                nn.Sigmoid(),
            )

        # Learnable step size for the update (allows the network to control
        # how aggressively it moves away from unary potentials).
        self.step_weight = nn.Parameter(torch.tensor(0.1))

    def forward(
        self,
        unary: torch.Tensor,
        spatial_features: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Run mean-field inference with convolutional message passing.

        Parameters
        ----------
        unary : (B, C, H, W)
            Raw logits from the base model (one channel per class).
        spatial_features : (B, 1, H, W), optional
            Texture IDs (float-cast) for bilateral filtering.
            Only used if ``use_bilateral=True``.

        Returns
        -------
        Q : (B, C, H, W)
            Refined logits (can be passed through softmax / sigmoid).
        """
        Q = F.softmax(unary, dim=1)

        for _ in range(self.num_iterations):
            # --- Spatial message passing ---
            msg = self.pairwise_conv(Q)

            # --- Optional bilateral message passing ---
            if self.use_bilateral and spatial_features is not None:
                gate = self.tex_gate(spatial_features)  # (B,1,H,W)
                bilateral_msg = self.bilateral_conv(Q)
                msg = msg + bilateral_msg * gate

            # --- Compatibility transform ---
            # msg: (B, C, H, W) -> einsum with (C, C) compatibility matrix
            msg = torch.einsum("bchw,cd->bdhw", msg, self.compatibility)

            # --- Update Q ---
            Q = F.softmax(unary + self.step_weight * msg, dim=1)

        # Return as logits (log-domain) for downstream loss computation.
        # Adding a small epsilon avoids log(0).
        return torch.log(Q + 1e-8)


class BlendCRFModule(nn.Module):
    """Wrapper holding separate CRF heads for presence and direction.

    Presence CRF
        2-class (no-blend, blend), kernel_size=5, 5 iterations.
    Direction CRF
        ``dir_num_classes``-class, kernel_size=3, 5 iterations.

    Neighbor mask is left untouched (no CRF applied).
    """

    def __init__(
        self,
        dir_num_classes: int,
        presence_kernel: int = 5,
        direction_kernel: int = 3,
        num_iterations: int = 5,
        use_bilateral: bool = False,
    ):
        super().__init__()
        self.blend_presence_crf = ConvCRF(
            num_classes=2,
            kernel_size=presence_kernel,
            num_iterations=num_iterations,
            use_bilateral=use_bilateral,
        )
        self.blend_direction_crf = ConvCRF(
            num_classes=dir_num_classes,
            kernel_size=direction_kernel,
            num_iterations=num_iterations,
            use_bilateral=use_bilateral,
        )
        self.se_presence_crf = ConvCRF(
            num_classes=2,
            kernel_size=presence_kernel,
            num_iterations=num_iterations,
            use_bilateral=use_bilateral,
        )
        self.se_direction_crf = ConvCRF(
            num_classes=dir_num_classes,
            kernel_size=direction_kernel,
            num_iterations=num_iterations,
            use_bilateral=use_bilateral,
        )

    def forward(
        self,
        blend_present_logits: torch.Tensor,
        blend_dir_logits: torch.Tensor,
        se_present_logits: torch.Tensor,
        se_dir_logits: torch.Tensor,
        spatial_features: Optional[torch.Tensor] = None,
    ) -> dict:
        """Refine all grid-level logits through CRF layers.

        All inputs are (B, C, H, W).
        Returns a dict of refined (B, C, H, W) log-probabilities.
        """
        return {
            "blend_present": self.blend_presence_crf(blend_present_logits, spatial_features),
            "blend_dir": self.blend_direction_crf(blend_dir_logits, spatial_features),
            "se_present": self.se_presence_crf(se_present_logits, spatial_features),
            "se_dir": self.se_direction_crf(se_dir_logits, spatial_features),
        }

    def param_count(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ---------------------------------------------------------------------------
# Batch inference over entire map -> (H, W) grids of logits
# ---------------------------------------------------------------------------

def _run_base_model_on_map(
    model: nn.Module,
    tex_grid: np.ndarray,
    elev_grid: Optional[np.ndarray],
    local_to_global: Dict[int, int],
    n_local_tex: int,
    elev_mean: float,
    elev_std: float,
    device: torch.device,
    batch_size: int = 256,
) -> dict:
    """Run base model over every cell in the map.

    Returns numpy arrays of raw logits shaped (W, H, C) for each head.
    """
    w, h = tex_grid.shape
    win = 5
    pad = win // 2

    tex_pad = np.pad(tex_grid, ((pad, pad), (pad, pad)), mode="edge")
    elev_pad = None
    if elev_grid is not None:
        elev_pad = np.pad(elev_grid.astype(np.float32), ((pad, pad), (pad, pad)), mode="edge")

    # Pre-allocate output grids
    out_b_present = np.zeros((w, h), dtype=np.float32)
    out_b_mask = np.zeros((w, h, 8), dtype=np.float32)
    out_b_dir = None  # allocated on first batch (need dir_num_classes)
    out_se_present = np.zeros((w, h), dtype=np.float32)
    out_se_mask = np.zeros((w, h, 8), dtype=np.float32)
    out_se_dir = None

    all_coords = [(x, y) for x in range(w) for y in range(h)]

    with torch.no_grad():
        for start in range(0, len(all_coords), batch_size):
            end = min(start + batch_size, len(all_coords))
            coords = all_coords[start:end]

            tex_windows = []
            elev_windows = []
            local_norm_windows = []

            for x, y in coords:
                px, py = x + pad, y + pad
                tw = tex_pad[px - pad : px + pad + 1, py - pad : py + pad + 1].flatten()
                tw_global = np.array([local_to_global.get(int(t), 0) for t in tw], dtype=np.int64)
                tex_windows.append(tw_global)

                if n_local_tex > 1:
                    lnorm = tex_pad[px - pad : px + pad + 1, py - pad : py + pad + 1].flatten().astype(np.float32) / (n_local_tex - 1)
                else:
                    lnorm = np.zeros(25, dtype=np.float32)
                local_norm_windows.append(lnorm)

                if elev_pad is not None:
                    ew = elev_pad[px - pad : px + pad + 1, py - pad : py + pad + 1].flatten()
                    elev_windows.append(ew)
                else:
                    elev_windows.append(np.zeros(25, dtype=np.float32))

            tex_t = torch.tensor(np.stack(tex_windows), dtype=torch.long, device=device)
            elev_t = torch.tensor(np.stack(elev_windows), dtype=torch.float32, device=device)
            lnorm_t = torch.tensor(np.stack(local_norm_windows), dtype=torch.float32, device=device)
            elev_z = (elev_t - elev_mean) / max(elev_std, 1e-6)

            outputs = model(tex=tex_t, elev_z=elev_z, tex_local_norm=lnorm_t)

            bp = outputs["logits_blend_present"].cpu().numpy()
            bm = outputs["logits_blend_mask"].cpu().numpy()
            bd = outputs["logits_blend_dir"].cpu().numpy()
            sp = outputs["logits_se_present"].cpu().numpy()
            sm = outputs["logits_se_mask"].cpu().numpy()
            sd = outputs["logits_se_dir"].cpu().numpy()

            if out_b_dir is None:
                dir_classes = bd.shape[1]
                out_b_dir = np.zeros((w, h, dir_classes), dtype=np.float32)
                out_se_dir = np.zeros((w, h, dir_classes), dtype=np.float32)

            for i, (x, y) in enumerate(coords):
                out_b_present[x, y] = bp[i]
                out_b_mask[x, y] = bm[i]
                out_b_dir[x, y] = bd[i]
                out_se_present[x, y] = sp[i]
                out_se_mask[x, y] = sm[i]
                out_se_dir[x, y] = sd[i]

            if end % 20000 == 0 or end == len(all_coords):
                pct = 100.0 * end / len(all_coords)
                _log(f"    Base model: {end}/{len(all_coords)} cells ({pct:.1f}%)")

    return {
        "blend_present": out_b_present,  # (W, H)
        "blend_mask": out_b_mask,        # (W, H, 8)
        "blend_dir": out_b_dir,          # (W, H, D)
        "se_present": out_se_present,
        "se_mask": out_se_mask,
        "se_dir": out_se_dir,
    }


def _logits_to_crf_input(
    present_logits_1d: np.ndarray,
    dir_logits: np.ndarray,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Convert per-cell raw logits to (1, C, H, W) tensors for CRF input.

    present_logits_1d: (W, H) single logit for blend_present
        -> convert to 2-class: [logit_no, logit_yes] where logit_no = 0.
    dir_logits: (W, H, D)
        -> transpose to (1, D, W, H) -- note: W is the first spatial dim.
    """
    W, H = present_logits_1d.shape

    # 2-class presence logits: class 0 = no-blend, class 1 = blend
    pres_2c = np.zeros((1, 2, W, H), dtype=np.float32)
    pres_2c[0, 0, :, :] = 0.0  # "no blend" reference logit
    pres_2c[0, 1, :, :] = present_logits_1d  # raw sigmoid-space logit

    # Direction logits: (W, H, D) -> (1, D, W, H)
    D = dir_logits.shape[2]
    dir_4d = dir_logits.transpose(2, 0, 1)[np.newaxis, :, :, :]  # (1, D, W, H)

    return (
        torch.from_numpy(pres_2c),
        torch.from_numpy(np.ascontiguousarray(dir_4d)),
    )


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def _compute_f1(pred: np.ndarray, gt: np.ndarray) -> Tuple[float, float, float]:
    """Binary F1, precision, recall from boolean arrays."""
    tp = np.logical_and(pred, gt).sum()
    fp = np.logical_and(pred, ~gt).sum()
    fn = np.logical_and(~pred, gt).sum()
    precision = float(tp) / max(float(tp + fp), 1.0)
    recall = float(tp) / max(float(tp + fn), 1.0)
    f1 = 2 * precision * recall / max(precision + recall, 1e-8)
    return f1, precision, recall


def _compute_dir_accuracy(pred_cls: np.ndarray, gt_cls: np.ndarray, present_mask: np.ndarray) -> float:
    """Direction accuracy only where blend is present in ground truth."""
    mask = present_mask > 0
    if mask.sum() == 0:
        return 0.0
    return float((pred_cls[mask] == gt_cls[mask]).mean())


def _evaluate_map(
    raw_logits: dict,
    gt: dict,
    crf_module: Optional[nn.Module],
    device: torch.device,
    tex_grid: Optional[np.ndarray] = None,
    threshold: float = 0.3,
) -> dict:
    """Evaluate base model (optionally + CRF) on a single map.

    Returns metrics dict with and without CRF.
    """
    results = {}

    # --- Base model metrics (no CRF) ---
    bp_pred = (1.0 / (1.0 + np.exp(-raw_logits["blend_present"]))) > threshold
    sp_pred = (1.0 / (1.0 + np.exp(-raw_logits["se_present"]))) > threshold
    bd_pred = raw_logits["blend_dir"].argmax(axis=-1)
    sd_pred = raw_logits["se_dir"].argmax(axis=-1)

    gt_bp = gt["blend_present"].astype(bool)
    gt_sp = gt["se_present"].astype(bool)
    gt_bd = gt["blend_dir"]
    gt_sd = gt["se_dir"]

    f1_bp, prec_bp, rec_bp = _compute_f1(bp_pred, gt_bp)
    f1_sp, prec_sp, rec_sp = _compute_f1(sp_pred, gt_sp)
    acc_bd = _compute_dir_accuracy(bd_pred, gt_bd, gt["blend_present"])
    acc_sd = _compute_dir_accuracy(sd_pred, gt_sd, gt["se_present"])

    results["base"] = {
        "blend_present_f1": f1_bp, "blend_present_prec": prec_bp, "blend_present_rec": rec_bp,
        "se_present_f1": f1_sp, "se_present_prec": prec_sp, "se_present_rec": rec_sp,
        "blend_dir_acc": acc_bd, "se_dir_acc": acc_sd,
    }

    # --- CRF-refined metrics ---
    if crf_module is not None:
        W, H = raw_logits["blend_present"].shape

        bp_in, bd_in = _logits_to_crf_input(raw_logits["blend_present"], raw_logits["blend_dir"])
        sp_in, sd_in = _logits_to_crf_input(raw_logits["se_present"], raw_logits["se_dir"])

        bp_in = bp_in.to(device)
        bd_in = bd_in.to(device)
        sp_in = sp_in.to(device)
        sd_in = sd_in.to(device)

        # Spatial features (texture IDs) for bilateral
        spatial = None
        if tex_grid is not None:
            spatial = torch.from_numpy(tex_grid.astype(np.float32)[np.newaxis, np.newaxis, :, :]).to(device)

        with torch.no_grad():
            refined = crf_module(bp_in, bd_in, sp_in, sd_in, spatial)

        # Decode CRF outputs
        bp_crf = refined["blend_present"][0].cpu().numpy()  # (2, W, H)
        bd_crf = refined["blend_dir"][0].cpu().numpy()       # (D, W, H)
        sp_crf = refined["se_present"][0].cpu().numpy()
        sd_crf = refined["se_dir"][0].cpu().numpy()

        # Presence: class 1 prob > threshold
        bp_crf_pred = np.exp(bp_crf[1]) > threshold
        sp_crf_pred = np.exp(sp_crf[1]) > threshold
        bd_crf_pred = bd_crf.argmax(axis=0)  # (W, H)
        sd_crf_pred = sd_crf.argmax(axis=0)

        f1_bp_c, prec_bp_c, rec_bp_c = _compute_f1(bp_crf_pred, gt_bp)
        f1_sp_c, prec_sp_c, rec_sp_c = _compute_f1(sp_crf_pred, gt_sp)
        acc_bd_c = _compute_dir_accuracy(bd_crf_pred, gt_bd, gt["blend_present"])
        acc_sd_c = _compute_dir_accuracy(sd_crf_pred, gt_sd, gt["se_present"])

        results["crf"] = {
            "blend_present_f1": f1_bp_c, "blend_present_prec": prec_bp_c, "blend_present_rec": rec_bp_c,
            "se_present_f1": f1_sp_c, "se_present_prec": prec_sp_c, "se_present_rec": rec_sp_c,
            "blend_dir_acc": acc_bd_c, "se_dir_acc": acc_sd_c,
        }

    return results


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def _log(msg: str):
    print(msg, flush=True)


def _discover_maps(maps_dir: Path) -> List[Path]:
    """Recursively find all .map files."""
    maps = sorted(maps_dir.rglob("*.map"))
    # Filter out files that look like derivatives (blendless, predicted, etc.)
    filtered = []
    for m in maps:
        stem_lower = m.stem.lower()
        if any(tag in stem_lower for tag in ("_blendless", "_predicted", "_original")):
            continue
        filtered.append(m)
    return filtered


def _parse_map_data(
    map_path: Path,
    global_tex_name_to_id: dict,
) -> Optional[dict]:
    """Parse a single map and return everything needed for CRF training.

    Returns None if the map cannot be parsed or has no blends.
    """
    try:
        m = Ra3Map(str(map_path))
        m.parse()
        ctx = m.get_context()
        blend = ctx.get_asset_by_type(BlendTileData)
        height = ctx.get_asset_by_type(HeightMapData)
        if blend is None:
            return None

        tex_grid = _decode_texture_grid(blend)
        w, h = tex_grid.shape

        elev_grid = None
        if height is not None and height.elevations is not None:
            elev_grid = np.asarray(height.elevations, dtype=np.float32)

        local_tex_names = [t.name for t in (blend.textures or [])]
        local_to_global = {}
        for i, name in enumerate(local_tex_names):
            local_to_global[i] = global_tex_name_to_id.get(name, 0)

        return {
            "path": map_path,
            "blend": blend,
            "tex_grid": tex_grid,
            "elev_grid": elev_grid,
            "local_to_global": local_to_global,
            "n_local_tex": len(local_tex_names),
            "w": w,
            "h": h,
        }
    except Exception as e:
        _log(f"  Warning: failed to parse {map_path}: {e}")
        return None


def train_crf(args):
    """Train CRF parameters on top of frozen base model predictions."""
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    _log(f"Device: {device}")

    # Load metadata
    data_dir = Path(args.data_dir)
    meta_path = data_dir / "prepared_meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    dir_values = meta["direction"]["values"]
    dir_num_classes = meta["direction"]["num_classes"]
    global_tex_name_to_id = meta["vocab"]["texture_name_to_id"]
    elev_mean = meta["elevation_norm"]["mean"]
    elev_std = meta["elevation_norm"]["std"]

    _log(f"Direction classes: {dir_num_classes}, Textures: {meta['vocab']['num_textures']}")

    # Load base model (frozen)
    checkpoint_dir = Path(args.model_path)
    _log(f"Loading base model from {checkpoint_dir} ...")
    base_model = _load_base_model(
        checkpoint_dir, meta, hidden=args.hidden, n_layers=args.layers
    )
    base_model = base_model.to(device)
    _log("  Base model loaded and frozen.")

    # Create CRF module
    crf = BlendCRFModule(
        dir_num_classes=dir_num_classes,
        presence_kernel=args.presence_kernel,
        direction_kernel=args.direction_kernel,
        num_iterations=args.crf_iterations,
        use_bilateral=args.bilateral,
    )
    crf = crf.to(device)
    _log(f"  CRF module: {crf.param_count()} trainable parameters")

    # Optimizer (only CRF params)
    optimizer = torch.optim.Adam(crf.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.lr * 0.01)

    # Discover maps
    maps_dir = Path(args.maps_dir)
    map_paths = _discover_maps(maps_dir)
    _log(f"Found {len(map_paths)} map files in {maps_dir}")

    if len(map_paths) == 0:
        _log("ERROR: No maps found. Check --maps_dir path.")
        return 1

    # Split into train/val
    rng = np.random.default_rng(args.seed)
    indices = np.arange(len(map_paths))
    rng.shuffle(indices)
    n_val = max(1, int(len(map_paths) * 0.15))
    val_indices = set(indices[:n_val].tolist())
    train_indices = [i for i in range(len(map_paths)) if i not in val_indices]
    val_indices_list = sorted(val_indices)

    _log(f"  Train: {len(train_indices)} maps, Val: {len(val_indices_list)} maps")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    best_val_f1 = -1.0
    history = []

    for epoch in range(args.epochs):
        _log(f"\n{'='*60}")
        _log(f"Epoch {epoch + 1}/{args.epochs}")
        _log(f"{'='*60}")

        # Shuffle training maps each epoch
        rng.shuffle(train_indices)

        crf.train()
        epoch_losses = []
        t0 = time.time()

        for mi, map_idx in enumerate(train_indices):
            map_path = map_paths[map_idx]
            _log(f"  [{mi+1}/{len(train_indices)}] Processing {map_path.name} ...")

            parsed = _parse_map_data(map_path, global_tex_name_to_id)
            if parsed is None:
                _log(f"    Skipped (parse error or no blends)")
                continue

            # Get ground truth
            gt = _extract_ground_truth(parsed["blend"], dir_values)
            if gt["blend_present"].sum() == 0 and gt["se_present"].sum() == 0:
                _log(f"    Skipped (no blends in ground truth)")
                continue

            # Run base model (frozen) over entire map
            raw = _run_base_model_on_map(
                base_model,
                parsed["tex_grid"],
                parsed["elev_grid"],
                parsed["local_to_global"],
                parsed["n_local_tex"],
                elev_mean, elev_std,
                device,
                batch_size=args.batch_size,
            )

            W, H = parsed["w"], parsed["h"]

            # Convert to CRF input tensors (1, C, W, H)
            bp_in, bd_in = _logits_to_crf_input(raw["blend_present"], raw["blend_dir"])
            sp_in, sd_in = _logits_to_crf_input(raw["se_present"], raw["se_dir"])

            bp_in = bp_in.to(device)
            bd_in = bd_in.to(device)
            sp_in = sp_in.to(device)
            sd_in = sd_in.to(device)

            spatial = None
            if args.bilateral:
                spatial = torch.from_numpy(
                    parsed["tex_grid"].astype(np.float32)[np.newaxis, np.newaxis, :, :]
                ).to(device)

            # Forward through CRF
            refined = crf(bp_in, bd_in, sp_in, sd_in, spatial)

            # Ground truth tensors
            gt_bp = torch.from_numpy(gt["blend_present"].astype(np.int64)).to(device)   # (W, H)
            gt_bd = torch.from_numpy(gt["blend_dir"].astype(np.int64)).to(device)       # (W, H)
            gt_sp = torch.from_numpy(gt["se_present"].astype(np.int64)).to(device)
            gt_sd = torch.from_numpy(gt["se_dir"].astype(np.int64)).to(device)

            # --- Compute losses ---
            # Presence loss: cross-entropy over 2 classes
            bp_refined = refined["blend_present"]  # (1, 2, W, H)
            sp_refined = refined["se_present"]

            loss_bp = F.cross_entropy(bp_refined, gt_bp.unsqueeze(0), reduction="mean")
            loss_sp = F.cross_entropy(sp_refined, gt_sp.unsqueeze(0), reduction="mean")

            # Direction loss: cross-entropy only where blend is present
            bd_refined = refined["blend_dir"]   # (1, D, W, H)
            sd_refined = refined["se_dir"]

            # Mask directions: set to -100 where not present
            gt_bd_masked = gt_bd.clone()
            gt_bd_masked[gt["blend_present"] == 0] = -100
            gt_bd_masked[gt_bd < 0] = -100

            gt_sd_masked = gt_sd.clone()
            gt_sd_masked[gt["se_present"] == 0] = -100
            gt_sd_masked[gt_sd < 0] = -100

            loss_bd = F.cross_entropy(
                bd_refined, gt_bd_masked.unsqueeze(0), ignore_index=-100, reduction="mean"
            )
            loss_sd = F.cross_entropy(
                sd_refined, gt_sd_masked.unsqueeze(0), ignore_index=-100, reduction="mean"
            )

            loss = (
                args.loss_weight_presence * (loss_bp + loss_sp)
                + args.loss_weight_direction * (loss_bd + loss_sd)
            )

            # Backward + update
            optimizer.zero_grad()
            loss.backward()

            # Gradient clipping (CRF can be unstable)
            torch.nn.utils.clip_grad_norm_(crf.parameters(), max_norm=1.0)

            optimizer.step()

            loss_val = loss.item()
            epoch_losses.append(loss_val)
            _log(f"    loss={loss_val:.4f} (pres={loss_bp.item() + loss_sp.item():.4f}, dir={loss_bd.item() + loss_sd.item():.4f})")

        scheduler.step()

        elapsed = time.time() - t0
        avg_loss = np.mean(epoch_losses) if epoch_losses else float("nan")
        _log(f"\n  Epoch {epoch+1} avg loss: {avg_loss:.4f} ({elapsed:.1f}s, lr={scheduler.get_last_lr()[0]:.2e})")

        # --- Validation ---
        _log(f"  Running validation on {len(val_indices_list)} maps ...")
        crf.eval()
        val_metrics = {
            "blend_present_f1": [], "se_present_f1": [],
            "blend_dir_acc": [], "se_dir_acc": [],
        }
        base_metrics = {
            "blend_present_f1": [], "se_present_f1": [],
            "blend_dir_acc": [], "se_dir_acc": [],
        }

        for vi, map_idx in enumerate(val_indices_list):
            map_path = map_paths[map_idx]
            parsed = _parse_map_data(map_path, global_tex_name_to_id)
            if parsed is None:
                continue

            gt = _extract_ground_truth(parsed["blend"], dir_values)
            if gt["blend_present"].sum() == 0 and gt["se_present"].sum() == 0:
                continue

            raw = _run_base_model_on_map(
                base_model,
                parsed["tex_grid"],
                parsed["elev_grid"],
                parsed["local_to_global"],
                parsed["n_local_tex"],
                elev_mean, elev_std,
                device,
                batch_size=args.batch_size,
            )

            eval_res = _evaluate_map(
                raw, gt, crf, device,
                tex_grid=parsed["tex_grid"] if args.bilateral else None,
                threshold=args.threshold,
            )

            for k in val_metrics:
                val_metrics[k].append(eval_res["crf"][k])
                base_metrics[k].append(eval_res["base"][k])

        # Summarize
        _log(f"\n  Validation results (epoch {epoch+1}):")
        _log(f"  {'Metric':<25} {'Base':>10} {'CRF':>10} {'Delta':>10}")
        _log(f"  {'-'*55}")
        val_f1_avg = 0.0
        for k in val_metrics:
            base_avg = np.mean(base_metrics[k]) if base_metrics[k] else 0.0
            crf_avg = np.mean(val_metrics[k]) if val_metrics[k] else 0.0
            delta = crf_avg - base_avg
            sign = "+" if delta >= 0 else ""
            _log(f"  {k:<25} {base_avg:>10.4f} {crf_avg:>10.4f} {sign}{delta:>9.4f}")
            if "f1" in k:
                val_f1_avg += crf_avg

        val_f1_avg /= 2.0  # average of blend and SE F1

        history.append({
            "epoch": epoch + 1,
            "train_loss": float(avg_loss),
            "val_blend_present_f1_base": float(np.mean(base_metrics["blend_present_f1"])) if base_metrics["blend_present_f1"] else 0.0,
            "val_blend_present_f1_crf": float(np.mean(val_metrics["blend_present_f1"])) if val_metrics["blend_present_f1"] else 0.0,
            "val_se_present_f1_base": float(np.mean(base_metrics["se_present_f1"])) if base_metrics["se_present_f1"] else 0.0,
            "val_se_present_f1_crf": float(np.mean(val_metrics["se_present_f1"])) if val_metrics["se_present_f1"] else 0.0,
            "val_blend_dir_acc_base": float(np.mean(base_metrics["blend_dir_acc"])) if base_metrics["blend_dir_acc"] else 0.0,
            "val_blend_dir_acc_crf": float(np.mean(val_metrics["blend_dir_acc"])) if val_metrics["blend_dir_acc"] else 0.0,
        })

        # Save best
        if val_f1_avg > best_val_f1:
            best_val_f1 = val_f1_avg
            save_path = out_dir / "crf_best.pt"
            torch.save(crf.state_dict(), str(save_path))
            _log(f"  ** New best avg F1: {best_val_f1:.4f} -> saved to {save_path}")

        # Always save latest
        torch.save(crf.state_dict(), str(out_dir / "crf_latest.pt"))

    # Save training history
    hist_path = out_dir / "crf_training_history.json"
    with open(str(hist_path), "w") as f:
        json.dump(history, f, indent=2)
    _log(f"\nTraining complete. History saved to {hist_path}")
    _log(f"Best validation avg F1: {best_val_f1:.4f}")

    # Save CRF config for loading
    cfg = {
        "dir_num_classes": dir_num_classes,
        "presence_kernel": args.presence_kernel,
        "direction_kernel": args.direction_kernel,
        "num_iterations": args.crf_iterations,
        "use_bilateral": args.bilateral,
    }
    cfg_path = out_dir / "crf_config.json"
    with open(str(cfg_path), "w") as f:
        json.dump(cfg, f, indent=2)
    _log(f"CRF config saved to {cfg_path}")

    return 0


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def eval_crf(args):
    """Evaluate base model with and without CRF on all maps."""
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    _log(f"Device: {device}")

    # Load metadata
    data_dir = Path(args.data_dir)
    meta = json.loads((data_dir / "prepared_meta.json").read_text(encoding="utf-8"))
    dir_values = meta["direction"]["values"]
    dir_num_classes = meta["direction"]["num_classes"]
    global_tex_name_to_id = meta["vocab"]["texture_name_to_id"]
    elev_mean = meta["elevation_norm"]["mean"]
    elev_std = meta["elevation_norm"]["std"]

    # Load base model
    checkpoint_dir = Path(args.model_path)
    _log(f"Loading base model from {checkpoint_dir} ...")
    base_model = _load_base_model(checkpoint_dir, meta, hidden=args.hidden, n_layers=args.layers)
    base_model = base_model.to(device)

    # Load CRF
    crf_path = Path(args.crf_path)
    crf = None
    if crf_path.suffix == ".pt":
        # Load config from same directory
        cfg_path = crf_path.parent / "crf_config.json"
        if cfg_path.exists():
            cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
        else:
            _log(f"Warning: no crf_config.json found, using defaults")
            cfg = {
                "dir_num_classes": dir_num_classes,
                "presence_kernel": 5,
                "direction_kernel": 3,
                "num_iterations": 5,
                "use_bilateral": False,
            }
        crf = BlendCRFModule(
            dir_num_classes=cfg["dir_num_classes"],
            presence_kernel=cfg["presence_kernel"],
            direction_kernel=cfg["direction_kernel"],
            num_iterations=cfg["num_iterations"],
            use_bilateral=cfg.get("use_bilateral", False),
        )
        state = torch.load(str(crf_path), map_location=device, weights_only=True)
        crf.load_state_dict(state)
        crf = crf.to(device)
        crf.eval()
        _log(f"Loaded CRF from {crf_path} ({crf.param_count()} params)")
    else:
        _log(f"WARNING: No CRF loaded (crf_path={crf_path}). Will only report base metrics.")

    # Discover maps
    maps_dir = Path(args.maps_dir)
    map_paths = _discover_maps(maps_dir)
    _log(f"Found {len(map_paths)} maps in {maps_dir}")

    all_base = {
        "blend_present_f1": [], "blend_present_prec": [], "blend_present_rec": [],
        "se_present_f1": [], "se_present_prec": [], "se_present_rec": [],
        "blend_dir_acc": [], "se_dir_acc": [],
    }
    all_crf = {k: [] for k in all_base}
    per_map_results = []

    for mi, map_path in enumerate(map_paths):
        _log(f"\n[{mi+1}/{len(map_paths)}] {map_path.name}")
        parsed = _parse_map_data(map_path, global_tex_name_to_id)
        if parsed is None:
            _log(f"  Skipped")
            continue

        gt = _extract_ground_truth(parsed["blend"], dir_values)
        n_blend = int(gt["blend_present"].sum())
        n_se = int(gt["se_present"].sum())
        total = parsed["w"] * parsed["h"]
        _log(f"  Size: {parsed['w']}x{parsed['h']} = {total} cells, "
             f"blends: {n_blend} ({100*n_blend/total:.1f}%), "
             f"SE: {n_se} ({100*n_se/total:.1f}%)")

        if n_blend == 0 and n_se == 0:
            _log(f"  No blends, skipping")
            continue

        raw = _run_base_model_on_map(
            base_model,
            parsed["tex_grid"],
            parsed["elev_grid"],
            parsed["local_to_global"],
            parsed["n_local_tex"],
            elev_mean, elev_std,
            device,
            batch_size=args.batch_size,
        )

        use_bilateral = crf is not None and hasattr(crf.blend_presence_crf, "bilateral_conv")
        eval_res = _evaluate_map(
            raw, gt, crf, device,
            tex_grid=parsed["tex_grid"] if use_bilateral else None,
            threshold=args.threshold,
        )

        # Print per-map
        _log(f"  {'Metric':<25} {'Base':>10} {'CRF':>10}")
        _log(f"  {'-'*45}")
        for k in all_base:
            base_v = eval_res["base"].get(k, 0.0)
            crf_v = eval_res.get("crf", {}).get(k, float("nan"))
            _log(f"  {k:<25} {base_v:>10.4f} {crf_v:>10.4f}")
            all_base[k].append(base_v)
            if "crf" in eval_res:
                all_crf[k].append(crf_v)

        per_map_results.append({
            "map": map_path.name,
            "size": f"{parsed['w']}x{parsed['h']}",
            "n_blend": n_blend,
            "n_se": n_se,
            "base": eval_res["base"],
            "crf": eval_res.get("crf"),
        })

    # --- Summary ---
    _log(f"\n{'='*60}")
    _log(f"SUMMARY ({len(per_map_results)} maps evaluated)")
    _log(f"{'='*60}")
    _log(f"  {'Metric':<25} {'Base (avg)':>12} {'CRF (avg)':>12} {'Delta':>10}")
    _log(f"  {'-'*60}")
    for k in all_base:
        base_avg = np.mean(all_base[k]) if all_base[k] else 0.0
        crf_avg = np.mean(all_crf[k]) if all_crf[k] else float("nan")
        delta = crf_avg - base_avg if not np.isnan(crf_avg) else float("nan")
        sign = "+" if delta >= 0 else ""
        _log(f"  {k:<25} {base_avg:>12.4f} {crf_avg:>12.4f} {sign}{delta:>9.4f}")

    # Save results
    if args.out_dir:
        out_dir = Path(args.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        results_path = out_dir / "crf_eval_results.json"
        summary = {
            "n_maps": len(per_map_results),
            "threshold": args.threshold,
            "base_avg": {k: float(np.mean(v)) for k, v in all_base.items() if v},
            "crf_avg": {k: float(np.mean(v)) for k, v in all_crf.items() if v},
            "per_map": per_map_results,
        }
        with open(str(results_path), "w") as f:
            json.dump(summary, f, indent=2, default=str)
        _log(f"\nResults saved to {results_path}")

    return 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description="ConvCRF post-processing for RA3 blend predictions.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train CRF on top of a pretrained model
  python crf_postprocess.py --mode train \\
      --model_path "../blendinfo dataset/_generated/checkpoints/best" \\
      --data_dir "../blendinfo dataset/_generated/prepared_w5_elev_full" \\
      --maps_dir "../RA3 Official maps" \\
      --out_dir "../blendinfo dataset/_generated/crf_output"

  # Evaluate CRF + base model
  python crf_postprocess.py --mode eval \\
      --model_path "../blendinfo dataset/_generated/checkpoints/best" \\
      --crf_path "../blendinfo dataset/_generated/crf_output/crf_best.pt" \\
      --data_dir "../blendinfo dataset/_generated/prepared_w5_elev_full" \\
      --maps_dir "../RA3 Official maps"
""",
    )

    # Required args
    ap.add_argument("--mode", required=True, choices=["train", "eval"],
                    help="Mode: train CRF parameters or evaluate.")
    ap.add_argument("--model_path", required=True,
                    help="Path to pretrained base model checkpoint directory.")
    ap.add_argument("--data_dir", required=True,
                    help="Path to prepared dataset directory (for metadata).")
    ap.add_argument("--maps_dir", required=True,
                    help="Path to directory containing .map files.")

    # Optional: CRF weights for eval mode
    ap.add_argument("--crf_path", default="",
                    help="Path to trained CRF weights (.pt file). Required for eval mode.")
    ap.add_argument("--out_dir", default="",
                    help="Output directory for trained CRF weights and results.")

    # Base model config
    ap.add_argument("--hidden", type=int, default=384,
                    help="Hidden size of the base Token Transformer (default: 384).")
    ap.add_argument("--layers", type=int, default=6,
                    help="Number of transformer layers in base model (default: 6).")
    ap.add_argument("--batch_size", type=int, default=256,
                    help="Batch size for running base model over map cells (default: 256).")

    # CRF architecture
    ap.add_argument("--presence_kernel", type=int, default=5,
                    help="Kernel size for presence CRF (default: 5).")
    ap.add_argument("--direction_kernel", type=int, default=3,
                    help="Kernel size for direction CRF (default: 3).")
    ap.add_argument("--crf_iterations", type=int, default=5,
                    help="Number of mean-field iterations in CRF (default: 5).")
    ap.add_argument("--bilateral", action="store_true",
                    help="Use bilateral CRF branch with texture features.")

    # Training hyperparams
    ap.add_argument("--lr", type=float, default=1e-3,
                    help="Learning rate for CRF parameters (default: 1e-3).")
    ap.add_argument("--weight_decay", type=float, default=1e-4,
                    help="Weight decay for optimizer (default: 1e-4).")
    ap.add_argument("--epochs", type=int, default=10,
                    help="Number of training epochs (default: 10).")
    ap.add_argument("--seed", type=int, default=42,
                    help="Random seed for train/val split (default: 42).")
    ap.add_argument("--loss_weight_presence", type=float, default=1.0,
                    help="Weight for presence CRF loss (default: 1.0).")
    ap.add_argument("--loss_weight_direction", type=float, default=1.0,
                    help="Weight for direction CRF loss (default: 1.0).")

    # Evaluation
    ap.add_argument("--threshold", type=float, default=0.3,
                    help="Presence decision threshold for evaluation (default: 0.3).")

    # Device
    ap.add_argument("--device", default="cuda",
                    help="Torch device (default: cuda).")

    args = ap.parse_args()

    if args.mode == "train":
        if not args.out_dir:
            ap.error("--out_dir is required for training mode.")
        return train_crf(args)
    elif args.mode == "eval":
        if not args.crf_path:
            _log("Warning: No --crf_path provided. Will only report base model metrics.")
        return eval_crf(args)
    else:
        ap.error(f"Unknown mode: {args.mode}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
