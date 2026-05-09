"""
Head-to-head evaluation of a trained BlendUNet against the TokenBlendModel
baseline (F1=0.54 from ablation_baseline).

For each held-out .npz map:
  1. Run sliding-window U-Net inference (with optional flipX TTA).
  2. Compute the SAME per-cell metrics as train_blend_unet.py:
       present F1 / Prec / Rec / Acc, mask exact, mask bit acc, dir acc.
  3. Aggregate over maps (micro-averaged via raw TP/FP/TN/FN counts).

The exact validation split is reproduced by passing the same --val_frac and --seed
that were used during training (defaults match train_blend_unet.py).

Outputs a JSON report next to the model checkpoint.

Usage:
  python scripts/eval_unet_vs_baseline.py \\
      --model_path  "../blendinfo dataset/_generated/unet_baseline_v1/best_model.pt" \\
      --prepared_dir "../blendinfo dataset/_generated/unet_data_v1" \\
      --val_frac 0.15 --seed 42 \\
      --tta            # optional flipX TTA
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

_SCRIPT_DIR = Path(__file__).resolve().parent
_ROOT = _SCRIPT_DIR.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from train_blend_unet import NUM_PATTERN_CODES  # noqa: E402


def load_prepared_npz(npz_path: str) -> Dict[str, np.ndarray]:
    d = np.load(npz_path)
    tex_grid = np.asarray(d["tex_grid"], dtype=np.int32)
    out = {
        "tex_grid": tex_grid,
        "elev_grid": np.asarray(d["elev_grid"], dtype=np.float32),
        "blend_present": np.asarray(d["blend_present"], dtype=np.uint8),
        "blend_mask": np.asarray(d["blend_mask"], dtype=np.uint8),
        "blend_dir": np.asarray(d["blend_dir"], dtype=np.int16),
    }
    if "pattern_code" in d.files:
        out["pattern_code"] = np.asarray(d["pattern_code"], dtype=np.int8)
    return out


def _sliding_inference(
    model,
    encode_input,
    tex_grid: np.ndarray,
    elev_grid: np.ndarray,
    patch_size: int,
    stride: int,
    device,
    num_dir_classes: int,
    elev_mean: float,
    elev_std: float,
    flip_tta: bool = False,
    dist_grid: Optional[np.ndarray] = None,
    pattern_grid: Optional[np.ndarray] = None,
    num_pattern_codes: int = 13,
    la_log_priors: Optional[np.ndarray] = None,
    la_tau: float = 0.0,
    style_vec: Optional[np.ndarray] = None,
    map_tex_ds: Optional[np.ndarray] = None,
    map_elev_ds: Optional[np.ndarray] = None,
):
    """Run sliding-window U-Net inference with Gaussian-weighted overlap.

    Returns:
        present_p [W, H] float32 - blend_present probability
        mask_p    [8, W, H] float32 - per-bit neighbor mask probability
        dir_p     [K, W, H] float32 - softmax direction class probability
    """
    import torch
    import torch.nn.functional as F

    w, h = tex_grid.shape
    elev_norm = (elev_grid.astype(np.float32) - elev_mean) / max(elev_std, 1e-6)

    # Pad so the sliding window covers everything
    pad_w = (patch_size - w % patch_size) % patch_size
    pad_h = (patch_size - h % patch_size) % patch_size
    tex_padded = np.pad(tex_grid, ((0, pad_w), (0, pad_h)), mode="edge")
    elev_padded = np.pad(elev_norm, ((0, pad_w), (0, pad_h)), mode="edge")
    if dist_grid is not None:
        dist_norm = 1.0 / (1.0 + dist_grid.astype(np.float32))
        dist_padded = np.pad(dist_norm, ((0, pad_w), (0, pad_h)), mode="edge")
    else:
        dist_padded = None
    if pattern_grid is not None:
        pattern_padded = np.pad(pattern_grid.astype(np.int64),
                                ((0, pad_w), (0, pad_h)), mode="edge")
    else:
        pattern_padded = None
    Wp, Hp = tex_padded.shape

    # FlipX permutation for pattern_code values that swap under horizontal mirror.
    # Codes: 0=none 1=L==T 2=R==T 3=R==B 4=L==B 5=L 6=R 7=T 8=B 9=TL 10=TR 11=BR 12=BL
    # Under flipX: 1<->2, 3<->4, 5<->6, 9<->10, 11<->12. 0,7,8 stay.
    pat_flipx_perm = np.array([0, 2, 1, 4, 3, 6, 5, 7, 8, 10, 9, 12, 11], dtype=np.int64)

    # Gaussian weighting kernel
    sigma = patch_size / 4.0
    ax = np.arange(patch_size, dtype=np.float32) - patch_size / 2 + 0.5
    xx, yy = np.meshgrid(ax, ax, indexing="ij")
    gauss = np.exp(-(xx**2 + yy**2) / (2 * sigma**2))
    gauss = gauss / gauss.max()

    present_acc = np.zeros((w, h), dtype=np.float64)
    mask_acc = np.zeros((8, w, h), dtype=np.float64)
    dir_acc = np.zeros((num_dir_classes, w, h), dtype=np.float64)
    weight_acc = np.zeros((w, h), dtype=np.float64)

    positions = []
    for x0 in range(0, Wp - patch_size + 1, stride):
        for y0 in range(0, Hp - patch_size + 1, stride):
            positions.append((x0, y0))

    # Bit-flip table for the 8-neighbor mask under flipX (matches _flip_mask_x)
    flipx_bit_perm = [2, 1, 0, 4, 3, 7, 6, 5]
    # Direction class flipX permutation
    DIRECTION_VALUES = [-1, 1, 2, 4, 8, 17, 18, 20, 24, 33, 34, 36, 40, 49, 50, 52, 56]
    DIR_VAL_TO_CLASS = {v: i for i, v in enumerate(DIRECTION_VALUES)}
    flipx_pairs = {
        -1: -1, 1: 17, 17: 1, 2: 2, 18: 18,
        4: 8, 8: 4, 20: 24, 24: 20,
        36: 40, 40: 36, 52: 56, 56: 52,
        33: 34, 34: 33, 49: 50, 50: 49,
    }
    dir_class_perm = list(range(num_dir_classes))
    for raw_from, raw_to in flipx_pairs.items():
        a = DIR_VAL_TO_CLASS.get(raw_from)
        b = DIR_VAL_TO_CLASS.get(raw_to)
        if a is not None and b is not None:
            dir_class_perm[a] = b

    batch_size = 32
    with torch.no_grad():
        for s in range(0, len(positions), batch_size):
            batch = positions[s:s + batch_size]
            tex_b = np.stack([tex_padded[x0:x0+patch_size, y0:y0+patch_size] for x0, y0 in batch])
            elev_b = np.stack([elev_padded[x0:x0+patch_size, y0:y0+patch_size] for x0, y0 in batch])
            tex_t = torch.from_numpy(tex_b).long().to(device)
            elev_t = torch.from_numpy(elev_b).float().to(device)
            extras_parts = []
            if dist_padded is not None:
                dist_b = np.stack([dist_padded[x0:x0+patch_size, y0:y0+patch_size] for x0, y0 in batch])
                extras_parts.append(torch.from_numpy(dist_b).float().unsqueeze(1).to(device))  # [B,1,P,P]
            if pattern_padded is not None:
                pat_b = np.stack([pattern_padded[x0:x0+patch_size, y0:y0+patch_size] for x0, y0 in batch])
                pat_t = torch.from_numpy(pat_b).long().clamp(0, num_pattern_codes - 1).to(device)
                pat_oh = torch.nn.functional.one_hot(pat_t, num_pattern_codes).permute(0, 3, 1, 2).float()
                extras_parts.append(pat_oh)  # [B,13,P,P]
            if style_vec is not None:
                B = tex_t.shape[0]
                sv_t = torch.from_numpy(style_vec).float().to(device).view(1, -1, 1, 1)
                sv_b = sv_t.expand(B, -1, patch_size, patch_size).contiguous()
                extras_parts.append(sv_b)
            extras_t = torch.cat(extras_parts, dim=1) if extras_parts else None
            # Map encoder downsampled inputs (same per batch).
            map_tex_t = map_elev_t = None
            if map_tex_ds is not None and map_elev_ds is not None:
                B = tex_t.shape[0]
                map_tex_t = torch.from_numpy(map_tex_ds).long().to(device).unsqueeze(0).expand(B, -1, -1).contiguous()
                map_elev_t = torch.from_numpy(map_elev_ds).float().to(device).unsqueeze(0).expand(B, -1, -1).contiguous()
            x = encode_input(tex_t, elev_t, extras_t, map_tex_ds=map_tex_t, map_elev_ds=map_elev_t)
            out = model(x)
            present_p = torch.sigmoid(out["present_logits"].squeeze(1)).cpu().numpy()  # [B,P,P]
            mask_p = torch.sigmoid(out["mask_logits"]).cpu().numpy()                    # [B,8,P,P]
            dir_logits_t = out["dir_logits"]
            if la_log_priors is not None and la_tau != 0.0:
                lp = torch.tensor(la_log_priors, device=device, dtype=dir_logits_t.dtype)
                dir_logits_t = dir_logits_t - la_tau * lp.view(1, -1, 1, 1)
            dir_p = F.softmax(dir_logits_t, dim=1).cpu().numpy()                        # [B,K,P,P]

            if flip_tta:
                tex_f = torch.from_numpy(np.flip(tex_b, axis=2).copy()).long().to(device)
                elev_f = torch.from_numpy(np.flip(elev_b, axis=2).copy()).float().to(device)
                # Build flipped extras (same composition as the forward pass).
                extras_f_parts = []
                if dist_padded is not None:
                    dist_f_b = np.flip(dist_b, axis=2).copy()
                    extras_f_parts.append(torch.from_numpy(dist_f_b).float().unsqueeze(1).to(device))
                if pattern_padded is not None:
                    pat_f_b = pat_flipx_perm[np.flip(pat_b, axis=2)].copy()
                    pat_f_t = torch.from_numpy(pat_f_b).long().clamp(0, num_pattern_codes - 1).to(device)
                    pat_f_oh = torch.nn.functional.one_hot(pat_f_t, num_pattern_codes).permute(0, 3, 1, 2).float()
                    extras_f_parts.append(pat_f_oh)
                if style_vec is not None:
                    # Style hist concatenation: [pat_hist (13)] + [tex_hist (64)] in 'both' mode.
                    # Under flipX, pat codes 1<->2, 3<->4, 5<->6, 9<->10, 11<->12 swap; the
                    # tex hist is flip-invariant (cell count by texture id is unaffected).
                    L = len(style_vec)
                    if L == NUM_PATTERN_CODES:
                        sv_flipped = style_vec[pat_flipx_perm]
                    elif L == NUM_PATTERN_CODES + 64:
                        sv_flipped = np.concatenate([
                            style_vec[:NUM_PATTERN_CODES][pat_flipx_perm],
                            style_vec[NUM_PATTERN_CODES:],
                        ])
                    else:
                        sv_flipped = style_vec  # tex-only or unknown layout: invariant
                    Bf = tex_f.shape[0]
                    sv_f_t = torch.from_numpy(sv_flipped).float().to(device).view(1, -1, 1, 1)
                    sv_f_b = sv_f_t.expand(Bf, -1, patch_size, patch_size).contiguous()
                    extras_f_parts.append(sv_f_b)
                extras_f_t = torch.cat(extras_f_parts, dim=1) if extras_f_parts else None
                # Map_ds inputs are global; flipping the patch doesn't flip the map identity.
                # Pass the SAME map_ds tensors (they describe the same map style).
                xf = encode_input(tex_f, elev_f, extras_f_t,
                                  map_tex_ds=map_tex_t, map_elev_ds=map_elev_t)
                outf = model(xf)
                present_pf = torch.sigmoid(outf["present_logits"].squeeze(1)).cpu().numpy()
                mask_pf = torch.sigmoid(outf["mask_logits"]).cpu().numpy()
                dir_logits_f = outf["dir_logits"]
                if la_log_priors is not None and la_tau != 0.0:
                    lp = torch.tensor(la_log_priors, device=device, dtype=dir_logits_f.dtype)
                    dir_logits_f = dir_logits_f - la_tau * lp.view(1, -1, 1, 1)
                dir_pf = F.softmax(dir_logits_f, dim=1).cpu().numpy()
                # Un-flip horizontally and remap mask bits / dir classes.
                present_pf = present_pf[:, :, ::-1]
                mask_pf_unflip = mask_pf[:, :, :, ::-1]
                # Remap bit channels: out_bit[i] = in_bit[perm[i]]
                mask_pf_remapped = np.empty_like(mask_pf_unflip)
                for i in range(8):
                    mask_pf_remapped[:, i] = mask_pf_unflip[:, flipx_bit_perm[i]]
                dir_pf_unflip = dir_pf[:, :, :, ::-1]
                dir_pf_remapped = np.empty_like(dir_pf_unflip)
                for i in range(num_dir_classes):
                    dir_pf_remapped[:, i] = dir_pf_unflip[:, dir_class_perm[i]]
                present_p = 0.5 * (present_p + present_pf)
                mask_p = 0.5 * (mask_p + mask_pf_remapped)
                dir_p = 0.5 * (dir_p + dir_pf_remapped)

            for i, (x0, y0) in enumerate(batch):
                xe = min(x0 + patch_size, w)
                ye = min(y0 + patch_size, h)
                if x0 >= w or y0 >= h:
                    continue
                px, py = xe - x0, ye - y0
                gw = gauss[:px, :py]
                present_acc[x0:xe, y0:ye] += present_p[i, :px, :py] * gw
                weight_acc[x0:xe, y0:ye] += gw
                for ch in range(8):
                    mask_acc[ch, x0:xe, y0:ye] += mask_p[i, ch, :px, :py] * gw
                for ch in range(num_dir_classes):
                    dir_acc[ch, x0:xe, y0:ye] += dir_p[i, ch, :px, :py] * gw

    weight_acc = np.maximum(weight_acc, 1e-8)
    present_p = (present_acc / weight_acc).astype(np.float32)
    mask_p_n = (mask_acc / weight_acc[None, :, :]).astype(np.float32)
    dir_p_n = (dir_acc / weight_acc[None, :, :]).astype(np.float32)
    return present_p, mask_p_n, dir_p_n


def _accumulate_metrics(
    present_p: np.ndarray,
    mask_p: np.ndarray,
    dir_p: np.ndarray,
    y_present: np.ndarray,
    y_mask: np.ndarray,
    y_dir: np.ndarray,
    threshold: float,
    counters: Dict[str, float],
):
    """Update aggregate counters in-place for one map."""
    pred_present = (present_p > threshold).astype(np.uint8)
    yp = y_present.astype(np.uint8)
    counters["tp"] += int(((pred_present == 1) & (yp == 1)).sum())
    counters["fp"] += int(((pred_present == 1) & (yp == 0)).sum())
    counters["fn"] += int(((pred_present == 0) & (yp == 1)).sum())
    counters["tn"] += int(((pred_present == 0) & (yp == 0)).sum())

    valid_mask = (y_mask != 255)
    if valid_mask.any():
        pred_bits = (mask_p > 0.5).astype(np.int32)             # [8, W, H]
        true_bits = np.stack([(y_mask.astype(np.int32) >> i) & 1 for i in range(8)], axis=0)
        ve = np.broadcast_to(valid_mask, pred_bits.shape)
        counters["mask_bit_correct"] += float(((pred_bits == true_bits) & ve).sum())
        counters["mask_bit_total"] += float(ve.sum())
        exact = (pred_bits == true_bits).all(axis=0) & valid_mask
        counters["mask_exact_correct"] += float(exact.sum())
        counters["mask_exact_total"] += float(valid_mask.sum())

    present_bool = (yp == 1)
    dir_valid = present_bool & (y_dir >= 0) & (y_dir < dir_p.shape[0])
    if dir_valid.any():
        dir_pred = dir_p.argmax(axis=0).astype(np.int16)
        counters["dir_correct"] += float(((dir_pred == y_dir) & dir_valid).sum())
        counters["dir_total"] += float(dir_valid.sum())
        # Per-class confusion (only when caller requested via 'cm' key)
        if "cm" in counters:
            cm = counters["cm"]
            K = dir_p.shape[0]
            yd = y_dir[dir_valid]
            pd_ = dir_pred[dir_valid]
            for c in range(K):
                ms = yd == c
                if ms.any():
                    pds = pd_[ms]
                    for k in range(K):
                        cm[c, k] += int((pds == k).sum())


def _finalize(counters: Dict[str, float], threshold: float) -> Dict[str, float]:
    tp, fp, fn, tn = counters["tp"], counters["fp"], counters["fn"], counters["tn"]
    prec = tp / (tp + fp + 1e-12)
    rec = tp / (tp + fn + 1e-12)
    f1 = 2 * prec * rec / (prec + rec + 1e-12)
    acc = (tp + tn) / max(1.0, tp + fp + fn + tn)
    return {
        "present_threshold": float(threshold),
        "present_f1": float(f1),
        "present_prec": float(prec),
        "present_rec": float(rec),
        "present_acc": float(acc),
        "present_pos_rate": float((tp + fn) / max(1.0, tp + fp + fn + tn)),
        "mask_bit_acc": float(counters["mask_bit_correct"] / max(1.0, counters["mask_bit_total"])),
        "mask_exact_acc": float(counters["mask_exact_correct"] / max(1.0, counters["mask_exact_total"])),
        "dir_acc": float(counters["dir_correct"] / max(1.0, counters["dir_total"])),
        "n_cells_eval": int(tp + fp + fn + tn),
        "n_blend_cells": int(tp + fn),
    }


def main():
    ap = argparse.ArgumentParser(description="Evaluate trained U-Net on held-out maps.")
    ap.add_argument("--model_path", required=True, help="Path to best_model.pt")
    ap.add_argument("--prepared_dir", required=True, help="Directory of .npz files")
    ap.add_argument("--val_frac", type=float, default=0.15, help="Match training val_frac")
    ap.add_argument("--seed", type=int, default=42, help="Match training seed")
    ap.add_argument("--threshold", type=float, default=0.5, help="Decision threshold for present")
    ap.add_argument("--scan_thresholds", action="store_true",
                    help="Scan thresholds 0.1..0.9 and report best F1")
    ap.add_argument("--tta", action="store_true", help="Enable flipX test-time augmentation")
    ap.add_argument("--la_tau", type=float, default=0.0,
                    help="Post-hoc logit adjustment temperature for direction head. "
                         "Subtracts tau*log(prior) from dir logits before softmax. "
                         "Menon et al., ICLR 2021. Use ~0.5 to lift rare classes.")
    ap.add_argument("--la_prior_dirs", type=str, default="",
                    help="Comma-separated prepared dirs to compute training priors from "
                         "(typically same as training dataset). Defaults to --prepared_dir.")
    ap.add_argument("--device", type=str, default="cuda", help="cuda/cpu")
    ap.add_argument("--out", type=str, default="", help="Optional output JSON path")
    args = ap.parse_args()

    import torch
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    ckpt = torch.load(args.model_path, map_location=device, weights_only=False)
    num_textures = ckpt["num_textures"]
    tex_embed_dim = ckpt["tex_embed_dim"]
    hidden_channels = ckpt["hidden_channels"]
    patch_size = int(ckpt.get("patch_size", 64))
    extra_in_ch = int(ckpt.get("extra_input_channels", 0))
    use_dist = bool(ckpt.get("use_dist_to_boundary", False))
    use_pattern = bool(ckpt.get("use_pattern_code", False))
    use_map_style = bool(ckpt.get("use_map_style", False))
    style_mode = str(ckpt.get("style_mode") or "pat")
    # Older ckpts didn't save style_mode but extra_input_channels reveals it.
    # Heuristic: if extras include 64+ texture-hist channels, style is "both" or "tex".
    use_neighbor_tex = bool(ckpt.get("use_neighbor_tex", False))
    map_emb_dim = int(ckpt.get("map_emb_dim", 0))
    map_ds_size = int(ckpt.get("map_ds_size", 32))
    dir_head_type = str(ckpt.get("dir_head_type", "linear"))

    # Re-import the model factory.
    from train_blend_unet import (_make_model, NUM_DIR_CLASSES,
                                   _compute_distance_to_boundary,
                                   _compute_pattern_code, NUM_PATTERN_CODES)
    model = _make_model(
        num_textures=num_textures,
        tex_embed_dim=tex_embed_dim,
        hidden_channels=hidden_channels,
        extra_input_channels=extra_in_ch,
        dir_head_type=dir_head_type,
        map_emb_dim=map_emb_dim,
        map_ds_size=map_ds_size,
        use_neighbor_tex=use_neighbor_tex,
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    encode_input = model.encode_input  # type: ignore
    print(f"Loaded model from {args.model_path} (epoch {ckpt.get('epoch', '?')}, "
          f"trained F1={ckpt.get('val_f1', '?')})")

    # Reproduce validation split.
    prep_dir = Path(args.prepared_dir).resolve()
    npz_paths = sorted(prep_dir.glob("*.npz"))
    rng = np.random.default_rng(args.seed)
    indices = np.arange(len(npz_paths))
    rng.shuffle(indices)
    n_val = max(1, int(len(npz_paths) * args.val_frac))
    val_indices = indices[:n_val]
    val_paths = [npz_paths[i] for i in val_indices]
    print(f"Eval set: {len(val_paths)} maps (val_frac={args.val_frac}, seed={args.seed})")

    elev_mean, elev_std = 168.5, 113.85
    threshold_grid = ([args.threshold] if not args.scan_thresholds
                      else [round(t, 2) for t in np.linspace(0.1, 0.9, 17)])

    # Build direction priors from training maps for logit adjustment.
    la_log_priors = None
    if args.la_tau != 0.0:
        prior_dirs = (args.la_prior_dirs.split(",") if args.la_prior_dirs
                      else [args.prepared_dir])
        prior_paths = []
        for d in prior_dirs:
            d = d.strip()
            if d:
                prior_paths.extend(sorted(Path(d).glob("*.npz")))
        # Train set = all prior_paths minus the val set we just sampled.
        val_names = {p.name for p in val_paths}
        train_paths = [p for p in prior_paths if p.name not in val_names]
        counts = np.zeros(NUM_DIR_CLASSES, dtype=np.float64)
        for p in train_paths:
            try:
                with np.load(p) as d:
                    bdir = d["blend_dir"]
                    pres = d["blend_present"] > 0
                    valid = pres & (bdir >= 0) & (bdir < NUM_DIR_CLASSES)
                    if not valid.any():
                        continue
                    v = bdir[valid]
                    for c in range(NUM_DIR_CLASSES):
                        counts[c] += int((v == c).sum())
            except Exception:
                pass
        smooth = counts + 1.0
        priors = smooth / smooth.sum()
        la_log_priors = np.log(priors)
        print(f"Logit adjustment: tau={args.la_tau}, priors min={priors.min():.6f} "
              f"max={priors.max():.4f} (from {len(train_paths)} train maps)")

    counters_per_threshold: Dict[float, Dict[str, float]] = {
        t: {"tp": 0.0, "fp": 0.0, "fn": 0.0, "tn": 0.0,
            "mask_bit_correct": 0.0, "mask_bit_total": 0.0,
            "mask_exact_correct": 0.0, "mask_exact_total": 0.0,
            "dir_correct": 0.0, "dir_total": 0.0}
        for t in threshold_grid
    }
    # Confusion matrix (only at primary threshold, for per-class diagnostic)
    confusion_matrix = np.zeros((NUM_DIR_CLASSES, NUM_DIR_CLASSES), dtype=np.int64)
    counters_per_threshold[args.threshold]["cm"] = confusion_matrix
    per_map_results: List[Dict] = []

    t_start = time.time()
    stride = max(1, patch_size // 2)

    for i, p in enumerate(val_paths):
        data = load_prepared_npz(str(p))
        dist_grid = _compute_distance_to_boundary(data["tex_grid"]) if use_dist else None
        if use_pattern or use_map_style:
            pattern_grid = data.get("pattern_code")
            if pattern_grid is None:
                pattern_grid = _compute_pattern_code(data["tex_grid"])
        else:
            pattern_grid = None
        # For map-style: compute style vector matching the trained style_mode.
        # Modes: "pat" (13-dim pattern hist), "tex" (64-dim tex hist),
        # "both" (concat -> 77-dim). tex_dim=64 matches training (decoupled
        # from per-dataset max_tex_id so train/val produce same shape).
        style_vec = None
        if use_map_style:
            parts = []
            if style_mode in ("pat", "both") and pattern_grid is not None:
                hp = np.bincount(pattern_grid.ravel().astype(np.int64),
                                 minlength=NUM_PATTERN_CODES).astype(np.float32)
                hp = hp / max(1.0, float(hp.sum()))
                parts.append(hp[:NUM_PATTERN_CODES])
            if style_mode in ("tex", "both"):
                tex_dim = 64
                tex = data["tex_grid"].astype(np.int64)
                tex = np.clip(tex, 0, tex_dim - 1)
                ht = np.bincount(tex.ravel(), minlength=tex_dim).astype(np.float32)[:tex_dim]
                ht = ht / max(1.0, float(ht.sum()))
                parts.append(ht)
            if parts:
                style_vec = np.concatenate(parts).astype(np.float32)
        # For map encoder: compute downsampled tex/elev arrays (same scheme as training).
        map_tex_ds_eval = None
        map_elev_ds_eval = None
        if map_emb_dim > 0:
            tex_full = data["tex_grid"]
            elev_full = data.get("elev_grid")
            if elev_full is None:
                elev_full = np.full(tex_full.shape, elev_mean, dtype=np.float32)
            w_, h_ = tex_full.shape
            xs = np.linspace(0, w_ - 1, map_ds_size).astype(np.int32)
            ys = np.linspace(0, h_ - 1, map_ds_size).astype(np.int32)
            map_tex_ds_eval = tex_full[xs[:, None], ys[None, :]].astype(np.int64)
            elev_ds = elev_full[xs[:, None], ys[None, :]].astype(np.float32)
            map_elev_ds_eval = (elev_ds - elev_mean) / max(elev_std, 1e-6)
        # Don't pass pattern_grid as per-cell extras unless use_pattern is on.
        pattern_grid_for_per_cell = pattern_grid if use_pattern else None
        present_p, mask_p, dir_p = _sliding_inference(
            model=model,
            encode_input=encode_input,
            tex_grid=data["tex_grid"],
            elev_grid=data["elev_grid"],
            patch_size=patch_size,
            stride=stride,
            device=device,
            num_dir_classes=NUM_DIR_CLASSES,
            elev_mean=elev_mean,
            elev_std=elev_std,
            flip_tta=args.tta,
            dist_grid=dist_grid,
            pattern_grid=pattern_grid_for_per_cell,
            num_pattern_codes=NUM_PATTERN_CODES,
            la_log_priors=la_log_priors,
            la_tau=args.la_tau,
            style_vec=style_vec,
            map_tex_ds=map_tex_ds_eval,
            map_elev_ds=map_elev_ds_eval,
        )

        # Per-map at the primary threshold (for logging).
        per_map_counters = {"tp": 0.0, "fp": 0.0, "fn": 0.0, "tn": 0.0,
                            "mask_bit_correct": 0.0, "mask_bit_total": 0.0,
                            "mask_exact_correct": 0.0, "mask_exact_total": 0.0,
                            "dir_correct": 0.0, "dir_total": 0.0}
        _accumulate_metrics(present_p, mask_p, dir_p,
                            data["blend_present"], data["blend_mask"], data["blend_dir"],
                            args.threshold, per_map_counters)
        per_map_summary = _finalize(per_map_counters, args.threshold)
        per_map_summary["map"] = p.name
        per_map_summary["w"] = int(data["tex_grid"].shape[0])
        per_map_summary["h"] = int(data["tex_grid"].shape[1])
        per_map_results.append(per_map_summary)

        # Aggregate per-threshold for the global scan.
        for t in threshold_grid:
            _accumulate_metrics(present_p, mask_p, dir_p,
                                data["blend_present"], data["blend_mask"], data["blend_dir"],
                                t, counters_per_threshold[t])

        elapsed = time.time() - t_start
        print(f"  [{i+1}/{len(val_paths)}] {p.name:40s} "
              f"F1={per_map_summary['present_f1']:.3f} dir={per_map_summary['dir_acc']:.3f} "
              f"mask_ex={per_map_summary['mask_exact_acc']:.3f} "
              f"({elapsed:.0f}s)", flush=True)

    # Aggregate.
    aggregate_per_threshold = {
        f"{t:.2f}": _finalize(counters_per_threshold[t], t) for t in threshold_grid
    }
    best_t = max(threshold_grid, key=lambda t: counters_per_threshold[t]["tp"] /
                 max(1e-9, counters_per_threshold[t]["tp"] + 0.5 *
                     (counters_per_threshold[t]["fp"] + counters_per_threshold[t]["fn"])))
    best_summary = _finalize(counters_per_threshold[best_t], best_t)

    print("\n" + "=" * 70)
    print(f"AGGREGATE (micro-averaged over {len(val_paths)} held-out maps)")
    print("=" * 70)
    primary = aggregate_per_threshold[f"{args.threshold:.2f}"]
    print(f"  thr={args.threshold:.2f}  "
          f"F1={primary['present_f1']:.4f}  "
          f"Prec={primary['present_prec']:.4f}  "
          f"Rec={primary['present_rec']:.4f}  "
          f"DirAcc={primary['dir_acc']:.4f}  "
          f"MaskBit={primary['mask_bit_acc']:.4f}  "
          f"MaskExact={primary['mask_exact_acc']:.4f}")
    if args.scan_thresholds:
        print(f"\n  best F1 at thr={best_t:.2f}: {best_summary['present_f1']:.4f}")
        print(f"  (Prec={best_summary['present_prec']:.4f}, "
              f"Rec={best_summary['present_rec']:.4f})")
    print(f"  TTA: {'on' if args.tta else 'off'}")
    print()
    print(f"  Token-model baseline (run_ablation.py 'baseline'): F1=0.5432, "
          f"best_F1=0.6462, dir_acc=0.8929, mask_exact=0.1900")

    # Per-class direction accuracy
    DIRECTION_VALUES = [-1, 1, 2, 4, 8, 17, 18, 20, 24, 33, 34, 36, 40, 49, 50, 52, 56]
    print("\n--- Per-class direction accuracy ---")
    print(f"  {'cls':>3} {'raw':>4} {'n':>9} {'acc':>7}  group")
    per_class_data = []
    for c in range(NUM_DIR_CLASSES):
        n_c = int(confusion_matrix[c].sum())
        acc_c = float(confusion_matrix[c, c]) / max(1, n_c)
        raw = DIRECTION_VALUES[c]
        if c == 0:
            grp = "(none)"
        elif raw in (1, 2, 4, 8):
            grp = "cardinal"
        elif raw in (17, 18, 20, 24):
            grp = "diagonal"
        elif raw in (36, 40, 52, 56):
            grp = "Except (compound)"
        elif raw in (33, 34, 49, 50):
            grp = "Rare compound"
        else:
            grp = "?"
        flag = "  <-- 0%" if (n_c > 0 and acc_c < 0.05) else ""
        print(f"  {c:>3} {raw:>4} {n_c:>9} {acc_c:>7.3f}  {grp}{flag}")
        per_class_data.append({"class": c, "raw": raw, "n": n_c, "acc": acc_c, "group": grp})

    # Save report.
    out_path = Path(args.out) if args.out else Path(args.model_path).parent / "eval_report.json"
    report = {
        "model_path": str(args.model_path),
        "prepared_dir": str(args.prepared_dir),
        "val_frac": args.val_frac,
        "seed": args.seed,
        "tta": bool(args.tta),
        "patch_size": patch_size,
        "stride": stride,
        "n_val_maps": len(val_paths),
        "primary_threshold": args.threshold,
        "primary_metrics": primary,
        "best_threshold": float(best_t),
        "best_metrics": best_summary,
        "per_threshold": aggregate_per_threshold,
        "per_map": per_map_results,
        "per_class": per_class_data,
        "confusion_matrix": confusion_matrix.tolist(),
        "token_baseline": {
            "f1": 0.5432, "best_f1": 0.6462, "dir_acc": 0.8929,
            "mask_exact": 0.1900, "mask_bit": 0.7641,
        },
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"\nReport: {out_path}")


if __name__ == "__main__":
    main()
