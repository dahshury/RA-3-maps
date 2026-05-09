#!/usr/bin/env python3
"""Cascaded e2e overfit: tile head first, blend heads conditioned on tile output.

Blend predictions are decomposed into 4 sub-targets (present, secondary tex,
direction, neighbor mask) and trained with class-balanced losses ported from
the existing scripts/train_blend_unet.py:
  - present       : BCE with pos_weight + optional boundary weighting
  - secondary_tex : CE masked to present=1, optional class-balanced focal
  - direction     : focal CE (or class-balanced focal) masked to present=1, with logit adjustment
  - neighbor_mask : Asymmetric Loss (ASL), masked to non-ignore cells

Same raw inputs as overfit_e2e.py / overfit_e2e_full.py: elev + water + coord
+ object stamps + style. Object dropout regularises against missing objects.

Usage:
  python scripts/overfit_e2e_decomp.py \
    --src "../RA3 Official maps/2 II/map_mp_2_rao1.map" \
    --epochs 2000 --obj_dropout 0.5
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import torch.nn.functional as F


def _python_tools_root() -> Path:
    return Path(__file__).resolve().parents[1]


sys.path.insert(0, str(_python_tools_root()))
from map_processor import Ra3Map  # noqa: E402
from map_processor.features import extract_raw_inputs  # noqa: E402
from map_processor.features.raw_inputs import (  # noqa: E402
    DIRECTION_VALUES, NUM_DIR_CLASSES,
)
from map_processor.models.end_to_end_unet import CascadeTextureNet  # noqa: E402


# -------------------------- Encoding helpers --------------------------

def _position_pattern(x: int, y: int) -> int:
    row_first = (y % 8 // 2) * 16 + (y % 2) * 2
    return (x % 8 // 2) * 4 + (x % 2) + row_first


def _make_position_pattern_grid(W: int, H: int) -> np.ndarray:
    g = np.zeros((W, H), dtype=np.uint16)
    for x in range(W):
        for y in range(H):
            g[x, y] = _position_pattern(x, y)
    return g


def _pad_t(t: torch.Tensor, multiple: int = 16):
    *_, W, H = t.shape
    pw = (-W) % multiple
    ph = (-H) % multiple
    if pw or ph:
        t = F.pad(t, (0, ph, 0, pw), mode="reflect")
    return t


def build_vocabs(raw):
    type_to_id, owner_to_id = {}, {}
    for obj in raw.objects:
        if obj.type_name not in type_to_id:
            type_to_id[obj.type_name] = len(type_to_id) + 1
        if obj.owner not in owner_to_id:
            owner_to_id[obj.owner] = len(owner_to_id) + 1
    return type_to_id, owner_to_id


def encode_objects(raw, type_to_id, owner_to_id):
    return [{
        "tile_x": o.tile_x, "tile_y": o.tile_y,
        "type_id": type_to_id.get(o.type_name, 0),
        "owner_id": owner_to_id.get(o.owner, 0),
        "angle_deg": o.angle_deg,
    } for o in raw.objects]


# -------------------------- Losses --------------------------

def focal_bce(logits: torch.Tensor, target: torch.Tensor, gamma: float = 2.0) -> torch.Tensor:
    """Focal binary cross-entropy. Handles class imbalance without hand-tuning pos_weight."""
    p = torch.sigmoid(logits)
    p_t = p * target + (1 - p) * (1 - target)
    focal = (1 - p_t).pow(gamma)
    bce = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    return (focal * bce).mean()


def soft_dice(logits: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """1 - soft Dice coefficient. Symmetric in FP/FN — robust under severe imbalance."""
    p = torch.sigmoid(logits)
    inter = (p * target).sum()
    denom = p.sum() + target.sum() + eps
    return 1.0 - (2 * inter + eps) / denom


def bce_dice(logits: torch.Tensor, target: torch.Tensor, gamma: float = 2.0) -> torch.Tensor:
    """Equal-weight focal BCE + soft Dice. Empirically cracks the rare-positive
    blend-present problem where focal-BCE alone plateaus at ~80% F1.
    """
    return 0.5 * focal_bce(logits, target, gamma=gamma) + 0.5 * soft_dice(logits, target)


def cb_focal_ce(logits: torch.Tensor, target: torch.Tensor,
                class_freqs: torch.Tensor, *, gamma: float = 2.0,
                tau: float = 1.0, beta: float = 0.999) -> torch.Tensor:
    """Class-balanced focal cross-entropy with logit adjustment.

    logits: (N, C); target: (N,) in [0, C). class_freqs: (C,) marginal freqs.
    """
    if class_freqs is not None and tau > 0:
        log_freqs = torch.log(class_freqs.clamp(min=1e-8)) * tau
        logits = logits - log_freqs.unsqueeze(0)
    log_probs = F.log_softmax(logits, dim=-1)
    log_pt = log_probs.gather(1, target.unsqueeze(1)).squeeze(1)
    pt = log_pt.exp().clamp(max=1.0 - 1e-6)
    focal = (1.0 - pt).pow(gamma) if gamma > 0 else 1.0
    per = -focal * log_pt
    # Class-balanced weights (Cui 2019)
    if class_freqs is not None and beta > 0:
        eff_n = (1.0 - beta ** class_freqs.clamp(min=1.0)) / (1.0 - beta + 1e-8)
        cls_w = 1.0 / eff_n.clamp(min=1e-8)
        cls_w = cls_w * (cls_w.numel() / cls_w.sum().clamp(min=1e-8))
        w = cls_w[target]
        return (per * w).sum() / w.sum().clamp(min=1e-8)
    return per.mean()


def asl_loss(logits: torch.Tensor, targets: torch.Tensor, *,
             gamma_pos: float = 0.0, gamma_neg: float = 4.0, m: float = 0.05) -> torch.Tensor:
    """Asymmetric Loss for multi-label classification."""
    probs = torch.sigmoid(logits)
    probs_neg = (probs - m).clamp(min=0)
    loss_pos = targets * torch.log(probs.clamp(min=1e-8))
    loss_neg = (1 - targets) * torch.log((1 - probs_neg).clamp(min=1e-8))
    if gamma_pos > 0:
        loss_pos = loss_pos * ((1 - probs) ** gamma_pos)
    if gamma_neg > 0:
        pt_neg = probs_neg.clamp(max=1.0 - 1e-8)
        loss_neg = loss_neg * (pt_neg ** gamma_neg)
    return -(loss_pos + loss_neg).mean()


def compute_blend_loss(blend_out: Dict[str, torch.Tensor], targets: Dict[str, torch.Tensor],
                       palette_size: int, dir_class_freqs: torch.Tensor,
                       sec_class_freqs: torch.Tensor, *,
                       present_focal_gamma: float = 2.0,
                       dir_focal_gamma: float = 2.0,
                       dir_logit_adj_tau: float = 0.3,
                       dir_cb_beta: float = 0.999,
                       w_present: float = 1.0, w_sec: float = 1.0,
                       w_dir: float = 1.0):
    """Returns (total_loss, dict_of_components).

    Uses only target labels — no hand-engineered weights derived from spatial
    structure of the targets (no boundary weighting, no neighbor-mask aux head).
    Imbalance is handled via focal losses + logit adjustment + class-balanced
    weighting on direction & secondary.
    """
    present_logits = blend_out["present"]                     # (B, 1, W, H)
    sec_logits = blend_out["secondary"]                       # (B, P, W, H)
    dir_logits = blend_out["direction"]                       # (B, D, W, H)
    y_present = targets["present"].float()                    # (B, W, H)
    y_sec = targets["secondary"].long()                       # (B, W, H), -1 = ignore
    y_dir = targets["direction"].long()                       # (B, W, H), -1 = ignore

    # Present: focal-BCE + soft Dice. Equal weights. Dice symmetrises FP/FN under
    # severe imbalance; focal-BCE keeps the easy negatives from dominating.
    loss_present = bce_dice(present_logits.squeeze(1), y_present, gamma=present_focal_gamma)

    # Mask to present=1 cells for secondary and direction.
    present_bool = (y_present > 0.5)

    def _flat(t):
        return t.permute(0, 2, 3, 1).reshape(-1, t.shape[1])

    sec_valid = present_bool & (y_sec >= 0) & (y_sec < palette_size)
    if sec_valid.any():
        sl = _flat(sec_logits)[sec_valid.reshape(-1)]
        st = y_sec.reshape(-1)[sec_valid.reshape(-1)]
        loss_sec = cb_focal_ce(sl, st, sec_class_freqs, gamma=dir_focal_gamma,
                               tau=dir_logit_adj_tau, beta=dir_cb_beta)
    else:
        loss_sec = present_logits.new_zeros(())

    dir_valid = present_bool & (y_dir >= 0) & (y_dir < NUM_DIR_CLASSES)
    if dir_valid.any():
        dl = _flat(dir_logits)[dir_valid.reshape(-1)]
        dt = y_dir.reshape(-1)[dir_valid.reshape(-1)]
        loss_dir = cb_focal_ce(dl, dt, dir_class_freqs, gamma=dir_focal_gamma,
                               tau=dir_logit_adj_tau, beta=dir_cb_beta)
    else:
        loss_dir = present_logits.new_zeros(())

    total = w_present * loss_present + w_sec * loss_sec + w_dir * loss_dir
    return total, {
        "present": float(loss_present.item()),
        "secondary": float(loss_sec.item()),
        "direction": float(loss_dir.item()),
    }


# -------------------------- Train --------------------------

def train_overfit_decomp(model: CascadeTextureNet,
                         inputs: dict, targets: dict, batch_objects: list,
                         *, epochs: int, lr: float, log_every: int,
                         obj_dropout: float = 0.5,
                         rng: np.random.Generator | None = None,
                         w_tile: float = 5.0):
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.0)
    print(f"Model: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M params, "
          f"in_ch={model.in_ch}, palette_size={model.palette_size}")
    print(f"Object dropout p={obj_dropout:.2f}; w_tile={w_tile} "
          f"(blend losses use focal-BCE + cb-focal-CE, no boundary/pos_weight tuning)")
    if rng is None:
        rng = np.random.default_rng(0)

    # Class frequencies for direction & secondary heads (used by CB-focal)
    dir_freqs = torch.zeros(NUM_DIR_CLASSES, device=inputs["elev"].device)
    sec_freqs = torch.zeros(model.palette_size, device=inputs["elev"].device)
    for blend_key in ("blend", "single_edge"):
        present = targets[blend_key]["present"]                 # (1, W', H')
        d = targets[blend_key]["direction"]
        s = targets[blend_key]["secondary"]
        valid_dir = (present > 0.5) & (d >= 0) & (d < NUM_DIR_CLASSES)
        valid_sec = (present > 0.5) & (s >= 0) & (s < model.palette_size)
        if valid_dir.any():
            for c in range(NUM_DIR_CLASSES):
                dir_freqs[c] += float(((d == c) & valid_dir).sum().item())
        if valid_sec.any():
            for c in range(model.palette_size):
                sec_freqs[c] += float(((s == c) & valid_sec).sum().item())
    dir_freqs = dir_freqs / dir_freqs.sum().clamp(min=1)
    sec_freqs = sec_freqs / sec_freqs.sum().clamp(min=1)

    full_objs = batch_objects
    n_objs = sum(len(b) for b in full_objs)

    t0 = time.time()
    model.train()
    for ep in range(1, epochs + 1):
        if obj_dropout > 0 and n_objs > 0:
            train_objs = []
            for b in full_objs:
                keep = rng.random(len(b)) > obj_dropout
                train_objs.append([o for o, k in zip(b, keep) if k])
        else:
            train_objs = full_objs

        opt.zero_grad()
        out = model(inputs["elev"], inputs["water"], inputs["coord"],
                    inputs["style_id"], train_objs)
        loss_tiles = F.cross_entropy(out["tiles"], targets["tiles"][None])
        l_blend, _ = compute_blend_loss(
            out["blend"], targets["blend"], model.palette_size, dir_freqs, sec_freqs,
        )
        l_se, _ = compute_blend_loss(
            out["single_edge"], targets["single_edge"], model.palette_size, dir_freqs, sec_freqs,
        )
        loss = w_tile * loss_tiles + l_blend + l_se
        loss.backward()
        opt.step()

        if ep == 1 or ep % log_every == 0 or ep == epochs:
            with torch.no_grad():
                eval_out = model(inputs["elev"], inputs["water"], inputs["coord"],
                                 inputs["style_id"], full_objs)
                W = inputs["W_orig"]; H = inputs["H_orig"]
                # Tile accuracy
                tpred = eval_out["tiles"].argmax(1)[0][:W, :H]
                ttgt = targets["tiles"][:W, :H]
                acc_tiles = (tpred == ttgt).float().mean().item()

                metrics = {"tiles": acc_tiles}
                for key in ("blend", "single_edge"):
                    bo = eval_out[key]
                    bt = targets[key]
                    pp = (torch.sigmoid(bo["present"][0, 0]) > 0.5).long()[:W, :H]
                    pgt = bt["present"][0][:W, :H].long()
                    tp = ((pp == 1) & (pgt == 1)).sum().item()
                    fp = ((pp == 1) & (pgt == 0)).sum().item()
                    fn = ((pp == 0) & (pgt == 1)).sum().item()
                    prec = tp / max(tp + fp, 1)
                    rec = tp / max(tp + fn, 1)
                    f1 = 2 * prec * rec / max(prec + rec, 1e-8)

                    sec_pred = bo["secondary"].argmax(1)[0][:W, :H]
                    dir_pred = bo["direction"].argmax(1)[0][:W, :H]
                    sec_tgt = bt["secondary"][0][:W, :H]
                    dir_tgt = bt["direction"][0][:W, :H]
                    valid_sec = (pgt == 1) & (sec_tgt >= 0)
                    valid_dir = (pgt == 1) & (dir_tgt >= 0)
                    sec_acc = ((sec_pred == sec_tgt) & valid_sec).sum().item() / max(valid_sec.sum().item(), 1)
                    dir_acc = ((dir_pred == dir_tgt) & valid_dir).sum().item() / max(valid_dir.sum().item(), 1)
                    metrics[f"{key}_f1"] = f1
                    metrics[f"{key}_sec"] = sec_acc
                    metrics[f"{key}_dir"] = dir_acc

            ms = " ".join(f"{k}={v*100:5.2f}%" if "tiles" in k or "sec" in k or "dir" in k or "f1" in k
                          else f"{k}={v:.4f}" for k, v in metrics.items())
            print(f"  ep {ep:5d}/{epochs}  loss={loss.item():.4f}  {ms}  ({time.time()-t0:.1f}s)")

    # Final eval
    model.eval()
    with torch.no_grad():
        out = model(inputs["elev"], inputs["water"], inputs["coord"],
                    inputs["style_id"], full_objs)
    return out


# -------------------------- Writeback --------------------------

def writeback_decomp(src_map: Ra3Map, out: dict, palette_size: int,
                     out_path: Path, *, present_threshold: float = 0.5):
    ctx = src_map.get_context()
    blend_asset = ctx.get_asset("BlendTileData")
    W, H = blend_asset.map_width, blend_asset.map_height
    pos_grid = _make_position_pattern_grid(W, H)

    # ---- Tiles
    tile_pred = out["tiles"].argmax(1)[0].cpu().numpy()[:W, :H]
    blend_asset.tiles = (tile_pred.astype(np.uint16) * 64 + pos_grid.astype(np.uint16))

    # Build a fast lookup over the existing blend_info.
    info_list = list(blend_asset.blend_info or [])
    # key: (sec_tex_id, dir_raw, pos_pattern) -> idx
    info_key: dict = {}
    for i, bi in enumerate(info_list):
        # We can't recover sec_tex_id without (x, y) since sec_tex_tile encodes
        # sec_tex_id*64+pos. So we key on (sec_tex_tile, dir_raw); the consumer
        # of info_key supplies the encoded sec_tex_tile directly.
        info_key[(int(bi.secondary_texture_tile), int(bi.blend_direction))] = i

    from map_processor.assets.terrain.blend_info import BlendInfo

    def _ensure_blend_entry(sec_tex_tile: int, dir_raw: int) -> int:
        """Return 1-based index for (sec_tex_tile, dir_raw); appending if needed."""
        key = (sec_tex_tile, dir_raw)
        if key in info_key:
            return info_key[key] + 1
        bi = BlendInfo()
        bi.secondary_texture_tile = sec_tex_tile
        bi.blend_direction = dir_raw  # may not be a BlendDirection enum but stored as int
        bi.i3 = 0
        bi.i4 = 0
        # Provide raw bytes so save_data writes consistently
        bi._blend_direction_raw = bi._from_blend_direction(dir_raw) if hasattr(bi, "_from_blend_direction") else bytes(6)
        info_list.append(bi)
        info_key[key] = len(info_list) - 1
        return len(info_list)

    def _write_blend_array(name: str, attr: str):
        bo = out[name]
        present = (torch.sigmoid(bo["present"][0, 0]) > present_threshold).cpu().numpy()[:W, :H]
        sec = bo["secondary"].argmax(1)[0].cpu().numpy()[:W, :H]
        d = bo["direction"].argmax(1)[0].cpu().numpy()[:W, :H]
        arr = np.zeros((W, H), dtype=np.uint16)
        for x in range(W):
            for y in range(H):
                if not present[x, y]:
                    continue
                sec_id = int(sec[x, y])
                dir_cls = int(d[x, y])
                if dir_cls < 0 or dir_cls >= NUM_DIR_CLASSES:
                    continue
                dir_raw = DIRECTION_VALUES[dir_cls]
                if dir_raw < 0:
                    continue
                sec_tile = sec_id * 64 + int(pos_grid[x, y])
                arr[x, y] = _ensure_blend_entry(sec_tile, dir_raw)
        setattr(blend_asset, attr, arr)

    _write_blend_array("blend", "blends")
    _write_blend_array("single_edge", "single_edge_blends")
    blend_asset.blend_info = info_list
    blend_asset.blends_count = len(info_list)

    # cliff_blends untouched (model does not predict them in this overfit)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    src_map.save(str(out_path), compress=True)


def render(map_path: Path, out_dir: Path):
    import subprocess
    cmd = [sys.executable, str(_python_tools_root() / "scripts" / "generate_map_image.py"),
           str(map_path), str(out_dir)]
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True, timeout=180)
        for png in out_dir.glob(f"{map_path.stem}_terrain_comprehensive.png"):
            return png
    except Exception as e:  # noqa: BLE001
        print(f"  [render-warn] {map_path.name}: {e}")
    return None


# -------------------------- Main --------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", type=Path, required=True)
    ap.add_argument("--epochs", type=int, default=2000)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--out_map", type=Path, default=None)
    ap.add_argument("--no_render", action="store_true")
    ap.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    ap.add_argument("--log_every", type=int, default=100)
    ap.add_argument("--obj_dropout", type=float, default=0.5)
    ap.add_argument("--w_tile", type=float, default=5.0,
                    help="Tile-loss weight; >1 keeps tiles from being washed out by blend losses.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--obj_embed_dim", type=int, default=32)
    ap.add_argument("--base", type=int, default=24,
                    help="Trunk base channels. base=24 → ~5M params; sufficient for one-map overfit.")
    ap.add_argument("--blend_hidden", type=int, default=128)
    args = ap.parse_args()

    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available()
                          else (args.device if args.device != "auto" else "cpu"))
    print(f"Device: {device}")
    print(f"Source: {args.src}")

    m = Ra3Map(str(args.src)); m.parse()
    raw = extract_raw_inputs(m, extract_target=True, style_id=0)
    palette_size = len(raw.palette)
    print(f"Map size: {raw.width} x {raw.height}, palette={palette_size}, objects={len(raw.objects)}")

    type_to_id, owner_to_id = build_vocabs(raw)
    print(f"Vocabs: {len(type_to_id)} unique types, {len(owner_to_id)} unique owners")
    batch_objects = [encode_objects(raw, type_to_id, owner_to_id)]

    model = CascadeTextureNet(
        n_types=max(len(type_to_id), 1),
        n_owners=max(len(owner_to_id), 1),
        palette_size=palette_size,
        n_styles=8,
        obj_embed_dim=args.obj_embed_dim,
        base=args.base,
        n_directions=NUM_DIR_CLASSES,
        blend_hidden=args.blend_hidden,
    ).to(device)

    # Tensors
    elev_t = torch.from_numpy(raw.elev).float().to(device)[None, None]
    water_t = torch.from_numpy(raw.water).float().to(device)[None, None]
    target_tiles = torch.from_numpy(raw.target_tiles).long().to(device)
    W, H = raw.width, raw.height
    xs = torch.linspace(-1.0, 1.0, W, device=device)
    ys = torch.linspace(-1.0, 1.0, H, device=device)
    coord_x = xs[None, None, :, None].expand(1, 1, W, H)
    coord_y = ys[None, None, None, :].expand(1, 1, W, H)
    coord_t = torch.cat([coord_x, coord_y], dim=1)

    elev_p = _pad_t(elev_t)
    water_p = _pad_t(water_t)
    coord_p = _pad_t(coord_t)
    target_tiles_p = _pad_t(target_tiles[None, None].float())[0, 0].long()

    def _pad_blend(b):
        out = {}
        out["present"] = _pad_t(torch.from_numpy(b.present).float().to(device)[None, None])[:, 0]
        out["secondary"] = _pad_t(torch.from_numpy(b.secondary_tex).float().to(device)[None, None])[:, 0].long()
        out["direction"] = _pad_t(torch.from_numpy(b.direction).float().to(device)[None, None])[:, 0].long()
        if b.neighbor_mask is not None:
            out["neighbor"] = _pad_t(torch.from_numpy(b.neighbor_mask).float().to(device)[None, None])[:, 0].long()
        return out

    targets = {
        "tiles": target_tiles_p,
        "blend": _pad_blend(raw.blends),
        "single_edge": _pad_blend(raw.single_edge_blends),
    }
    inputs = {"elev": elev_p, "water": water_p, "coord": coord_p,
              "style_id": torch.tensor([raw.style_id or 0], dtype=torch.long, device=device),
              "W_orig": W, "H_orig": H}

    rng = np.random.default_rng(args.seed)
    out = train_overfit_decomp(
        model, inputs, targets, batch_objects,
        epochs=args.epochs, lr=args.lr, log_every=args.log_every,
        obj_dropout=args.obj_dropout, rng=rng, w_tile=args.w_tile,
    )

    out_map = args.out_map or (args.src.parent / "skinned_full" / "overfit_predicted_e2e_decomp.map")
    print(f"\nWriting predicted map: {out_map}")
    src2 = Ra3Map(str(args.src)); src2.parse()
    writeback_decomp(src2, out, palette_size, out_map)
    if not args.no_render:
        out_render_dir = out_map.parent / "_renders" / out_map.stem
        png = render(out_map, out_render_dir)
        print(f"Rendered: {png}")
    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
