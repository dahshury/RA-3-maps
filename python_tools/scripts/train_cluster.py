#!/usr/bin/env python3
"""Multi-map training across one biome cluster, then eval on a held-out map.

Usage:
  python scripts/train_cluster.py \
    --cluster_json style_clusters/browser/clusters_50.json \
    --target_map map_mp_2_rao1 \
    --target_path "../RA3 Official maps/2 II/map_mp_2_rao1.map" \
    --holdout_patterns rao1 infinity_isle ii_2 \
    --epochs 400 \
    --predict_on "../RA3 Official maps/2 IS/map_mp_2_feasel6.map"

Reuses the v3 cascade recipe (bce_dice + decomposed cascade + class-balanced
focal CE + object dropout). Builds UNION vocabularies for object types, owners,
and texture palette across all training maps in the cluster, so the model has
a single output space across the whole cluster.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

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


# -------------------------- Encoding --------------------------

def _position_pattern_grid(W: int, H: int) -> np.ndarray:
    g = np.zeros((W, H), dtype=np.uint16)
    for x in range(W):
        for y in range(H):
            row_first = (y % 8 // 2) * 16 + (y % 2) * 2
            g[x, y] = (x % 8 // 2) * 4 + (x % 2) + row_first
    return g


def _pad_t(t: torch.Tensor, multiple: int = 16):
    *_, W, H = t.shape
    pw = (-W) % multiple
    ph = (-H) % multiple
    if pw or ph:
        t = F.pad(t, (0, ph, 0, pw), mode="reflect")
    return t


# -------------------------- Losses --------------------------

def focal_bce(logits, target, gamma=2.0):
    p = torch.sigmoid(logits)
    pt = p * target + (1 - p) * (1 - target)
    focal = (1 - pt).pow(gamma)
    bce = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    return (focal * bce).mean()


def soft_dice(logits, target, eps=1e-6):
    p = torch.sigmoid(logits)
    inter = (p * target).sum()
    denom = p.sum() + target.sum() + eps
    return 1.0 - (2 * inter + eps) / denom


def bce_dice(logits, target, gamma=2.0):
    return 0.5 * focal_bce(logits, target, gamma) + 0.5 * soft_dice(logits, target)


def cb_focal_ce(logits, target, class_freqs, *, gamma=2.0, tau=0.3, beta=0.999):
    if class_freqs is not None and tau > 0:
        log_freqs = torch.log(class_freqs.clamp(min=1e-8)) * tau
        logits = logits - log_freqs.unsqueeze(0)
    log_probs = F.log_softmax(logits, dim=-1)
    log_pt = log_probs.gather(1, target.unsqueeze(1)).squeeze(1)
    pt = log_pt.exp().clamp(max=1.0 - 1e-6)
    focal = (1.0 - pt).pow(gamma) if gamma > 0 else 1.0
    per = -focal * log_pt
    if class_freqs is not None and beta > 0:
        eff_n = (1.0 - beta ** class_freqs.clamp(min=1.0)) / (1.0 - beta + 1e-8)
        cls_w = 1.0 / eff_n.clamp(min=1e-8)
        cls_w = cls_w * (cls_w.numel() / cls_w.sum().clamp(min=1e-8))
        w = cls_w[target]
        return (per * w).sum() / w.sum().clamp(min=1e-8)
    return per.mean()


def compute_blend_loss(blend_out, targets, palette_size, dir_freqs, sec_freqs, *,
                       w_present=1.0, w_sec=1.0, w_dir=1.0):
    present_logits = blend_out["present"]
    sec_logits = blend_out["secondary"]
    dir_logits = blend_out["direction"]
    y_present = targets["present"].float()
    y_sec = targets["secondary"].long()
    y_dir = targets["direction"].long()

    loss_present = bce_dice(present_logits.squeeze(1), y_present)
    present_bool = (y_present > 0.5)

    def _flat(t):
        return t.permute(0, 2, 3, 1).reshape(-1, t.shape[1])

    sec_valid = present_bool & (y_sec >= 0) & (y_sec < palette_size)
    if sec_valid.any():
        sl = _flat(sec_logits)[sec_valid.reshape(-1)]
        st = y_sec.reshape(-1)[sec_valid.reshape(-1)]
        loss_sec = cb_focal_ce(sl, st, sec_freqs)
    else:
        loss_sec = present_logits.new_zeros(())

    dir_valid = present_bool & (y_dir >= 0) & (y_dir < NUM_DIR_CLASSES)
    if dir_valid.any():
        dl = _flat(dir_logits)[dir_valid.reshape(-1)]
        dt = y_dir.reshape(-1)[dir_valid.reshape(-1)]
        loss_dir = cb_focal_ce(dl, dt, dir_freqs)
    else:
        loss_dir = present_logits.new_zeros(())

    return w_present * loss_present + w_sec * loss_sec + w_dir * loss_dir


# -------------------------- Cluster + holdout filter --------------------------

def filter_training_maps(cluster_assignments: List[dict], target_cluster: int,
                         holdout_patterns: List[str]) -> List[dict]:
    """Keep only maps in target_cluster whose name doesn't match any holdout pattern (case-insensitive substring)."""
    pats = [p.lower() for p in holdout_patterns]
    out = []
    for a in cluster_assignments:
        if int(a["cluster"]) != target_cluster:
            continue
        nm = a["map_name"].lower()
        if any(p in nm for p in pats):
            continue
        out.append(a)
    return out


# -------------------------- Per-map prep + union vocabs --------------------------

class TrainMap:
    """Container for one map's parsed raw inputs and re-encoded targets."""
    def __init__(self, raw, encoded_objects, target_tiles, target_blends, target_single):
        self.raw = raw
        self.encoded_objects = encoded_objects
        self.target_tiles = target_tiles
        self.target_blends = target_blends
        self.target_single = target_single
        self.W = raw.width
        self.H = raw.height


def re_encode(raw, type_to_id, owner_to_id, palette_to_id, device):
    """Build TrainMap with the union-vocab-encoded targets and torch tensors."""
    enc_objs = [{
        "tile_x": o.tile_x, "tile_y": o.tile_y,
        "type_id": type_to_id.get(o.type_name, 0),
        "owner_id": owner_to_id.get(o.owner, 0),
        "angle_deg": o.angle_deg,
    } for o in raw.objects]

    # Re-encode tiles: old palette idx -> name -> union idx.
    tile_remap = np.full(len(raw.palette), 0, dtype=np.int32)
    for i, name in enumerate(raw.palette):
        tile_remap[i] = palette_to_id.get(name, 0)
    tt = np.where(raw.target_tiles >= 0, tile_remap[raw.target_tiles.clip(min=0)], 0)

    def _remap_blend(blend):
        if blend is None:
            return None
        new_sec = np.where(blend.secondary_tex >= 0,
                           tile_remap[blend.secondary_tex.clip(min=0)], -1)
        return {
            "present": torch.from_numpy(blend.present.astype(np.int32)).to(device),
            "secondary": torch.from_numpy(new_sec.astype(np.int32)).to(device),
            "direction": torch.from_numpy(blend.direction.astype(np.int32)).to(device),
        }

    return TrainMap(
        raw=raw,
        encoded_objects=enc_objs,
        target_tiles=torch.from_numpy(tt.astype(np.int64)).to(device),
        target_blends=_remap_blend(raw.blends),
        target_single=_remap_blend(raw.single_edge_blends),
    )


def build_input_tensors(raw, device):
    """Build (elev, water, coord) padded tensors for one map."""
    elev = torch.from_numpy(raw.elev).float().to(device)[None, None]
    water = torch.from_numpy(raw.water).float().to(device)[None, None]
    W, H = raw.width, raw.height
    xs = torch.linspace(-1.0, 1.0, W, device=device)
    ys = torch.linspace(-1.0, 1.0, H, device=device)
    coord = torch.cat([
        xs[None, None, :, None].expand(1, 1, W, H),
        ys[None, None, None, :].expand(1, 1, W, H),
    ], dim=1)
    return _pad_t(elev), _pad_t(water), _pad_t(coord)


# -------------------------- Train + Eval --------------------------

def train(model, training_maps, *, epochs, lr, log_every,
          obj_dropout, w_tile, palette_size, device, rng):
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.0)
    print(f"Model: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M params, "
          f"palette={palette_size}, training_maps={len(training_maps)}")

    # Pre-pad targets and pre-encode inputs once per map (kept on device).
    cache = []
    dir_freqs = torch.zeros(NUM_DIR_CLASSES, device=device)
    sec_freqs = torch.zeros(palette_size, device=device)
    for tm in training_maps:
        elev, water, coord = build_input_tensors(tm.raw, device)
        target_tiles_p = _pad_t(tm.target_tiles[None, None].float())[0, 0].long()
        targets = {"tiles": target_tiles_p}
        for key, blend in (("blend", tm.target_blends), ("single_edge", tm.target_single)):
            targets[key] = {
                "present": _pad_t(blend["present"][None, None].float())[:, 0],
                "secondary": _pad_t(blend["secondary"][None, None].float())[:, 0].long(),
                "direction": _pad_t(blend["direction"][None, None].float())[:, 0].long(),
            }
            present = targets[key]["present"]
            d = targets[key]["direction"]
            s = targets[key]["secondary"]
            valid_dir = (present > 0.5) & (d >= 0) & (d < NUM_DIR_CLASSES)
            valid_sec = (present > 0.5) & (s >= 0) & (s < palette_size)
            if valid_dir.any():
                for c in range(NUM_DIR_CLASSES):
                    dir_freqs[c] += float(((d == c) & valid_dir).sum().item())
            if valid_sec.any():
                for c in range(palette_size):
                    sec_freqs[c] += float(((s == c) & valid_sec).sum().item())
        cache.append({
            "elev": elev, "water": water, "coord": coord,
            "style_id": torch.tensor([tm.raw.style_id or 0], dtype=torch.long, device=device),
            "objects": tm.encoded_objects,
            "targets": targets,
            "W_orig": tm.W, "H_orig": tm.H,
        })
    dir_freqs = dir_freqs / dir_freqs.sum().clamp(min=1)
    sec_freqs = sec_freqs / sec_freqs.sum().clamp(min=1)

    t0 = time.time()
    model.train()
    perm = list(range(len(cache)))
    for ep in range(1, epochs + 1):
        rng.shuffle(perm)
        epoch_loss = 0.0
        for mi in perm:
            c = cache[mi]
            objs = c["objects"]
            if obj_dropout > 0 and objs:
                keep = rng.random(len(objs)) > obj_dropout
                train_objs = [o for o, k in zip(objs, keep) if k]
            else:
                train_objs = objs
            opt.zero_grad()
            out = model(c["elev"], c["water"], c["coord"], c["style_id"], [train_objs])
            t = c["targets"]
            loss_tiles = F.cross_entropy(out["tiles"], t["tiles"][None])
            loss_blend = compute_blend_loss(out["blend"], t["blend"], palette_size, dir_freqs, sec_freqs)
            loss_single = compute_blend_loss(out["single_edge"], t["single_edge"], palette_size, dir_freqs, sec_freqs)
            loss = w_tile * loss_tiles + loss_blend + loss_single
            loss.backward()
            opt.step()
            epoch_loss += float(loss.item())
        if ep == 1 or ep % log_every == 0 or ep == epochs:
            avg = epoch_loss / max(len(cache), 1)
            # Sample one map for an accuracy snapshot.
            with torch.no_grad():
                sample = cache[0]
                out = model(sample["elev"], sample["water"], sample["coord"],
                            sample["style_id"], [sample["objects"]])
                W = sample["W_orig"]; H = sample["H_orig"]
                tt = sample["targets"]["tiles"][:W, :H]
                tile_acc = (out["tiles"].argmax(1)[0][:W, :H] == tt).float().mean().item()
                bp = (torch.sigmoid(out["blend"]["present"][0, 0]) > 0.5).long()[:W, :H]
                bgt = sample["targets"]["blend"]["present"][0][:W, :H].long()
                tp = ((bp == 1) & (bgt == 1)).sum().item()
                fp = ((bp == 1) & (bgt == 0)).sum().item()
                fn = ((bp == 0) & (bgt == 1)).sum().item()
                bf1 = (2 * tp / max(2 * tp + fp + fn, 1))
            print(f"  ep {ep:4d}/{epochs}  avg_loss={avg:.4f}  "
                  f"sample_tile_acc={tile_acc*100:.2f}%  sample_blend_F1={bf1*100:.2f}%  ({time.time()-t0:.1f}s)")
    return cache, dir_freqs, sec_freqs


def evaluate_on_map(model, target_path, type_to_id, owner_to_id, palette_to_id, palette_list,
                    style_id, device, label="EVAL"):
    """Predict on a held-out map (against its own ground truth, if available),
    and write back a re-textured copy. Returns metrics dict + writeback path.
    """
    print(f"\n=== {label}: {target_path} ===")
    m = Ra3Map(str(target_path)); m.parse()
    raw = extract_raw_inputs(m, extract_target=True, style_id=style_id)
    print(f"  size={raw.width}x{raw.height} palette={len(raw.palette)} objects={len(raw.objects)}")

    palette_size = len(palette_list)
    tm = re_encode(raw, type_to_id, owner_to_id, palette_to_id, device)
    elev, water, coord = build_input_tensors(raw, device)
    style_id_t = torch.tensor([style_id or 0], dtype=torch.long, device=device)

    model.eval()
    with torch.no_grad():
        out = model(elev, water, coord, style_id_t, [tm.encoded_objects])

    W, H = raw.width, raw.height
    pred_tiles = out["tiles"].argmax(1)[0][:W, :H].cpu().numpy()
    gt_tiles = tm.target_tiles[:W, :H].cpu().numpy()
    tile_acc = (pred_tiles == gt_tiles).mean()

    metrics = {"tile_acc": float(tile_acc)}
    for key in ("blend", "single_edge"):
        pp = (torch.sigmoid(out[key]["present"][0, 0]) > 0.5).long()[:W, :H]
        gt_blend = tm.target_blends if key == "blend" else tm.target_single
        pgt = gt_blend["present"][:W, :H].long()
        tp = ((pp == 1) & (pgt == 1)).sum().item()
        fp = ((pp == 1) & (pgt == 0)).sum().item()
        fn = ((pp == 0) & (pgt == 1)).sum().item()
        f1 = 2 * tp / max(2 * tp + fp + fn, 1)
        metrics[f"{key}_f1"] = float(f1)
    print(f"  tile_acc={tile_acc*100:.2f}%  "
          f"blend_f1={metrics['blend_f1']*100:.2f}%  "
          f"single_edge_f1={metrics['single_edge_f1']*100:.2f}%")
    return metrics, raw, out, pred_tiles


def writeback_predictions(target_path, raw, out, pred_tiles, palette_list,
                          out_path: Path):
    """Replace target's tiles + palette with predictions over the union palette,
    and overwrite blends/single_edge_blends per the predicted (present, sec, dir).
    """
    from map_processor.assets.terrain.texture import Texture
    from map_processor.assets.terrain.blend_info import BlendInfo

    src = Ra3Map(str(target_path)); src.parse()
    ctx = src.get_context()
    blend = ctx.get_asset("BlendTileData")
    W, H = blend.map_width, blend.map_height
    pos = _position_pattern_grid(W, H)

    # Replace palette with the union palette. Use defaults for cell_count/size/magic
    # — these are decorative metadata and the engine recomputes from the texture
    # name; copy from existing entries when available, else use a sane default.
    base_meta = blend.textures[0] if blend.textures else None
    new_textures = []
    for name in palette_list:
        t = Texture()
        t.cell_start = 0
        t.cell_count = base_meta.cell_count if base_meta is not None else 16
        t.cell_size = base_meta.cell_size if base_meta is not None else 4
        t.magic_value = base_meta.magic_value if base_meta is not None else 0
        t.name = name
        new_textures.append(t)
    blend.textures = new_textures

    # Tiles
    blend.tiles = (pred_tiles.astype(np.uint16) * 64 + pos.astype(np.uint16))

    # Blends
    info_list: List[BlendInfo] = []
    info_key: Dict[Tuple[int, int], int] = {}

    def _ensure_blend_entry(sec_tex_tile: int, dir_raw: int) -> int:
        key = (sec_tex_tile, dir_raw)
        if key in info_key:
            return info_key[key] + 1
        bi = BlendInfo()
        bi.secondary_texture_tile = sec_tex_tile
        bi.blend_direction = dir_raw
        bi.i3 = 0; bi.i4 = 0
        bi._blend_direction_raw = bi._from_blend_direction(dir_raw) if hasattr(bi, "_from_blend_direction") else bytes(6)
        info_list.append(bi)
        info_key[key] = len(info_list) - 1
        return len(info_list)

    def _write_blend_array(name: str, attr: str):
        bo = out[name]
        present = (torch.sigmoid(bo["present"][0, 0]) > 0.5).cpu().numpy()[:W, :H]
        sec = bo["secondary"].argmax(1)[0].cpu().numpy()[:W, :H]
        d = bo["direction"].argmax(1)[0].cpu().numpy()[:W, :H]
        arr = np.zeros((W, H), dtype=np.uint16)
        for x in range(W):
            for y in range(H):
                if not present[x, y]:
                    continue
                dir_cls = int(d[x, y])
                if dir_cls < 0 or dir_cls >= NUM_DIR_CLASSES:
                    continue
                dir_raw = DIRECTION_VALUES[dir_cls]
                if dir_raw < 0:
                    continue
                sec_tile = int(sec[x, y]) * 64 + int(pos[x, y])
                arr[x, y] = _ensure_blend_entry(sec_tile, dir_raw)
        setattr(blend, attr, arr)

    _write_blend_array("blend", "blends")
    _write_blend_array("single_edge", "single_edge_blends")
    blend.blend_info = info_list
    blend.blends_count = len(info_list)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    src.save(str(out_path), compress=True)
    print(f"  Wrote: {out_path}")


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
    ap.add_argument("--cluster_json", type=Path,
                    default=_python_tools_root() / "style_clusters" / "browser" / "clusters_50.json")
    ap.add_argument("--target_map", required=True,
                    help="Held-out map_name. Its cluster is auto-detected.")
    ap.add_argument("--target_path", required=True, type=Path,
                    help="Path to held-out .map file.")
    ap.add_argument("--holdout_patterns", nargs="*", default=[],
                    help="Substring patterns (case-insensitive) of map_names to exclude from training.")
    ap.add_argument("--epochs", type=int, default=400)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--log_every", type=int, default=10)
    ap.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    ap.add_argument("--obj_dropout", type=float, default=0.5)
    ap.add_argument("--w_tile", type=float, default=5.0)
    ap.add_argument("--obj_embed_dim", type=int, default=32)
    ap.add_argument("--base", type=int, default=24)
    ap.add_argument("--blend_hidden", type=int, default=128)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out_map", type=Path, default=None,
                    help="Output .map for the held-out target prediction.")
    ap.add_argument("--predict_on", type=Path, default=None,
                    help="Optional second map to predict on after training (e.g., 2 IS).")
    ap.add_argument("--no_render", action="store_true")
    args = ap.parse_args()

    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available()
                          else (args.device if args.device != "auto" else "cpu"))
    print(f"Device: {device}")

    cluster_data = json.loads(args.cluster_json.read_text(encoding="utf-8"))
    target_match = next((a for a in cluster_data["assignments"] if a["map_name"] == args.target_map), None)
    if not target_match:
        raise SystemExit(f"Target {args.target_map} not found in {args.cluster_json}")
    cluster_id = int(target_match["cluster"])
    print(f"Target {args.target_map} -> cluster {cluster_id}")
    holdout = list(args.holdout_patterns) + [args.target_map]
    train_members = filter_training_maps(cluster_data["assignments"], cluster_id, holdout)
    print(f"Training maps after filter ({len(holdout)} holdout patterns):")
    for m in train_members:
        print(f"  {m['map_name']:40s}  ({m['folder']})")

    # Parse all training maps + build union vocabs.
    raws = []
    for m in train_members:
        ra3 = Ra3Map(m["map_file"]); ra3.parse()
        raw = extract_raw_inputs(ra3, extract_target=True, style_id=cluster_id)
        raws.append(raw)
    union_types: set = set()
    union_owners: set = set()
    union_palette: set = set()
    for raw in raws:
        union_types.update(o.type_name for o in raw.objects)
        union_owners.update(o.owner for o in raw.objects)
        union_palette.update(raw.palette)
    palette_list = sorted(union_palette)
    type_to_id = {t: i + 1 for i, t in enumerate(sorted(union_types))}
    owner_to_id = {o: i + 1 for i, o in enumerate(sorted(union_owners))}
    palette_to_id = {t: i for i, t in enumerate(palette_list)}
    print(f"Union vocabs: types={len(type_to_id)} owners={len(owner_to_id)} palette={len(palette_list)}")

    training_maps = [re_encode(r, type_to_id, owner_to_id, palette_to_id, device) for r in raws]

    model = CascadeTextureNet(
        n_types=max(len(type_to_id), 1),
        n_owners=max(len(owner_to_id), 1),
        palette_size=len(palette_list),
        n_styles=max(cluster_id + 1, 8),
        obj_embed_dim=args.obj_embed_dim,
        base=args.base,
        n_directions=NUM_DIR_CLASSES,
        blend_hidden=args.blend_hidden,
    ).to(device)

    rng = np.random.default_rng(args.seed)
    train(model, training_maps,
          epochs=args.epochs, lr=args.lr, log_every=args.log_every,
          obj_dropout=args.obj_dropout, w_tile=args.w_tile,
          palette_size=len(palette_list), device=device, rng=rng)

    # Eval on held-out target.
    out_map = args.out_map or (args.target_path.parent / "skinned_full" /
                               f"cluster_predicted_{args.target_map}.map")
    metrics, raw_eval, out_eval, pred_eval = evaluate_on_map(
        model, args.target_path, type_to_id, owner_to_id, palette_to_id, palette_list,
        style_id=cluster_id, device=device, label="EVAL on held-out target",
    )
    writeback_predictions(args.target_path, raw_eval, out_eval, pred_eval,
                          palette_list, out_map)
    if not args.no_render:
        render(out_map, out_map.parent / "_renders" / out_map.stem)

    if args.predict_on is not None and args.predict_on.exists():
        out_other = args.predict_on.parent / "skinned_full" / f"cluster_predicted_{args.predict_on.stem}.map"
        m2, raw2, out2, pred2 = evaluate_on_map(
            model, args.predict_on, type_to_id, owner_to_id, palette_to_id, palette_list,
            style_id=cluster_id, device=device, label="PREDICT on out-of-cluster map",
        )
        writeback_predictions(args.predict_on, raw2, out2, pred2, palette_list, out_other)
        if not args.no_render:
            render(out_other, out_other.parent / "_renders" / out_other.stem)

    print("\nDone.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
