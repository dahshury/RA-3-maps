#!/usr/bin/env python3
"""Multi-cluster training across the full 945-map clustered set.

Each map's style_id = its cluster id from clusters_50.json. The trunk learns
biome-invariant placement features; the style embedding modulates which
palette entry fires per tile. Random-crop step-based loop with D4
augmentation (flip-X, flip-Y, rot-180, rot-90-CCW) for shift- and
rotation-invariance over the conv backbone.

Filter (applied to clusters_50 assignments):
  - exclude:  archon, banmode, ban_, _generated, _pruned, duplicate,
              test_archon, battlebase
  - exclude derivative outputs: *_blendless, *_original, *_predicted

Usage:
  python scripts/train_official.py \
    --target_path "../RA3 Official maps/2 II/map_mp_2_rao1.map" \
    --predict_on "../RA3 Official maps/2 IS/map_mp_2_feasel6.map" \
    --steps 30000
"""
from __future__ import annotations

import argparse
import json
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

# Reuse losses/encoding/writeback helpers from train_cluster.py via import.
from train_cluster import (  # noqa: E402
    _position_pattern_grid, _pad_t,
    bce_dice, cb_focal_ce, compute_blend_loss,
    re_encode, build_input_tensors,
    evaluate_on_map, writeback_predictions, render,
)


EXCLUDE_PATTERNS = [
    "archon", "banmode", "ban_", "_generated", "_pruned", "duplicate",
    "test_archon", "battlebase",
]
DERIV_SUFFIXES = ["_blendless", "_original", "_predicted"]


def is_includable(a: dict) -> bool:
    path = (a.get("map_file") or "").lower()
    if any(x in path for x in EXCLUDE_PATTERNS):
        return False
    name = (a.get("map_name") or "").lower()
    if any(name.endswith(s) for s in DERIV_SUFFIXES):
        return False
    return True


def filter_assignments(assignments: List[dict], holdout_names: List[str]) -> Tuple[List[dict], List[dict]]:
    """Return (training, held_out)."""
    holdouts = {n.lower() for n in holdout_names}
    train, held = [], []
    for a in assignments:
        if not is_includable(a):
            continue
        if a["map_name"].lower() in holdouts:
            held.append(a)
        else:
            train.append(a)
    return train, held


# ---- D4 augmentation: BlendDirection class-index remaps. -------------------
# DIRECTION_VALUES = [-1, 1, 2, 4, 8, 17, 18, 20, 24, 33, 34, 36, 40, 49, 50, 52, 56]
# class:               0  1  2  3  4   5   6   7   8   9  10  11  12  13  14  15  16
# Canonical (1, 2, 4, 8, 17, 18, 20, 24, 36, 40, 52, 56) come from the
# BlendDirection IntFlag (Left/Bottom/.../TopRight). Values 33, 34, 49, 50
# are non-enum bit combos that appear in real maps; we use the class-index
# mapping observed in map_processor/utils/map_rotation.py for rotations,
# and identity for the flips on those four (their semantics are unknown and
# they're rare — accepting tiny inconsistency keeps the augmentation lossless
# for the canonical 12 enum values which dominate the data).
# Cycles under 90-degree CW rotation (derived from canonical tile-edge geometry):
#   Edges:    Left -> Top -> Right -> Bottom -> Left            (1, 18, 17, 2)
#   Excepts:  4 -> 20 -> 24 -> 8 -> 4                            ("Except<corner>")
#   Corners:  36 -> 52 -> 56 -> 40 -> 36                          (BL, TL, TR, BR)
#   Unknown:  33 -> 50 -> 49 -> 34 -> 33                          (kept as observed)
# Note: this fixes a cycle-direction inconsistency in the historical
# map_processor/utils/map_rotation._ROT_90 table for the Except* group;
# our derivation guarantees r90^2 == r180 and r90^4 == identity.
DIR_CLASS_FLIP_X = [0, 5, 2, 4, 3, 1, 6, 8, 7, 13, 14, 12, 11, 9, 10, 16, 15]
DIR_CLASS_FLIP_Y = [0, 1, 6, 7, 8, 5, 2, 3, 4, 9, 10, 15, 16, 13, 14, 11, 12]
DIR_CLASS_ROT180 = [0, 5, 6, 8, 7, 1, 2, 4, 3, 13, 14, 16, 15, 9, 10, 12, 11]
DIR_CLASS_ROT90_CCW = [0, 2, 5, 4, 8, 6, 1, 3, 7, 10, 13, 12, 16, 14, 9, 11, 15]


def _augment_crop(elev, water, coord, target_tiles, blend, single_edge,
                  objects, *, aug: str):
    """Apply a D4 augmentation to all spatial channels and to object positions
    + angles. `aug` in {'identity', 'flip_x', 'flip_y', 'rot180', 'rot90_ccw'}.

    Tensors are (1, C, W, H) for inputs and (W, H) for targets. flip-X reverses
    the W axis; flip-Y reverses the H axis. rot-90-CCW transposes (W, H) ->
    (H, W) (only valid when caller guarantees a square crop).
    """
    if aug == "identity":
        return elev, water, coord, target_tiles, blend, single_edge, objects

    _, _, W, H = elev.shape

    def _flip_spatial(t, *, w_axis: int, h_axis: int, flip_w: bool, flip_h: bool):
        if flip_w:
            t = t.flip(w_axis)
        if flip_h:
            t = t.flip(h_axis)
        return t

    def _flip_blend(b, *, flip_w: bool, flip_h: bool, dir_remap):
        out = {}
        for k, v in b.items():
            ax_w, ax_h = (0, 1) if v.dim() == 2 else (-2, -1)
            v2 = _flip_spatial(v, w_axis=ax_w, h_axis=ax_h, flip_w=flip_w, flip_h=flip_h)
            if k == "direction" and dir_remap is not None:
                lut = torch.tensor(dir_remap, dtype=v2.dtype, device=v2.device)
                # direction holds -1 in "no blend" cells; only remap valid classes.
                valid = (v2 >= 0) & (v2 < lut.numel())
                v2 = torch.where(valid, lut[v2.clamp(min=0)], v2)
            out[k] = v2
        return out

    if aug in ("flip_x", "flip_y", "rot180"):
        flip_w = aug in ("flip_x", "rot180")
        flip_h = aug in ("flip_y", "rot180")
        if aug == "flip_x":
            dir_lut = DIR_CLASS_FLIP_X
            angle_fn = lambda a: (180.0 - a) % 360.0
        elif aug == "flip_y":
            dir_lut = DIR_CLASS_FLIP_Y
            angle_fn = lambda a: (360.0 - a) % 360.0
        else:
            dir_lut = DIR_CLASS_ROT180
            angle_fn = lambda a: (a - 180.0) % 360.0

        elev_a = _flip_spatial(elev, w_axis=-2, h_axis=-1, flip_w=flip_w, flip_h=flip_h)
        water_a = _flip_spatial(water, w_axis=-2, h_axis=-1, flip_w=flip_w, flip_h=flip_h)
        # coord channel 0 = x_coord (varies along W); flipping W negates it. ch1 = y_coord.
        coord_a = _flip_spatial(coord.clone(), w_axis=-2, h_axis=-1, flip_w=flip_w, flip_h=flip_h)
        if flip_w:
            coord_a[:, 0] = -coord_a[:, 0]
        if flip_h:
            coord_a[:, 1] = -coord_a[:, 1]
        tt_a = _flip_spatial(target_tiles, w_axis=0, h_axis=1, flip_w=flip_w, flip_h=flip_h)
        blend_a = _flip_blend(blend, flip_w=flip_w, flip_h=flip_h, dir_remap=dir_lut)
        single_a = _flip_blend(single_edge, flip_w=flip_w, flip_h=flip_h, dir_remap=dir_lut)
        objs_a = []
        for o in objects:
            tx, ty = o["tile_x"], o["tile_y"]
            if flip_w:
                tx = (W - 1) - tx
            if flip_h:
                ty = (H - 1) - ty
            objs_a.append({**o, "tile_x": tx, "tile_y": ty,
                           "angle_deg": angle_fn(o["angle_deg"])})
        return elev_a, water_a, coord_a, tt_a, blend_a, single_a, objs_a

    if aug == "rot90_ccw":
        if W != H:
            return elev, water, coord, target_tiles, blend, single_edge, objects
        # CCW on a (W, H)=(x, y) array: new[y, W-1-x] = old[x, y]; equivalently
        # new = transpose(old) then flip along the new W axis (the old y).
        # In tensor form: rot90 on the last two dims with k=1 (CCW) — but
        # torch.rot90's "CCW" matches numpy: rot90(A, k=1, dims=(-2,-1)).
        elev_a = torch.rot90(elev, k=1, dims=(-2, -1))
        water_a = torch.rot90(water, k=1, dims=(-2, -1))
        coord_a = torch.rot90(coord.clone(), k=1, dims=(-2, -1))
        # After rot-90-CCW: new x_coord = old y_coord, new y_coord = -old x_coord.
        ch0 = coord_a[:, 0].clone()
        ch1 = coord_a[:, 1].clone()
        # rot90 already permuted spatially; we still need to swap channels +
        # negate appropriately so each channel matches its meaning post-rotation.
        coord_a[:, 0] = ch1
        coord_a[:, 1] = -ch0

        def _rot_blend(b):
            out = {}
            for k, v in b.items():
                if v.dim() == 2:
                    v2 = torch.rot90(v, k=1, dims=(0, 1))
                else:
                    v2 = torch.rot90(v, k=1, dims=(-2, -1))
                if k == "direction":
                    lut = torch.tensor(DIR_CLASS_ROT90_CCW, dtype=v2.dtype, device=v2.device)
                    valid = (v2 >= 0) & (v2 < lut.numel())
                    v2 = torch.where(valid, lut[v2.clamp(min=0)], v2)
                out[k] = v2
            return out

        tt_a = torch.rot90(target_tiles, k=1, dims=(0, 1))
        blend_a = _rot_blend(blend)
        single_a = _rot_blend(single_edge)
        # rot-90-CCW on tile indices: new_x = old_y, new_y = (W-1) - old_x.
        objs_a = []
        for o in objects:
            tx, ty = o["tile_x"], o["tile_y"]
            new_x = ty
            new_y = (W - 1) - tx
            objs_a.append({**o, "tile_x": new_x, "tile_y": new_y,
                           "angle_deg": (o["angle_deg"] + 90.0) % 360.0})
        return elev_a, water_a, coord_a, tt_a, blend_a, single_a, objs_a

    raise ValueError(f"Unknown aug: {aug}")


def _crop_view(elev, water, coord, target_tiles_full, target_blend_full,
               target_single_full, encoded_objects, *, crop, rng):
    """Random crop of size `crop` (multiple of 16). Returns the cropped tensors
    and an objects list translated to crop-local coords. If a map is smaller
    than `crop` along an axis, that axis is taken in full (no crop)."""
    _, _, W, H = elev.shape
    cw = min(crop, W)
    ch = min(crop, H)
    cw -= cw % 16
    ch -= ch % 16
    if cw == 0 or ch == 0:
        cw = max(16, W - W % 16); ch = max(16, H - H % 16)
    cx = int(rng.integers(0, max(W - cw + 1, 1))) if W > cw else 0
    cy = int(rng.integers(0, max(H - ch + 1, 1))) if H > ch else 0
    elev_c = elev[..., cx:cx + cw, cy:cy + ch]
    water_c = water[..., cx:cx + cw, cy:cy + ch]
    coord_c = coord[..., cx:cx + cw, cy:cy + ch]
    tiles_c = target_tiles_full[cx:cx + cw, cy:cy + ch]
    blend_c = {k: v[cx:cx + cw, cy:cy + ch] if v.dim() == 2 else v[..., cx:cx + cw, cy:cy + ch]
               for k, v in target_blend_full.items()}
    single_c = {k: v[cx:cx + cw, cy:cy + ch] if v.dim() == 2 else v[..., cx:cx + cw, cy:cy + ch]
                for k, v in target_single_full.items()}
    objs_c = []
    for o in encoded_objects:
        if cx <= o["tile_x"] < cx + cw and cy <= o["tile_y"] < cy + ch:
            objs_c.append({**o, "tile_x": o["tile_x"] - cx, "tile_y": o["tile_y"] - cy})
    return elev_c, water_c, coord_c, tiles_c, blend_c, single_c, objs_c


def train(model, training_maps, *, steps, lr, log_every,
          obj_dropout, w_tile, palette_size, device, rng,
          crop: int = 256, augment: bool = True):
    """Random-crop step-based training across all training_maps.

    Each step picks a random map and a random crop_size×crop_size window from
    it, runs forward+backward on the crop. Avoids the cost of a full-map
    forward on 720×720 maps and gives uniform per-step cost.
    """
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.0)
    print(f"Model: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M params, "
          f"palette={palette_size}, training_maps={len(training_maps)}, crop={crop}, steps={steps}")

    # Pre-pad and cache full-map tensors. Sec/dir freqs come from raw targets.
    cache = []
    dir_freqs_np = np.zeros(NUM_DIR_CLASSES, dtype=np.float64)
    sec_freqs_np = np.zeros(palette_size, dtype=np.float64)
    print("Pre-padding inputs/targets per map...")

    def _pad_target(t):
        """Pad a (W, H) tensor to (W', H') with multiple-of-16 sides (constant 0)."""
        x = t[None, None].float()
        return _pad_t(x, 16)[0, 0].to(t.dtype)

    for tm in training_maps:
        elev, water, coord = build_input_tensors(tm.raw, device)
        # Pad all targets to match the padded spatial dims of elev.
        full = {
            "tiles": _pad_target(tm.target_tiles),
            "blend": {
                "present": _pad_target(tm.target_blends["present"]),
                "secondary": _pad_target(tm.target_blends["secondary"]),
                "direction": _pad_target(tm.target_blends["direction"]),
            },
            "single_edge": {
                "present": _pad_target(tm.target_single["present"]),
                "secondary": _pad_target(tm.target_single["secondary"]),
                "direction": _pad_target(tm.target_single["direction"]),
            },
        }
        # Class-frequency accumulation via numpy on CPU (fast).
        for key in ("blend", "single_edge"):
            present = full[key]["present"].cpu().numpy()
            d = full[key]["direction"].cpu().numpy()
            s = full[key]["secondary"].cpu().numpy()
            mask_p = present > 0.5
            md = mask_p & (d >= 0) & (d < NUM_DIR_CLASSES)
            if md.any():
                dir_freqs_np += np.bincount(d[md].astype(np.int64), minlength=NUM_DIR_CLASSES)
            ms = mask_p & (s >= 0) & (s < palette_size)
            if ms.any():
                sec_freqs_np += np.bincount(s[ms].astype(np.int64), minlength=palette_size)
        cache.append({
            "elev": elev, "water": water, "coord": coord,
            "style_id": torch.tensor([tm.raw.style_id or 0], dtype=torch.long, device=device),
            "objects": tm.encoded_objects,
            "full": full,
            "W": tm.W, "H": tm.H,
        })
    dir_freqs = torch.from_numpy(dir_freqs_np / max(dir_freqs_np.sum(), 1)).float().to(device)
    sec_freqs = torch.from_numpy(sec_freqs_np / max(sec_freqs_np.sum(), 1)).float().to(device)
    print(f"  freqs: dir nonzero classes={int((dir_freqs > 0).sum())}, "
          f"sec nonzero classes={int((sec_freqs > 0).sum())}")

    t0 = time.time()
    model.train()
    losses = []
    for step in range(1, steps + 1):
        mi = int(rng.integers(0, len(cache)))
        c = cache[mi]
        elev_c, water_c, coord_c, tiles_c, blend_c, single_c, objs_c = _crop_view(
            c["elev"], c["water"], c["coord"], c["full"]["tiles"],
            c["full"]["blend"], c["full"]["single_edge"], c["objects"], crop=crop, rng=rng,
        )
        if augment:
            _, _, cw, ch = elev_c.shape
            choices = ["identity", "flip_x", "flip_y", "rot180"]
            if cw == ch:
                choices.append("rot90_ccw")
            aug = choices[int(rng.integers(0, len(choices)))]
            elev_c, water_c, coord_c, tiles_c, blend_c, single_c, objs_c = _augment_crop(
                elev_c, water_c, coord_c, tiles_c, blend_c, single_c, objs_c, aug=aug,
            )
        if obj_dropout > 0 and objs_c:
            keep = rng.random(len(objs_c)) > obj_dropout
            objs_c = [o for o, k in zip(objs_c, keep) if k]

        opt.zero_grad()
        out = model(elev_c, water_c, coord_c, c["style_id"], [objs_c])
        loss_tiles = F.cross_entropy(out["tiles"], tiles_c.long()[None])
        targets_blend = {k: v[None] if v.dim() == 2 else v for k, v in blend_c.items()}
        targets_single = {k: v[None] if v.dim() == 2 else v for k, v in single_c.items()}
        l_b = compute_blend_loss(out["blend"], targets_blend, palette_size, dir_freqs, sec_freqs)
        l_s = compute_blend_loss(out["single_edge"], targets_single, palette_size, dir_freqs, sec_freqs)
        loss = w_tile * loss_tiles + l_b + l_s
        loss.backward()
        opt.step()
        losses.append(float(loss.item()))

        if step == 1 or step % log_every == 0 or step == steps:
            recent = float(np.mean(losses[-log_every:]))
            with torch.no_grad():
                # Sample full-map eval on a random training map (un-padded slice).
                mi_eval = int(rng.integers(0, len(cache)))
                ce = cache[mi_eval]
                out_full = model(ce["elev"], ce["water"], ce["coord"], ce["style_id"], [ce["objects"]])
                W = ce["W"]; H = ce["H"]
                pred_t = out_full["tiles"].argmax(1)[0][:W, :H]
                tgt_t = ce["full"]["tiles"][:W, :H]
                tile_acc = (pred_t == tgt_t).float().mean().item()
                bp = (torch.sigmoid(out_full["blend"]["present"][0, 0]) > 0.5).long()[:W, :H]
                bgt = ce["full"]["blend"]["present"][:W, :H].long()
                tp = ((bp == 1) & (bgt == 1)).sum().item()
                fp = ((bp == 1) & (bgt == 0)).sum().item()
                fn = ((bp == 0) & (bgt == 1)).sum().item()
                bf1 = 2 * tp / max(2 * tp + fp + fn, 1)
            print(f"  step {step:5d}/{steps}  recent_loss={recent:.4f}  "
                  f"sample_tile_acc={tile_acc*100:.2f}%  sample_blend_F1={bf1*100:.2f}%  ({time.time()-t0:.1f}s)")
    return cache, dir_freqs, sec_freqs


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cluster_json", type=Path,
                    default=_python_tools_root() / "style_clusters" / "browser" / "clusters_50.json")
    ap.add_argument("--target_path", type=Path, required=True,
                    help="Held-out .map file. Its map_name is auto-derived for filtering.")
    ap.add_argument("--predict_on", type=Path, default=None,
                    help="Optional second map to predict on after training.")
    ap.add_argument("--steps", type=int, default=20000,
                    help="Total optimisation steps (random-crop sampling). 1 step = 1 random crop from 1 random map.")
    ap.add_argument("--crop", type=int, default=256,
                    help="Crop window side (in tiles). Must be a multiple of 16.")
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--log_every", type=int, default=200)
    ap.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    ap.add_argument("--obj_dropout", type=float, default=0.5)
    ap.add_argument("--w_tile", type=float, default=5.0)
    ap.add_argument("--obj_embed_dim", type=int, default=32)
    ap.add_argument("--base", type=int, default=24)
    ap.add_argument("--blend_hidden", type=int, default=128)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out_map", type=Path, default=None)
    ap.add_argument("--no_render", action="store_true")
    ap.add_argument("--no_augment", action="store_true",
                    help="Disable D4 augmentation (flip/rot).")
    ap.add_argument("--max_train_maps", type=int, default=0,
                    help="If >0, randomly subsample training maps to this many (debug).")
    args = ap.parse_args()

    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available()
                          else (args.device if args.device != "auto" else "cpu"))
    print(f"Device: {device}")

    cluster_data = json.loads(args.cluster_json.read_text(encoding="utf-8"))
    target_name = args.target_path.stem
    extra_holdouts = [args.predict_on.stem] if args.predict_on is not None else []
    train_assignments, held_assignments = filter_assignments(
        cluster_data["assignments"], holdout_names=[target_name] + extra_holdouts,
    )
    print(f"Cluster set: train={len(train_assignments)}  held_out={len(held_assignments)}")
    held_target = next((a for a in held_assignments if a["map_name"] == target_name), None)
    if held_target is None:
        # Target may not be in clusters_50 (e.g. unparseable map). Look up its
        # cluster id from the full assignment list before bailing.
        any_match = next((a for a in cluster_data["assignments"]
                          if a["map_name"] == target_name), None)
        if any_match is None:
            raise SystemExit(f"Target {target_name} not found in clusters_50.json.")
        held_target = any_match
    target_cluster = int(held_target["cluster"])
    print(f"Target {target_name}: cluster={target_cluster}")

    if args.max_train_maps and args.max_train_maps > 0 and len(train_assignments) > args.max_train_maps:
        rng_subset = np.random.default_rng(args.seed)
        idxs = rng_subset.choice(len(train_assignments), size=args.max_train_maps, replace=False)
        train_assignments = [train_assignments[i] for i in idxs]
        print(f"  subsampled training set to {len(train_assignments)} maps")

    # Cluster sizes among training maps
    from collections import Counter
    cluster_sizes = Counter(a["cluster"] for a in train_assignments)
    print(f"Training cluster distribution: {len(cluster_sizes)} unique style ids; "
          f"max size={max(cluster_sizes.values())}, singletons={sum(1 for v in cluster_sizes.values() if v == 1)}")

    # Parse all training maps + build union vocabs.
    print(f"\nParsing {len(train_assignments)} training maps ...")
    raws = []
    skipped = 0
    for ti, m in enumerate(train_assignments):
        try:
            ra3 = Ra3Map(m["map_file"]); ra3.parse()
            raw = extract_raw_inputs(ra3, extract_target=True, style_id=int(m["cluster"]))
        except Exception as ex:
            print(f"  SKIP {m['map_name']}: parse error: {ex}")
            skipped += 1
            continue
        # Validate: target_tiles indices must fit the parsed palette. Some
        # community maps have texture-index overflow (palette tiny, indices
        # large) — skip rather than train on garbage targets.
        n_pal = len(raw.palette or [])
        if n_pal == 0 or raw.target_tiles is None:
            print(f"  SKIP {m['map_name']}: empty palette / no target tiles")
            skipped += 1
            continue
        max_idx = int(raw.target_tiles.max(initial=-1))
        if max_idx >= n_pal:
            print(f"  SKIP {m['map_name']}: target_tile idx {max_idx} >= palette size {n_pal}")
            skipped += 1
            continue
        # Same safety check for blend secondary textures.
        bad_blend = False
        for bd in (raw.blends, raw.single_edge_blends):
            if bd is None:
                continue
            sec = bd.secondary_tex
            if sec.size and int(sec.max(initial=-1)) >= n_pal:
                bad_blend = True
                break
        if bad_blend:
            print(f"  SKIP {m['map_name']}: blend secondary_tex idx >= palette size {n_pal}")
            skipped += 1
            continue
        raws.append(raw)
        if (ti + 1) % 10 == 0:
            print(f"  parsed {ti + 1}/{len(train_assignments)}  (kept={len(raws)} skipped={skipped})")
    print(f"  parsed: kept={len(raws)} skipped={skipped}")
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
    n_styles = max(int(a["cluster"]) for a in train_assignments) + 1
    n_styles = max(n_styles, target_cluster + 1)
    print(f"Union vocabs: types={len(type_to_id)} owners={len(owner_to_id)} palette={len(palette_list)} "
          f"n_styles={n_styles}")

    training_maps = [re_encode(r, type_to_id, owner_to_id, palette_to_id, device) for r in raws]

    model = CascadeTextureNet(
        n_types=max(len(type_to_id), 1),
        n_owners=max(len(owner_to_id), 1),
        palette_size=len(palette_list),
        n_styles=n_styles,
        obj_embed_dim=args.obj_embed_dim,
        base=args.base,
        n_directions=NUM_DIR_CLASSES,
        blend_hidden=args.blend_hidden,
    ).to(device)

    rng = np.random.default_rng(args.seed)
    train(model, training_maps,
          steps=args.steps, lr=args.lr, log_every=args.log_every,
          obj_dropout=args.obj_dropout, w_tile=args.w_tile,
          palette_size=len(palette_list), device=device, rng=rng,
          crop=args.crop, augment=(not args.no_augment))

    # Eval on held-out target with its true cluster id.
    out_map = args.out_map or (args.target_path.parent / "skinned_full" /
                               f"official_predicted_{target_name}.map")
    metrics, raw_eval, out_eval, pred_eval = evaluate_on_map(
        model, args.target_path, type_to_id, owner_to_id, palette_to_id, palette_list,
        style_id=target_cluster, device=device,
        label=f"EVAL on held-out target (style_id={target_cluster})",
    )
    writeback_predictions(args.target_path, raw_eval, out_eval, pred_eval, palette_list, out_map)
    if not args.no_render:
        render(out_map, out_map.parent / "_renders" / out_map.stem)

    if args.predict_on is not None and args.predict_on.exists():
        # Look up the predict_on map's cluster id (it might be in training).
        pname = args.predict_on.stem
        a2 = next((a for a in cluster_data["assignments"] if a["map_name"] == pname), None)
        sid = int(a2["cluster"]) if a2 else target_cluster
        out_other = args.predict_on.parent / "skinned_full" / f"official_predicted_{pname}.map"
        metrics2, raw2, out2, pred2 = evaluate_on_map(
            model, args.predict_on, type_to_id, owner_to_id, palette_to_id, palette_list,
            style_id=sid, device=device,
            label=f"PREDICT on {pname} (style_id={sid})",
        )
        writeback_predictions(args.predict_on, raw2, out2, pred2, palette_list, out_other)
        if not args.no_render:
            render(out_other, out_other.parent / "_renders" / out_other.stem)

    print("\nDone.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
