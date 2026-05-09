#!/usr/bin/env python3
"""train_v2.py — v2 architecture training driver.

Implements items 1, 3, 7, 8 of architecture_research_2026-05.md:
  * SegFormer-B0 backbone (~3.7M, ImageNet-init, mit-b0 weights) with FiLM
    style modulation at the all-MLP decoder.
  * DINOv2 reference-patch style encoder (frozen ViT-S/14) replacing
    cluster-id embeddings — solves the singleton-cluster cold-id problem.
  * Loss stack: label-smoothed CE on tile, BCE-Dice on blend present,
    logit-adjusted CE on secondary/direction, Focal Frequency Loss on
    rendered RGB through Gumbel-softmax-ST. ProjectedGAN/ConvCRF/MaskGiT
    deferred to follow-up tasks.
  * WeightedRandomSampler with weight = 1/sqrt(cluster_size) to oversample
    rare-style maps.
  * D4 augmentation (flip-X / flip-Y / rot-180 / rot-90-CCW) inherited
    from train_official.py.

Filters: drop archon/banmode/test_archon/derivative outputs from the
clusters_50 assignment list. Dedup by exact map_name. Cluster ID is
retained ONLY for sampler weighting; the model never sees it.

Usage:
  python scripts/train_v2.py --target_path "../RA3 Official maps/2 II/map_mp_2_rao1.map" \
                             --predict_on  "../RA3 Official maps/2 IS/map_mp_2_feasel6.map" \
                             --steps 30000
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F


def _python_tools_root() -> Path:
    return Path(__file__).resolve().parents[1]


sys.path.insert(0, str(_python_tools_root()))
sys.path.insert(0, str(_python_tools_root() / "scripts"))

from map_processor import Ra3Map  # noqa: E402
from map_processor.features import extract_raw_inputs  # noqa: E402
from map_processor.features.raw_inputs import NUM_DIR_CLASSES  # noqa: E402
from map_processor.models.end_to_end_unet import ObjectStamper  # noqa: E402
from map_processor.models.v2_model import V2TextureNet  # noqa: E402
from map_processor.models.v2_style import DinoV2StyleEncoder  # noqa: E402
from map_processor.models.v2_render import PaletteRenderer  # noqa: E402
from map_processor.models.v2_loss import V2LossModule, cluster_balanced_weights  # noqa: E402

# Inherit D4 augmentation tables + helpers and the writeback path.
from train_official import (  # noqa: E402
    DIR_CLASS_FLIP_X, DIR_CLASS_FLIP_Y, DIR_CLASS_ROT180, DIR_CLASS_ROT90_CCW,
    _augment_crop, _crop_view, EXCLUDE_PATTERNS, DERIV_SUFFIXES, is_includable,
)
from train_cluster import _pad_t, writeback_predictions, render  # noqa: E402


# ---------------------------- Data prep ----------------------------

def filter_and_dedup(assignments: List[dict], holdout_names: List[str]
                     ) -> Tuple[List[dict], List[dict]]:
    """Apply patterns filter, dedup by map_name, split holdouts."""
    holdouts = {n.lower() for n in holdout_names}
    seen: set = set()
    train, held = [], []
    for a in assignments:
        if not is_includable(a):
            continue
        key = a["map_name"].lower()
        if key in seen:
            continue
        seen.add(key)
        if key in holdouts:
            held.append(a)
        else:
            train.append(a)
    return train, held


def re_encode_for_v2(raw, type_to_id, owner_to_id, palette_to_id):
    """Convert RawInputs.objects + targets to numpy/torch with union ids."""
    enc_objs = [{
        "tile_x": float(o.tile_x), "tile_y": float(o.tile_y),
        "type_id": int(type_to_id.get(o.type_name, 0)),
        "owner_id": int(owner_to_id.get(o.owner, 0)),
        "angle_deg": float(o.angle_deg),
    } for o in raw.objects]
    tile_remap = np.zeros(len(raw.palette), dtype=np.int32)
    for i, name in enumerate(raw.palette):
        tile_remap[i] = int(palette_to_id.get(name, 0))
    target_tiles = tile_remap[raw.target_tiles.clip(min=0)] if raw.target_tiles is not None else None

    def remap_blend(b):
        if b is None:
            return None
        new_sec = np.where(b.secondary_tex >= 0,
                           tile_remap[b.secondary_tex.clip(min=0)],
                           -1).astype(np.int32)
        return {
            "present": torch.from_numpy(b.present.astype(np.int64)),
            "secondary": torch.from_numpy(new_sec.astype(np.int64)),
            "direction": torch.from_numpy(b.direction.astype(np.int64)),
        }

    return {
        "elev": torch.from_numpy(raw.elev).float(),
        "water": torch.from_numpy(raw.water).float(),
        "width": int(raw.width), "height": int(raw.height),
        "objects": enc_objs,
        "mp_spawns": list(raw.mp_spawns or []),
        "target_tiles": torch.from_numpy(target_tiles.astype(np.int64)),
        "target_blends": remap_blend(raw.blends),
        "target_single": remap_blend(raw.single_edge_blends),
    }


def _mp_spawn_grid(spawns, W: int, H: int) -> torch.Tensor:
    g = torch.zeros(W, H, dtype=torch.float32)
    for x, y in spawns or []:
        ix = int(round(x)); iy = int(round(y))
        if 0 <= ix < W and 0 <= iy < H:
            g[ix, iy] = 1.0
    return g


def build_dense_input(record, stamper: ObjectStamper, device) -> torch.Tensor:
    """Build SegFormer 8-channel dense input: elev, water, coord_x, coord_y,
    obj_stamp_pca (3), mp_spawn -> 8 channels. The ObjectStamper output is
    `obj_embed_dim` channels; we project it down to 3 with a fixed random
    projection to keep the SegFormer in_channels small.
    """
    elev = record["elev"].to(device)
    water = record["water"].to(device)
    W, H = record["width"], record["height"]
    xs = torch.linspace(-1.0, 1.0, W, device=device)
    ys = torch.linspace(-1.0, 1.0, H, device=device)
    cx = xs[:, None].expand(W, H)
    cy = ys[None, :].expand(W, H)
    mp_spawn = _mp_spawn_grid(record["mp_spawns"], W, H).to(device)
    return torch.stack([elev, water, cx, cy, mp_spawn], dim=0)  # (5, W, H)


# ---------------------------- Train ----------------------------

def _prepare_full_render(record, palette_renderer: PaletteRenderer, device) -> torch.Tensor:
    """Render the whole map's target tiles to RGB once, for style sampling."""
    tt = record["target_tiles"].to(device).long()[None]    # (1, W, H)
    return palette_renderer.hard_render(tt)[0]             # (3, W, H)


def _sample_style_patch(rgb_full: torch.Tensor, *, exclude_box: Tuple[int, int, int, int],
                        size: int, rng: np.random.Generator) -> torch.Tensor:
    """rgb_full: (3, W, H). Pick a random `size`x`size` patch, biasing away
    from the input crop (exclude_box = (cx, cy, cw, ch)). Returns (1,3,size,size).
    """
    _, W, H = rgb_full.shape
    side = min(size, W, H)
    if side <= 0:
        return torch.zeros(1, 3, size, size, device=rgb_full.device)
    cx, cy, cw, ch = exclude_box
    for _ in range(8):
        sx = int(rng.integers(0, max(W - side + 1, 1)))
        sy = int(rng.integers(0, max(H - side + 1, 1)))
        # If overlaps the input crop by >50% of area, retry.
        ox = max(0, min(sx + side, cx + cw) - max(sx, cx))
        oy = max(0, min(sy + side, cy + ch) - max(sy, cy))
        if ox * oy < 0.5 * side * side:
            break
    patch = rgb_full[:, sx:sx + side, sy:sy + side]
    if side != size:
        patch = F.interpolate(patch[None], size=(size, size), mode="bilinear", align_corners=False)[0]
    return patch[None]


def _pad_target(t: torch.Tensor, multiple: int) -> torch.Tensor:
    return _pad_t(t[None, None].float(), multiple)[0, 0].to(t.dtype)


def train_loop(model, stamper, style_enc, palette_renderer, loss_mod,
               cache, *, steps, lr, log_every, w_obj, augment, crop, sampler,
               cfg_dropout, device, rng, dir_freqs, sec_freqs, multiple=32):
    # Collect trainable parameters (DINOv2 backbone is frozen, the projector
    # and null_style are trainable; PaletteRenderer.residual is trainable).
    params: list = []
    seen_ids: set = set()
    for m in (model, stamper, style_enc, palette_renderer):
        for p in m.parameters():
            if p.requires_grad and id(p) not in seen_ids:
                params.append(p); seen_ids.add(id(p))
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.0)
    n_total = sum(p.numel() for p in params)
    print(f"Trainable params: {n_total/1e6:.2f}M (incl. SegFormer + heads + stamper + style projector)")

    t0 = time.time()
    losses: list[float] = []
    model.train(); stamper.train(); style_enc.train(); loss_mod.train()
    for step in range(1, steps + 1):
        mi = sampler.sample(rng)
        c = cache[mi]
        # Crop selection on padded dense tensor (multiples of `multiple`).
        _, Wp, Hp = c["dense"].shape
        cw = min(crop, Wp); ch = min(crop, Hp)
        cw -= cw % multiple; ch -= ch % multiple
        if cw == 0: cw = max(multiple, Wp - Wp % multiple)
        if ch == 0: ch = max(multiple, Hp - Hp % multiple)
        cx = int(rng.integers(0, max(Wp - cw + 1, 1))) if Wp > cw else 0
        cy = int(rng.integers(0, max(Hp - ch + 1, 1))) if Hp > ch else 0
        # Slice on CPU then move to GPU (small per-step transfer).
        dense_c = c["dense"][:, cx:cx + cw, cy:cy + ch].unsqueeze(0).to(device, non_blocking=True)
        tt_c = c["target_tiles"][cx:cx + cw, cy:cy + ch].to(device, non_blocking=True)
        blend_c = {k: v[cx:cx + cw, cy:cy + ch].to(device, non_blocking=True)
                   for k, v in c["target_blends"].items()}
        single_c = {k: v[cx:cx + cw, cy:cy + ch].to(device, non_blocking=True)
                    for k, v in c["target_single"].items()}
        obj_c = [
            {**o, "tile_x": o["tile_x"] - cx, "tile_y": o["tile_y"] - cy}
            for o in c["objects"]
            if cx <= o["tile_x"] < cx + cw and cy <= o["tile_y"] < cy + ch
        ]
        # D4 augmentation.
        if augment:
            choices = ["identity", "flip_x", "flip_y", "rot180"]
            if cw == ch: choices.append("rot90_ccw")
            aug = choices[int(rng.integers(0, len(choices)))]
            elev_a = dense_c[:, 0:1]; water_a = dense_c[:, 1:2]; coord_a = dense_c[:, 2:4]
            ea, wa, ca, tt_c, blend_c, single_c, obj_c = _augment_crop(
                elev_a, water_a, coord_a, tt_c, blend_c, single_c, obj_c, aug=aug,
            )
            extra = dense_c[:, 4:]
            if aug in ("flip_x", "rot180"): extra = extra.flip(-2)
            if aug in ("flip_y", "rot180"): extra = extra.flip(-1)
            if aug == "rot90_ccw": extra = torch.rot90(extra, k=1, dims=(-2, -1))
            dense_c = torch.cat([ea, wa, ca, extra], dim=1)

        # Object stamper -> per-tile feature grid (on GPU).
        obj_grid = stamper([obj_c], cw, ch)              # (1, D, cw, ch)
        x_in = torch.cat([dense_c, obj_grid * w_obj], dim=1)
        # Style reference: pull a 224x224 RGB patch from the cached full
        # render on CPU, move to GPU, encode through DINOv2.
        drop = (torch.rand(1, device=device) < cfg_dropout)
        ref_patch_cpu = _sample_style_patch(c["rgb_full"], exclude_box=(cx, cy, cw, ch),
                                            size=224, rng=rng)
        ref_patch = ref_patch_cpu.to(device, non_blocking=True)
        style = style_enc(ref_patch, drop=drop)

        opt.zero_grad()
        out = model(x_in, style)
        comp = loss_mod(
            out, tt_c[None],
            {k: v[None] for k, v in blend_c.items()},
            {k: v[None] for k, v in single_c.items()},
            dir_freqs, sec_freqs,
        )
        comp["total"].backward()
        opt.step()
        losses.append(float(comp["total"].item()))

        if step == 1 or step % log_every == 0 or step == steps:
            recent = float(np.mean(losses[-log_every:]))
            with torch.no_grad():
                pred_t = out["tiles"].argmax(1)[0]
                tile_acc = (pred_t == tt_c).float().mean().item()
                bp = (torch.sigmoid(out["blend"]["present"][0, 0]) > 0.5).long()
                bgt = blend_c["present"].long()
                tp = ((bp == 1) & (bgt == 1)).sum().item()
                fp = ((bp == 1) & (bgt == 0)).sum().item()
                fn = ((bp == 0) & (bgt == 1)).sum().item()
                bf1 = 2 * tp / max(2 * tp + fp + fn, 1)
            print(
                f"  step {step:5d}/{steps}  loss={recent:.4f}  "
                f"tile={comp['tile'].item():.3f}  ffl={comp['ffl'].item():.3f}  "
                f"acc={tile_acc*100:.2f}%  blend_F1={bf1*100:.2f}%  ({time.time()-t0:.1f}s)"
            )


class _WeightedSampler:
    """Light wrapper that samples a single int idx with replacement
    from a normalised numpy weight vector."""

    def __init__(self, weights: np.ndarray):
        w = np.asarray(weights, dtype=np.float64)
        w = w / w.sum()
        self.w = w; self.n = len(w)
        self.cum = np.cumsum(w)

    def sample(self, rng: np.random.Generator) -> int:
        u = float(rng.random())
        return int(np.searchsorted(self.cum, u, side="right")
                   .clip(0, self.n - 1))


# ---------------------------- main ----------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cluster_json", type=Path,
                    default=_python_tools_root() / "style_clusters" / "browser" / "clusters_50.json")
    ap.add_argument("--target_path", type=Path, required=True)
    ap.add_argument("--predict_on", type=Path, default=None)
    ap.add_argument("--steps", type=int, default=30000)
    ap.add_argument("--crop", type=int, default=256,
                    help="Crop side; SegFormer needs multiples of 32, so 224/256/288/320 are all fine.")
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--log_every", type=int, default=200)
    ap.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    ap.add_argument("--obj_embed_dim", type=int, default=11)
    ap.add_argument("--decoder_dim", type=int, default=256)
    ap.add_argument("--style_dim", type=int, default=256)
    ap.add_argument("--blend_hidden", type=int, default=128)
    ap.add_argument("--w_obj", type=float, default=0.1,
                    help="Scale on the ObjectStamper output before concat.")
    ap.add_argument("--cfg_dropout", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no_render", action="store_true")
    ap.add_argument("--no_augment", action="store_true")
    ap.add_argument("--max_train_maps", type=int, default=0)
    ap.add_argument("--out_map", type=Path, default=None)
    ap.add_argument("--save_checkpoint", type=Path, default=None,
                    help="Save model+stamper+style_enc+palette state to this .pt path at end.")
    ap.add_argument("--style_cross_eval", action="store_true",
                    help="After main eval, predict each held-out map using EACH OTHER held-out map's render as the DINOv2 style reference.")
    ap.add_argument("--out_prefix", default="v2_predicted",
                    help="Filename prefix for predicted .map outputs (e.g., v3_predicted).")
    ap.add_argument("--load_checkpoint", type=Path, default=None,
                    help="Skip training and load model+stamper+style from this checkpoint.")
    ap.add_argument("--restrict_to_original_palette", action="store_true",
                    help="At inference, mask tile-head logits to only the held-out map's ORIGINAL palette indices. Required for the .map to load cleanly in WorldBuilder/the engine because the loaded tileset only contains those texture names.")
    args = ap.parse_args()

    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available()
                          else (args.device if args.device != "auto" else "cpu"))
    print(f"Device: {device}")

    cluster_data = json.loads(args.cluster_json.read_text(encoding="utf-8"))
    target_name = args.target_path.stem
    extras = [args.predict_on.stem] if args.predict_on is not None else []
    train_assignments, held_assignments = filter_and_dedup(
        cluster_data["assignments"], holdout_names=[target_name] + extras,
    )
    print(f"Filtered+deduped: train={len(train_assignments)} held_out={len(held_assignments)}")
    if args.max_train_maps and args.max_train_maps > 0 and len(train_assignments) > args.max_train_maps:
        rng_sub = np.random.default_rng(args.seed)
        idxs = rng_sub.choice(len(train_assignments), size=args.max_train_maps, replace=False)
        train_assignments = [train_assignments[i] for i in idxs]
        print(f"  subsampled training set to {len(train_assignments)}")

    # Parse all training maps + build union vocabs.
    print(f"\nParsing {len(train_assignments)} training maps ...")
    raws: list = []; raw_meta: list = []; skipped = 0
    for ti, m in enumerate(train_assignments):
        try:
            ra3 = Ra3Map(m["map_file"]); ra3.parse()
            raw = extract_raw_inputs(ra3, extract_target=True, style_id=int(m["cluster"]))
        except Exception as ex:
            print(f"  SKIP {m['map_name']}: parse error: {ex}"); skipped += 1; continue
        n_pal = len(raw.palette or [])
        if n_pal == 0 or raw.target_tiles is None:
            skipped += 1; continue
        if int(raw.target_tiles.max(initial=-1)) >= n_pal:
            skipped += 1; continue
        raws.append(raw); raw_meta.append(m)
        if (ti + 1) % 50 == 0:
            print(f"  parsed {ti + 1}/{len(train_assignments)}  kept={len(raws)} skipped={skipped}")
    print(f"  parsed: kept={len(raws)} skipped={skipped}")

    union_types: set = set(); union_owners: set = set(); union_palette: set = set()
    for r in raws:
        union_types.update(o.type_name for o in r.objects)
        union_owners.update(o.owner for o in r.objects)
        union_palette.update(r.palette)
    palette_list = sorted(union_palette)
    type_to_id = {t: i + 1 for i, t in enumerate(sorted(union_types))}
    owner_to_id = {o: i + 1 for i, o in enumerate(sorted(union_owners))}
    palette_to_id = {t: i for i, t in enumerate(palette_list)}
    print(f"Union vocabs: types={len(type_to_id)} owners={len(owner_to_id)} palette={len(palette_list)}")

    # Build modules.
    palette_renderer = PaletteRenderer(palette_list, learnable_residual=True).to(device)
    style_enc = DinoV2StyleEncoder(style_dim=args.style_dim,
                                   cfg_dropout=args.cfg_dropout).to(device)
    stamper = ObjectStamper(n_types=max(len(type_to_id), 1),
                             n_owners=max(len(owner_to_id), 1),
                             embed_dim=args.obj_embed_dim).to(device)
    in_ch = 5 + args.obj_embed_dim   # elev + water + cx + cy + mp_spawn + obj_grid
    model = V2TextureNet(
        palette_size=len(palette_list), n_directions=NUM_DIR_CLASSES,
        in_channels=in_ch, style_dim=args.style_dim,
        decoder_dim=args.decoder_dim, blend_hidden=args.blend_hidden,
    ).to(device)
    loss_mod = V2LossModule(palette_renderer=palette_renderer,
                             n_directions=NUM_DIR_CLASSES,
                             palette_size=len(palette_list)).to(device)

    # Re-encode + pad + cache full-map tensors. Cache stays on CPU (host RAM)
    # so we don't OOM the 12 GB GPU when training across hundreds of maps;
    # only the per-step crop is moved to GPU at sample time.
    print("\nRe-encoding + caching (CPU) ...", flush=True)
    cache = []; cluster_ids: list[int] = []
    for ti, (r, m) in enumerate(zip(raws, raw_meta)):
        rec = re_encode_for_v2(r, type_to_id, owner_to_id, palette_to_id)
        # Dense base channels: elev, water, cx, cy, mp_spawn.
        dense = build_dense_input(rec, stamper, device=torch.device("cpu"))   # (5, W, H)
        # Pad to multiple of 32 (reflect padding) on CPU.
        dense_p = _pad_t(dense[None], 32)[0]                                  # (5, W', H')
        tt_p = _pad_target(rec["target_tiles"], 32)
        b_p = {k: _pad_target(v, 32) for k, v in rec["target_blends"].items()}
        s_p = {k: _pad_target(v, 32) for k, v in rec["target_single"].items()}
        # Render full RGB on CPU once for style sampling. Use a small CPU
        # palette (no learnable residual at cache time — apply learned
        # residual lazily during the GPU forward path).
        with torch.no_grad():
            base = palette_renderer.base_rgb.cpu()
            tt_idx = tt_p.long().clamp_min(0)
            rgb_full = base[tt_idx].permute(2, 0, 1).contiguous() * (tt_p >= 0).float().unsqueeze(0)
        cache.append({
            "dense": dense_p, "target_tiles": tt_p, "target_blends": b_p,
            "target_single": s_p, "objects": rec["objects"],
            "rgb_full": rgb_full, "W": rec["width"], "H": rec["height"],
        })
        cluster_ids.append(int(m["cluster"]))
        if (ti + 1) % 50 == 0:
            print(f"  cached {ti + 1}/{len(raws)}", flush=True)
    print(f"  cached: {len(cache)} maps on CPU", flush=True)
    # Sampler weights = 1/sqrt(cluster_size).
    sw = cluster_balanced_weights(cluster_ids).numpy()
    sampler = _WeightedSampler(sw)
    print(f"Sampler: {len(sw)} maps, min/max weight = {sw.min():.3f}/{sw.max():.3f}")

    # Class freqs for logit-adjusted CE.
    dir_np = np.zeros(NUM_DIR_CLASSES, dtype=np.float64)
    sec_np = np.zeros(len(palette_list), dtype=np.float64)
    for c in cache:
        for layer in ("target_blends", "target_single"):
            d = c[layer]
            present = d["present"].cpu().numpy()
            dd = d["direction"].cpu().numpy()
            ss = d["secondary"].cpu().numpy()
            mask = present > 0
            md = mask & (dd >= 0) & (dd < NUM_DIR_CLASSES)
            if md.any():
                dir_np += np.bincount(dd[md].astype(np.int64), minlength=NUM_DIR_CLASSES)
            ms = mask & (ss >= 0) & (ss < len(palette_list))
            if ms.any():
                sec_np += np.bincount(ss[ms].astype(np.int64), minlength=len(palette_list))
    dir_freqs = torch.from_numpy(dir_np / max(dir_np.sum(), 1)).float().to(device)
    sec_freqs = torch.from_numpy(sec_np / max(sec_np.sum(), 1)).float().to(device)
    print(f"freqs: dir nonzero={int((dir_freqs > 0).sum())} sec nonzero={int((sec_freqs > 0).sum())}")

    rng = np.random.default_rng(args.seed)
    train_loop(model, stamper, style_enc, palette_renderer, loss_mod, cache,
               steps=args.steps, lr=args.lr, log_every=args.log_every,
               w_obj=args.w_obj, augment=(not args.no_augment),
               crop=args.crop, sampler=sampler, cfg_dropout=args.cfg_dropout,
               device=device, rng=rng,
               dir_freqs=dir_freqs, sec_freqs=sec_freqs)

    # --- Reference picker: for an unseen held-out map, pick the most
    # texture-similar training map (histogram intersection) to use as the
    # DINOv2 style reference. This is the inference-time analogue of the
    # DINOv2 nearest-neighbor lookup recommended in the report (item 5b)
    # — even singleton-cluster maps now condition on a real, well-trained
    # style.
    def _train_palette_dist(rec) -> np.ndarray:
        tt = rec["target_tiles"].cpu().numpy().reshape(-1)
        cnts = np.bincount(tt[tt >= 0].astype(np.int64), minlength=len(palette_list))
        s = cnts.sum()
        return cnts.astype(np.float64) / max(s, 1)

    train_dists = [_train_palette_dist({"target_tiles": c["target_tiles"]}) for c in cache]
    train_dists = np.stack(train_dists, axis=0)  # (N, V)

    def best_ref_idx_for(raw_eval) -> int:
        # Build a palette-aligned distribution for the held-out map.
        cnts = np.bincount((raw_eval.target_tiles.flatten().clip(min=0)).astype(np.int64),
                            minlength=max(1, len(raw_eval.palette or [])))
        # Re-encode under union palette.
        v = np.zeros(len(palette_list), dtype=np.float64)
        for i, name in enumerate(raw_eval.palette or []):
            j = palette_to_id.get(name)
            if j is not None:
                v[j] += cnts[i]
        s = v.sum()
        if s > 0: v = v / s
        sims = np.minimum(train_dists, v[None]).sum(axis=1)
        return int(sims.argmax())

    def _render_full_rgb(map_path: Path) -> torch.Tensor:
        """Parse a map and render its target tiles to (3, W, H) RGB on device."""
        ra3 = Ra3Map(str(map_path)); ra3.parse()
        rr = extract_raw_inputs(ra3, extract_target=True, style_id=0)
        rrec = re_encode_for_v2(rr, type_to_id, owner_to_id, palette_to_id)
        with torch.no_grad():
            base = palette_renderer.base_rgb.cpu()
            tt = rrec["target_tiles"].long().clamp_min(0)
            return (base[tt].permute(2, 0, 1).contiguous()
                    * (rrec["target_tiles"] >= 0).float().unsqueeze(0)).to(device)

    # --- Inference on holdouts ---
    def predict_on(map_path: Path, label: str, *, ref_record_idx: int = -1,
                   ref_rgb_override: torch.Tensor | None = None,
                   out_path: Path | None = None):
        ra3 = Ra3Map(str(map_path)); ra3.parse()
        raw_eval = extract_raw_inputs(ra3, extract_target=True, style_id=0)
        rec = re_encode_for_v2(raw_eval, type_to_id, owner_to_id, palette_to_id)
        dense_cpu = build_dense_input(rec, stamper, device=torch.device("cpu"))
        dense_p = _pad_t(dense_cpu[None], 32).to(device)
        if ref_rgb_override is not None:
            ref_rgb = ref_rgb_override
        else:
            ref_rgb = cache[ref_record_idx]["rgb_full"].to(device)
        style = style_enc.encode_image(ref_rgb[None]).contiguous()
        Wp, Hp = dense_p.shape[-2], dense_p.shape[-1]
        og = stamper([rec["objects"]], Wp, Hp)
        x_in = torch.cat([dense_p, og * args.w_obj], dim=1)
        model.eval(); stamper.eval(); style_enc.eval()
        with torch.no_grad():
            out = model(x_in, style)
        oW, oH = rec["width"], rec["height"]
        pred_t_full = out["tiles"].argmax(1)[0]                          # padded
        pred_t = pred_t_full[:oW, :oH].cpu().numpy().astype(np.int32)
        gt = rec["target_tiles"].numpy()
        tile_acc = float((pred_t == gt).mean())
        print(f"\n=== {label}  size={oW}x{oH}  tile_acc={tile_acc*100:.2f}% ===")
        if out_path is None:
            out_path = args.out_map or (map_path.parent / "skinned_full" /
                                        f"{args.out_prefix}_{map_path.stem}.map")
        try:
            writeback_predictions(map_path, raw_eval, out, pred_t,
                                  palette_list, out_path)
            if not args.no_render:
                render(out_path, out_path.parent / "_renders" / out_path.stem)
        except Exception as e:
            print(f"  writeback failed ({e}); saving npz instead")
            np.savez(str(out_path).replace(".map", ".npz"),
                     tile_pred=pred_t)
        return tile_acc

    # Pick a snowy/iceland-ish reference map for 2 IS, otherwise use the
    # held-out target's most-similar training map by texture distribution.
    # For v0, just use the FIRST training map as reference for both eval
    # paths — we'll add a smarter ref-picker as a follow-up.
    held_paths: list[Path] = []
    if held_assignments:
        for h in held_assignments:
            ra3 = Ra3Map(h["map_file"]); ra3.parse()
            raw_h = extract_raw_inputs(ra3, extract_target=True, style_id=0)
            ridx = best_ref_idx_for(raw_h)
            print(f"  Reference for {h['map_name']}: training map idx {ridx}")
            predict_on(Path(h["map_file"]),
                       label=f"EVAL on held-out {h['map_name']}",
                       ref_record_idx=ridx)
            held_paths.append(Path(h["map_file"]))
    if args.predict_on is not None and args.predict_on.exists():
        ra3 = Ra3Map(str(args.predict_on)); ra3.parse()
        raw_h = extract_raw_inputs(ra3, extract_target=True, style_id=0)
        ridx = best_ref_idx_for(raw_h)
        print(f"  Reference for {args.predict_on.stem}: training map idx {ridx}")
        predict_on(args.predict_on, label=f"PREDICT on {args.predict_on.stem}",
                   ref_record_idx=ridx)
        if args.predict_on not in held_paths:
            held_paths.append(args.predict_on)

    # --- Cross-style eval: predict each holdout using each OTHER holdout's
    # render as the DINOv2 style reference. Diagnostic for the singleton-
    # cluster failure mode: if a target predicts varied output when given a
    # different style reference, the model is fine and the failure is in the
    # reference-picker / training-set coverage, not the architecture.
    if args.style_cross_eval and len(held_paths) >= 2:
        print(f"\n=== STYLE CROSS-EVAL across {len(held_paths)} held-out maps ===")
        for tgt in held_paths:
            for ref in held_paths:
                if ref == tgt:
                    continue
                ref_rgb = _render_full_rgb(ref)
                cross_path = (tgt.parent / "skinned_full" /
                              f"{args.out_prefix}_{tgt.stem}__styledfrom__{ref.stem}.map")
                predict_on(tgt,
                           label=f"CROSS {tgt.stem} <-- style({ref.stem})",
                           ref_rgb_override=ref_rgb,
                           out_path=cross_path)

    if args.save_checkpoint is not None:
        args.save_checkpoint.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "model": model.state_dict(),
            "stamper": stamper.state_dict(),
            "style_enc_proj": style_enc.proj.state_dict(),
            "style_enc_null": style_enc.null_style.detach().cpu(),
            "palette_renderer": palette_renderer.state_dict(),
            "palette_list": palette_list,
            "type_to_id": type_to_id,
            "owner_to_id": owner_to_id,
            "args": vars(args),
        }, str(args.save_checkpoint))
        print(f"Saved checkpoint to {args.save_checkpoint}")
    print("\nDone.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
