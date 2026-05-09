#!/usr/bin/env python3
"""MVP: tiny U-Net texture predictor with temperature sampling at inference.

Why this exists: v3 (SegFormer + DINOv2 style + heavy losses) trained for
~50 min and produced 1-tile-dominated outputs. Argmax over a 375-class
softmax always collapses to the dataset's mode texture for the easy
regions of a map; rare textures (roads, transitions) never win the argmax
even if their probability is non-trivial.

This MVP fixes that with two structural choices:
  1. **Sample, don't argmax**, at inference. Each predicted tile is drawn
     from the per-pixel softmax with temperature `tau`. Rare textures are
     emitted in proportion to their soft probability -> guaranteed
     variation in the output.
  2. **Tiny U-Net per cluster**, not a global model. Trains on 10-30 maps
     of ONE cluster; palette is the cluster-local union (~30-60 textures,
     all loadable in that cluster's environment). 5-10 min to train.

Per-tile inputs (8 channels):
  elev, water, coord_x, coord_y, mp_spawn,
  obj_density_road, obj_density_civ, obj_density_other.

Outputs: tile-class softmax over the cluster-local palette.
Blends are NOT predicted -- copied from the held-out's original to keep
roads/edges intact (the user's complaint that v3 nuked roads). Easy to
upgrade later.

Usage:
  python scripts/mvp_predict.py \
    --target_path "../RA3 Official maps/2 IS/map_mp_2_feasel6.map" \
    --steps 5000
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
import torch.nn as nn
import torch.nn.functional as F


def _python_tools_root() -> Path:
    return Path(__file__).resolve().parents[1]


sys.path.insert(0, str(_python_tools_root()))
sys.path.insert(0, str(_python_tools_root() / "scripts"))

from map_processor import Ra3Map  # noqa: E402
from map_processor.features import extract_raw_inputs  # noqa: E402
from train_official import _augment_crop, EXCLUDE_PATTERNS, DERIV_SUFFIXES, is_includable  # noqa: E402
from train_cluster import _pad_t, _position_pattern_grid, render  # noqa: E402


# Game-mode rule-variant suffixes — these maps are IDENTICAL to the
# canonical-named map in terrain/textures/objects, only differing in gameplay
# rules. Including them in training is data leakage (model sees the same map
# multiple times) and is the cause of v3-MVP's "original underneath +
# sprinkled noise" failure mode the user identified.
# Order longest-first so compound suffixes match before their components.
RULE_SUFFIXES = sorted(
    ["_inf_only", "_noair", "_nosw_noair", "_nosw", "_tanks_only"],
    key=len, reverse=True,
)


def canonical_name(map_name: str) -> str:
    """Return the canonical (rule-suffix-stripped) name. Strip greedily so
    e.g. `_nosw_noair` is removed in one step rather than only `_noair`."""
    n = map_name
    changed = True
    while changed:
        changed = False
        for s in RULE_SUFFIXES:
            if n.endswith(s):
                n = n[:-len(s)]
                changed = True
                break
    return n


# ---------------- Tiny U-Net ----------------

class _DConv(nn.Module):
    def __init__(self, ci: int, co: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(ci, co, 3, padding=1), nn.GroupNorm(8, co), nn.GELU(),
            nn.Conv2d(co, co, 3, padding=1), nn.GroupNorm(8, co), nn.GELU(),
        )

    def forward(self, x): return self.net(x)


class TinyUNet(nn.Module):
    def __init__(self, in_ch: int, n_classes: int, base: int = 24):
        super().__init__()
        self.e1 = _DConv(in_ch, base)
        self.e2 = _DConv(base, base * 2)
        self.e3 = _DConv(base * 2, base * 4)
        self.e4 = _DConv(base * 4, base * 8)
        self.bot = _DConv(base * 8, base * 16)
        self.u4 = nn.ConvTranspose2d(base * 16, base * 8, 2, stride=2)
        self.d4 = _DConv(base * 16, base * 8)
        self.u3 = nn.ConvTranspose2d(base * 8, base * 4, 2, stride=2)
        self.d3 = _DConv(base * 8, base * 4)
        self.u2 = nn.ConvTranspose2d(base * 4, base * 2, 2, stride=2)
        self.d2 = _DConv(base * 4, base * 2)
        self.u1 = nn.ConvTranspose2d(base * 2, base, 2, stride=2)
        self.d1 = _DConv(base * 2, base)
        self.head = nn.Conv2d(base, n_classes, 1)

    def forward(self, x):
        e1 = self.e1(x)
        e2 = self.e2(F.max_pool2d(e1, 2))
        e3 = self.e3(F.max_pool2d(e2, 2))
        e4 = self.e4(F.max_pool2d(e3, 2))
        b = self.bot(F.max_pool2d(e4, 2))
        d4 = self.d4(torch.cat([self.u4(b), e4], dim=1))
        d3 = self.d3(torch.cat([self.u3(d4), e3], dim=1))
        d2 = self.d2(torch.cat([self.u2(d3), e2], dim=1))
        d1 = self.d1(torch.cat([self.u1(d2), e1], dim=1))
        return self.head(d1)


# ---------------- Inputs ----------------

def _classify_obj(name: str) -> int:
    """0 = road/path, 1 = civilian/building, 2 = other."""
    n = name.lower()
    if "road" in n or "path" in n or "bridge" in n:
        return 0
    if "civ" in n or "build" in n or "house" in n or "warehouse" in n:
        return 1
    return 2


def _density_grids(objects, W: int, H: int) -> np.ndarray:
    """3 channels: road/civ/other density per tile (counts, normalised)."""
    out = np.zeros((3, W, H), dtype=np.float32)
    for o in objects:
        ix = int(round(o.tile_x)); iy = int(round(o.tile_y))
        if 0 <= ix < W and 0 <= iy < H:
            ch = _classify_obj(o.type_name)
            out[ch, ix, iy] += 1.0
    if out.sum() > 0:
        out = np.tanh(out)
    return out


def _mp_grid(spawns, W: int, H: int) -> np.ndarray:
    g = np.zeros((W, H), dtype=np.float32)
    for x, y in spawns or []:
        ix = int(round(x)); iy = int(round(y))
        if 0 <= ix < W and 0 <= iy < H:
            g[ix, iy] = 1.0
    return g


def build_inputs(raw, palette_to_id: Dict[str, int], device) -> Tuple[torch.Tensor, torch.Tensor]:
    """Returns (x_dense [1,8,W,H], target_tiles_remapped [W,H])."""
    W, H = raw.width, raw.height
    elev = torch.from_numpy(raw.elev).float()[None]                 # (1, W, H)
    water = torch.from_numpy(raw.water).float()[None]
    xs = torch.linspace(-1, 1, W); ys = torch.linspace(-1, 1, H)
    cx = xs[:, None].expand(W, H)[None]
    cy = ys[None, :].expand(W, H)[None]
    mp = torch.from_numpy(_mp_grid(raw.mp_spawns, W, H))[None]
    dens = torch.from_numpy(_density_grids(raw.objects, W, H))      # (3, W, H)
    x = torch.cat([elev, water, cx, cy, mp, dens], dim=0)[None]     # (1, 8, W, H)

    tile_remap = np.zeros(len(raw.palette or []), dtype=np.int32)
    for i, name in enumerate(raw.palette or []):
        tile_remap[i] = palette_to_id.get(name, 0)
    tt = tile_remap[raw.target_tiles.clip(min=0)] if raw.target_tiles is not None else None
    target = torch.from_numpy(tt.astype(np.int64)) if tt is not None else None
    return x.to(device), (target.to(device) if target is not None else None)


# ---------------- Training ----------------

def train(model, cache, *, steps, lr, log_every, class_weights, device, rng,
          crop=128):
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.0)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {n_params/1e6:.2f}M params", flush=True)
    t0 = time.time()
    losses = []
    model.train()
    multiple = 16
    for step in range(1, steps + 1):
        mi = int(rng.integers(0, len(cache)))
        x_full, tt_full = cache[mi]                # (1, 8, W, H), (W, H)
        _, _, W, H = x_full.shape
        cw = min(crop, W); ch = min(crop, H)
        cw -= cw % multiple; ch -= ch % multiple
        if cw == 0: cw = max(multiple, W - W % multiple)
        if ch == 0: ch = max(multiple, H - H % multiple)
        cx = int(rng.integers(0, max(W - cw + 1, 1))) if W > cw else 0
        cy = int(rng.integers(0, max(H - ch + 1, 1))) if H > ch else 0
        x = x_full[:, :, cx:cx + cw, cy:cy + ch]
        t = tt_full[cx:cx + cw, cy:cy + ch]

        # D4 augmentation. Build dummy "elev/water/coord" args for _augment_crop;
        # treat the remaining 4 channels (mp + 3 density) as extras to flip together.
        choices = ["identity", "flip_x", "flip_y", "rot180"]
        if cw == ch: choices.append("rot90_ccw")
        aug = choices[int(rng.integers(0, len(choices)))]
        if aug != "identity":
            elev_a = x[:, 0:1]; water_a = x[:, 1:2]; coord_a = x[:, 2:4]
            empty_b = {"present": torch.zeros_like(t),
                       "secondary": torch.full_like(t, -1),
                       "direction": torch.full_like(t, -1)}
            ea, wa, ca, t, _, _, _ = _augment_crop(
                elev_a, water_a, coord_a, t, empty_b, empty_b, [], aug=aug,
            )
            extra = x[:, 4:]
            if aug in ("flip_x", "rot180"): extra = extra.flip(-2)
            if aug in ("flip_y", "rot180"): extra = extra.flip(-1)
            if aug == "rot90_ccw": extra = torch.rot90(extra, k=1, dims=(-2, -1))
            x = torch.cat([ea, wa, ca, extra], dim=1)

        opt.zero_grad()
        logits = model(x)
        loss = F.cross_entropy(logits, t.long()[None], weight=class_weights)
        loss.backward()
        opt.step()
        losses.append(float(loss.item()))

        if step == 1 or step % log_every == 0 or step == steps:
            recent = float(np.mean(losses[-log_every:]))
            with torch.no_grad():
                pred = logits.argmax(1)[0]
                acc = (pred == t).float().mean().item()
                # Diversity proxy: # distinct argmax classes in this crop.
                n_distinct = int(pred.unique().numel())
            print(f"  step {step:5d}/{steps}  loss={recent:.4f}  "
                  f"argmax_acc={acc*100:.1f}%  distinct={n_distinct}  ({time.time()-t0:.1f}s)",
                  flush=True)


# ---------------- Inference ----------------

def sample_predict(model, x_full: torch.Tensor, *, tau: float, rng_seed: int) -> np.ndarray:
    """Sample tile indices with temperature `tau` from the per-tile softmax."""
    model.eval()
    with torch.no_grad():
        logits = model(x_full)                                 # (1, V, W, H)
    if tau <= 0:
        return logits.argmax(1)[0].cpu().numpy().astype(np.int32)
    probs = F.softmax(logits / tau, dim=1)
    B, V, W, H = probs.shape
    flat = probs.permute(0, 2, 3, 1).reshape(-1, V)
    g = torch.Generator(device=flat.device); g.manual_seed(rng_seed)
    samples = torch.multinomial(flat, num_samples=1, generator=g)
    return samples.reshape(B, W, H)[0].cpu().numpy().astype(np.int32)


def writeback_tiles_only(target_path: Path, pred_local: np.ndarray,
                         palette_list: List[str], out_path: Path,
                         restrict_to_original: bool = True) -> None:
    """Write a copy of `target_path` with `blend.tiles` replaced by sampled
    cluster-local indices. Blends + single_edges + objects are preserved
    from the original, so roads/transitions stay intact (this addresses the
    user-reported regression where v3 nuked roads).

    palette_list is the CLUSTER palette. If restrict_to_original, we
    additionally clip to the held-out's actual loaded textures (set of
    intersect names) so WorldBuilder can resolve every entry.
    """
    from map_processor.assets.terrain.texture import Texture
    src = Ra3Map(str(target_path)); src.parse()
    blend = src.get_context().get_asset("BlendTileData")
    W, H = blend.map_width, blend.map_height
    pos = _position_pattern_grid(W, H)

    orig_names = {t.name for t in (blend.textures or [])}
    if restrict_to_original:
        # remap: cluster-palette idx -> original-palette idx (for names that
        # exist in the original); for names not in the original, drop to the
        # most common texture in the original palette for the predicted region.
        orig_palette = [t.name for t in blend.textures]
        name_to_orig = {n: i for i, n in enumerate(orig_palette)}
        cluster_to_orig = np.full(len(palette_list), -1, dtype=np.int64)
        for ci, name in enumerate(palette_list):
            if name in name_to_orig:
                cluster_to_orig[ci] = name_to_orig[name]
        # For unmappable predictions, fall back to the dominant ORIGINAL
        # texture present in the prediction's neighborhood (use idx 0 as a
        # safe default).
        pred_orig = np.where(cluster_to_orig[pred_local] >= 0,
                             cluster_to_orig[pred_local], 0).astype(np.int32)
        pred_to_write = pred_orig
        new_textures = blend.textures   # leave original palette intact
    else:
        # Use cluster palette directly. Compact to used.
        used = sorted(set(int(v) for v in pred_local.flatten().tolist()))
        compact = [palette_list[i] for i in used if 0 <= i < len(palette_list)]
        remap = np.zeros(len(palette_list), dtype=np.int32)
        for new_i, old_i in enumerate(used):
            remap[old_i] = new_i
        pred_to_write = remap[pred_local]
        base_meta = blend.textures[0] if blend.textures else None
        new_textures = []
        for name in compact:
            t = Texture()
            t.cell_start = 0
            t.cell_count = base_meta.cell_count if base_meta else 16
            t.cell_size = base_meta.cell_size if base_meta else 4
            t.magic_value = base_meta.magic_value if base_meta else 0
            t.name = name
            new_textures.append(t)
        blend.textures = new_textures

    # Crop predictions to map dims.
    pw, ph = pred_to_write.shape
    pred_to_write = pred_to_write[:W, :H]
    blend.tiles = (pred_to_write.astype(np.uint16) * 64 + pos.astype(np.uint16))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    src.save(str(out_path), compress=True)
    print(f"  Wrote {out_path}", flush=True)


# ---------------- Main ----------------

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cluster_json", type=Path,
                    default=_python_tools_root() / "style_clusters" / "browser" / "clusters_50.json")
    ap.add_argument("--target_path", type=Path, required=True)
    ap.add_argument("--steps", type=int, default=5000)
    ap.add_argument("--crop", type=int, default=128)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--base", type=int, default=24)
    ap.add_argument("--log_every", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tau", type=float, default=0.7,
                    help="Inference sampling temperature. 0 = argmax, 1 = vanilla softmax sample.")
    ap.add_argument("--no_render", action="store_true")
    ap.add_argument("--out_path", type=Path, default=None)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}", flush=True)

    # Find cluster of target, gather sibling maps.
    cd = json.loads(args.cluster_json.read_text(encoding="utf-8"))
    target_name = args.target_path.stem
    a_target = next((a for a in cd["assignments"] if a["map_name"] == target_name), None)
    if a_target is None:
        raise SystemExit(f"Target {target_name} not in clusters_50.")
    cid = int(a_target["cluster"])
    print(f"Target cluster: {cid}", flush=True)
    target_canon = canonical_name(target_name)
    raw_siblings = [a for a in cd["assignments"]
                    if a["cluster"] == cid and is_includable(a)]
    # Dedup by canonical name. Drop any rule-variant of the TARGET itself
    # (incl. the target). For other base-names, keep ONE representative
    # (preferably the canonical / suffix-less form).
    by_canon: Dict[str, dict] = {}
    for a in raw_siblings:
        cn = canonical_name(a["map_name"])
        if cn == target_canon:
            continue                                # leak: same map in any rule-variant form
        existing = by_canon.get(cn)
        # Prefer the suffix-less (canonical) form when available.
        if existing is None or (a["map_name"] == cn and existing["map_name"] != cn):
            by_canon[cn] = a
    siblings = list(by_canon.values())
    print(f"Siblings raw={len(raw_siblings)} after canonical-dedup={len(siblings)} "
          f"(rule-variants of target excluded; one representative per base-name)",
          flush=True)
    if siblings:
        print(f"  kept: {[a['map_name'] for a in siblings]}", flush=True)

    # Parse all + build cluster-local palette.
    raws = []; raw_meta = []
    for ti, m in enumerate(siblings):
        try:
            r = Ra3Map(m["map_file"]); r.parse()
            raw = extract_raw_inputs(r, extract_target=True, style_id=cid)
        except Exception as ex:
            print(f"  SKIP {m['map_name']}: {ex}", flush=True)
            continue
        n_pal = len(raw.palette or [])
        if n_pal == 0 or raw.target_tiles is None:
            continue
        if int(raw.target_tiles.max(initial=-1)) >= n_pal:
            continue
        raws.append(raw); raw_meta.append(m)
    print(f"Parsed {len(raws)} sibling maps", flush=True)
    if len(raws) < 2:
        raise SystemExit("Need >= 2 sibling maps to train; cluster too small.")

    union = sorted({n for r in raws for n in r.palette})
    palette_to_id = {n: i for i, n in enumerate(union)}
    print(f"Cluster palette size: {len(union)}", flush=True)

    # Build cached inputs+targets, all on GPU (small clusters fit easily).
    cache = []
    for r in raws:
        x, t = build_inputs(r, palette_to_id, device)
        x = _pad_t(x, 16); t = _pad_t(t[None, None].float(), 16)[0, 0].long()
        cache.append((x, t))

    # Class weights = 1/sqrt(freq) so rare textures (roads, rare grasses)
    # actually receive gradient signal.
    class_counts = np.zeros(len(union), dtype=np.float64)
    for _, t in cache:
        bc = np.bincount(t.cpu().numpy().flatten(), minlength=len(union))
        class_counts += bc
    class_freqs = class_counts / class_counts.sum().clip(min=1)
    cw_np = 1.0 / np.sqrt(class_freqs.clip(min=1e-6))
    # Normalise so mean weight is 1 (keeps overall loss scale stable).
    cw_np = cw_np * (len(cw_np) / cw_np.sum())
    class_weights = torch.from_numpy(cw_np.astype(np.float32)).to(device)

    model = TinyUNet(in_ch=8, n_classes=len(union), base=args.base).to(device)
    rng = np.random.default_rng(args.seed)
    train(model, cache, steps=args.steps, lr=args.lr, log_every=args.log_every,
          class_weights=class_weights, device=device, rng=rng, crop=args.crop)

    # Predict on the held-out target.
    print(f"\n=== PREDICT on held-out {target_name} ===", flush=True)
    r_t = Ra3Map(str(args.target_path)); r_t.parse()
    raw_t = extract_raw_inputs(r_t, extract_target=True, style_id=cid)
    x_t, t_t = build_inputs(raw_t, palette_to_id, device)
    x_t_p = _pad_t(x_t, 16)
    pred_local = sample_predict(model, x_t_p, tau=args.tau, rng_seed=args.seed)
    pred_local_argmax = sample_predict(model, x_t_p, tau=0.0, rng_seed=args.seed)
    W, H = raw_t.width, raw_t.height
    pred_local = pred_local[:W, :H]
    pred_local_argmax = pred_local_argmax[:W, :H]
    distinct_sample = int(np.unique(pred_local).size)
    distinct_argmax = int(np.unique(pred_local_argmax).size)
    print(f"  argmax distinct classes: {distinct_argmax}", flush=True)
    print(f"  sample (tau={args.tau}) distinct classes: {distinct_sample}", flush=True)

    out_path = args.out_path or (args.target_path.parent / "skinned_full" /
                                 f"mvp_predicted_{target_name}.map")
    writeback_tiles_only(args.target_path, pred_local, union, out_path,
                         restrict_to_original=True)
    if not args.no_render:
        render(out_path, out_path.parent / "_renders" / out_path.stem)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
