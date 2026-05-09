#!/usr/bin/env python3
"""v3 retexture: SPADE U-Net + per-biome lookup constraint + CRF post-pass.

Inputs: source map (heightmap+masks+objects only — textures discarded).
Output: fully-painted-from-scratch map in target style.

Pipeline:
  1. Parse source. Discard tiles, blends arrays, textures palette.
  2. Extract 10-channel context (heightmap, slope, water, buildability,
     passability, 5x object densities). NEVER reads source textures.
  3. Run SPADE U-Net with target style_id -> raw logits per tile.
  4. (Optional) Constrain logits via per-biome lookup table: classes not
     present for the (style, seg_class) get a large negative bias.
  5. (Optional) Spatial-coherence CRF post-pass on softmax probs.
  6. Argmax -> texture index per tile.
  7. Build new tile array + textures palette + zero blends arrays.
  8. Save .map.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import List

import numpy as np
import torch


def _python_tools_root() -> Path:
    return Path(__file__).resolve().parents[1]


sys.path.insert(0, str(_python_tools_root()))
from map_processor import Ra3Map  # noqa: E402
from map_processor.models.spade_texture_unet import SPADEUNet, discretize_input  # noqa: E402
from map_processor.utils.style_features import extract_input_channels  # noqa: E402
from map_processor.utils.crf_post import crf_smooth, _build_blocking_mask  # noqa: E402


def _default_data_dir() -> Path:
    return _python_tools_root() / "training_outputs" / "texture_transfer"


@torch.no_grad()
def predict_tiled(model, X: np.ndarray, style_id: int, device: str,
                  tile: int = 256, overlap: int = 32, never_trained_mask: np.ndarray | None = None):
    """Tile-and-blend logits over the full map. Returns (vocab, W, H) float32."""
    C, W, H = X.shape
    n_classes = model.out_conv.out_channels
    accum = torch.zeros((n_classes, W, H), device=device, dtype=torch.float32)
    counts = torch.zeros((W, H), device=device, dtype=torch.float32)
    step = max(1, tile - overlap)
    xs = list(range(0, max(1, W - tile + 1), step))
    ys = list(range(0, max(1, H - tile + 1), step))
    if not xs or xs[-1] + tile < W: xs.append(max(0, W - tile))
    if not ys or ys[-1] + tile < H: ys.append(max(0, H - tile))
    if W < tile: xs = [0]
    if H < tile: ys = [0]
    xs = sorted(set(xs)); ys = sorted(set(ys))
    s_tensor = torch.tensor([style_id], dtype=torch.long, device=device)
    for x0 in xs:
        for y0 in ys:
            x1 = min(x0 + tile, W); y1 = min(y0 + tile, H)
            patch = X[:, x0:x1, y0:y1]
            ph, pw = patch.shape[1], patch.shape[2]
            if ph < tile or pw < tile:
                pad = np.zeros((C, tile, tile), dtype=np.float32)
                pad[:, :ph, :pw] = patch
                patch = pad
            xt = torch.from_numpy(patch).unsqueeze(0).float().to(device)
            logits = model(xt, s_tensor)[0]
            ph_e = min(tile, x1 - x0); pw_e = min(tile, y1 - y0)
            accum[:, x0:x0 + ph_e, y0:y0 + pw_e] += logits[:, :ph_e, :pw_e]
            counts[x0:x0 + ph_e, y0:y0 + pw_e] += 1.0
    counts = counts.clamp_min(1e-6)
    avg = accum / counts.unsqueeze(0)
    if never_trained_mask is not None:
        ntm = torch.from_numpy(never_trained_mask).to(device).bool()
        avg = avg.masked_fill(ntm[:, None, None], float("-inf"))
    return avg.cpu().numpy()  # (V, W, H)


def apply_biome_constraint(
    logits: np.ndarray, X: np.ndarray, style_id: int, lookup: dict, mode: str = "soft", k_keep: int = 12,
) -> np.ndarray:
    """Bias logits toward textures observed for (style, seg_class).

    mode='soft': add log-prior from empirical (style, seg) distribution.
    mode='hard': zero out (set to -inf) any class not in top_k of (style, seg).
    """
    if str(style_id) not in lookup:
        return logits
    style_lookup = lookup[str(style_id)]
    V, W, H = logits.shape
    seg_t = discretize_input(torch.from_numpy(X).unsqueeze(0).float())[0].numpy()  # (W, H)
    out = logits.copy()
    seg_classes = np.unique(seg_t)
    for sc in seg_classes:
        cell = style_lookup.get(str(int(sc)))
        if cell is None:
            continue
        mask = seg_t == sc
        if not mask.any():
            continue
        if mode == "hard":
            allowed = np.zeros(V, dtype=bool)
            allowed[cell["top_k"][:k_keep]] = True
            ar = np.where(allowed, 0.0, -1e6)
            out[:, mask] = out[:, mask] + ar[:, None]
        else:  # soft - add log-prior bonus to seen textures
            log_prior = np.full(V, -3.0, dtype=np.float32)  # base mild penalty for unseen
            for tex, prob in cell["probs"]:
                log_prior[tex] = float(np.log(max(prob, 1e-6)))
            out[:, mask] = out[:, mask] + log_prior[:, None] * 0.5  # soft weight
    return out


def quantize_palette(pred_idx: np.ndarray, vocab: List[str], max_palette: int = 64):
    counts = Counter(pred_idx.reshape(-1).tolist())
    top = [vid for vid, _ in counts.most_common(max_palette)]
    if len(counts) <= max_palette:
        return top, pred_idx
    keep = set(top)
    fam = {v: (vocab[v].split("_", 1)[0] if "_" in vocab[v] else vocab[v]) for v in range(len(vocab))}
    family_keepers = {}
    for kv in top:
        family_keepers.setdefault(fam[kv], []).append(kv)
    remap = {v: v for v in top}
    for v in counts:
        if v in keep: continue
        f = fam[v]
        remap[v] = family_keepers[f][0] if f in family_keepers else top[0]
    out = np.vectorize(remap.get)(pred_idx).astype(pred_idx.dtype)
    return top, out


def fresh_repaint(blend, pred_idx: np.ndarray, vocab: List[str],
                  pattern_mode: str = "random", seed: int = 1337) -> List[str]:
    W, H = pred_idx.shape
    palette_ids, pred_idx_q = quantize_palette(pred_idx, vocab, max_palette=64)
    palette_names = [vocab[v] for v in palette_ids]
    name_to_pidx = {n: i for i, n in enumerate(palette_names)}
    pred_palette_idx = np.vectorize(lambda v: name_to_pidx[vocab[v]])(pred_idx_q).astype(np.uint16)
    if pattern_mode == "random":
        rs = np.random.RandomState(seed)
        pattern = rs.randint(0, 64, size=(W, H), dtype=np.int32).astype(np.uint16)
    else:
        pattern = np.zeros((W, H), dtype=np.uint16)
    new_tile_ids = (pred_palette_idx.astype(np.uint32) * 64 + pattern.astype(np.uint32)).astype(np.uint16)
    full_tiles = np.zeros_like(blend.tiles, dtype=np.uint16)
    full_tiles[:W, :H] = new_tile_ids
    blend.tiles = full_tiles
    blend.blends = np.zeros_like(blend.blends)
    blend.single_edge_blends = np.zeros_like(blend.single_edge_blends)
    blend.cliff_blends = np.zeros_like(blend.cliff_blends)
    from map_processor.assets.terrain.texture import Texture
    new_textures = []
    for i, name in enumerate(palette_names):
        t = Texture()
        t.cell_start = i * 16; t.cell_count = 16; t.cell_size = 4
        t.magic_value = 0; t.name = name
        new_textures.append(t)
    blend.textures = new_textures
    return palette_names


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--src", type=Path, required=True)
    ap.add_argument("--style", type=int, required=True)
    ap.add_argument("--ckpt", type=Path, default=_default_data_dir() / "ckpt_v3" / "best.pt")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--data_dir", type=Path, default=_default_data_dir())
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--biome_constraint", choices=["off", "soft", "hard"], default="soft")
    ap.add_argument("--crf", action="store_true", help="Apply CRF post-pass")
    ap.add_argument("--crf_sigma", type=float, default=1.5)
    ap.add_argument("--crf_iters", type=int, default=3)
    ap.add_argument("--pattern", choices=["random", "zero"], default="random")
    ap.add_argument("--tile", type=int, default=256)
    ap.add_argument("--overlap", type=int, default=32)
    ap.add_argument("--seed", type=int, default=1337)
    args = ap.parse_args()

    if not args.src.exists():
        raise SystemExit(f"Source not found: {args.src}")
    if not args.ckpt.exists():
        raise SystemExit(f"Checkpoint not found: {args.ckpt}")
    out_path = args.out or args.src.with_name(f"{args.src.stem}_v3_style{args.style}.map")

    vocab = json.loads((args.data_dir / "vocab.json").read_text(encoding="utf-8"))
    index_file = args.data_dir / "curated_index.json"
    if not index_file.exists():
        index_file = args.data_dir / "index.json"
    index = json.loads(index_file.read_text(encoding="utf-8"))
    n_chan = index["n_channels"]; n_styles = index["n_styles"]; vocab_size = index["vocab_size"]

    class_freq = np.array(index.get("class_freq_train", []), dtype=np.int64)
    never_trained = (class_freq == 0).astype(np.bool_) if class_freq.size == vocab_size else np.zeros(vocab_size, dtype=bool)

    biome_lookup = {}
    biome_path = args.data_dir / "biome_lookup.json"
    if args.biome_constraint != "off" and biome_path.exists():
        biome_lookup = json.loads(biome_path.read_text(encoding="utf-8"))

    ckpt = torch.load(args.ckpt, map_location=args.device, weights_only=False)
    base = ckpt.get("base", 32) if "base" in ckpt else (
        ckpt["args"].get("base", 32) if isinstance(ckpt.get("args"), dict) else 32)
    model = SPADEUNet(in_channels=n_chan, n_styles=n_styles,
                      vocab_size=vocab_size, base=base).to(args.device)
    model.load_state_dict(ckpt["model"]); model.eval()
    print(f"Model loaded: {args.ckpt} (epoch={ckpt.get('epoch','?')})")

    print(f"Parsing source: {args.src}")
    m = Ra3Map(str(args.src)); m.parse(); ctx = m.get_context()
    blend = ctx.get_asset("BlendTileData")
    h_asset = ctx.get_asset("HeightMapData")
    objs = ctx.get_asset("ObjectsList")
    print(f"  source map {blend.map_width}x{blend.map_height} - "
          f"DISCARDING tiles, blends, textures palette")

    X, W, H = extract_input_channels(blend, h_asset, objs,
                                     world_to_tile=index["world_to_tile"],
                                     sigma=index["object_sigma"])
    print(f"  context channels: {X.shape}, predicting style={args.style}")
    logits = predict_tiled(model, X, args.style, args.device,
                           tile=args.tile, overlap=args.overlap,
                           never_trained_mask=never_trained)
    print(f"  logits: {logits.shape}")

    if args.biome_constraint != "off" and biome_lookup:
        logits = apply_biome_constraint(logits, X, args.style, biome_lookup, mode=args.biome_constraint)
        print(f"  applied biome lookup constraint mode={args.biome_constraint}")

    if args.crf:
        # Softmax + CRF + argmax
        m_l = logits.max(axis=0, keepdims=True)
        e = np.exp(logits - m_l); probs = e / e.sum(axis=0, keepdims=True)
        feat = X[[1, 2, 3, 4, 9]]
        boundary = _build_blocking_mask(feat)
        probs = crf_smooth(probs, boundary, sigma=args.crf_sigma,
                           iterations=args.crf_iters, blocking_strength=0.7)
        pred_idx = probs.argmax(axis=0)
        print(f"  CRF post-pass: sigma={args.crf_sigma} iters={args.crf_iters}")
    else:
        pred_idx = logits.argmax(axis=0)

    n_uniq = len(set(pred_idx.reshape(-1).tolist()))
    print(f"  predicted: {pred_idx.shape}, {n_uniq} unique classes")

    palette = fresh_repaint(blend, pred_idx, vocab, pattern_mode=args.pattern, seed=args.seed)
    print(f"  new palette ({len(palette)}): "
          f"{', '.join(palette[:5])}{'...' if len(palette) > 5 else ''}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    m.save(str(out_path), compress=True)
    print(f"Wrote: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
