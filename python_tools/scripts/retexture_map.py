#!/usr/bin/env python3
"""Retexture a map from scratch using the trained model.

Discards the source's tile patterns, blends arrays, and texture palette
ENTIRELY. The model paints a fresh texture map from non-texture context only:
heightmap, slope, water/buildability/passability masks, and per-category
object density. Style id selects the target style.

Usage:
  python scripts/retexture_map.py --src MAP.map --style N --ckpt PATH

Output:
  <src_dir>/<src_stem>_retex_style<N>.map  (or --out)
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
from map_processor.models.texture_transfer_unet import TextureTransferUNet  # noqa: E402
from map_processor.utils.style_features import extract_input_channels  # noqa: E402


def _default_data_dir() -> Path:
    return _python_tools_root() / "training_outputs" / "texture_transfer"


@torch.no_grad()
def predict_full_map(
    model, X: np.ndarray, style_id: int, device: str,
    tile: int = 256, overlap: int = 32, never_trained_mask: np.ndarray | None = None,
):
    """Tile-and-blend logits across the whole map. Mask never-trained classes."""
    C, W, H = X.shape
    n_classes = model.out_conv.out_channels
    accum = torch.zeros((n_classes, W, H), device=device, dtype=torch.float32)
    counts = torch.zeros((W, H), device=device, dtype=torch.float32)
    step = tile - overlap
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
            ph_eff = min(tile, x1 - x0); pw_eff = min(tile, y1 - y0)
            accum[:, x0:x0 + ph_eff, y0:y0 + pw_eff] += logits[:, :ph_eff, :pw_eff]
            counts[x0:x0 + ph_eff, y0:y0 + pw_eff] += 1.0
    counts = counts.clamp_min(1e-6)
    avg = accum / counts.unsqueeze(0)

    if never_trained_mask is not None:
        # Mask classes with zero training data so they cannot be predicted
        ntm = torch.from_numpy(never_trained_mask).to(device)
        avg = avg.masked_fill(ntm[:, None, None].bool(), float("-inf"))

    return avg.argmax(dim=0).cpu().numpy()  # (W, H) int64


def quantize_palette(pred_idx: np.ndarray, vocab: List[str], max_palette: int = 64):
    """Pick top-N predicted textures as the new palette; remap rare picks to nearest by family."""
    counts = Counter(pred_idx.reshape(-1).tolist())
    top = [vid for vid, _ in counts.most_common(max_palette)]
    if len(counts) <= max_palette:
        return top, pred_idx
    keep = set(top)
    fam_map = {vid: (name.split("_", 1)[0] if "_" in name else name) for vid, name in enumerate(vocab)}
    family_to_keepers: dict[str, list[int]] = {}
    for kvid in top:
        family_to_keepers.setdefault(fam_map[kvid], []).append(kvid)
    remap = {vid: vid for vid in top}
    for vid in counts:
        if vid in keep:
            continue
        fam = fam_map[vid]
        if fam in family_to_keepers:
            remap[vid] = family_to_keepers[fam][0]
        else:
            remap[vid] = top[0]
    out = np.vectorize(remap.get)(pred_idx).astype(pred_idx.dtype)
    return top, out


def fresh_repaint(blend, pred_idx: np.ndarray, vocab: List[str], pattern_mode: str = "random",
                  seed: int = 1337) -> List[str]:
    """Build new tiles + textures palette FROM SCRATCH. Wipe blends arrays.

    Source pattern indices are NOT carried over.
    """
    W, H = pred_idx.shape
    palette_ids, pred_idx_q = quantize_palette(pred_idx, vocab, max_palette=64)
    palette_names = [vocab[v] for v in palette_ids]
    name_to_pidx = {n: i for i, n in enumerate(palette_names)}

    # Per-tile palette index (uint16)
    pred_palette_idx = np.vectorize(lambda v: name_to_pidx[vocab[v]])(pred_idx_q).astype(np.uint16)

    # Pattern indices: pick fresh, do NOT use source patterns
    if pattern_mode == "random":
        rs = np.random.RandomState(seed)
        pattern = rs.randint(0, 64, size=(W, H), dtype=np.int32).astype(np.uint16)
    elif pattern_mode == "zero":
        pattern = np.zeros((W, H), dtype=np.uint16)
    else:
        raise ValueError(f"Unknown pattern_mode={pattern_mode}")

    new_tile_ids = (pred_palette_idx.astype(np.uint32) * 64 + pattern.astype(np.uint32)).astype(np.uint16)

    # Replace blend.tiles entirely (preserve shape - may include source border padding)
    full_tiles = np.zeros_like(blend.tiles, dtype=np.uint16)
    full_tiles[:W, :H] = new_tile_ids
    blend.tiles = full_tiles

    # Wipe blends arrays - the model is painting from scratch.
    blend.blends = np.zeros_like(blend.blends)
    blend.single_edge_blends = np.zeros_like(blend.single_edge_blends)
    blend.cliff_blends = np.zeros_like(blend.cliff_blends)

    # Replace texture palette
    from map_processor.assets.terrain.texture import Texture
    new_textures = []
    for i, name in enumerate(palette_names):
        t = Texture()
        t.cell_start = i * 16
        t.cell_count = 16
        t.cell_size = 4
        t.magic_value = 0
        t.name = name
        new_textures.append(t)
    blend.textures = new_textures
    return palette_names


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--src", type=Path, required=True)
    ap.add_argument("--style", type=int, required=True)
    ap.add_argument("--ckpt", type=Path, default=_default_data_dir() / "ckpt" / "best.pt")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--data_dir", type=Path, default=_default_data_dir())
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--tile", type=int, default=256)
    ap.add_argument("--overlap", type=int, default=32)
    ap.add_argument("--pattern", choices=["random", "zero"], default="random")
    ap.add_argument("--seed", type=int, default=1337)
    args = ap.parse_args()

    if not args.src.exists():
        raise SystemExit(f"Source not found: {args.src}")
    if not args.ckpt.exists():
        raise SystemExit(f"Checkpoint not found: {args.ckpt}")

    out_path = args.out or args.src.with_name(f"{args.src.stem}_retex_style{args.style}.map")
    vocab = json.loads((args.data_dir / "vocab.json").read_text(encoding="utf-8"))
    index = json.loads((args.data_dir / "index.json").read_text(encoding="utf-8"))
    n_chan = index["n_channels"]; n_styles = index["n_styles"]; vocab_size = index["vocab_size"]

    # Mask classes that never appeared in training (so model cannot predict them)
    class_freq = np.array(index.get("class_freq_train", []), dtype=np.int64)
    never_trained = (class_freq == 0).astype(np.bool_)
    if class_freq.size != vocab_size:
        never_trained = np.zeros(vocab_size, dtype=np.bool_)
    print(f"Vocab: {vocab_size}, never-trained masked: {int(never_trained.sum())}")

    ckpt = torch.load(args.ckpt, map_location=args.device, weights_only=False)
    base = ckpt["args"].get("base", 32) if isinstance(ckpt.get("args"), dict) else 32
    model = TextureTransferUNet(in_channels=n_chan, n_styles=n_styles,
                                vocab_size=vocab_size, base=base).to(args.device)
    model.load_state_dict(ckpt["model"]); model.eval()
    print(f"Model loaded: {args.ckpt} (epoch={ckpt.get('epoch','?')}, "
          f"val top1={ckpt.get('metrics', {}).get('top1', '?')})")

    print(f"Parsing source: {args.src}")
    m = Ra3Map(str(args.src)); m.parse(); ctx = m.get_context()
    blend = ctx.get_asset("BlendTileData")
    h_asset = ctx.get_asset("HeightMapData")
    objs = ctx.get_asset("ObjectsList")
    print(f"  source map {blend.map_width}x{blend.map_height} - "
          f"DISCARDING tiles, blends, single_edge_blends, cliff_blends, textures palette")

    X, W, H = extract_input_channels(blend, h_asset, objs,
                                     world_to_tile=index["world_to_tile"],
                                     sigma=index["object_sigma"])
    print(f"  context channels: {X.shape}, predicting style={args.style}")
    pred_idx = predict_full_map(model, X, args.style, args.device,
                                tile=args.tile, overlap=args.overlap,
                                never_trained_mask=never_trained)
    print(f"  predicted: {pred_idx.shape}, "
          f"{len(set(pred_idx.reshape(-1).tolist()))} unique classes")

    palette = fresh_repaint(blend, pred_idx, vocab, pattern_mode=args.pattern, seed=args.seed)
    print(f"  new palette ({len(palette)}): "
          f"{', '.join(palette[:5])}{'...' if len(palette) > 5 else ''}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    m.save(str(out_path), compress=True)
    print(f"Wrote: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
