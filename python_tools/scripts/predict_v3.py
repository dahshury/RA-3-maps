#!/usr/bin/env python3
"""Inference-only script for v3 checkpoint.

Fixes the WorldBuilder "weird tile" issue exposed by the v3 cross-eval:
  * Restricts the model's tile + blend.secondary argmax to the held-out
    map's ORIGINAL palette indices, so every predicted name is part of the
    tileset the engine actually loads for that map.
  * Compacts the writeback palette to only the texture names actually used
    in the output, instead of dumping the entire 375-entry training-union.

Usage:
  python scripts/predict_v3.py \
    --checkpoint checkpoints/v3_run1.pt \
    --target_path "../RA3 Official maps/2 IS/map_mp_2_feasel6.map" \
    --style_ref "../RA3 Official maps/2 II/map_mp_2_rao1.map"
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import torch.nn.functional as F


def _python_tools_root() -> Path:
    return Path(__file__).resolve().parents[1]


sys.path.insert(0, str(_python_tools_root()))
sys.path.insert(0, str(_python_tools_root() / "scripts"))

from map_processor import Ra3Map
from map_processor.features import extract_raw_inputs
from map_processor.features.raw_inputs import NUM_DIR_CLASSES, DIRECTION_VALUES
from map_processor.models.end_to_end_unet import ObjectStamper
from map_processor.models.v2_model import V2TextureNet
from map_processor.models.v2_style import DinoV2StyleEncoder
from map_processor.models.v2_render import PaletteRenderer
from train_v2 import re_encode_for_v2, build_dense_input
from train_cluster import _pad_t, _position_pattern_grid, render


def _pad_target(t: torch.Tensor, multiple: int) -> torch.Tensor:
    return _pad_t(t[None, None].float(), multiple)[0, 0].to(t.dtype)


def _render_full_rgb(map_path: Path, type_to_id: dict, owner_to_id: dict,
                     palette_to_id: dict, palette_renderer: PaletteRenderer,
                     device) -> torch.Tensor:
    ra3 = Ra3Map(str(map_path)); ra3.parse()
    rr = extract_raw_inputs(ra3, extract_target=True, style_id=0)
    rrec = re_encode_for_v2(rr, type_to_id, owner_to_id, palette_to_id)
    with torch.no_grad():
        base = palette_renderer.base_rgb.cpu()
        tt = rrec["target_tiles"].long().clamp_min(0)
        rgb = (base[tt].permute(2, 0, 1).contiguous()
               * (rrec["target_tiles"] >= 0).float().unsqueeze(0))
    return rgb.to(device)


def write_constrained(target_path: Path, raw_eval, model_out: Dict,
                      palette_list: List[str], palette_to_id: Dict[str, int],
                      out_path: Path, restrict_to_original: bool = True) -> None:
    """Write a .map where palette is compacted to used names AND (optionally)
    predictions are restricted to the held-out's original palette.
    """
    from map_processor.assets.terrain.texture import Texture
    from map_processor.assets.terrain.blend_info import BlendInfo

    src = Ra3Map(str(target_path)); src.parse()
    ctx = src.get_context()
    blend = ctx.get_asset("BlendTileData")
    W, H = blend.map_width, blend.map_height
    pos = _position_pattern_grid(W, H)

    V = len(palette_list)
    device = model_out["tiles"].device
    if restrict_to_original:
        allowed = [palette_to_id[n] for n in (raw_eval.palette or []) if n in palette_to_id]
        mask_vec = torch.full((V,), float("-inf"), device=device)
        if allowed:
            mask_vec[torch.tensor(allowed, device=device)] = 0.0
        # broadcast over (B, V, H, W)
        mask_b = mask_vec.view(1, V, 1, 1)
        tile_logits = model_out["tiles"] + mask_b
        sec_b = model_out["blend"]["secondary"] + mask_b
        sec_s = model_out["single_edge"]["secondary"] + mask_b
    else:
        tile_logits = model_out["tiles"]
        sec_b = model_out["blend"]["secondary"]
        sec_s = model_out["single_edge"]["secondary"]

    # Argmax in union space.
    tile_idx_u = tile_logits.argmax(1)[0][:W, :H].cpu().numpy().astype(np.int64)
    sec_b_u = sec_b.argmax(1)[0][:W, :H].cpu().numpy().astype(np.int64)
    sec_s_u = sec_s.argmax(1)[0][:W, :H].cpu().numpy().astype(np.int64)
    pres_b = (torch.sigmoid(model_out["blend"]["present"][0, 0]) > 0.5).cpu().numpy()[:W, :H]
    pres_s = (torch.sigmoid(model_out["single_edge"]["present"][0, 0]) > 0.5).cpu().numpy()[:W, :H]
    dir_b = model_out["blend"]["direction"].argmax(1)[0][:W, :H].cpu().numpy().astype(np.int64)
    dir_s = model_out["single_edge"]["direction"].argmax(1)[0][:W, :H].cpu().numpy().astype(np.int64)

    # Collect all union indices that are actually USED.
    used = set(np.unique(tile_idx_u).tolist())
    used.update(np.unique(sec_b_u[pres_b]).tolist())
    used.update(np.unique(sec_s_u[pres_s]).tolist())
    used = sorted(i for i in used if 0 <= i < V)

    compact_palette = [palette_list[i] for i in used]
    remap = np.full(V, 0, dtype=np.int32)
    for new_i, old_i in enumerate(used):
        remap[old_i] = new_i
    tile_idx_c = remap[tile_idx_u]
    sec_b_c = remap[sec_b_u]
    sec_s_c = remap[sec_s_u]

    # Write palette.
    base_meta = blend.textures[0] if blend.textures else None
    new_textures = []
    for name in compact_palette:
        # Try to copy metadata from an original texture with the same name to
        # match the engine's expected cell layout; else use base_meta defaults.
        meta = next((t for t in (blend.textures or []) if t.name == name), base_meta)
        t = Texture()
        t.cell_start = 0
        t.cell_count = meta.cell_count if meta is not None else 16
        t.cell_size = meta.cell_size if meta is not None else 4
        t.magic_value = meta.magic_value if meta is not None else 0
        t.name = name
        new_textures.append(t)
    blend.textures = new_textures

    # Tiles.
    blend.tiles = (tile_idx_c.astype(np.uint16) * 64 + pos.astype(np.uint16))

    # Blends.
    info_list: List = []
    info_key: Dict = {}

    def _ensure(sec_tex_tile: int, dir_raw: int) -> int:
        key = (int(sec_tex_tile), int(dir_raw))
        if key in info_key:
            return info_key[key] + 1
        bi = BlendInfo()
        bi.secondary_texture_tile = int(sec_tex_tile)
        bi.blend_direction = int(dir_raw)
        bi.i3 = 0; bi.i4 = 0
        bi._blend_direction_raw = (
            bi._from_blend_direction(dir_raw)
            if hasattr(bi, "_from_blend_direction")
            else bytes(6)
        )
        info_list.append(bi)
        info_key[key] = len(info_list) - 1
        return len(info_list)

    def _emit(present, sec_c, dir_arr, attr):
        out = np.zeros((W, H), dtype=np.uint16)
        for x in range(W):
            for y in range(H):
                if not present[x, y]:
                    continue
                dc = int(dir_arr[x, y])
                if dc < 0 or dc >= NUM_DIR_CLASSES:
                    continue
                dr = DIRECTION_VALUES[dc]
                if dr < 0:
                    continue
                tile = int(sec_c[x, y]) * 64 + int(pos[x, y])
                out[x, y] = _ensure(tile, dr)
        setattr(blend, attr, out)

    _emit(pres_b, sec_b_c, dir_b, "blends")
    _emit(pres_s, sec_s_c, dir_s, "single_edge_blends")
    blend.blend_info = info_list
    blend.blends_count = len(info_list)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    src.save(str(out_path), compress=True)
    print(f"  Wrote {out_path}  (palette={len(compact_palette)} entries, "
          f"original={len(raw_eval.palette or [])}, restricted={restrict_to_original})")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--target_path", type=Path, required=True,
                    help="Path to the .map file to predict on.")
    ap.add_argument("--style_ref", type=Path, default=None,
                    help="Optional: path to a .map whose render is used as DINOv2 style. Defaults to the target itself (oracle).")
    ap.add_argument("--out_path", type=Path, default=None)
    ap.add_argument("--no_restrict", action="store_true",
                    help="Don't mask logits to the original palette (debug).")
    ap.add_argument("--no_render", action="store_true")
    ap.add_argument("--obj_embed_dim", type=int, default=11)
    ap.add_argument("--decoder_dim", type=int, default=256)
    ap.add_argument("--style_dim", type=int, default=256)
    ap.add_argument("--blend_hidden", type=int, default=128)
    ap.add_argument("--w_obj", type=float, default=0.1)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Loading checkpoint: {args.checkpoint}")
    ckpt = torch.load(str(args.checkpoint), map_location=device, weights_only=False)
    palette_list: List[str] = list(ckpt["palette_list"])
    type_to_id: Dict[str, int] = dict(ckpt["type_to_id"])
    owner_to_id: Dict[str, int] = dict(ckpt["owner_to_id"])
    palette_to_id = {n: i for i, n in enumerate(palette_list)}
    print(f"  palette={len(palette_list)} types={len(type_to_id)} owners={len(owner_to_id)}")

    palette_renderer = PaletteRenderer(palette_list, learnable_residual=True).to(device)
    palette_renderer.load_state_dict(ckpt["palette_renderer"])
    style_enc = DinoV2StyleEncoder(style_dim=args.style_dim, cfg_dropout=0.0).to(device)
    style_enc.proj.load_state_dict(ckpt["style_enc_proj"])
    style_enc.null_style.data.copy_(ckpt["style_enc_null"].to(device))
    stamper = ObjectStamper(n_types=max(len(type_to_id), 1),
                             n_owners=max(len(owner_to_id), 1),
                             embed_dim=args.obj_embed_dim).to(device)
    stamper.load_state_dict(ckpt["stamper"])
    in_ch = 5 + args.obj_embed_dim
    model = V2TextureNet(
        palette_size=len(palette_list), n_directions=NUM_DIR_CLASSES,
        in_channels=in_ch, style_dim=args.style_dim,
        decoder_dim=args.decoder_dim, blend_hidden=args.blend_hidden,
    ).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval(); stamper.eval(); style_enc.eval()

    # --- Inference ---
    ra3 = Ra3Map(str(args.target_path)); ra3.parse()
    raw_eval = extract_raw_inputs(ra3, extract_target=True, style_id=0)
    rec = re_encode_for_v2(raw_eval, type_to_id, owner_to_id, palette_to_id)
    dense_cpu = build_dense_input(rec, stamper, device=torch.device("cpu"))
    dense_p = _pad_t(dense_cpu[None], 32).to(device)

    # Style reference: own render by default (oracle), else from --style_ref.
    style_src = args.style_ref if args.style_ref else args.target_path
    print(f"  Style reference: {style_src}")
    ref_rgb = _render_full_rgb(Path(style_src), type_to_id, owner_to_id,
                                palette_to_id, palette_renderer, device)
    style = style_enc.encode_image(ref_rgb[None]).contiguous()

    Wp, Hp = dense_p.shape[-2], dense_p.shape[-1]
    og = stamper([rec["objects"]], Wp, Hp)
    x_in = torch.cat([dense_p, og * args.w_obj], dim=1)
    with torch.no_grad():
        out = model(x_in, style)
    oW, oH = rec["width"], rec["height"]
    out["tiles"] = out["tiles"][:, :, :oW, :oH]
    out["blend"] = {k: v[:, :, :oW, :oH] for k, v in out["blend"].items()}
    out["single_edge"] = {k: v[:, :, :oW, :oH] for k, v in out["single_edge"].items()}

    # Compute tile_acc with and without the restriction (diagnostic).
    gt = rec["target_tiles"].numpy()
    pred_un = out["tiles"].argmax(1)[0].cpu().numpy()
    print(f"  Unrestricted tile_acc:  {100*(pred_un == gt).mean():.2f}%")
    if not args.no_restrict:
        allowed = [palette_to_id[n] for n in (raw_eval.palette or []) if n in palette_to_id]
        mask = torch.full((1, len(palette_list), 1, 1), float("-inf"), device=device)
        if allowed:
            mask[0, torch.tensor(allowed, device=device), 0, 0] = 0.0
        pred_re = (out["tiles"] + mask).argmax(1)[0].cpu().numpy()
        print(f"  Restricted   tile_acc:  {100*(pred_re == gt).mean():.2f}%   "
              f"(allowed_palette_in_union={len(allowed)})")

    out_path = args.out_path or (args.target_path.parent / "skinned_full" /
                                 f"v3p_predicted_{args.target_path.stem}.map")
    write_constrained(args.target_path, raw_eval, out, palette_list, palette_to_id,
                      out_path, restrict_to_original=(not args.no_restrict))
    if not args.no_render:
        render(out_path, out_path.parent / "_renders" / out_path.stem)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
