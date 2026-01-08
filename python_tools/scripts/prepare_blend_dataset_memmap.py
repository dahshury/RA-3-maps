"""
Prepare a large blend dataset for training.

Why this exists
--------------
The generator writes a single compressed .npz. That's convenient, but very slow for training
because you can't memory-map compressed NPZ members efficiently.

This script converts:
  - X (50-d: 25 texture indices + 25 elevation) + labels + map_id
into memmap-friendly .npy files and, critically, remaps *local per-map texture indices*
into a *global texture vocabulary* using the metadata JSON.

Output directory contains:
  - tex.npy               int32 [N, 25]  global texture IDs for the 5x5 window
  - tex_local_norm.npy    float32 [N, 25] local (per-map) texture index normalized to [0..1]
  - elev.npy              float32 [N, 25] elevation window
  - y_blend_present.npy   uint8  [N]
  - y_blend_mask.npy      uint8  [N] 8-bit neighbor mask (bit i => neighbor i equals secondary), 255=ignore (if present in NPZ)
  - y_blend_sec.npy       int8   [N] legacy: first matching neighbor index (0-7) or -1
  - y_blend_dir.npy       int16  [N] direction class index (or -1 when not present)
  - y_se_present.npy      uint8  [N]
  - y_se_mask.npy         uint8  [N] 8-bit neighbor mask, 255=ignore (if present in NPZ)
  - y_se_sec.npy          int8   [N] legacy: first matching neighbor index (0-7) or -1
  - y_se_dir.npy          int16  [N] direction class index (or -1 when not present)
  - map_id.npy            int32  [N]
  - map_style.npy         float32 [N, K] per-sample map-style features (regime hints, optional)
  - prepared_meta.json    global vocab + direction vocab + stats

Usage
-----
python scripts/prepare_blend_dataset_memmap.py \
  --npz "../blendinfo dataset/_generated/blend_dataset_w5_elev_full.npz" \
  --meta "../blendinfo dataset/_generated/blend_dataset_w5_elev_full.json" \
  --out-dir "../blendinfo dataset/_generated/prepared_w5_elev_full"
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np


def _load_meta(meta_path: Path) -> dict:
    return json.loads(meta_path.read_text(encoding="utf-8"))


def _build_texture_vocab(meta: dict) -> Tuple[Dict[str, int], List[List[int]]]:
    """
    Returns (name_to_id, per_map_local_to_global_ids).
    per_map_local_to_global_ids[m][local_idx] = global_id
    """
    name_to_id: Dict[str, int] = {}
    per_map_local_to_global: List[List[int]] = []

    pairs = meta.get("pairs") or []
    for entry in pairs:
        tex_names = entry.get("textures") or []
        local_to_global: List[int] = []
        for nm in tex_names:
            if nm not in name_to_id:
                name_to_id[nm] = len(name_to_id)
            local_to_global.append(int(name_to_id[nm]))
        per_map_local_to_global.append(local_to_global)

    return name_to_id, per_map_local_to_global


def _pad_mapping(per_map: List[List[int]]) -> np.ndarray:
    max_len = max((len(x) for x in per_map), default=0)
    out = np.full((len(per_map), max_len), -1, dtype=np.int32)
    for i, row in enumerate(per_map):
        if not row:
            continue
        out[i, : len(row)] = np.asarray(row, dtype=np.int32)
    return out


def _dir_vocab_from_arrays(*dirs: np.ndarray) -> Tuple[List[int], np.ndarray]:
    """
    Build direction vocabulary from raw arrays and return (values, lut) where:
      - values: sorted unique raw direction values (int)
      - lut: int16 LUT of size 65536 mapping uint16(raw) -> class idx, default -1
    """
    uniq: np.ndarray = np.unique(np.concatenate([d.astype(np.int32, copy=False) for d in dirs], axis=0))
    values = [int(v) for v in uniq.tolist()]

    lut = np.full((65536,), -1, dtype=np.int16)
    for i, v in enumerate(values):
        if 0 <= v < 65536:
            lut[v] = np.int16(i)
        else:
            # out-of-range raw values shouldn't happen; ignore
            pass
    return values, lut


def main() -> int:
    ap = argparse.ArgumentParser(description="Convert compressed blend dataset NPZ to memmap-friendly .npy files.")
    ap.add_argument("--npz", required=True, help="Path to dataset .npz")
    ap.add_argument("--meta", default="", help="Path to dataset metadata .json (default: <npz>.json)")
    ap.add_argument("--out-dir", required=True, help="Output directory to write .npy + prepared_meta.json")
    ap.add_argument("--force", action="store_true", help="Overwrite outputs if they exist")
    ap.add_argument("--seed", type=int, default=123, help="RNG seed for shuffling")
    ap.add_argument(
        "--no-shuffle",
        action="store_true",
        help="Disable shuffling before writing .npy files (not recommended).",
    )
    args = ap.parse_args()

    npz_path = Path(args.npz)
    meta_path = Path(args.meta) if args.meta else npz_path.with_suffix(".json")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    prepared_meta_path = out_dir / "prepared_meta.json"
    if prepared_meta_path.exists() and not args.force:
        raise SystemExit(f"Refusing to overwrite existing prepared dataset: {prepared_meta_path} (use --force)")

    meta = _load_meta(meta_path)
    tex_name_to_id, per_map_local_to_global = _build_texture_vocab(meta)
    map_tex_map = _pad_mapping(per_map_local_to_global)  # [num_maps, max_local_tex]

    # NPZ is compressed: this will decompress into memory. That's intended as a one-time conversion step.
    data = np.load(npz_path)
    X = np.asarray(data["X"], dtype=np.float32)
    map_id = np.asarray(data["map_id"], dtype=np.int32)

    # labels
    ybp = np.asarray(data["y_blend_present"], dtype=np.uint8)
    # NEW (optional): 8-bit neighbor mask (bit i => neighbor i equals secondary), 255=ignore
    ybm = np.asarray(data["y_blend_mask"], dtype=np.uint8) if ("y_blend_mask" in data) else None
    # Legacy: first matching neighbor index (0-7), -1 if unknown
    ybs_neighbor = np.asarray(data["y_blend_sec"], dtype=np.int8)
    ybd_raw = np.asarray(data["y_blend_dir"], dtype=np.int32)

    ysp = np.asarray(data["y_se_present"], dtype=np.uint8)
    ysm = np.asarray(data["y_se_mask"], dtype=np.uint8) if ("y_se_mask" in data) else None
    yss_neighbor = np.asarray(data["y_se_sec"], dtype=np.int8)
    ysd_raw = np.asarray(data["y_se_dir"], dtype=np.int32)
    
    # Check if new neighbor-based encoding is present
    label_encoding = meta.get("label_encoding", {})
    uses_neighbor_idx = label_encoding.get("y_blend_sec") == "neighbor_index"

    window = int(meta.get("window", 5))
    include_elevation = bool(meta.get("include_elevation", True))
    base_tex = int(window) * int(window)
    base_elev = base_tex if include_elevation else 0

    n = int(X.shape[0])
    if int(X.shape[1]) < (base_tex + base_elev):
        raise SystemExit(
            f"X has too few dims for window={window} include_elevation={include_elevation}: "
            f"need >= {base_tex + base_elev}, got {X.shape[1]}"
        )
    extra_dim = int(X.shape[1]) - int(base_tex + base_elev)

    # split X into (tex_local, elev, extra)
    tex_local = np.rint(X[:, :base_tex]).astype(np.int32, copy=False)  # [N, base_tex], local map texture indices
    elev = None
    if include_elevation:
        elev = np.asarray(X[:, base_tex : base_tex + base_elev], dtype=np.float32)
    extra = None
    if extra_dim > 0:
        extra = np.asarray(X[:, base_tex + base_elev : base_tex + base_elev + extra_dim], dtype=np.float32)

    # remap local->global using padded mapping: map_tex_map[map_id, local_idx]
    mid = map_id.astype(np.int64, copy=False)
    tex_global = map_tex_map[mid[:, None], tex_local]  # [N, base_tex]

    # NEW: local palette-order signal (this is what the blend tool compares!)
    # Normalize per-map local texture index to [0..1] using the per-map texture count from meta.
    pairs = meta.get("pairs") or []
    per_map_ntex = np.ones((int(map_tex_map.shape[0]),), dtype=np.float32)
    for i in range(min(len(pairs), int(map_tex_map.shape[0]))):
        try:
            per_map_ntex[i] = float(max(1, int(len(pairs[i].get("textures") or []))))
        except Exception:
            per_map_ntex[i] = 1.0
    denom = np.maximum(per_map_ntex[mid], 1.0) - 1.0
    denom = np.where(denom > 0.0, denom, 1.0).astype(np.float32, copy=False)
    tex_local_norm = (tex_local.astype(np.float32, copy=False) / denom[:, None]).astype(np.float32, copy=False)

    # y_blend_sec / y_se_sec: now neighbor indices (0-7), NOT texture IDs
    # No remapping needed - just pass through
    # (The model predicts WHICH neighbor, then we look up that neighbor's texture from tex_global)
    ybs_out = ybs_neighbor  # int8 [N], 0-7 or -1
    yss_out = yss_neighbor  # int8 [N], 0-7 or -1

    # direction vocab (shared between blend + se)
    dir_values, dir_lut = _dir_vocab_from_arrays(ybd_raw, ysd_raw)

    # map raw dir -> class idx, but ignore when not present
    ybd_cls = dir_lut[(ybd_raw.astype(np.int32) & 0xFFFF)].astype(np.int16, copy=False)
    ysd_cls = dir_lut[(ysd_raw.astype(np.int32) & 0xFFFF)].astype(np.int16, copy=False)
    ybd_cls = np.where(ybp > 0, ybd_cls, np.int16(-1)).astype(np.int16, copy=False)
    ysd_cls = np.where(ysp > 0, ysd_cls, np.int16(-1)).astype(np.int16, copy=False)

    # stats (for elevation normalization)
    if include_elevation and elev is not None:
        elev_mean = float(elev.mean(dtype=np.float64))
        elev_std = float(elev.std(dtype=np.float64) + 1e-8)
    else:
        elev_mean = 0.0
        elev_std = 1.0

    # Strong global shuffle (training should not depend on file order).
    # Keep the permutation for reproducibility/debugging.
    shuffle_perm = None
    if not bool(args.no_shuffle):
        rng = np.random.default_rng(int(args.seed))
        shuffle_perm = rng.permutation(n).astype(np.int64, copy=False)

        tex_global = tex_global[shuffle_perm]
        tex_local_norm = tex_local_norm[shuffle_perm]
        if include_elevation and elev is not None:
            elev = elev[shuffle_perm]
        if extra is not None:
            extra = extra[shuffle_perm]
        ybp = ybp[shuffle_perm]
        if ybm is not None:
            ybm = ybm[shuffle_perm]
        ybs_out = ybs_out[shuffle_perm]
        ybd_cls = ybd_cls[shuffle_perm]
        ysp = ysp[shuffle_perm]
        if ysm is not None:
            ysm = ysm[shuffle_perm]
        yss_out = yss_out[shuffle_perm]
        ysd_cls = ysd_cls[shuffle_perm]
        map_id = map_id[shuffle_perm]

    # write outputs
    def _save(name: str, arr: np.ndarray) -> None:
        p = out_dir / name
        if p.exists() and not args.force:
            raise SystemExit(f"Refusing to overwrite: {p} (use --force)")
        np.save(p, arr)

    _save("tex.npy", tex_global.astype(np.int32, copy=False))
    _save("tex_local_norm.npy", tex_local_norm.astype(np.float32, copy=False))
    if include_elevation and elev is not None:
        _save("elev.npy", elev.astype(np.float32, copy=False))
    if extra is not None:
        _save("extra.npy", extra.astype(np.float32, copy=False))
    _save("y_blend_present.npy", ybp)
    if ybm is not None:
        _save("y_blend_mask.npy", ybm.astype(np.uint8, copy=False))
    _save("y_blend_sec.npy", ybs_out.astype(np.int8, copy=False))  # neighbor index 0-7
    _save("y_blend_dir.npy", ybd_cls)
    _save("y_se_present.npy", ysp)
    if ysm is not None:
        _save("y_se_mask.npy", ysm.astype(np.uint8, copy=False))
    _save("y_se_sec.npy", yss_out.astype(np.int8, copy=False))  # neighbor index 0-7
    _save("y_se_dir.npy", ysd_cls)
    _save("map_id.npy", map_id)

    # --- map-style features (regime hints; avoids separate models) ---
    map_style_names: List[str] = []
    map_style = None
    try:
        pairs = meta.get("pairs") or []
        num_maps = int(map_id.max()) + 1 if map_id.size else 0
        map_style_names = [
            "blend_pos_rate",
            "se_pos_rate",
            "blend_mask_valid_rate",
            "se_mask_valid_rate",
            "rule_adherence",  # 1.0 = consistent map (center<secondary always), <0.7 = inconsistent
        ]
        K = len(map_style_names)
        per_map = np.zeros((num_maps, K), dtype=np.float32)
        for mi in range(min(num_maps, len(pairs))):
            st = (pairs[mi].get("stats") or {}) if isinstance(pairs[mi], dict) else {}
            per_map[mi, 0] = float(st.get("blend_pos_rate", 0.0))
            per_map[mi, 1] = float(st.get("se_pos_rate", 0.0))
            per_map[mi, 2] = float(st.get("blend_mask_valid_rate", 0.0))
            per_map[mi, 3] = float(st.get("se_mask_valid_rate", 0.0))
            per_map[mi, 4] = float(st.get("rule_adherence", 0.5))  # Default 0.5 if unknown
        map_style = per_map[map_id.astype(np.int64, copy=False)]
        _save("map_style.npy", map_style.astype(np.float32, copy=False))
    except Exception:
        map_style_names = []
        map_style = None

    # Get label encoding info from source meta
    label_encoding = meta.get("label_encoding", {})
    neighbor_names = label_encoding.get("neighbor_names", ["TL", "T", "TR", "L", "R", "BL", "B", "BR"])
    
    prepared_meta = {
        "source_npz": str(npz_path),
        "source_meta": str(meta_path),
        "n_samples": n,
        "window": int(window),
        "include_elevation": bool(include_elevation),
        "extra_dim": int(extra_dim),
        "extra_feature_names": meta.get("extra_feature_names", []),
        "map_style_dim": int(map_style.shape[1]) if map_style is not None and getattr(map_style, "ndim", 0) == 2 else 0,
        "map_style_names": list(map_style_names or []),
        "vocab": {
            "texture_name_to_id": tex_name_to_id,
            "num_textures": len(tex_name_to_id),
            "map_local_to_global_padded_shape": [int(map_tex_map.shape[0]), int(map_tex_map.shape[1])],
        },
        "local_tex": {
            "encoding": "local_index_norm",
            "source": "X[:,:window*window] local palette indices normalized by meta.pairs[i].textures.length",
        },
        # NEW: neighbor-based secondary texture prediction
        "neighbor": {
            "names": neighbor_names,  # ["TL", "T", "TR", "L", "R", "BL", "B", "BR"]
            "num_classes": 8,
            "ignore_index": -1,
        },
        "mask": {
            "encoding": "neighbor_mask8_u8",
            "ignore_value": 255,
            "enabled": bool(ybm is not None and ysm is not None),
        },
        "direction": {
            "values": dir_values,
            "num_classes": len(dir_values),
            "ignore_index": -1,
        },
        "elevation_norm": {"mean": elev_mean, "std": elev_std},
        "shuffle": {"enabled": (not bool(args.no_shuffle)), "seed": int(args.seed)},
        # Document texture semantic vocabularies (for reference)
        "texture_type_vocab": meta.get("texture_type_vocab", []),
        "texture_biome_vocab": meta.get("texture_biome_vocab", []),
    }
    if shuffle_perm is not None:
        _save("shuffle_perm.npy", shuffle_perm)
    prepared_meta_path.write_text(json.dumps(prepared_meta, indent=2), encoding="utf-8")

    print(f"Wrote prepared dataset to: {out_dir}")
    print(f"- samples: {n}")
    print(f"- global textures: {len(tex_name_to_id)}")
    print(f"- neighbor classes: 8 (TL,T,TR,L,R,BL,B,BR)")
    print(f"- direction classes: {len(dir_values)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())





