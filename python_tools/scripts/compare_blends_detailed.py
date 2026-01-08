"""
Detailed comparison of predicted vs original blends.

Extracts and compares:
- Blend presence (true/false positives/negatives)
- Secondary texture (which neighbor)
- Direction
- Tile type breakdown
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple, Optional
import sys

import numpy as np

# Add parent to path for imports
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from map_processor.ra3map import Ra3Map
from map_processor.assets.terrain.blend_tile_data import BlendTileData
from map_processor.assets.terrain.blend_info import BlendInfo
from map_processor.assets.terrain.blend_direction import BlendDirection


# Neighbor offsets matching the mask bits (row, col order)
_NEIGHBOR_OFFSETS = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
_NEIGHBOR_NAMES = ["TL", "T", "TR", "L", "R", "BL", "B", "BR"]


def _get_texture_from_tile(x: int, y: int, tile_value: int) -> int:
    """C# BlendTileData.GetTexture inverse"""
    row_first = (y % 8) // 2 * 16 + (y % 2) * 2
    current = (x % 8) // 2 * 4 + (x % 2) + row_first
    return (int(tile_value) - current) // 64


def _decode_texture_grid(blend: BlendTileData) -> np.ndarray:
    """Convert tile grid to texture id grid."""
    tiles = np.asarray(blend.tiles, dtype=np.int32)
    w, h = tiles.shape
    tex = np.zeros((w, h), dtype=np.int32)
    for x in range(w):
        for y in range(h):
            tex[x, y] = _get_texture_from_tile(x, y, int(tiles[x, y]))
    return tex


def extract_blend_data(map_path: Path) -> dict:
    """Extract all blend data from a map."""
    m = Ra3Map(str(map_path))
    m.parse()
    ctx = m.get_context()
    blend = ctx.get_asset_by_type(BlendTileData)
    
    if blend is None:
        raise ValueError(f"BlendTileData not found in {map_path}")
    
    w, h = blend.tiles.shape
    tex_grid = _decode_texture_grid(blend)
    
    # Build texture name lookup
    tex_names = {i: t.name for i, t in enumerate(blend.textures or [])}
    
    # Build blend_info lookup
    blend_info_list = list(blend.blend_info) if blend.blend_info else []
    
    data = {
        "map_path": str(map_path),
        "size": (w, h),
        "tex_names": tex_names,
        "blend_info_count": len(blend_info_list),
        "cells": {},  # (x,y) -> cell data
    }
    
    blends_arr = np.asarray(blend.blends)
    se_arr = np.asarray(blend.single_edge_blends)
    
    for x in range(w):
        for y in range(h):
            center_tex = int(tex_grid[x, y])
            blend_idx = int(blends_arr[x, y])
            se_idx = int(se_arr[x, y])
            
            cell = {
                "center_tex": center_tex,
                "center_tex_name": tex_names.get(center_tex, f"tex_{center_tex}"),
                "blend_present": blend_idx > 0,
                "se_present": se_idx > 0,
            }
            
            if blend_idx > 0 and blend_idx <= len(blend_info_list):
                bi = blend_info_list[blend_idx - 1]
                sec_tex = _get_texture_from_tile(x, y, bi.secondary_texture_tile)
                cell["blend"] = {
                    "secondary_tex": sec_tex,
                    "secondary_tex_name": tex_names.get(sec_tex, f"tex_{sec_tex}"),
                    "direction": bi.blend_direction.value if bi.blend_direction else 0,
                    "direction_name": bi.blend_direction.name if bi.blend_direction else "Unknown",
                    "secondary_tile_raw": bi.secondary_texture_tile,
                }
                # Find which neighbor has this texture
                for ni, (dx, dy) in enumerate(_NEIGHBOR_OFFSETS):
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < w and 0 <= ny < h:
                        if tex_grid[nx, ny] == sec_tex:
                            cell["blend"]["neighbor_idx"] = ni
                            cell["blend"]["neighbor_name"] = _NEIGHBOR_NAMES[ni]
                            break
                else:
                    cell["blend"]["neighbor_idx"] = -1
                    cell["blend"]["neighbor_name"] = "NONE"
            
            if se_idx > 0 and se_idx <= len(blend_info_list):
                bi = blend_info_list[se_idx - 1]
                sec_tex = _get_texture_from_tile(x, y, bi.secondary_texture_tile)
                cell["se"] = {
                    "secondary_tex": sec_tex,
                    "secondary_tex_name": tex_names.get(sec_tex, f"tex_{sec_tex}"),
                    "direction": bi.blend_direction.value if bi.blend_direction else 0,
                    "direction_name": bi.blend_direction.name if bi.blend_direction else "Unknown",
                }
                for ni, (dx, dy) in enumerate(_NEIGHBOR_OFFSETS):
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < w and 0 <= ny < h:
                        if tex_grid[nx, ny] == sec_tex:
                            cell["se"]["neighbor_idx"] = ni
                            cell["se"]["neighbor_name"] = _NEIGHBOR_NAMES[ni]
                            break
                else:
                    cell["se"]["neighbor_idx"] = -1
                    cell["se"]["neighbor_name"] = "NONE"
            
            # Only store cells with blend or SE
            if blend_idx > 0 or se_idx > 0:
                data["cells"][(x, y)] = cell
    
    return data


def compare_blends(orig_data: dict, pred_data: dict) -> dict:
    """Compare original and predicted blend data in detail."""
    w, h = orig_data["size"]
    
    stats = {
        "total_cells": w * h,
        "blend": {
            "true_positive": 0,
            "true_negative": 0,
            "false_positive": 0,
            "false_negative": 0,
            # When both have blend
            "both_present": {
                "total": 0,
                "same_secondary": 0,
                "same_direction": 0,
                "same_neighbor": 0,
                "exact_match": 0,
            },
            # Direction breakdown
            "direction_confusion": defaultdict(lambda: defaultdict(int)),
            # Neighbor breakdown
            "neighbor_confusion": defaultdict(lambda: defaultdict(int)),
        },
        "se": {
            "true_positive": 0,
            "true_negative": 0,
            "false_positive": 0,
            "false_negative": 0,
        },
        "examples": {
            "false_positives": [],
            "false_negatives": [],
            "wrong_secondary": [],
            "wrong_direction": [],
            "wrong_neighbor": [],
        }
    }
    
    orig_cells = orig_data["cells"]
    pred_cells = pred_data["cells"]
    
    all_coords = set()
    all_coords.update(orig_cells.keys())
    all_coords.update(pred_cells.keys())
    
    for x in range(w):
        for y in range(h):
            key = (x, y)
            orig = orig_cells.get(key, {})
            pred = pred_cells.get(key, {})
            
            orig_blend = orig.get("blend_present", False)
            pred_blend = pred.get("blend_present", False)
            
            # Blend presence classification
            if orig_blend and pred_blend:
                stats["blend"]["true_positive"] += 1
                stats["blend"]["both_present"]["total"] += 1
                
                # Compare details
                ob = orig.get("blend", {})
                pb = pred.get("blend", {})
                
                same_sec = ob.get("secondary_tex") == pb.get("secondary_tex")
                same_dir = ob.get("direction") == pb.get("direction")
                same_neigh = ob.get("neighbor_idx") == pb.get("neighbor_idx")
                
                if same_sec:
                    stats["blend"]["both_present"]["same_secondary"] += 1
                if same_dir:
                    stats["blend"]["both_present"]["same_direction"] += 1
                if same_neigh:
                    stats["blend"]["both_present"]["same_neighbor"] += 1
                if same_sec and same_dir:
                    stats["blend"]["both_present"]["exact_match"] += 1
                
                # Track confusion
                orig_dir = ob.get("direction_name", "Unknown")
                pred_dir = pb.get("direction_name", "Unknown")
                stats["blend"]["direction_confusion"][orig_dir][pred_dir] += 1
                
                orig_neigh = ob.get("neighbor_name", "Unknown")
                pred_neigh = pb.get("neighbor_name", "Unknown")
                stats["blend"]["neighbor_confusion"][orig_neigh][pred_neigh] += 1
                
                # Examples of mismatches
                if not same_sec and len(stats["examples"]["wrong_secondary"]) < 20:
                    stats["examples"]["wrong_secondary"].append({
                        "pos": key,
                        "orig_sec": ob.get("secondary_tex_name"),
                        "pred_sec": pb.get("secondary_tex_name"),
                        "orig_neigh": ob.get("neighbor_name"),
                        "pred_neigh": pb.get("neighbor_name"),
                    })
                if not same_dir and len(stats["examples"]["wrong_direction"]) < 20:
                    stats["examples"]["wrong_direction"].append({
                        "pos": key,
                        "orig_dir": orig_dir,
                        "pred_dir": pred_dir,
                    })
                if not same_neigh and len(stats["examples"]["wrong_neighbor"]) < 20:
                    stats["examples"]["wrong_neighbor"].append({
                        "pos": key,
                        "orig_neigh": orig_neigh,
                        "pred_neigh": pred_neigh,
                        "orig_sec": ob.get("secondary_tex_name"),
                        "pred_sec": pb.get("secondary_tex_name"),
                    })
                    
            elif orig_blend and not pred_blend:
                stats["blend"]["false_negative"] += 1
                if len(stats["examples"]["false_negatives"]) < 20:
                    ob = orig.get("blend", {})
                    stats["examples"]["false_negatives"].append({
                        "pos": key,
                        "orig_sec": ob.get("secondary_tex_name"),
                        "orig_dir": ob.get("direction_name"),
                        "orig_neigh": ob.get("neighbor_name"),
                    })
            elif not orig_blend and pred_blend:
                stats["blend"]["false_positive"] += 1
                if len(stats["examples"]["false_positives"]) < 20:
                    pb = pred.get("blend", {})
                    stats["examples"]["false_positives"].append({
                        "pos": key,
                        "pred_sec": pb.get("secondary_tex_name"),
                        "pred_dir": pb.get("direction_name"),
                        "pred_neigh": pb.get("neighbor_name"),
                    })
            else:
                stats["blend"]["true_negative"] += 1
            
            # SE classification
            orig_se = orig.get("se_present", False)
            pred_se = pred.get("se_present", False)
            
            if orig_se and pred_se:
                stats["se"]["true_positive"] += 1
            elif orig_se and not pred_se:
                stats["se"]["false_negative"] += 1
            elif not orig_se and pred_se:
                stats["se"]["false_positive"] += 1
            else:
                stats["se"]["true_negative"] += 1
    
    # Convert defaultdicts to regular dicts for JSON serialization
    stats["blend"]["direction_confusion"] = {
        k: dict(v) for k, v in stats["blend"]["direction_confusion"].items()
    }
    stats["blend"]["neighbor_confusion"] = {
        k: dict(v) for k, v in stats["blend"]["neighbor_confusion"].items()
    }
    
    # Compute derived metrics
    bp = stats["blend"]
    total_pos = bp["true_positive"] + bp["false_negative"]
    total_pred_pos = bp["true_positive"] + bp["false_positive"]
    
    stats["blend"]["metrics"] = {
        "precision": bp["true_positive"] / total_pred_pos if total_pred_pos > 0 else 0,
        "recall": bp["true_positive"] / total_pos if total_pos > 0 else 0,
        "accuracy": (bp["true_positive"] + bp["true_negative"]) / stats["total_cells"],
    }
    
    # When both present, what % are correct
    both = bp["both_present"]
    if both["total"] > 0:
        stats["blend"]["both_present"]["pct_same_secondary"] = both["same_secondary"] / both["total"]
        stats["blend"]["both_present"]["pct_same_direction"] = both["same_direction"] / both["total"]
        stats["blend"]["both_present"]["pct_same_neighbor"] = both["same_neighbor"] / both["total"]
        stats["blend"]["both_present"]["pct_exact_match"] = both["exact_match"] / both["total"]
    
    return stats


def print_report(stats: dict):
    """Print a human-readable report."""
    print("\n" + "="*80)
    print("BLEND COMPARISON REPORT")
    print("="*80)
    
    bp = stats["blend"]
    print(f"\n--- Blend Presence ---")
    print(f"True Positives:  {bp['true_positive']:,}")
    print(f"True Negatives:  {bp['true_negative']:,}")
    print(f"False Positives: {bp['false_positive']:,} (predicted blend where none exists)")
    print(f"False Negatives: {bp['false_negative']:,} (missed blends)")
    
    m = bp["metrics"]
    print(f"\nPrecision: {m['precision']*100:.1f}%")
    print(f"Recall:    {m['recall']*100:.1f}%")
    print(f"Accuracy:  {m['accuracy']*100:.1f}%")
    
    print(f"\n--- When Both Have Blend (True Positives) ---")
    both = bp["both_present"]
    if both["total"] > 0:
        print(f"Total: {both['total']:,}")
        print(f"Same Secondary Texture: {both['same_secondary']:,} ({both['pct_same_secondary']*100:.1f}%)")
        print(f"Same Direction:         {both['same_direction']:,} ({both['pct_same_direction']*100:.1f}%)")
        print(f"Same Neighbor:          {both['same_neighbor']:,} ({both['pct_same_neighbor']*100:.1f}%)")
        print(f"Exact Match (sec+dir):  {both['exact_match']:,} ({both['pct_exact_match']*100:.1f}%)")
    
    print(f"\n--- Direction Confusion Matrix ---")
    dir_conf = bp["direction_confusion"]
    all_dirs = sorted(set(dir_conf.keys()) | set(d for v in dir_conf.values() for d in v.keys()))
    
    # Find most common directions
    dir_totals = defaultdict(int)
    for orig_dir, preds in dir_conf.items():
        for pred_dir, count in preds.items():
            dir_totals[orig_dir] += count
    
    top_dirs = sorted(dir_totals.keys(), key=lambda d: dir_totals[d], reverse=True)[:10]
    
    print(f"Top directions (orig -> pred counts):")
    for orig_dir in top_dirs:
        preds = dir_conf.get(orig_dir, {})
        total = sum(preds.values())
        correct = preds.get(orig_dir, 0)
        if total > 0:
            top_preds = sorted(preds.items(), key=lambda x: x[1], reverse=True)[:3]
            top_str = ", ".join(f"{p}:{c}" for p, c in top_preds)
            print(f"  {orig_dir}: {correct}/{total} correct ({correct/total*100:.1f}%) | top: {top_str}")
    
    print(f"\n--- Neighbor Confusion Matrix ---")
    neigh_conf = bp["neighbor_confusion"]
    print("(Original neighbor -> Predicted neighbor)")
    for orig_neigh in _NEIGHBOR_NAMES + ["NONE"]:
        preds = neigh_conf.get(orig_neigh, {})
        if preds:
            total = sum(preds.values())
            correct = preds.get(orig_neigh, 0)
            top_preds = sorted(preds.items(), key=lambda x: x[1], reverse=True)[:3]
            top_str = ", ".join(f"{p}:{c}" for p, c in top_preds)
            print(f"  {orig_neigh:4s}: {correct}/{total} correct ({correct/total*100:.1f}%) | top: {top_str}")
    
    print(f"\n--- Example Errors ---")
    ex = stats["examples"]
    
    if ex["false_positives"]:
        print(f"\nFalse Positives (predicted blend where none exists):")
        for e in ex["false_positives"][:5]:
            print(f"  {e['pos']}: pred sec={e['pred_sec']}, dir={e['pred_dir']}, neigh={e['pred_neigh']}")
    
    if ex["false_negatives"]:
        print(f"\nFalse Negatives (missed blends):")
        for e in ex["false_negatives"][:5]:
            print(f"  {e['pos']}: orig sec={e['orig_sec']}, dir={e['orig_dir']}, neigh={e['orig_neigh']}")
    
    if ex["wrong_neighbor"]:
        print(f"\nWrong Neighbor (same/different secondary):")
        for e in ex["wrong_neighbor"][:10]:
            print(f"  {e['pos']}: orig={e['orig_neigh']} ({e['orig_sec']}), pred={e['pred_neigh']} ({e['pred_sec']})")
    
    if ex["wrong_direction"]:
        print(f"\nWrong Direction:")
        for e in ex["wrong_direction"][:10]:
            print(f"  {e['pos']}: orig={e['orig_dir']}, pred={e['pred_dir']}")
    
    print("\n" + "="*80)


def main():
    ap = argparse.ArgumentParser(description="Compare predicted vs original blends in detail")
    ap.add_argument("--original", required=True, help="Path to original map")
    ap.add_argument("--predicted", required=True, help="Path to predicted map")
    ap.add_argument("--out", help="Output JSON path for full comparison data")
    args = ap.parse_args()
    
    print(f"Loading original: {args.original}")
    orig_data = extract_blend_data(Path(args.original))
    print(f"  -> {len(orig_data['cells'])} cells with blends")
    
    print(f"Loading predicted: {args.predicted}")
    pred_data = extract_blend_data(Path(args.predicted))
    print(f"  -> {len(pred_data['cells'])} cells with blends")
    
    print("Comparing...")
    stats = compare_blends(orig_data, pred_data)
    
    print_report(stats)
    
    if args.out:
        # Convert tuple keys to strings for JSON
        stats_json = stats.copy()
        for key in ["false_positives", "false_negatives", "wrong_secondary", "wrong_direction", "wrong_neighbor"]:
            for item in stats_json["examples"][key]:
                item["pos"] = list(item["pos"])
        
        Path(args.out).write_text(json.dumps(stats_json, indent=2))
        print(f"\nSaved full stats to: {args.out}")


if __name__ == "__main__":
    main()




