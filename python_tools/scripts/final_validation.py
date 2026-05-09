"""
Final validation: Apply rules learned from original maps to blendless maps.
"""

import sys
import numpy as np
from pathlib import Path
from collections import Counter, defaultdict

sys.path.insert(0, str(Path(__file__).parent.parent))
from map_processor import Ra3Map


def decode_tile_texture(tile_value: int, texture_cell_starts: list) -> int:
    if tile_value == 0:
        return 0
    for i in range(len(texture_cell_starts) - 1, -1, -1):
        if tile_value >= texture_cell_starts[i]:
            return i
    return 0


def extract_config_stats(map_path: Path):
    """Extract configuration statistics from a map."""
    ra3map = Ra3Map(str(map_path))
    ra3map.parse()
    ctx = ra3map.get_context()

    blend_tile = ctx.get_asset("BlendTileData")
    w, h = blend_tile.map_width, blend_tile.map_height
    texture_names = [t.name for t in blend_tile.textures]
    texture_cell_starts = [t.cell_start for t in blend_tile.textures]

    tiles = np.array(blend_tile.tiles).reshape(w, h)
    blends = np.array(blend_tile.blends).reshape(w, h)

    tex_grid = np.zeros((w, h), dtype=np.int32)
    for x in range(w):
        for y in range(h):
            tex_grid[x, y] = decode_tile_texture(tiles[x, y], texture_cell_starts)

    config_stats = defaultdict(lambda: {'blended': 0, 'total': 0})

    for x in range(1, w-1):
        for y in range(1, h-1):
            center = tex_grid[x, y]
            center_name = texture_names[center] if center < len(texture_names) else f"t{center}"

            diff_neighbors = []
            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                n = tex_grid[x+dx, y+dy]
                if n != center:
                    n_name = texture_names[n] if n < len(texture_names) else f"t{n}"
                    diff_neighbors.append(n_name)

            if not diff_neighbors:
                continue

            same_count = sum(1 for dx in [-1,0,1] for dy in [-1,0,1]
                            if (dx,dy) != (0,0) and tex_grid[x+dx, y+dy] == center)

            primary_diff = Counter(diff_neighbors).most_common(1)[0][0]
            config = (center_name, primary_diff, same_count)

            config_stats[config]['total'] += 1
            if blends[x, y] > 0:
                config_stats[config]['blended'] += 1

    total_blend = np.sum(blends > 0)
    total_cells = w * h
    blend_rate = total_blend / total_cells

    return config_stats, blend_rate


def main():
    orig_dir = Path(r"E:\DL\Projects\Ra3 texture gen\RA 3 maps\blendinfo dataset\_restored")
    blendless_dir = Path(r"E:\DL\Projects\Ra3 texture gen\RA 3 maps\blendinfo dataset\_restored\_generated\blendless")

    # Find original/blendless pairs
    orig_maps = sorted(orig_dir.glob("*.map"))
    print(f"Found {len(orig_maps)} original maps")

    pairs = []
    for orig_map in orig_maps:
        bl_name = orig_map.stem + "_blendless.map"
        bl_map = blendless_dir / bl_name
        if bl_map.exists():
            pairs.append((orig_map, bl_map))

    print(f"Found {len(pairs)} original/blendless pairs\n")

    # Aggregate config stats from all original maps
    print("Aggregating config stats from original maps...")
    aggregated_stats = defaultdict(lambda: {'blended': 0, 'total': 0})
    original_blend_rates = []

    for orig_map, _ in pairs[:50]:  # Use first 50 pairs
        try:
            stats, blend_rate = extract_config_stats(orig_map)
            original_blend_rates.append(blend_rate)
            for config, s in stats.items():
                aggregated_stats[config]['blended'] += s['blended']
                aggregated_stats[config]['total'] += s['total']
            print(f"  {orig_map.name[:40]:40s}: blend_rate={blend_rate*100:.1f}%")
        except Exception as e:
            print(f"  {orig_map.name[:40]:40s}: ERROR - {str(e)[:30]}")

    print(f"\nAggregated {len(aggregated_stats)} unique configurations")
    print(f"Average original blend rate: {np.mean(original_blend_rates)*100:.1f}%")

    # Now evaluate: use aggregated rules to predict blendless maps
    # Compare prediction to actual original blend pattern
    print("\n" + "="*80)
    print("EVALUATING RULE-BASED PREDICTION")
    print("="*80)

    results = []
    for orig_map, bl_map in pairs[:20]:  # Test on first 20 pairs
        try:
            # Get ground truth from original
            orig_stats, _ = extract_config_stats(orig_map)

            # Parse blendless map for texture grid
            ra3map = Ra3Map(str(bl_map))
            ra3map.parse()
            ctx = ra3map.get_context()

            blend_tile = ctx.get_asset("BlendTileData")
            w, h = blend_tile.map_width, blend_tile.map_height
            texture_names = [t.name for t in blend_tile.textures]
            texture_cell_starts = [t.cell_start for t in blend_tile.textures]

            tiles = np.array(blend_tile.tiles).reshape(w, h)

            tex_grid = np.zeros((w, h), dtype=np.int32)
            for x in range(w):
                for y in range(h):
                    tex_grid[x, y] = decode_tile_texture(tiles[x, y], texture_cell_starts)

            # Get ground truth blends from original
            orig_ra3map = Ra3Map(str(orig_map))
            orig_ra3map.parse()
            orig_ctx = orig_ra3map.get_context()
            orig_blends = np.array(orig_ctx.get_asset("BlendTileData").blends).reshape(w, h)
            actual_blend = orig_blends > 0

            # Predict using aggregated rules
            tp, fp, tn, fn = 0, 0, 0, 0

            for x in range(1, w-1):
                for y in range(1, h-1):
                    center = tex_grid[x, y]
                    center_name = texture_names[center] if center < len(texture_names) else f"t{center}"

                    diff_neighbors = []
                    for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        n = tex_grid[x+dx, y+dy]
                        if n != center:
                            n_name = texture_names[n] if n < len(texture_names) else f"t{n}"
                            diff_neighbors.append(n_name)

                    if not diff_neighbors:
                        continue

                    same_count = sum(1 for dx in [-1,0,1] for dy in [-1,0,1]
                                    if (dx,dy) != (0,0) and tex_grid[x+dx, y+dy] == center)

                    primary_diff = Counter(diff_neighbors).most_common(1)[0][0]
                    config = (center_name, primary_diff, same_count)

                    # Predict based on aggregated stats
                    if config in aggregated_stats:
                        s = aggregated_stats[config]
                        rate = s['blended'] / s['total'] if s['total'] > 0 else 0
                        pred = rate >= 0.5
                    else:
                        # Unknown config - use same_count heuristic
                        pred = same_count <= 4

                    actual = actual_blend[x, y]

                    if pred and actual: tp += 1
                    elif pred and not actual: fp += 1
                    elif not pred and actual: fn += 1
                    else: tn += 1

            total = tp + fp + tn + fn
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            acc = (tp + tn) / total

            results.append({
                'name': orig_map.stem,
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'accuracy': acc,
            })

            print(f"{orig_map.stem[:40]:40s}: F1={f1:.3f}, Acc={acc:.3f}, P={precision:.3f}, R={recall:.3f}")

        except Exception as e:
            print(f"{orig_map.stem[:40]:40s}: ERROR - {str(e)[:50]}")

    # Summary
    if results:
        print("\n" + "="*80)
        print("SUMMARY")
        print("="*80)

        avg_f1 = np.mean([r['f1'] for r in results])
        avg_acc = np.mean([r['accuracy'] for r in results])
        avg_p = np.mean([r['precision'] for r in results])
        avg_r = np.mean([r['recall'] for r in results])

        print(f"Average F1:        {avg_f1:.3f}")
        print(f"Average Accuracy:  {avg_acc:.3f}")
        print(f"Average Precision: {avg_p:.3f}")
        print(f"Average Recall:    {avg_r:.3f}")

        f1_scores = [r['f1'] for r in results]
        print(f"\nF1 distribution:")
        print(f"  Min:    {min(f1_scores):.3f}")
        print(f"  Median: {np.median(f1_scores):.3f}")
        print(f"  Max:    {max(f1_scores):.3f}")


if __name__ == "__main__":
    main()
