"""
Test if config-specific rules generalize across maps.
Train on one map, test on others.
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

    # Use texture NAMES as keys (not IDs, since IDs vary between maps)
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

    return config_stats, texture_names, tex_grid, blends


def evaluate_with_rules(map_path: Path, config_rules: dict, threshold: float = 0.5):
    """Evaluate using pre-computed configuration rules."""
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

    tp, fp, tn, fn = 0, 0, 0, 0
    unknown_configs = 0

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

            # Get prediction from rules
            if config in config_rules:
                stats = config_rules[config]
                rate = stats['blended'] / stats['total'] if stats['total'] > 0 else 0
                pred = rate >= threshold
            else:
                # Unknown config - use default based on same_count
                # Straight edges (same=5) rarely blend, corners usually blend
                pred = same_count <= 4
                unknown_configs += 1

            actual = blends[x, y] > 0

            if pred and actual: tp += 1
            elif pred and not actual: fp += 1
            elif not pred and actual: fn += 1
            else: tn += 1

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    acc = (tp + tn) / (tp + fp + tn + fn)

    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'accuracy': acc,
        'unknown_configs': unknown_configs,
        'total_boundary': tp + fp + tn + fn,
    }


def main():
    maps_dir = Path(r"E:\DL\Projects\Ra3 texture gen\RA 3 maps\RA3 Official maps")

    # Find all original maps
    all_maps = sorted(maps_dir.rglob("*_original.map"))
    print(f"Found {len(all_maps)} original maps")

    # Train on first map
    train_map = all_maps[0]
    print(f"\n=== Training on: {train_map.parent.name} ===")

    config_stats, _, _, _ = extract_config_stats(train_map)
    print(f"Extracted {len(config_stats)} unique configurations")

    # Self-evaluation
    print("\n--- Self-evaluation (train=test) ---")
    result = evaluate_with_rules(train_map, config_stats, threshold=0.5)
    print(f"  P={result['precision']:.3f}, R={result['recall']:.3f}, F1={result['f1']:.3f}, Acc={result['accuracy']:.3f}")

    # Test on other maps
    print("\n--- Cross-map evaluation ---")

    all_results = []
    for test_map in all_maps[:10]:  # Test on first 10 maps
        result = evaluate_with_rules(test_map, config_stats, threshold=0.5)
        result['name'] = test_map.parent.name
        all_results.append(result)
        print(f"  {test_map.parent.name}: P={result['precision']:.3f}, R={result['recall']:.3f}, "
              f"F1={result['f1']:.3f}, Acc={result['accuracy']:.3f}, "
              f"unknown={result['unknown_configs']}/{result['total_boundary']}")

    # Average performance
    print("\n--- Average performance ---")
    avg_p = np.mean([r['precision'] for r in all_results])
    avg_r = np.mean([r['recall'] for r in all_results])
    avg_f1 = np.mean([r['f1'] for r in all_results])
    avg_acc = np.mean([r['accuracy'] for r in all_results])
    print(f"  Average: P={avg_p:.3f}, R={avg_r:.3f}, F1={avg_f1:.3f}, Acc={avg_acc:.3f}")

    # Now try aggregating stats from ALL maps for training
    print("\n=== Training on ALL maps (aggregated) ===")

    aggregated_stats = defaultdict(lambda: {'blended': 0, 'total': 0})
    for train_map in all_maps:
        try:
            stats, _, _, _ = extract_config_stats(train_map)
            for config, s in stats.items():
                aggregated_stats[config]['blended'] += s['blended']
                aggregated_stats[config]['total'] += s['total']
        except Exception as e:
            print(f"  Error with {train_map.parent.name}: {e}")

    print(f"Aggregated {len(aggregated_stats)} unique configurations")

    # Evaluate with aggregated rules
    print("\n--- Cross-map evaluation with aggregated rules ---")

    all_results_agg = []
    for test_map in all_maps[:10]:
        result = evaluate_with_rules(test_map, aggregated_stats, threshold=0.5)
        result['name'] = test_map.parent.name
        all_results_agg.append(result)
        print(f"  {test_map.parent.name}: P={result['precision']:.3f}, R={result['recall']:.3f}, "
              f"F1={result['f1']:.3f}, Acc={result['accuracy']:.3f}, "
              f"unknown={result['unknown_configs']}/{result['total_boundary']}")

    print("\n--- Average with aggregated rules ---")
    avg_p = np.mean([r['precision'] for r in all_results_agg])
    avg_r = np.mean([r['recall'] for r in all_results_agg])
    avg_f1 = np.mean([r['f1'] for r in all_results_agg])
    avg_acc = np.mean([r['accuracy'] for r in all_results_agg])
    print(f"  Average: P={avg_p:.3f}, R={avg_r:.3f}, F1={avg_f1:.3f}, Acc={avg_acc:.3f}")


if __name__ == "__main__":
    main()
