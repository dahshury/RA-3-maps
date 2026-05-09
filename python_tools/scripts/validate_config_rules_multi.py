"""
Validate config-based rules on multiple maps from the dataset.
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


def extract_and_evaluate(map_path: Path, threshold: float = 0.5):
    """Extract config stats and self-evaluate."""
    try:
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

        # Extract config stats
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

        # Evaluate
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

                if config in config_stats:
                    stats = config_stats[config]
                    rate = stats['blended'] / stats['total'] if stats['total'] > 0 else 0
                    pred = rate >= threshold
                else:
                    pred = same_count <= 4

                actual = blends[x, y] > 0

                if pred and actual: tp += 1
                elif pred and not actual: fp += 1
                elif not pred and actual: fn += 1
                else: tn += 1

        total = tp + fp + tn + fn
        if total == 0:
            return None

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        acc = (tp + tn) / total
        blend_rate = (tp + fn) / total

        return {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'accuracy': acc,
            'blend_rate': blend_rate,
            'total': total,
            'configs': len(config_stats),
        }

    except Exception as e:
        return {'error': str(e)}


def main():
    # Use restored maps
    maps_dir = Path(r"E:\DL\Projects\Ra3 texture gen\RA 3 maps\blendinfo dataset\_restored")

    all_maps = sorted(maps_dir.rglob("*.map"))[:50]  # Test on first 50 maps
    print(f"Testing on {len(all_maps)} maps...\n")

    results = []
    for i, map_path in enumerate(all_maps):
        result = extract_and_evaluate(map_path)
        if result and 'error' not in result:
            results.append(result)
            if result['blend_rate'] > 0.01:  # Only report maps with actual blends
                print(f"{i+1:3d}. {map_path.name[:40]:40s}: F1={result['f1']:.3f}, Acc={result['accuracy']:.3f}, "
                      f"P={result['precision']:.3f}, R={result['recall']:.3f}, blend_rate={result['blend_rate']:.2f}")
        elif result:
            print(f"{i+1:3d}. {map_path.name[:40]:40s}: ERROR - {result['error'][:50]}")

    # Filter to maps with meaningful blends
    meaningful = [r for r in results if r['blend_rate'] > 0.05]

    if meaningful:
        print(f"\n{'='*80}")
        print(f"SUMMARY (maps with >5% blend rate: {len(meaningful)})")
        print(f"{'='*80}")

        avg_f1 = np.mean([r['f1'] for r in meaningful])
        avg_acc = np.mean([r['accuracy'] for r in meaningful])
        avg_p = np.mean([r['precision'] for r in meaningful])
        avg_r = np.mean([r['recall'] for r in meaningful])

        print(f"Average F1:        {avg_f1:.3f}")
        print(f"Average Accuracy:  {avg_acc:.3f}")
        print(f"Average Precision: {avg_p:.3f}")
        print(f"Average Recall:    {avg_r:.3f}")

        # Distribution
        f1_scores = [r['f1'] for r in meaningful]
        print(f"\nF1 distribution:")
        print(f"  Min:    {min(f1_scores):.3f}")
        print(f"  25%:    {np.percentile(f1_scores, 25):.3f}")
        print(f"  Median: {np.percentile(f1_scores, 50):.3f}")
        print(f"  75%:    {np.percentile(f1_scores, 75):.3f}")
        print(f"  Max:    {max(f1_scores):.3f}")


if __name__ == "__main__":
    main()
