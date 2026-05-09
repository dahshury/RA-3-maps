"""
Find rules for blend SEEDS - what starts the propagation.

Propagation rule found: blend if blended_neighbors >= 2
Now find: what cells are the SEEDS (start with 0-1 blended neighbors but still blend)?
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


def find_seeds(map_path: Path):
    """Find blend seed patterns."""
    print(f"\n{'='*80}")
    print(f"FINDING BLEND SEED RULES: {map_path.name}")
    print(f"{'='*80}")

    ra3map = Ra3Map(str(map_path))
    ra3map.parse()
    ctx = ra3map.get_context()

    blend_tile = ctx.get_asset("BlendTileData")
    height_map = ctx.get_asset("HeightMapData")

    w, h = blend_tile.map_width, blend_tile.map_height
    texture_names = [t.name for t in blend_tile.textures]
    texture_cell_starts = [t.cell_start for t in blend_tile.textures]

    tiles = np.array(blend_tile.tiles).reshape(w, h)
    blends = np.array(blend_tile.blends).reshape(w, h)
    elevations = height_map.elevations

    tex_grid = np.zeros((w, h), dtype=np.int32)
    for x in range(w):
        for y in range(h):
            tex_grid[x, y] = decode_tile_texture(tiles[x, y], texture_cell_starts)

    blend_mask = blends > 0

    # Find seed cells: blended cells with 0 or 1 blended neighbors
    seeds = []
    non_seeds_blended = []  # blended with 2+ neighbors (propagated, not seeds)

    for x in range(1, w-1):
        for y in range(1, h-1):
            if not blend_mask[x, y]:
                continue

            # Count blended neighbors
            blend_neighbor_count = 0
            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                if blend_mask[x+dx, y+dy]:
                    blend_neighbor_count += 1

            if blend_neighbor_count <= 1:
                seeds.append((x, y))
            else:
                non_seeds_blended.append((x, y))

    print(f"Seed cells (blended with 0-1 blended neighbors): {len(seeds)}")
    print(f"Propagated cells (blended with 2+ blended neighbors): {len(non_seeds_blended)}")

    if len(seeds) == 0:
        print("No seeds found - all blends have 2+ blended neighbors!")
        return

    # ===== Analyze seed characteristics =====
    print("\n--- SEED CHARACTERISTICS ---")

    def get_features(x, y):
        center = tex_grid[x, y]
        center_name = texture_names[center] if center < len(texture_names) else ""

        # Different cardinal neighbors
        diff_count = 0
        diff_textures = []
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            n = tex_grid[x+dx, y+dy]
            if n != center:
                diff_count += 1
                diff_textures.append(n)

        # Same 8-neighbors
        same_count = sum(1 for dx in [-1,0,1] for dy in [-1,0,1]
                        if (dx,dy) != (0,0) and tex_grid[x+dx, y+dy] == center)

        # Gradient
        gx = elevations[x+1, y] - elevations[x-1, y]
        gy = elevations[x, y+1] - elevations[x, y-1]
        grad = np.sqrt(gx**2 + gy**2)

        # Blended 8-neighbors
        blend_8 = sum(1 for dx in [-1,0,1] for dy in [-1,0,1]
                     if (dx,dy) != (0,0) and blend_mask[x+dx, y+dy])

        return {
            'diff_count': diff_count,
            'same_count': same_count,
            'gradient': grad,
            'blend_8': blend_8,
            'center_name': center_name,
            'prefix': center_name.split('_')[0],
        }

    seed_features = [get_features(x, y) for x, y in seeds]

    # Distribution of diff_count for seeds
    print("\nSeeds by cardinal different-neighbor count:")
    diff_counts = Counter(f['diff_count'] for f in seed_features)
    for n, count in sorted(diff_counts.items()):
        print(f"  {n} different neighbors: {count} seeds ({100*count/len(seeds):.1f}%)")

    # Distribution of same_count for seeds
    print("\nSeeds by same-texture 8-neighbor count:")
    same_counts = Counter(f['same_count'] for f in seed_features)
    for n, count in sorted(same_counts.items()):
        print(f"  {n} same neighbors: {count} seeds ({100*count/len(seeds):.1f}%)")

    # Blended 8-neighbors
    print("\nSeeds by blended 8-neighbor count:")
    blend8_counts = Counter(f['blend_8'] for f in seed_features)
    for n, count in sorted(blend8_counts.items()):
        print(f"  {n} blended 8-neighbors: {count} seeds ({100*count/len(seeds):.1f}%)")

    # Gradient distribution
    print("\nSeed gradient distribution:")
    grads = np.array([f['gradient'] for f in seed_features])
    print(f"  Mean: {grads.mean():.2f}, Median: {np.median(grads):.2f}")
    for thresh in [0.5, 1.0, 2.0, 5.0, 10.0]:
        pct = 100 * np.sum(grads > thresh) / len(grads)
        print(f"  Gradient > {thresh}: {pct:.1f}%")

    # Texture prefix
    print("\nSeeds by texture prefix:")
    prefix_counts = Counter(f['prefix'] for f in seed_features)
    for prefix, count in prefix_counts.most_common(10):
        print(f"  {prefix}: {count} ({100*count/len(seeds):.1f}%)")

    # ===== Find non-seeds at boundaries for comparison =====
    print("\n--- COMPARING SEEDS vs NON-SEEDS AT BOUNDARIES ---")

    non_seed_boundary = []
    for x in range(1, w-1):
        for y in range(1, h-1):
            if blend_mask[x, y]:
                continue  # Skip blended cells

            center = tex_grid[x, y]
            at_boundary = any(tex_grid[x+dx, y+dy] != center
                            for dx, dy in [(-1,0), (1,0), (0,-1), (0,1)])
            if at_boundary:
                # Has 0-1 blended cardinal neighbors (same as seed criteria)
                bn = sum(1 for dx, dy in [(-1,0), (1,0), (0,-1), (0,1)]
                        if blend_mask[x+dx, y+dy])
                if bn <= 1:
                    non_seed_boundary.append((x, y))

    print(f"Non-seed boundary cells (0-1 blended neighbors, not blended): {len(non_seed_boundary)}")

    non_seed_features = [get_features(x, y) for x, y in non_seed_boundary[:5000]]

    # Compare diff_count
    print("\nDifferent-neighbor count comparison:")
    non_seed_diff = Counter(f['diff_count'] for f in non_seed_features)
    for n in range(5):
        s = diff_counts.get(n, 0)
        ns = non_seed_diff.get(n, 0)
        total = s + ns
        seed_rate = 100 * s / total if total > 0 else 0
        print(f"  {n} diff neighbors: seeds={s}, non-seeds={ns}, seed_rate={seed_rate:.1f}%")

    # Compare same_count
    print("\nSame-neighbor count comparison:")
    non_seed_same = Counter(f['same_count'] for f in non_seed_features)
    for n in range(9):
        s = same_counts.get(n, 0)
        ns = non_seed_same.get(n, 0)
        total = s + ns
        seed_rate = 100 * s / total if total > 0 else 0
        if total > 0:
            print(f"  {n} same neighbors: seeds={s}, non-seeds={ns}, seed_rate={seed_rate:.1f}%")

    # Compare gradient
    print("\nGradient comparison:")
    non_seed_grads = np.array([f['gradient'] for f in non_seed_features])
    print(f"  Seeds: mean={grads.mean():.2f}, median={np.median(grads):.2f}")
    print(f"  Non-seeds: mean={non_seed_grads.mean():.2f}, median={np.median(non_seed_grads):.2f}")

    # ===== Test seed rules =====
    print("\n--- TESTING SEED IDENTIFICATION RULES ---")

    # Combine seeds and non-seed boundaries for testing
    all_candidates = [(x, y, True) for x, y in seeds] + [(x, y, False) for x, y in non_seed_boundary[:len(seeds)*2]]

    # Rule: seed if diff_count >= threshold
    for thresh in [2, 3, 4]:
        tp, fp, tn, fn = 0, 0, 0, 0
        for x, y, is_seed in all_candidates:
            f = get_features(x, y)
            predict_seed = f['diff_count'] >= thresh

            if predict_seed and is_seed:
                tp += 1
            elif predict_seed and not is_seed:
                fp += 1
            elif not predict_seed and is_seed:
                fn += 1
            else:
                tn += 1

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        print(f"  diff_count >= {thresh}: P={precision:.3f}, R={recall:.3f}, F1={f1:.3f}")

    # Rule: seed if same_count <= threshold
    for thresh in [3, 4, 5]:
        tp, fp, tn, fn = 0, 0, 0, 0
        for x, y, is_seed in all_candidates:
            f = get_features(x, y)
            predict_seed = f['same_count'] <= thresh

            if predict_seed and is_seed:
                tp += 1
            elif predict_seed and not is_seed:
                fp += 1
            elif not predict_seed and is_seed:
                fn += 1
            else:
                tn += 1

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        print(f"  same_count <= {thresh}: P={precision:.3f}, R={recall:.3f}, F1={f1:.3f}")

    # Combined rule
    print("\n  Combined rules:")
    for diff_thresh in [2, 3]:
        for same_thresh in [3, 4, 5]:
            tp, fp, tn, fn = 0, 0, 0, 0
            for x, y, is_seed in all_candidates:
                f = get_features(x, y)
                predict_seed = f['diff_count'] >= diff_thresh and f['same_count'] <= same_thresh

                if predict_seed and is_seed:
                    tp += 1
                elif predict_seed and not is_seed:
                    fp += 1
                elif not predict_seed and is_seed:
                    fn += 1
                else:
                    tn += 1

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

            print(f"    diff>={diff_thresh} AND same<={same_thresh}: P={precision:.3f}, R={recall:.3f}, F1={f1:.3f}")


def main():
    map_path = Path(r"E:\DL\Projects\Ra3 texture gen\RA 3 maps\RA3 Official maps\2 II\map_mp_2_rao1_original.map")
    find_seeds(map_path)


if __name__ == "__main__":
    main()
