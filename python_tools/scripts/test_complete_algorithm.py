"""
Test complete blend prediction algorithm:
1. Seed identification
2. Propagation (iterative)
"""

import sys
import numpy as np
from pathlib import Path
from collections import deque

sys.path.insert(0, str(Path(__file__).parent.parent))
from map_processor import Ra3Map


def decode_tile_texture(tile_value: int, texture_cell_starts: list) -> int:
    if tile_value == 0:
        return 0
    for i in range(len(texture_cell_starts) - 1, -1, -1):
        if tile_value >= texture_cell_starts[i]:
            return i
    return 0


def test_algorithm(map_path: Path):
    """Test complete blend prediction algorithm."""
    print(f"\n{'='*80}")
    print(f"TESTING COMPLETE BLEND ALGORITHM: {map_path.name}")
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

    actual_blend = blends > 0

    # Precompute features
    print("Precomputing features...")

    # Same-texture 8-neighbor count
    same_count = np.zeros((w, h), dtype=np.int32)
    for x in range(1, w-1):
        for y in range(1, h-1):
            center = tex_grid[x, y]
            count = 0
            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    if (dx, dy) != (0, 0) and tex_grid[x+dx, y+dy] == center:
                        count += 1
            same_count[x, y] = count

    # Different-texture cardinal neighbor count
    diff_count = np.zeros((w, h), dtype=np.int32)
    for x in range(1, w-1):
        for y in range(1, h-1):
            center = tex_grid[x, y]
            count = 0
            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                if tex_grid[x+dx, y+dy] != center:
                    count += 1
            diff_count[x, y] = count

    # Is Transition texture
    is_transition = np.zeros((w, h), dtype=bool)
    for x in range(w):
        for y in range(h):
            tex_name = texture_names[tex_grid[x, y]] if tex_grid[x, y] < len(texture_names) else ""
            is_transition[x, y] = tex_name.startswith("Transition")

    # ===== Algorithm 1: Simple seed + propagation =====
    print("\n--- ALGORITHM 1: Corner seeds + propagation ---")

    def run_algorithm(seed_rule, prop_threshold=2, max_iterations=100):
        """
        Run the seed + propagation algorithm.
        seed_rule: function(x, y) -> bool
        prop_threshold: minimum blended neighbors to propagate
        """
        predicted = np.zeros((w, h), dtype=bool)

        # Step 1: Identify seeds
        seeds = []
        for x in range(1, w-1):
            for y in range(1, h-1):
                if seed_rule(x, y):
                    seeds.append((x, y))
                    predicted[x, y] = True

        print(f"  Seeds identified: {len(seeds)}")

        # Step 2: Propagate
        iterations = 0
        changed = True
        while changed and iterations < max_iterations:
            changed = False
            iterations += 1
            new_predicted = predicted.copy()

            for x in range(1, w-1):
                for y in range(1, h-1):
                    if predicted[x, y]:
                        continue  # Already blended

                    # Count blended neighbors
                    blend_neighbors = 0
                    for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        if predicted[x+dx, y+dy]:
                            blend_neighbors += 1

                    if blend_neighbors >= prop_threshold:
                        new_predicted[x, y] = True
                        changed = True

            predicted = new_predicted

        print(f"  Iterations: {iterations}")

        # Evaluate
        tp = np.sum(predicted & actual_blend)
        fp = np.sum(predicted & ~actual_blend)
        fn = np.sum(~predicted & actual_blend)
        tn = np.sum(~predicted & ~actual_blend)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        acc = (tp + tn) / (tp + fp + tn + fn)

        print(f"  TP={tp}, FP={fp}, FN={fn}, TN={tn}")
        print(f"  Precision={precision:.3f}, Recall={recall:.3f}, F1={f1:.3f}, Accuracy={acc:.3f}")

        return predicted

    # Rule 1a: Seeds are corners (diff >= 3)
    print("\nRule 1a: Seeds = diff_count >= 3")
    rule1a = lambda x, y: diff_count[x, y] >= 3
    run_algorithm(rule1a)

    # Rule 1b: Seeds are corners OR Transition uniform regions
    print("\nRule 1b: Seeds = (diff_count >= 3) OR (is_transition AND diff_count == 0)")
    rule1b = lambda x, y: diff_count[x, y] >= 3 or (is_transition[x, y] and diff_count[x, y] == 0)
    run_algorithm(rule1b)

    # Rule 1c: Seeds are same_count != 5 at boundaries
    print("\nRule 1c: Seeds = (diff_count > 0 AND same_count != 5)")
    rule1c = lambda x, y: diff_count[x, y] > 0 and same_count[x, y] != 5
    run_algorithm(rule1c)

    # Rule 1d: More aggressive - all corners (same <= 4) at boundaries
    print("\nRule 1d: Seeds = (diff_count > 0 AND same_count <= 4)")
    rule1d = lambda x, y: diff_count[x, y] > 0 and same_count[x, y] <= 4
    run_algorithm(rule1d)

    # ===== Algorithm 2: Include self-blend regions =====
    print("\n--- ALGORITHM 2: Include Transition self-blends ---")

    # Rule 2a: Seeds = corners + Transition regions
    print("\nRule 2a: Seeds = (diff >= 3) OR (is_transition)")
    rule2a = lambda x, y: diff_count[x, y] >= 3 or is_transition[x, y]
    run_algorithm(rule2a)

    # Rule 2b: Seeds = corners + partial Transition (sparse seeding)
    print("\nRule 2b: Seeds = (diff >= 3) OR (is_transition AND (x+y) % 4 == 0)")
    rule2b = lambda x, y: diff_count[x, y] >= 3 or (is_transition[x, y] and (x + y) % 4 == 0)
    run_algorithm(rule2b)

    # ===== Algorithm 3: Different propagation thresholds =====
    print("\n--- ALGORITHM 3: Vary propagation threshold ---")

    best_rule = lambda x, y: diff_count[x, y] >= 3 or (is_transition[x, y] and diff_count[x, y] == 0)

    for prop_thresh in [1, 2, 3]:
        print(f"\nPropagation threshold = {prop_thresh}")
        run_algorithm(best_rule, prop_threshold=prop_thresh)

    # ===== Algorithm 4: Consider boundary type for propagation =====
    print("\n--- ALGORITHM 4: Boundary-aware propagation ---")

    def run_boundary_aware(seed_rule, max_iterations=100):
        """Propagation depends on boundary type."""
        predicted = np.zeros((w, h), dtype=bool)

        # Seeds
        seeds = []
        for x in range(1, w-1):
            for y in range(1, h-1):
                if seed_rule(x, y):
                    seeds.append((x, y))
                    predicted[x, y] = True

        print(f"  Seeds: {len(seeds)}")

        # Propagate with boundary-aware rules
        iterations = 0
        changed = True
        while changed and iterations < max_iterations:
            changed = False
            iterations += 1
            new_predicted = predicted.copy()

            for x in range(1, w-1):
                for y in range(1, h-1):
                    if predicted[x, y]:
                        continue

                    # Count blended neighbors
                    blend_neighbors = 0
                    for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        if predicted[x+dx, y+dy]:
                            blend_neighbors += 1

                    # Rule: propagate if:
                    # - 2+ blended neighbors, OR
                    # - 1 blended neighbor AND not a straight edge (same != 5)
                    should_prop = (blend_neighbors >= 2) or \
                                 (blend_neighbors >= 1 and same_count[x, y] != 5)

                    # Only propagate along boundaries or in Transition regions
                    at_boundary = diff_count[x, y] > 0
                    in_transition = is_transition[x, y]

                    if should_prop and (at_boundary or in_transition):
                        new_predicted[x, y] = True
                        changed = True

            predicted = new_predicted

        print(f"  Iterations: {iterations}")

        # Evaluate
        tp = np.sum(predicted & actual_blend)
        fp = np.sum(predicted & ~actual_blend)
        fn = np.sum(~predicted & actual_blend)
        tn = np.sum(~predicted & ~actual_blend)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        acc = (tp + tn) / (tp + fp + tn + fn)

        print(f"  TP={tp}, FP={fp}, FN={fn}, TN={tn}")
        print(f"  Precision={precision:.3f}, Recall={recall:.3f}, F1={f1:.3f}, Accuracy={acc:.3f}")

    print("\nRule 4: Boundary-aware propagation")
    run_boundary_aware(best_rule)


def main():
    map_path = Path(r"E:\DL\Projects\Ra3 texture gen\RA 3 maps\RA3 Official maps\2 II\map_mp_2_rao1_original.map")
    test_algorithm(map_path)


if __name__ == "__main__":
    main()
