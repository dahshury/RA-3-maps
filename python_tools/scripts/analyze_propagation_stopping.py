"""
Analyze what makes blends STOP propagating.
We know blends propagate when there's a blended neighbor.
What determines where they stop?
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


def analyze_stopping(map_path: Path):
    """Analyze what stops blend propagation."""
    print(f"\n{'='*80}")
    print(f"ANALYZING BLEND PROPAGATION STOPPING: {map_path.name}")
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

    # Find cells where propagation STOPPED
    # These are boundary cells that:
    # 1. Have a blended neighbor
    # 2. Are NOT blended themselves

    stopped_cells = []
    continued_cells = []

    for x in range(1, w-1):
        for y in range(1, h-1):
            center = tex_grid[x, y]

            # Check if at boundary
            at_boundary = False
            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                if tex_grid[x+dx, y+dy] != center:
                    at_boundary = True
                    break

            if not at_boundary:
                continue

            # Check if has blended neighbor
            has_blend_neighbor = False
            blend_neighbor_count = 0
            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                if blend_mask[x+dx, y+dy]:
                    has_blend_neighbor = True
                    blend_neighbor_count += 1

            if not has_blend_neighbor:
                continue  # No blended neighbor, not a propagation candidate

            # This cell COULD have been blended via propagation
            # Was it?
            if blend_mask[x, y]:
                continued_cells.append((x, y, blend_neighbor_count))
            else:
                stopped_cells.append((x, y, blend_neighbor_count))

    print(f"Cells where propagation continued: {len(continued_cells)}")
    print(f"Cells where propagation STOPPED: {len(stopped_cells)}")
    print(f"Stop rate: {100*len(stopped_cells)/(len(stopped_cells)+len(continued_cells)):.1f}%")

    # ===== Analyze what's different about stopped cells =====
    print("\n--- COMPARING STOPPED vs CONTINUED ---")

    def get_cell_features(x, y):
        center = tex_grid[x, y]
        center_name = texture_names[center] if center < len(texture_names) else ""

        # Same neighbors
        same_count = 0
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                if (dx, dy) != (0, 0) and tex_grid[x+dx, y+dy] == center:
                    same_count += 1

        # Different neighbors (cardinal)
        diff_count = 0
        diff_textures = set()
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            n = tex_grid[x+dx, y+dy]
            if n != center:
                diff_count += 1
                diff_textures.add(n)

        # Blended neighbors count
        blend_neighbor_count = 0
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            if blend_mask[x+dx, y+dy]:
                blend_neighbor_count += 1

        # Blended neighbors (8-connected)
        blend_neighbor_8 = 0
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                if (dx, dy) != (0, 0) and blend_mask[x+dx, y+dy]:
                    blend_neighbor_8 += 1

        # Elevation gradient
        gx = elevations[x+1, y] - elevations[x-1, y]
        gy = elevations[x, y+1] - elevations[x, y-1]
        grad = np.sqrt(gx**2 + gy**2)

        # Texture prefix
        prefix = center_name.split('_')[0]

        return {
            'same_count': same_count,
            'diff_count': diff_count,
            'blend_neighbor_count': blend_neighbor_count,
            'blend_neighbor_8': blend_neighbor_8,
            'gradient': grad,
            'texture_prefix': prefix,
            'texture_name': center_name,
        }

    # Analyze stopped cells
    stopped_features = [get_cell_features(x, y) for x, y, _ in stopped_cells[:5000]]
    continued_features = [get_cell_features(x, y) for x, y, _ in continued_cells[:5000]]

    # Compare same_count
    print("\nSame-texture neighbor count:")
    stopped_same = Counter(f['same_count'] for f in stopped_features)
    continued_same = Counter(f['same_count'] for f in continued_features)
    for n in range(9):
        s = stopped_same.get(n, 0)
        c = continued_same.get(n, 0)
        total = s + c
        stop_rate = 100 * s / total if total > 0 else 0
        print(f"  {n} same neighbors: stopped={s:5d}, continued={c:5d}, stop_rate={stop_rate:.1f}%")

    # Compare diff_count
    print("\nDifferent-texture neighbor count (cardinal):")
    stopped_diff = Counter(f['diff_count'] for f in stopped_features)
    continued_diff = Counter(f['diff_count'] for f in continued_features)
    for n in range(5):
        s = stopped_diff.get(n, 0)
        c = continued_diff.get(n, 0)
        total = s + c
        stop_rate = 100 * s / total if total > 0 else 0
        print(f"  {n} diff neighbors: stopped={s:5d}, continued={c:5d}, stop_rate={stop_rate:.1f}%")

    # Compare blend_neighbor_count
    print("\nBlended neighbor count (cardinal):")
    stopped_bn = Counter(f['blend_neighbor_count'] for f in stopped_features)
    continued_bn = Counter(f['blend_neighbor_count'] for f in continued_features)
    for n in range(5):
        s = stopped_bn.get(n, 0)
        c = continued_bn.get(n, 0)
        total = s + c
        stop_rate = 100 * s / total if total > 0 else 0
        print(f"  {n} blended neighbors: stopped={s:5d}, continued={c:5d}, stop_rate={stop_rate:.1f}%")

    # Compare gradient
    print("\nElevation gradient:")
    stopped_grad = np.array([f['gradient'] for f in stopped_features])
    continued_grad = np.array([f['gradient'] for f in continued_features])
    print(f"  Stopped: mean={stopped_grad.mean():.2f}, median={np.median(stopped_grad):.2f}")
    print(f"  Continued: mean={continued_grad.mean():.2f}, median={np.median(continued_grad):.2f}")

    for thresh in [0.5, 1.0, 2.0, 5.0, 10.0]:
        s_above = 100 * np.sum(stopped_grad > thresh) / len(stopped_grad)
        c_above = 100 * np.sum(continued_grad > thresh) / len(continued_grad)
        print(f"    Grad > {thresh}: stopped={s_above:.1f}%, continued={c_above:.1f}%")

    # Compare texture prefix
    print("\nTexture prefix (top 5):")
    stopped_prefix = Counter(f['texture_prefix'] for f in stopped_features)
    continued_prefix = Counter(f['texture_prefix'] for f in continued_features)
    all_prefixes = set(stopped_prefix.keys()) | set(continued_prefix.keys())
    prefix_stats = []
    for prefix in all_prefixes:
        s = stopped_prefix.get(prefix, 0)
        c = continued_prefix.get(prefix, 0)
        total = s + c
        if total > 100:
            stop_rate = 100 * s / total
            prefix_stats.append((prefix, s, c, stop_rate))

    for prefix, s, c, stop_rate in sorted(prefix_stats, key=lambda x: -x[1])[:10]:
        print(f"  {prefix}: stopped={s:5d}, continued={c:5d}, stop_rate={stop_rate:.1f}%")

    # ===== Test rule: stop based on gradient =====
    print("\n--- TESTING STOPPING RULES ---")

    # Rule: propagate only if gradient > threshold
    for thresh in [0.0, 0.5, 1.0, 2.0, 5.0]:
        tp, fp, tn, fn = 0, 0, 0, 0

        for x, y, _ in stopped_cells + continued_cells:
            gx = elevations[x+1, y] - elevations[x-1, y]
            gy = elevations[x, y+1] - elevations[x, y-1]
            grad = np.sqrt(gx**2 + gy**2)

            predict_continue = grad > thresh
            actual_continue = blend_mask[x, y]

            if predict_continue and actual_continue:
                tp += 1
            elif predict_continue and not actual_continue:
                fp += 1
            elif not predict_continue and actual_continue:
                fn += 1
            else:
                tn += 1

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        acc = (tp + tn) / (tp + fp + tn + fn)

        print(f"  Grad > {thresh}: P={precision:.3f}, R={recall:.3f}, F1={f1:.3f}, Acc={acc:.3f}")

    # ===== Test rule: stop based on blended neighbor count =====
    print("\n--- TESTING: Propagate if blended_neighbors >= threshold ---")

    for thresh in [1, 2, 3]:
        tp, fp, tn, fn = 0, 0, 0, 0

        for x, y, _ in stopped_cells + continued_cells:
            blend_neighbor_count = 0
            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                if blend_mask[x+dx, y+dy]:
                    blend_neighbor_count += 1

            predict_continue = blend_neighbor_count >= thresh
            actual_continue = blend_mask[x, y]

            if predict_continue and actual_continue:
                tp += 1
            elif predict_continue and not actual_continue:
                fp += 1
            elif not predict_continue and actual_continue:
                fn += 1
            else:
                tn += 1

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        acc = (tp + tn) / (tp + fp + tn + fn)

        print(f"  Blended neighbors >= {thresh}: P={precision:.3f}, R={recall:.3f}, F1={f1:.3f}, Acc={acc:.3f}")

    # ===== Combined: blended_neighbors + gradient =====
    print("\n--- TESTING: Propagate if (blended_neighbors >= 2) OR (grad > thresh) ---")

    for grad_thresh in [0.5, 1.0, 2.0, 5.0]:
        tp, fp, tn, fn = 0, 0, 0, 0

        for x, y, _ in stopped_cells + continued_cells:
            blend_neighbor_count = 0
            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                if blend_mask[x+dx, y+dy]:
                    blend_neighbor_count += 1

            gx = elevations[x+1, y] - elevations[x-1, y]
            gy = elevations[x, y+1] - elevations[x, y-1]
            grad = np.sqrt(gx**2 + gy**2)

            predict_continue = (blend_neighbor_count >= 2) or (grad > grad_thresh)
            actual_continue = blend_mask[x, y]

            if predict_continue and actual_continue:
                tp += 1
            elif predict_continue and not actual_continue:
                fp += 1
            elif not predict_continue and actual_continue:
                fn += 1
            else:
                tn += 1

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        acc = (tp + tn) / (tp + fp + tn + fn)

        print(f"  bn>=2 OR grad>{grad_thresh}: P={precision:.3f}, R={recall:.3f}, F1={f1:.3f}, Acc={acc:.3f}")


def main():
    map_path = Path(r"E:\DL\Projects\Ra3 texture gen\RA 3 maps\RA3 Official maps\2 II\map_mp_2_rao1_original.map")
    analyze_stopping(map_path)


if __name__ == "__main__":
    main()
