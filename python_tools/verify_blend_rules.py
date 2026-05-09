"""
Verify whether the 12 deterministic blend rules from MapGenerator.cs
are actually deterministic across real RA3 maps.

The 12 rules (priority order - first match wins):
1. left == top && top != center -> BottomRight (0x28)
2. right == top && top != center -> BottomLeft (0x24)
3. right == bottom && bottom != center -> TopLeft (0x34)
4. left == bottom && bottom != center -> TopRight (0x38)
5. left != center -> Right (0x11)
6. right != center -> Left (0x01)
7. top != center -> Bottom (0x02)
8. bottom != center -> Top (0x12)
9. topLeft != center -> ExceptTopLeft (0x08)
10. topRight != center -> ExceptTopRight (0x04)
11. bottomRight != center -> ExceptBottomRight (0x14)
12. bottomLeft != center -> ExceptBottomLeft (0x18)

PLUS: blend only created if centerTexture <= secondaryTexture (palette priority)
"""

import sys
import os
from collections import defaultdict

sys.path.insert(0, r"E:\DL\Projects\Ra3 texture gen\RA 3 maps\python_tools")

from map_processor.core.ra3map import Ra3Map
from map_processor.assets.terrain.blend_tile_data import BlendTileData
from map_processor.assets.terrain.blend_direction import BlendDirection


# Map files to test - diverse selection from different directories
MAP_FILES = [
    # 2-player maps
    r"E:\DL\Projects\Ra3 texture gen\RA 3 maps\RA3 Official maps\2 CE\map_mp_2_feasel7.map",
    r"E:\DL\Projects\Ra3 texture gen\RA 3 maps\RA3 Official maps\2 COC\map_mp_2_feasel2.map",
    r"E:\DL\Projects\Ra3 texture gen\RA 3 maps\RA3 Official maps\2 CR\map_mp_2_feasel1.map",
    r"E:\DL\Projects\Ra3 texture gen\RA 3 maps\RA3 Official maps\2 FI\map_mp_2_feasel8.map",
    r"E:\DL\Projects\Ra3 texture gen\RA 3 maps\RA3 Official maps\2 II\map_mp_2_rao1_original.map",
    r"E:\DL\Projects\Ra3 texture gen\RA 3 maps\RA3 Official maps\2 IS\map_mp_2_feasel6.map",
    r"E:\DL\Projects\Ra3 texture gen\RA 3 maps\RA3 Official maps\2 SS\map_mp_2_feasel3.map",
    r"E:\DL\Projects\Ra3 texture gen\RA 3 maps\RA3 Official maps\2 TP\map_mp_2_black1b.map",
    # 3-player
    r"E:\DL\Projects\Ra3 texture gen\RA 3 maps\RA3 Official maps\3 Caledra of Chaos\map_mp_3_feasel2.map",
    r"E:\DL\Projects\Ra3 texture gen\RA 3 maps\RA3 Official maps\3 HF\map_mp_3_feasel3.map",
    # 4-player maps
    r"E:\DL\Projects\Ra3 texture gen\RA 3 maps\RA3 Official maps\4 Death Aquatic\map_mp_4_ssmith2-remix.map",
    r"E:\DL\Projects\Ra3 texture gen\RA 3 maps\RA3 Official maps\4 Pool Party\map_mp_4_feasel6.map",
    r"E:\DL\Projects\Ra3 texture gen\RA 3 maps\RA3 Official maps\4 Rock Ridge\map_mp_4_feasel2.map",
    r"E:\DL\Projects\Ra3 texture gen\RA 3 maps\RA3 Official maps\4 Ring of Fire\map_mp_4_feasel7.map",
    # 5/6-player maps
    r"E:\DL\Projects\Ra3 texture gen\RA 3 maps\RA3 Official maps\5 Circus Maximus\map_mp_5_feasel3.map",
    r"E:\DL\Projects\Ra3 texture gen\RA 3 maps\RA3 Official maps\6 Burnt out Paradise\map_mp_6_feasel1.map",
    r"E:\DL\Projects\Ra3 texture gen\RA 3 maps\RA3 Official maps\6 Magmageddon\map_mp_6_feasel4.map",
    # Snow maps
    r"E:\DL\Projects\Ra3 texture gen\RA 3 maps\RA3 Official maps\Snow\4 Cold Showdown\map_mp_4_stewart_1.map",
    r"E:\DL\Projects\Ra3 texture gen\RA 3 maps\RA3 Official maps\Snow\6 Sub Zero Hour\map_mp_6_feasel3.map",
]


def decode_texture(tile_value, x, y):
    """Decode texture index from tile value at position (x, y)."""
    position_offset = (y % 8 // 2) * 16 + (y % 2) * 2 + (x % 8 // 2) * 4 + (x % 2)
    return (tile_value - position_offset) // 64


def apply_rules(center, left, right, top, bottom, top_left, top_right, bottom_left, bottom_right):
    """
    Apply the 12 MapGenerator rules in priority order.
    Returns (direction, secondary_texture) or (None, None) if no rule matches.

    NOTE: The C# code uses `tex` as both secondary texture tracker and condition checker.
    Each rule sets `tex` to the texture it checks against.
    """
    # Rule 1: left == top && top != center -> BottomRight
    if left == top and top != center:
        return BlendDirection.BottomRight, top
    # Rule 2: right == top && top != center -> BottomLeft
    if right == top and top != center:
        return BlendDirection.BottomLeft, top
    # Rule 3: right == bottom && bottom != center -> TopLeft
    if right == bottom and bottom != center:
        return BlendDirection.TopLeft, bottom
    # Rule 4: left == bottom && bottom != center -> TopRight
    if left == bottom and bottom != center:
        return BlendDirection.TopRight, bottom
    # Rule 5: left != center -> Right
    if left != center:
        return BlendDirection.Right, left
    # Rule 6: right != center -> Left
    if right != center:
        return BlendDirection.Left, right
    # Rule 7: top != center -> Bottom
    if top != center:
        return BlendDirection.Bottom, top
    # Rule 8: bottom != center -> Top
    if bottom != center:
        return BlendDirection.Top, bottom
    # Rule 9: topLeft != center -> ExceptTopLeft
    if top_left != center:
        return BlendDirection.ExceptTopLeft, top_left
    # Rule 10: topRight != center -> ExceptTopRight
    if top_right != center:
        return BlendDirection.ExceptTopRight, top_right
    # Rule 11: bottomRight != center -> ExceptBottomRight
    if bottom_right != center:
        return BlendDirection.ExceptBottomRight, bottom_right
    # Rule 12: bottomLeft != center -> ExceptBottomLeft
    if bottom_left != center:
        return BlendDirection.ExceptBottomLeft, bottom_left

    return None, None


def blend_direction_name(bd):
    """Get human-readable name for a BlendDirection value."""
    try:
        return BlendDirection(bd).name
    except ValueError:
        return f"Unknown(0x{bd:02x})"


def analyze_map(map_path):
    """Analyze a single map file and return statistics."""
    map_name = os.path.basename(map_path)
    print(f"\n{'='*80}")
    print(f"Analyzing: {map_name}")
    print(f"  Path: {map_path}")

    try:
        ra3map = Ra3Map(map_path)
        ra3map.parse()
    except Exception as e:
        print(f"  ERROR parsing map: {e}")
        return None

    ctx = ra3map.get_context()
    btd = ctx.get_asset_by_type(BlendTileData)

    if btd is None:
        print(f"  ERROR: No BlendTileData found")
        return None

    w = btd.map_width
    h = btd.map_height
    tiles = btd.tiles
    blends = btd.blends
    blend_info_list = btd.blend_info
    textures = btd.textures

    print(f"  Map size: {w}x{h}")
    print(f"  Textures ({len(textures)}): {[t.name for t in textures]}")
    print(f"  Blend info entries: {len(blend_info_list)}")

    # Count blended cells
    blend_count = int((blends > 0).sum())
    print(f"  Cells with blends: {blend_count}")

    stats = {
        'map_name': map_name,
        'map_size': (w, h),
        'num_textures': len(textures),
        'total_blend_cells': blend_count,
        'total_no_blend_cells': 0,
        # For cells WITH blends
        'direction_match': 0,
        'direction_mismatch': 0,
        'secondary_match': 0,
        'secondary_mismatch': 0,
        'palette_rule_holds': 0,
        'palette_rule_violated': 0,
        'rules_predict_no_blend_but_has_blend': 0,
        # For cells WITHOUT blends
        'no_blend_correct': 0,  # rules also predict no blend
        'no_blend_false_positive': 0,  # rules predict blend but none exists
        'no_blend_suppressed_by_palette': 0,  # rules predict blend but palette suppresses
        # Direction-level stats
        'direction_confusion': defaultdict(lambda: defaultdict(int)),
        # Mismatch examples
        'mismatch_examples': [],
        'false_positive_examples': [],
        'palette_violation_examples': [],
        # Multi-blend investigation
        'multi_neighbor_diff': 0,
    }

    for y in range(1, h - 1):
        for x in range(1, w - 1):
            center = decode_texture(int(tiles[x, y]), x, y)
            left = decode_texture(int(tiles[x - 1, y]), x - 1, y)
            right = decode_texture(int(tiles[x + 1, y]), x + 1, y)
            top = decode_texture(int(tiles[x, y + 1]), x, y + 1)
            bottom = decode_texture(int(tiles[x, y - 1]), x, y - 1)
            top_left = decode_texture(int(tiles[x - 1, y + 1]), x - 1, y + 1)
            top_right = decode_texture(int(tiles[x + 1, y + 1]), x + 1, y + 1)
            bottom_left = decode_texture(int(tiles[x - 1, y - 1]), x - 1, y - 1)
            bottom_right = decode_texture(int(tiles[x + 1, y - 1]), x + 1, y - 1)

            pred_dir, pred_tex = apply_rules(
                center, left, right, top, bottom,
                top_left, top_right, bottom_left, bottom_right
            )

            blend_idx = int(blends[x, y])

            if blend_idx > 0:
                # Cell HAS a blend
                bi = blend_info_list[blend_idx - 1]
                actual_dir = int(bi.blend_direction)
                actual_secondary_tile = bi.secondary_texture_tile
                actual_secondary_tex = decode_texture(actual_secondary_tile, x, y)

                # Check palette priority rule
                if center <= actual_secondary_tex:
                    stats['palette_rule_holds'] += 1
                else:
                    stats['palette_rule_violated'] += 1
                    if len(stats['palette_violation_examples']) < 5:
                        stats['palette_violation_examples'].append({
                            'pos': (x, y),
                            'center_tex': center,
                            'secondary_tex': actual_secondary_tex,
                            'center_name': textures[center].name if center < len(textures) else '?',
                            'secondary_name': textures[actual_secondary_tex].name if actual_secondary_tex < len(textures) else '?',
                            'direction': blend_direction_name(actual_dir),
                        })

                if pred_dir is None:
                    # Rules predict no blend, but one exists
                    stats['rules_predict_no_blend_but_has_blend'] += 1
                    if len(stats['mismatch_examples']) < 5:
                        stats['mismatch_examples'].append({
                            'pos': (x, y),
                            'type': 'rules_predict_none',
                            'actual_dir': blend_direction_name(actual_dir),
                            'actual_secondary': actual_secondary_tex,
                            'center': center,
                            'neighbors': {
                                'left': left, 'right': right, 'top': top, 'bottom': bottom,
                                'tl': top_left, 'tr': top_right, 'bl': bottom_left, 'br': bottom_right
                            }
                        })
                else:
                    # Rules predict a blend - check direction
                    if int(pred_dir) == actual_dir:
                        stats['direction_match'] += 1
                    else:
                        stats['direction_mismatch'] += 1
                        stats['direction_confusion'][blend_direction_name(actual_dir)][blend_direction_name(int(pred_dir))] += 1
                        if len(stats['mismatch_examples']) < 10:
                            stats['mismatch_examples'].append({
                                'pos': (x, y),
                                'type': 'direction_mismatch',
                                'predicted_dir': blend_direction_name(int(pred_dir)),
                                'actual_dir': blend_direction_name(actual_dir),
                                'predicted_tex': pred_tex,
                                'actual_secondary': actual_secondary_tex,
                                'center': center,
                                'neighbors': {
                                    'left': left, 'right': right, 'top': top, 'bottom': bottom,
                                    'tl': top_left, 'tr': top_right, 'bl': bottom_left, 'br': bottom_right
                                }
                            })

                    # Check secondary texture
                    if pred_tex == actual_secondary_tex:
                        stats['secondary_match'] += 1
                    else:
                        stats['secondary_mismatch'] += 1
            else:
                # Cell has NO blend
                stats['total_no_blend_cells'] += 1
                if pred_dir is None:
                    stats['no_blend_correct'] += 1
                else:
                    # Rules predict a blend but none exists
                    # Check if it's suppressed by palette priority
                    if center > pred_tex:
                        stats['no_blend_suppressed_by_palette'] += 1
                    else:
                        stats['no_blend_false_positive'] += 1
                        if len(stats['false_positive_examples']) < 5:
                            stats['false_positive_examples'].append({
                                'pos': (x, y),
                                'predicted_dir': blend_direction_name(int(pred_dir)),
                                'predicted_tex': pred_tex,
                                'center': center,
                                'neighbors': {
                                    'left': left, 'right': right, 'top': top, 'bottom': bottom,
                                    'tl': top_left, 'tr': top_right, 'bl': bottom_left, 'br': bottom_right
                                }
                            })

    # Print per-map summary
    total_with_blend = stats['direction_match'] + stats['direction_mismatch'] + stats['rules_predict_no_blend_but_has_blend']

    print(f"\n  --- Results for {map_name} ---")
    print(f"  Cells with blends (interior only): {total_with_blend}")

    if total_with_blend > 0:
        match_pct = stats['direction_match'] / total_with_blend * 100
        print(f"    Direction match:    {stats['direction_match']:6d} ({match_pct:.2f}%)")
        print(f"    Direction mismatch: {stats['direction_mismatch']:6d} ({stats['direction_mismatch']/total_with_blend*100:.2f}%)")
        print(f"    Rules predict none: {stats['rules_predict_no_blend_but_has_blend']:6d} ({stats['rules_predict_no_blend_but_has_blend']/total_with_blend*100:.2f}%)")

        if stats['direction_match'] + stats['direction_mismatch'] > 0:
            sec_total = stats['secondary_match'] + stats['secondary_mismatch']
            print(f"    Secondary tex match:    {stats['secondary_match']:6d} / {sec_total}")
            print(f"    Secondary tex mismatch: {stats['secondary_mismatch']:6d} / {sec_total}")

        print(f"    Palette rule holds:    {stats['palette_rule_holds']:6d}")
        print(f"    Palette rule violated: {stats['palette_rule_violated']:6d}")

    no_blend_total = stats['no_blend_correct'] + stats['no_blend_false_positive'] + stats['no_blend_suppressed_by_palette']
    if no_blend_total > 0:
        print(f"\n  Cells without blends (interior): {no_blend_total}")
        print(f"    Correctly predicted no blend: {stats['no_blend_correct']:6d} ({stats['no_blend_correct']/no_blend_total*100:.2f}%)")
        print(f"    Suppressed by palette rule:   {stats['no_blend_suppressed_by_palette']:6d} ({stats['no_blend_suppressed_by_palette']/no_blend_total*100:.2f}%)")
        print(f"    False positive (unexpected):  {stats['no_blend_false_positive']:6d} ({stats['no_blend_false_positive']/no_blend_total*100:.2f}%)")

    if stats['mismatch_examples']:
        print(f"\n  Sample mismatches:")
        for ex in stats['mismatch_examples'][:5]:
            print(f"    pos={ex['pos']}, type={ex['type']}")
            if ex['type'] == 'direction_mismatch':
                print(f"      predicted={ex['predicted_dir']}, actual={ex['actual_dir']}")
                print(f"      predicted_tex={ex['predicted_tex']}, actual_tex={ex['actual_secondary']}, center={ex['center']}")
            else:
                print(f"      actual_dir={ex['actual_dir']}, actual_secondary={ex.get('actual_secondary','?')}, center={ex['center']}")
            print(f"      neighbors={ex['neighbors']}")

    if stats['palette_violation_examples']:
        print(f"\n  Palette violation examples:")
        for ex in stats['palette_violation_examples'][:3]:
            print(f"    pos={ex['pos']}: center={ex['center_name']}({ex['center_tex']}) > secondary={ex['secondary_name']}({ex['secondary_tex']}), dir={ex['direction']}")

    if stats['false_positive_examples']:
        print(f"\n  False positive examples (rules say blend, map says no blend, palette allows):")
        for ex in stats['false_positive_examples'][:3]:
            print(f"    pos={ex['pos']}: predicted_dir={ex['predicted_dir']}, predicted_tex={ex['predicted_tex']}, center={ex['center']}")
            print(f"      neighbors={ex['neighbors']}")

    if stats['direction_confusion']:
        print(f"\n  Direction confusion matrix (actual -> predicted):")
        for actual, preds in sorted(stats['direction_confusion'].items()):
            for pred, count in sorted(preds.items(), key=lambda x: -x[1]):
                print(f"    {actual:25s} -> {pred:25s}: {count}")

    return stats


def main():
    print("=" * 80)
    print("BLEND RULE VERIFICATION ACROSS REAL RA3 MAPS")
    print("=" * 80)
    print(f"\nTesting {len(MAP_FILES)} maps...")

    # Check which files exist
    existing_maps = []
    for path in MAP_FILES:
        if os.path.exists(path):
            existing_maps.append(path)
        else:
            print(f"  WARNING: Map not found: {path}")

    print(f"  Found {len(existing_maps)} / {len(MAP_FILES)} maps")

    all_stats = []
    for map_path in existing_maps:
        stats = analyze_map(map_path)
        if stats is not None:
            all_stats.append(stats)

    # Overall summary
    print("\n" + "=" * 80)
    print("OVERALL SUMMARY")
    print("=" * 80)

    total_direction_match = sum(s['direction_match'] for s in all_stats)
    total_direction_mismatch = sum(s['direction_mismatch'] for s in all_stats)
    total_predict_none = sum(s['rules_predict_no_blend_but_has_blend'] for s in all_stats)
    total_with_blend = total_direction_match + total_direction_mismatch + total_predict_none

    total_sec_match = sum(s['secondary_match'] for s in all_stats)
    total_sec_mismatch = sum(s['secondary_mismatch'] for s in all_stats)

    total_palette_holds = sum(s['palette_rule_holds'] for s in all_stats)
    total_palette_violated = sum(s['palette_rule_violated'] for s in all_stats)

    total_no_blend_correct = sum(s['no_blend_correct'] for s in all_stats)
    total_no_blend_suppressed = sum(s['no_blend_suppressed_by_palette'] for s in all_stats)
    total_no_blend_fp = sum(s['no_blend_false_positive'] for s in all_stats)
    total_no_blend = total_no_blend_correct + total_no_blend_suppressed + total_no_blend_fp

    print(f"\nMaps analyzed: {len(all_stats)}")
    print(f"\n--- CELLS WITH BLENDS ---")
    print(f"Total blend cells analyzed: {total_with_blend}")
    if total_with_blend > 0:
        print(f"  Direction match:         {total_direction_match:8d} ({total_direction_match/total_with_blend*100:.4f}%)")
        print(f"  Direction mismatch:      {total_direction_mismatch:8d} ({total_direction_mismatch/total_with_blend*100:.4f}%)")
        print(f"  Rules predict no blend:  {total_predict_none:8d} ({total_predict_none/total_with_blend*100:.4f}%)")

    print(f"\n--- SECONDARY TEXTURE ---")
    sec_total = total_sec_match + total_sec_mismatch
    if sec_total > 0:
        print(f"  Secondary match:    {total_sec_match:8d} ({total_sec_match/sec_total*100:.4f}%)")
        print(f"  Secondary mismatch: {total_sec_mismatch:8d} ({total_sec_mismatch/sec_total*100:.4f}%)")

    print(f"\n--- PALETTE PRIORITY RULE (center <= secondary) ---")
    palette_total = total_palette_holds + total_palette_violated
    if palette_total > 0:
        print(f"  Holds:    {total_palette_holds:8d} ({total_palette_holds/palette_total*100:.4f}%)")
        print(f"  Violated: {total_palette_violated:8d} ({total_palette_violated/palette_total*100:.4f}%)")

    print(f"\n--- CELLS WITHOUT BLENDS ---")
    print(f"Total no-blend cells (interior): {total_no_blend}")
    if total_no_blend > 0:
        print(f"  Correctly no blend:         {total_no_blend_correct:8d} ({total_no_blend_correct/total_no_blend*100:.4f}%)")
        print(f"  Suppressed by palette rule: {total_no_blend_suppressed:8d} ({total_no_blend_suppressed/total_no_blend*100:.4f}%)")
        print(f"  False positive:             {total_no_blend_fp:8d} ({total_no_blend_fp/total_no_blend*100:.4f}%)")

    # Overall accuracy metrics
    print(f"\n--- OVERALL ACCURACY ---")
    total_cells = total_with_blend + total_no_blend
    correctly_predicted = total_direction_match + total_no_blend_correct + total_no_blend_suppressed
    print(f"Total interior cells analyzed: {total_cells}")
    if total_cells > 0:
        print(f"Correctly predicted (direction or no-blend): {correctly_predicted} ({correctly_predicted/total_cells*100:.4f}%)")

    if total_with_blend > 0:
        # What fraction of blend cells can the rules explain?
        explainable = total_direction_match
        print(f"\nBlend cells where rules exactly predict direction: {explainable}/{total_with_blend} ({explainable/total_with_blend*100:.4f}%)")

        # Including secondary texture match
        both_match = total_sec_match  # secondary match only counted when direction also assessed
        print(f"Blend cells where rules predict direction AND secondary texture: {both_match}/{total_with_blend} ({both_match/total_with_blend*100:.4f}%)")

    # Aggregate confusion matrix
    print(f"\n--- AGGREGATE DIRECTION CONFUSION (actual -> predicted, top 20) ---")
    agg_confusion = defaultdict(lambda: defaultdict(int))
    for s in all_stats:
        for actual, preds in s['direction_confusion'].items():
            for pred, count in preds.items():
                agg_confusion[actual][pred] += count

    confusion_rows = []
    for actual, preds in agg_confusion.items():
        for pred, count in preds.items():
            confusion_rows.append((count, actual, pred))
    confusion_rows.sort(reverse=True)

    for count, actual, pred in confusion_rows[:20]:
        print(f"  {actual:25s} -> {pred:25s}: {count}")

    # Per-map summary table
    print(f"\n--- PER-MAP SUMMARY ---")
    print(f"{'Map':<40s} {'BlendCells':>10s} {'DirMatch%':>10s} {'DirMismatch':>12s} {'PredNone':>10s} {'FP':>6s} {'PalViol':>8s}")
    for s in all_stats:
        total = s['direction_match'] + s['direction_mismatch'] + s['rules_predict_no_blend_but_has_blend']
        if total > 0:
            pct = s['direction_match'] / total * 100
        else:
            pct = 0
        print(f"{s['map_name']:<40s} {total:>10d} {pct:>9.2f}% {s['direction_mismatch']:>12d} {s['rules_predict_no_blend_but_has_blend']:>10d} {s['no_blend_false_positive']:>6d} {s['palette_rule_violated']:>8d}")


if __name__ == '__main__':
    main()
