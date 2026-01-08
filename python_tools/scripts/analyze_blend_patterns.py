"""
Deep analysis of blend patterns in the prepared dataset.
Goal: understand what actually determines blend_present=1 vs 0.
"""
from __future__ import annotations
import numpy as np
import json
from pathlib import Path
from collections import Counter
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, required=True)
    parser.add_argument("--sample-size", type=int, default=50000)
    args = parser.parse_args()

    data_dir = Path(args.data_dir)

    # Load data
    tex = np.load(data_dir / "tex.npy", mmap_mode="r")  # [N,25] global tex IDs
    tex_local = np.load(data_dir / "tex_local_norm.npy", mmap_mode="r")  # [N,25] normalized local indices
    elev = np.load(data_dir / "elev.npy", mmap_mode="r")  # [N,25]
    bp = np.load(data_dir / "y_blend_present.npy", mmap_mode="r")  # [N]
    bm = np.load(data_dir / "y_blend_mask.npy", mmap_mode="r")  # [N] uint8
    bd = np.load(data_dir / "y_blend_dir.npy", mmap_mode="r")  # [N]
    sp = np.load(data_dir / "y_se_present.npy", mmap_mode="r")  # [N]
    sm = np.load(data_dir / "y_se_mask.npy", mmap_mode="r")  # [N]

    with open(data_dir / "prepared_meta.json") as f:
        meta = json.load(f)

    N = len(tex)
    print(f"Dataset: {N:,} samples")
    print(f"Blend positive rate: {bp.sum()/N:.4f} ({bp.sum():,} samples)")
    print(f"SE positive rate: {sp.sum()/N:.4f} ({sp.sum():,} samples)")

    # 5x5 window layout (row-major):
    # 0  1  2  3  4
    # 5  6  7  8  9
    # 10 11 12 13 14
    # 15 16 17 18 19
    # 20 21 22 23 24
    # Center = 12
    # 8 neighbors of center: TL=6, T=7, TR=8, L=11, R=13, BL=16, B=17, BR=18
    NEIGHBOR_IDXS = [6, 7, 8, 11, 13, 16, 17, 18]
    NEIGHBOR_NAMES = ["TL", "T", "TR", "L", "R", "BL", "B", "BR"]
    CENTER_IDX = 12

    print("\n" + "="*60)
    print("ANALYSIS 1: What texture patterns trigger blend_present=1?")
    print("="*60)

    # Sample analysis
    sample_size = min(args.sample_size, N)
    rng = np.random.default_rng(42)
    sample_idx = rng.choice(N, sample_size, replace=False)

    tex_s = np.array(tex[sample_idx])
    tex_local_s = np.array(tex_local[sample_idx])
    bp_s = np.array(bp[sample_idx])
    bm_s = np.array(bm[sample_idx])
    elev_s = np.array(elev[sample_idx])

    center_tex = tex_s[:, CENTER_IDX]
    center_local = tex_local_s[:, CENTER_IDX]
    neighbor_tex = tex_s[:, NEIGHBOR_IDXS]  # [S,8]
    neighbor_local = tex_local_s[:, NEIGHBOR_IDXS]  # [S,8]

    # Q1: Is blend triggered when center differs from any neighbor?
    has_different_neighbor = np.any(neighbor_tex != center_tex[:, None], axis=1)
    print(f"\nQ1: Does 'center differs from any neighbor' predict blend?")
    print(f"  Samples with different neighbor: {has_different_neighbor.sum():,} / {sample_size:,}")
    diff_blend_rate = bp_s[has_different_neighbor].mean() if has_different_neighbor.sum() > 0 else 0
    print(f"  Of those, blend_present=1: {bp_s[has_different_neighbor].sum():,} ({diff_blend_rate:.4f})")
    same_mask = ~has_different_neighbor
    same_blend_rate = bp_s[same_mask].mean() if same_mask.sum() > 0 else 0
    print(f"  Samples with ALL same neighbors: {same_mask.sum():,}")
    print(f"  Of those, blend_present=1: {bp_s[same_mask].sum():,} ({same_blend_rate:.4f})")

    # Q2: For blend=1 cases, what's the mask pattern?
    blend_pos_mask = bp_s == 1
    print(f"\nQ2: For blend_present=1 samples, analyze the mask:")
    print(f"  Total blend_present=1: {blend_pos_mask.sum():,}")
    masks_present = bm_s[blend_pos_mask]
    mask_counts = Counter(masks_present.tolist())
    print(f"  Unique mask values: {len(mask_counts)}")
    print(f"  Top 10 mask patterns:")
    for mv, cnt in mask_counts.most_common(10):
        bits = format(mv, '08b')[::-1]  # reverse to get bit 0 first
        neigh_str = ",".join(NEIGHBOR_NAMES[i] for i in range(8) if (mv >> i) & 1)
        print(f"    mask={mv:3d} ({bits}) => [{neigh_str}]: {cnt:,} ({cnt/blend_pos_mask.sum()*100:.1f}%)")

    # Q3: When blend=1, what's the relationship between center and secondary (mask) textures?
    print(f"\nQ3: For blend_present=1, is center_local < secondary_local?")
    center_lt_sec = 0
    center_gt_sec = 0
    center_eq_sec = 0
    blend_indices = np.where(blend_pos_mask)[0]
    for idx in blend_indices:
        mask = bm_s[idx]
        if mask == 0 or mask == 255:
            continue
        c_local = center_local[idx]
        # Find which neighbors are in the mask
        sec_locals = []
        for bit in range(8):
            if (mask >> bit) & 1:
                sec_locals.append(neighbor_local[idx, bit])
        if len(sec_locals) == 0:
            continue
        sec_local = sec_locals[0]  # first secondary
        if c_local < sec_local:
            center_lt_sec += 1
        elif c_local > sec_local:
            center_gt_sec += 1
        else:
            center_eq_sec += 1

    total = center_lt_sec + center_gt_sec + center_eq_sec
    if total > 0:
        print(f"  center_local < secondary_local: {center_lt_sec:,} ({center_lt_sec/total*100:.1f}%)")
        print(f"  center_local > secondary_local: {center_gt_sec:,} ({center_gt_sec/total*100:.1f}%)")
        print(f"  center_local = secondary_local: {center_eq_sec:,} ({center_eq_sec/total*100:.1f}%)")

    # Q4: For blend=0 cases with different neighbors, why no blend?
    print(f"\nQ4: For blend_present=0 but has different neighbors, analyze:")
    no_blend_diff = (~bp_s.astype(bool)) & has_different_neighbor
    print(f"  Total: {no_blend_diff.sum():,}")

    # Check if all different neighbors have lower local index than center
    if no_blend_diff.sum() > 0:
        center_highest_count = 0
        center_not_highest_count = 0
        no_blend_indices = np.where(no_blend_diff)[0][:5000]  # sample
        for idx in no_blend_indices:
            c_local = center_local[idx]
            n_locals = neighbor_local[idx]
            c_tex = center_tex[idx]
            n_tex = neighbor_tex[idx]
            # Different neighbors only
            diff_mask = n_tex != c_tex
            if not diff_mask.any():
                continue
            diff_n_locals = n_locals[diff_mask]
            if np.all(diff_n_locals < c_local):
                center_highest_count += 1
            else:
                center_not_highest_count += 1
        print(f"  Center has highest local idx among all diff neighbors: {center_highest_count:,}")
        print(f"  Center does NOT have highest local idx: {center_not_highest_count:,}")
        if center_highest_count + center_not_highest_count > 0:
            pct = center_highest_count / (center_highest_count + center_not_highest_count) * 100
            print(f"  => {pct:.1f}% of no-blend cases have center as highest local idx")

    print("\n" + "="*60)
    print("ANALYSIS 2: Elevation patterns")
    print("="*60)

    center_elev = elev_s[:, CENTER_IDX]
    neighbor_elev = elev_s[:, NEIGHBOR_IDXS]

    # Does elevation difference correlate with blend?
    elev_diff_max = np.max(np.abs(neighbor_elev - center_elev[:, None]), axis=1)
    blend1_elev = elev_diff_max[bp_s==1]
    blend0_elev = elev_diff_max[bp_s==0]
    print(f"\nMax elevation diff from center to any neighbor:")
    if len(blend1_elev) > 0:
        print(f"  Blend=1 mean: {blend1_elev.mean():.4f}, std: {blend1_elev.std():.4f}")
    if len(blend0_elev) > 0:
        print(f"  Blend=0 mean: {blend0_elev.mean():.4f}, std: {blend0_elev.std():.4f}")

    print("\n" + "="*60)
    print("ANALYSIS 3: Check specific failure modes")
    print("="*60)

    # Look at cases where model would predict blend (different neighbors, center < some neighbor)
    # but ground truth says no blend
    potential_blend = np.zeros(sample_size, dtype=bool)
    for i in range(sample_size):
        c_local = center_local[i]
        c_tex = center_tex[i]
        for j in range(8):
            if neighbor_tex[i, j] != c_tex and neighbor_local[i, j] > c_local:
                potential_blend[i] = True
                break

    print(f"\nPotential blend (center < some different neighbor): {potential_blend.sum():,}")
    pot_blend_actual = (potential_blend & (bp_s==1)).sum()
    pot_blend_not = (potential_blend & (bp_s==0)).sum()
    if potential_blend.sum() > 0:
        print(f"  Actually blend=1: {pot_blend_actual:,} ({pot_blend_actual/potential_blend.sum()*100:.1f}%)")
        print(f"  Actually blend=0: {pot_blend_not:,} ({pot_blend_not/potential_blend.sum()*100:.1f}%)")

    # The key question: what distinguishes potential_blend cases that ARE vs AREN'T blended?
    is_blend = potential_blend & (bp_s == 1)
    not_blend = potential_blend & (bp_s == 0)

    if is_blend.sum() > 0 and not_blend.sum() > 0:
        print(f"\n  Comparing IS_BLEND vs NOT_BLEND within potential_blend:")
        print(f"    IS_BLEND center_local mean: {center_local[is_blend].mean():.4f}")
        print(f"    NOT_BLEND center_local mean: {center_local[not_blend].mean():.4f}")
        
        # Count unique neighbor patterns
        def get_pattern(idx):
            c = center_tex[idx]
            n = neighbor_tex[idx]
            return tuple(1 if n[j] != c else 0 for j in range(8))
        
        is_blend_indices = np.where(is_blend)[0][:3000]
        not_blend_indices = np.where(not_blend)[0][:3000]
        
        is_blend_patterns = Counter(get_pattern(i) for i in is_blend_indices)
        not_blend_patterns = Counter(get_pattern(i) for i in not_blend_indices)
        
        print(f"\n    Top patterns for IS_BLEND:")
        for pat, cnt in is_blend_patterns.most_common(5):
            neigh_str = ",".join(NEIGHBOR_NAMES[j] for j in range(8) if pat[j])
            print(f"      [{neigh_str}]: {cnt}")
        
        print(f"\n    Top patterns for NOT_BLEND:")
        for pat, cnt in not_blend_patterns.most_common(5):
            neigh_str = ",".join(NEIGHBOR_NAMES[j] for j in range(8) if pat[j])
            print(f"      [{neigh_str}]: {cnt}")

    print("\n" + "="*60)
    print("ANALYSIS 4: Check mask vs actual neighbor textures")
    print("="*60)

    # For blend=1, does the mask actually point to neighbors that differ from center?
    mask_matches_diff = 0
    mask_mismatches = 0
    for idx in blend_indices[:3000]:
        mask = bm_s[idx]
        if mask == 0 or mask == 255:
            continue
        c_tex = center_tex[idx]
        for bit in range(8):
            if (mask >> bit) & 1:
                n_tex = neighbor_tex[idx, bit]
                if n_tex != c_tex:
                    mask_matches_diff += 1
                else:
                    mask_mismatches += 1
    print(f"\nDo mask bits point to neighbors different from center?")
    print(f"  Mask bit -> different neighbor: {mask_matches_diff:,}")
    print(f"  Mask bit -> SAME neighbor (bug?): {mask_mismatches:,}")
    if mask_matches_diff + mask_mismatches > 0:
        print(f"  Match rate: {mask_matches_diff/(mask_matches_diff+mask_mismatches)*100:.1f}%")

    print("\n" + "="*60)
    print("ANALYSIS 5: What makes potential_blend=True but blend_present=0?")
    print("="*60)

    # These are the FALSE POSITIVES the model would make
    # Let's look at specific examples
    not_blend_indices = np.where(not_blend)[0][:20]
    print(f"\nSample of potential_blend=True but blend_present=0:")
    for idx in not_blend_indices[:10]:
        c_tex_val = center_tex[idx]
        c_local_val = center_local[idx]
        print(f"\n  Sample {idx}:")
        print(f"    Center tex={c_tex_val}, local={c_local_val:.4f}")
        for j in range(8):
            n_tex_val = neighbor_tex[idx, j]
            n_local_val = neighbor_local[idx, j]
            diff_str = " <-- DIFF" if n_tex_val != c_tex_val else ""
            higher_str = " (higher local)" if n_local_val > c_local_val else ""
            print(f"    {NEIGHBOR_NAMES[j]:3s}: tex={n_tex_val}, local={n_local_val:.4f}{diff_str}{higher_str}")

    # Check if there's a pattern with the direction of the texture change
    print("\n" + "="*60)
    print("ANALYSIS 6: Texture ID relationships")
    print("="*60)

    # For potential_blend cases, check if secondary texture is always higher global ID
    pot_blend_indices = np.where(potential_blend)[0]
    sec_higher_global = 0
    sec_lower_global = 0
    for idx in pot_blend_indices[:5000]:
        c_tex_val = center_tex[idx]
        c_local_val = center_local[idx]
        is_actually_blend = bp_s[idx] == 1
        for j in range(8):
            n_tex_val = neighbor_tex[idx, j]
            n_local_val = neighbor_local[idx, j]
            if n_tex_val != c_tex_val and n_local_val > c_local_val:
                if n_tex_val > c_tex_val:
                    sec_higher_global += 1
                else:
                    sec_lower_global += 1
                break
    print(f"\nFor potential_blend, is secondary (higher local) also higher global ID?")
    print(f"  Secondary has higher global ID: {sec_higher_global:,}")
    print(f"  Secondary has lower global ID: {sec_lower_global:,}")


if __name__ == "__main__":
    main()




