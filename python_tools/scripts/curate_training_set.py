#!/usr/bin/env python3
"""Curate the training set for v3 (SPADE U-Net + FFL + LPIPS).

Three filters applied in order:
  (1) MinHash near-duplicate detection on per-map texture histograms
      -> drop duplicates, keep the official version when conflict
  (2) Cluster-centroid distance filter
      -> drop maps whose texture-histogram is >threshold from cluster centroid
         (flags compstomp/joke maps that landed in a competitive cluster)
  (3) Heuristic quality score
      -> drop maps with too few unique textures used (likely template-spam)
      -> drop maps with valid_frac < 0.7 (mostly-custom-texture community maps)

Output: training_outputs/texture_transfer/curated_index.json with the same
schema as index.json but with reduced records list. Also writes
curation_report.json with per-map decisions for transparency.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List

import numpy as np


def _python_tools_root() -> Path:
    return Path(__file__).resolve().parents[1]


def shingle_hash(texture_counts: Dict[str, int], num_perm: int = 64) -> List[int]:
    """Cheap MinHash sketch: top-N texture names normalized by usage rank."""
    sorted_names = [n for n, _ in sorted(texture_counts.items(), key=lambda kv: -kv[1])]
    shingles = sorted_names[:32]
    sketch = []
    for seed in range(num_perm):
        h_min = None
        for s in shingles:
            h = int(hashlib.md5(f"{seed}:{s}".encode()).hexdigest()[:8], 16)
            if h_min is None or h < h_min:
                h_min = h
        sketch.append(h_min if h_min is not None else 0)
    return sketch


def jaccard_estimate(a: List[int], b: List[int]) -> float:
    if not a or not b:
        return 0.0
    matches = sum(1 for x, y in zip(a, b) if x == y)
    return matches / len(a)


def histogram_l1_dist(a: Dict[str, int], b: Dict[str, int]) -> float:
    """L1 distance between normalized texture histograms."""
    ta = sum(a.values()) or 1
    tb = sum(b.values()) or 1
    keys = set(a.keys()) | set(b.keys())
    return sum(abs(a.get(k, 0) / ta - b.get(k, 0) / tb) for k in keys)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data_dir", type=Path,
                    default=_python_tools_root() / "training_outputs" / "texture_transfer")
    ap.add_argument("--browser_dir", type=Path,
                    default=_python_tools_root() / "style_clusters" / "browser")
    ap.add_argument("--minhash_threshold", type=float, default=0.85,
                    help="Jaccard threshold for declaring near-duplicate.")
    ap.add_argument("--centroid_z", type=float, default=2.0,
                    help="Drop maps whose distance from cluster centroid > z*std.")
    ap.add_argument("--min_textures", type=int, default=10,
                    help="Drop maps using fewer than this many unique textures.")
    ap.add_argument("--min_valid_frac", type=float, default=0.7)
    args = ap.parse_args()

    index_path = args.data_dir / "index.json"
    cache_path = args.browser_dir / "feature_cache.json"
    if not index_path.exists():
        raise SystemExit(f"Missing {index_path}")
    if not cache_path.exists():
        raise SystemExit(f"Missing {cache_path}")

    index = json.loads(index_path.read_text(encoding="utf-8"))
    cache = json.loads(cache_path.read_text(encoding="utf-8"))
    cache_by_path = {r["map_file"]: r for r in cache["records"]}

    records = list(index["records"])
    decisions = {}
    keep = {r["id"]: True for r in records}
    reason = {r["id"]: "kept" for r in records}

    # ---- Filter 3 first (cheapest): valid_frac + texture diversity
    for r in records:
        if not keep[r["id"]]:
            continue
        cache_rec = cache_by_path.get(r["map_file"])
        if cache_rec is None:
            keep[r["id"]] = False; reason[r["id"]] = "no_cache_record"; continue
        ntex = len(cache_rec["texture_counts"])
        if ntex < args.min_textures:
            keep[r["id"]] = False; reason[r["id"]] = f"too_few_textures({ntex})"; continue
        if r.get("valid_frac", 1.0) < args.min_valid_frac:
            keep[r["id"]] = False; reason[r["id"]] = f"low_valid_frac({r['valid_frac']:.2f})"
    n_after_qual = sum(keep.values())
    print(f"After quality filter: {n_after_qual}/{len(records)} retained")

    # ---- Filter 2: cluster-centroid distance
    by_cluster: Dict[int, List[dict]] = defaultdict(list)
    for r in records:
        if keep[r["id"]]:
            by_cluster[r["style_id"]].append(r)
    for cid, recs in by_cluster.items():
        # Compute centroid as mean normalized texture histogram across cluster
        # Use top-K texture names as the feature vocab for distance.
        global_counts: Counter = Counter()
        for r in recs:
            cr = cache_by_path[r["map_file"]]
            global_counts.update(cr["texture_counts"])
        vocab = [n for n, _ in global_counts.most_common(64)]
        centroid = np.zeros(len(vocab), dtype=np.float64)
        for r in recs:
            cr = cache_by_path[r["map_file"]]
            tot = sum(cr["texture_counts"].values()) or 1
            v = np.array([cr["texture_counts"].get(n, 0) / tot for n in vocab])
            centroid += v
        centroid /= max(len(recs), 1)
        # Per-map distance + std
        dists = []
        for r in recs:
            cr = cache_by_path[r["map_file"]]
            tot = sum(cr["texture_counts"].values()) or 1
            v = np.array([cr["texture_counts"].get(n, 0) / tot for n in vocab])
            d = float(np.linalg.norm(v - centroid))
            dists.append((r["id"], d))
        ds = np.array([d for _, d in dists])
        if len(ds) < 4:
            continue
        thresh = ds.mean() + args.centroid_z * ds.std()
        for rid, d in dists:
            if d > thresh:
                keep[rid] = False
                reason[rid] = f"centroid_outlier(d={d:.3f}>thresh={thresh:.3f})"
    n_after_cent = sum(keep.values())
    print(f"After centroid filter: {n_after_cent}/{len(records)} retained")

    # ---- Filter 1: MinHash near-duplicate
    # Compute sketches only for surviving maps; bucket by first signature value.
    surviving = [r for r in records if keep[r["id"]]]
    sketches = {}
    for r in surviving:
        cr = cache_by_path[r["map_file"]]
        sketches[r["id"]] = shingle_hash(cr["texture_counts"])

    def is_official(p: str) -> bool:
        return "RA3 Official maps" in p.replace("\\", "/")

    # Pairwise within same cluster only (most likely candidates)
    by_cluster_keep: Dict[int, List[dict]] = defaultdict(list)
    for r in surviving:
        by_cluster_keep[r["style_id"]].append(r)
    drop_set = set()
    for cid, recs in by_cluster_keep.items():
        for i in range(len(recs)):
            ri = recs[i]
            if ri["id"] in drop_set or not keep[ri["id"]]:
                continue
            for j in range(i + 1, len(recs)):
                rj = recs[j]
                if rj["id"] in drop_set or not keep[rj["id"]]:
                    continue
                jacc = jaccard_estimate(sketches[ri["id"]], sketches[rj["id"]])
                if jacc < args.minhash_threshold:
                    continue
                # Also confirm with histogram L1 to avoid sketch false positives
                hi = cache_by_path[ri["map_file"]]["texture_counts"]
                hj = cache_by_path[rj["map_file"]]["texture_counts"]
                if histogram_l1_dist(hi, hj) > 0.25:
                    continue
                # Decide which to drop: prefer keeping official > community
                if is_official(ri["map_file"]) and not is_official(rj["map_file"]):
                    drop_set.add(rj["id"])
                    reason[rj["id"]] = f"dup_of_id={ri['id']}(jacc={jacc:.2f})"
                elif is_official(rj["map_file"]) and not is_official(ri["map_file"]):
                    drop_set.add(ri["id"])
                    reason[ri["id"]] = f"dup_of_id={rj['id']}(jacc={jacc:.2f})"
                    break
                else:
                    # Both same tier: drop the higher id (later)
                    drop_set.add(rj["id"])
                    reason[rj["id"]] = f"dup_of_id={ri['id']}(jacc={jacc:.2f})"
    for rid in drop_set:
        keep[rid] = False
    n_final = sum(keep.values())
    print(f"After dedup: {n_final}/{len(records)} retained")

    # Build curated index
    new_records = [r for r in records if keep[r["id"]]]

    # Ensure minimum val count per style: promote some train -> val if a style has <2 val
    min_val_per_style = 2
    rng = np.random.default_rng(1337)
    by_style: Dict[int, dict] = defaultdict(lambda: {"train": [], "val": []})
    for r in new_records:
        by_style[r["style_id"]][r["split"]].append(r)
    for cid, splits in by_style.items():
        if len(splits["val"]) < min_val_per_style and len(splits["train"]) > min_val_per_style * 4:
            need = min_val_per_style - len(splits["val"])
            promote = rng.choice(splits["train"], size=need, replace=False).tolist()
            for r in promote:
                r["split"] = "val"

    new_train = sum(1 for r in new_records if r["split"] == "train")
    new_val = sum(1 for r in new_records if r["split"] == "val")

    # Recompute class frequencies from training portion only
    class_freq = np.zeros(index["vocab_size"], dtype=np.int64)
    for r in new_records:
        if r["split"] != "train":
            continue
        npz_path = args.data_dir / r["npz"]
        d = np.load(npz_path)
        ys = d["y"].reshape(-1)
        ys = ys[ys != index["ignore_index"]]
        for k in np.unique(ys):
            class_freq[int(k)] += int((ys == k).sum())

    curated = dict(index)
    curated["records"] = new_records
    curated["n_train"] = new_train
    curated["n_val"] = new_val
    curated["class_freq_train"] = class_freq.tolist()
    curated["curation"] = {
        "minhash_threshold": args.minhash_threshold,
        "centroid_z": args.centroid_z,
        "min_textures": args.min_textures,
        "min_valid_frac": args.min_valid_frac,
        "n_input_records": len(records),
        "n_output_records": len(new_records),
    }
    out = args.data_dir / "curated_index.json"
    out.write_text(json.dumps(curated, indent=2), encoding="utf-8")

    # Curation report (which maps got dropped + why)
    drops = [{"id": r["id"], "map": r["map_file"], "reason": reason[r["id"]]}
             for r in records if not keep[r["id"]]]
    (args.data_dir / "curation_report.json").write_text(
        json.dumps({"dropped": drops, "n_kept": len(new_records)}, indent=2),
        encoding="utf-8",
    )

    # Per-style summary
    print("\nPer-style after curation:")
    by_style_train = Counter(r["style_id"] for r in new_records if r["split"] == "train")
    by_style_val = Counter(r["style_id"] for r in new_records if r["split"] == "val")
    for s in range(index["n_styles"]):
        print(f"  style {s}:  train={by_style_train.get(s, 0):3d}  val={by_style_val.get(s, 0):3d}")
    print(f"\nTotal: train={new_train}  val={new_val}")
    print(f"Wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
