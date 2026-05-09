#!/usr/bin/env python3
"""Re-cluster RA3 maps with the 70% texture-overlap rule.

Rule: every pair of maps inside a cluster must share at least 70% of their
texture distribution (histogram intersection). Maps that don't satisfy this
constraint within their existing K=8 cluster get split out into their own
group(s).

Implementation:
  1. Load each map's per-texture tile counts (from feature_cache.json).
  2. Convert to texture-fraction probability vectors aligned to the union vocab.
  3. Pairwise similarity = histogram intersection: sum_t min(p_A[t], p_B[t])
     in [0, 1]. 1.0 = identical distribution; 0.0 = no shared textures.
  4. Audit: for the existing K=8 clusters, print which maps are <0.70 similar
     to all their cluster-mates.
  5. Re-cluster: scipy hierarchical agglomerative with complete linkage and
     distance threshold = 1 - 0.70. Complete linkage guarantees that EVERY
     pair within a cluster has similarity >= 0.70.
  6. Save new assignments to <out>/clusters_70.json (same schema as the K=8
     cache so downstream tooling can switch over by changing one path).

Usage:
  python scripts/recluster_styles_70.py
"""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List

import numpy as np


def _python_tools_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _default_cache() -> Path:
    return _python_tools_root() / "style_clusters" / "browser" / "feature_cache.json"


def _default_k8() -> Path:
    return _python_tools_root() / "style_clusters" / "browser" / "clusters_k8.json"


def _default_out() -> Path:
    return _python_tools_root() / "style_clusters" / "browser" / "clusters_70.json"


def load_features(cache_path: Path) -> List[dict]:
    raw = json.loads(cache_path.read_text(encoding="utf-8"))
    records = raw["records"]
    out = []
    for r in records:
        counts = r.get("texture_counts") or {}
        total = sum(counts.values())
        if total == 0:
            continue
        out.append({
            "map_name": r["map_name"],
            "map_file": r["map_file"],
            "folder": r["folder"],
            "width": r["width"],
            "height": r["height"],
            "total_tiles": total,
            "fractions": {k: v / total for k, v in counts.items()},
        })
    return out


def build_aligned_matrix(records: List[dict]) -> tuple[np.ndarray, list[str]]:
    """Return (N, V) float32 matrix of per-tile texture fractions, plus the
    aligned texture vocabulary."""
    vocab = sorted({t for r in records for t in r["fractions"].keys()})
    idx = {t: i for i, t in enumerate(vocab)}
    M = np.zeros((len(records), len(vocab)), dtype=np.float32)
    for ri, r in enumerate(records):
        for t, f in r["fractions"].items():
            M[ri, idx[t]] = f
    return M, vocab


def histogram_intersection_matrix(M: np.ndarray) -> np.ndarray:
    """Pairwise histogram intersection. (N, V) -> (N, N) in [0, 1].

    sim(A, B) = sum_t min(M[A,t], M[B,t]). For probability vectors (each row
    sums to 1), this is in [0, 1].
    """
    N = M.shape[0]
    sim = np.zeros((N, N), dtype=np.float32)
    # vectorise per-batch to avoid blowing memory on a 952^2 outer product
    block = 64
    for i in range(0, N, block):
        a = M[i:i + block]                                # (B, V)
        for j in range(0, N, block):
            b = M[j:j + block]                            # (C, V)
            mins = np.minimum(a[:, None, :], b[None, :, :]).sum(axis=-1)
            sim[i:i + block, j:j + block] = mins
    np.fill_diagonal(sim, 1.0)
    return sim


def audit_clusters(records: List[dict], sim: np.ndarray,
                   k8_assignments: List[dict], threshold: float = 0.7) -> None:
    """Print which maps are below `threshold` similarity to all cluster-mates."""
    name_to_idx = {r["map_name"]: i for i, r in enumerate(records)}
    cluster_to_idxs: dict[int, list[int]] = defaultdict(list)
    for a in k8_assignments:
        i = name_to_idx.get(a["map_name"])
        if i is not None:
            cluster_to_idxs[int(a["cluster"])].append(i)

    print(f"\n=== AUDIT existing K=8 clusters @ threshold {threshold} ===")
    print(f"  Maps in audit: {sum(len(v) for v in cluster_to_idxs.values())}")
    for c in sorted(cluster_to_idxs):
        idxs = cluster_to_idxs[c]
        if len(idxs) < 2:
            continue
        sub = sim[np.ix_(idxs, idxs)]
        np.fill_diagonal(sub, np.nan)
        with np.errstate(invalid="ignore"):
            min_to_others = np.nanmin(sub, axis=1)
            mean_to_others = np.nanmean(sub, axis=1)
        below = [i for i, m in enumerate(min_to_others) if m < threshold]
        worst = sorted(zip(idxs, min_to_others, mean_to_others),
                       key=lambda x: x[1])[:5]
        print(f"\n  cluster {c:2d}  size={len(idxs):3d}  "
              f"avg_min_pair={float(np.nanmean(min_to_others)):.3f}  "
              f"below_threshold={len(below)}/{len(idxs)}")
        for gi, mn, av in worst:
            mark = " <-- BELOW" if mn < threshold else ""
            print(f"    min={mn:.3f} avg={av:.3f}  {records[gi]['map_name']}{mark}")


def recluster_complete_linkage(sim: np.ndarray, threshold: float) -> np.ndarray:
    """Complete-linkage agglomerative clustering with similarity threshold.

    Two maps end up in the same cluster only if EVERY pair in the cluster has
    similarity >= threshold.
    """
    from scipy.cluster.hierarchy import linkage, fcluster
    from scipy.spatial.distance import squareform

    dist = 1.0 - sim
    np.fill_diagonal(dist, 0.0)
    dist = np.clip(dist, 0.0, 1.0)
    cond = squareform(dist, checks=False)
    Z = linkage(cond, method="complete")
    labels = fcluster(Z, t=1.0 - threshold, criterion="distance")
    return labels.astype(np.int32) - 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cache", type=Path, default=_default_cache())
    ap.add_argument("--k8", type=Path, default=_default_k8(),
                    help="Existing K=8 clustering for the audit phase.")
    ap.add_argument("--out", type=Path, default=_default_out())
    ap.add_argument("--threshold", type=float, default=0.70,
                    help="Pairwise similarity threshold (histogram intersection).")
    ap.add_argument("--no-audit", action="store_true")
    args = ap.parse_args()

    print(f"Loading feature cache: {args.cache}")
    records = load_features(args.cache)
    print(f"Loaded {len(records)} maps with non-zero texture counts.")
    M, vocab = build_aligned_matrix(records)
    print(f"Vocab size: {len(vocab)}; matrix: {M.shape}")

    print("Computing pairwise similarity (histogram intersection) ...")
    sim = histogram_intersection_matrix(M)

    if not args.no_audit and args.k8.exists():
        k8_data = json.loads(args.k8.read_text(encoding="utf-8"))
        audit_clusters(records, sim, k8_data["assignments"], threshold=args.threshold)

    print(f"\n=== RE-CLUSTERING at similarity >= {args.threshold} (complete linkage) ===")
    labels = recluster_complete_linkage(sim, threshold=args.threshold)
    K = int(labels.max()) + 1

    # Stable cluster ids: sort clusters by descending size.
    counts = Counter(int(x) for x in labels.tolist())
    ordered = [c for c, _ in counts.most_common()]
    remap = {old: new for new, old in enumerate(ordered)}
    labels2 = np.array([remap[int(x)] for x in labels.tolist()], dtype=np.int32)

    sizes = Counter(int(x) for x in labels2.tolist())
    print(f"Discovered K = {K}")
    print(f"Cluster sizes (top 30): {sorted(sizes.items(), key=lambda kv: -kv[1])[:30]}")
    n_singletons = sum(1 for v in sizes.values() if v == 1)
    print(f"Singleton clusters: {n_singletons} ({n_singletons*100/K:.1f}% of clusters)")

    # Cluster summary: for each cluster compute min/avg pairwise similarity, top textures.
    clusters_summary = []
    for cid in sorted(set(labels2.tolist())):
        members = [i for i, l in enumerate(labels2.tolist()) if l == cid]
        if len(members) > 1:
            sub = sim[np.ix_(members, members)]
            np.fill_diagonal(sub, np.nan)
            with np.errstate(invalid="ignore"):
                min_pair = float(np.nanmin(sub))
                mean_pair = float(np.nanmean(sub))
        else:
            min_pair = 1.0
            mean_pair = 1.0
        # Aggregate texture distribution
        agg = M[members].mean(axis=0)
        top = sorted(zip(vocab, agg.tolist()), key=lambda kv: -kv[1])[:10]
        clusters_summary.append({
            "cluster": cid,
            "size": len(members),
            "min_pair_similarity": min_pair,
            "mean_pair_similarity": mean_pair,
            "top_textures": [{"name": n, "share": round(s, 4)} for n, s in top if s > 1e-4],
        })

    assignments = []
    for r, lab in zip(records, labels2.tolist()):
        assignments.append({
            "cluster": int(lab),
            "map_name": r["map_name"],
            "map_file": r["map_file"],
            "folder": r["folder"],
            "width": r["width"],
            "height": r["height"],
        })

    out = {
        "k": K,
        "threshold": float(args.threshold),
        "method": "complete_linkage_hist_intersection",
        "num_maps": len(records),
        "cluster_sizes": {str(k): int(v) for k, v in sizes.items()},
        "clusters_summary": clusters_summary,
        "assignments": assignments,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nWritten: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
