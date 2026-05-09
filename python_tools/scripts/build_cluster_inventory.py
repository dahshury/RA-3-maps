#!/usr/bin/env python3
"""Build per-cluster cliff biome inventory from OFFICIAL maps only.

Reads cluster assignments + parses each cluster member that lives under
`RA3 Official maps/` (community maps under `side/` are ignored when computing
the inventory, because we only want known-valid game asset types).

Output: style_clusters/browser/cluster_inventory.json
  {
    "<cluster_id>": {
      "n_official_members": int,
      "cliff_prefixes": {"YU": 5000, ...},     # by frequency
      "dominant_cliff_biome": "YU",
      "cliff_suffixes_by_prefix": {"YU": ["CLIFFWALL01", "SEACLIFFWALL01", ...]}
    }, ...
  }

Run after changing K or rebuilding clusters_k*_full.json.
"""
from __future__ import annotations

import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path


def _python_tools_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _import_ra3map():
    tools_root = _python_tools_root()
    if str(tools_root) not in sys.path:
        sys.path.insert(0, str(tools_root))
    from map_processor import Ra3Map  # noqa: WPS433
    return Ra3Map


CLIFF_RE = re.compile(r"^(.*?)_?(CLIFFWALL\d+|SEACLIFFWALL\d+)$", re.I)
OFFICIAL_DIR_NAME = "RA3 Official maps"


def _is_official(path_str: str) -> bool:
    return OFFICIAL_DIR_NAME in path_str.replace("\\", "/")


def main() -> int:
    browser = _python_tools_root() / "style_clusters" / "browser"
    clusters_path = browser / "clusters_k8_full.json"
    if not clusters_path.exists():
        raise SystemExit(f"Missing {clusters_path}")

    Ra3Map = _import_ra3map()
    clusters = json.loads(clusters_path.read_text(encoding="utf-8"))

    by_cluster: dict[int, list[str]] = defaultdict(list)
    for a in clusters["assignments"]:
        if _is_official(a["map_file"]):
            by_cluster[int(a["cluster"])].append(a["map_file"])

    out: dict[str, dict] = {}
    for cid in sorted(by_cluster):
        files = by_cluster[cid]
        prefix_counts: Counter = Counter()
        suf_by_pref: dict[str, set[str]] = defaultdict(set)

        for f in files:
            try:
                m = Ra3Map(f); m.parse()
                objs = m.get_context().get_asset("ObjectsList")
                for obj in objs.map_objects:
                    mm = CLIFF_RE.match(obj.type_name)
                    if not mm:
                        continue
                    prefix = mm.group(1).upper()
                    suffix = mm.group(2).upper()
                    prefix_counts[prefix] += 1
                    suf_by_pref[prefix].add(suffix)
            except Exception as e:  # noqa: BLE001
                print(f"  [warn] {f}: {e}")

        dom = prefix_counts.most_common(1)[0][0] if prefix_counts else None
        out[str(cid)] = {
            "n_official_members": len(files),
            "cliff_prefixes": dict(prefix_counts.most_common()),
            "dominant_cliff_biome": dom,
            "cliff_suffixes_by_prefix": {p: sorted(s) for p, s in suf_by_pref.items()},
        }
        print(f"cluster {cid}: n={len(files):3d}  dominant={dom}  prefixes={dict(prefix_counts.most_common(5))}")

    out_path = browser / "cluster_inventory.json"
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nWrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
