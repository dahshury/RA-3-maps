"""
Rotate an RA3 `.map` file by right angles (0/90/180/270/360).

Example:
  python scripts/rotate_map.py --in "path/to/map.map" --degrees 90 --out "path/to/map_rot90.map"
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

# Allow running this file directly (python scripts/rotate_map.py) by ensuring the
# project root (python_tools/) is on sys.path so `import map_processor` works.
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from map_processor.ra3map import Ra3Map
from map_processor.utils.map_rotation import rotate_context_right_angles


def main() -> int:
    p = argparse.ArgumentParser(description="Rotate an RA3 .map (right-angle rotations).")
    p.add_argument("--in", dest="in_path", required=True, help="Input .map path")
    p.add_argument("--out", dest="out_path", required=True, help="Output .map path")
    p.add_argument(
        "--degrees",
        type=int,
        default=90,
        help="Rotation in degrees (must be multiple of 90). 360 is treated as 0.",
    )
    p.add_argument(
        "--ccw",
        action="store_true",
        help="Interpret degrees as counter-clockwise (default is clockwise).",
    )
    p.add_argument(
        "--no-compress",
        action="store_true",
        help="Write uncompressed output (note: compressor is not implemented; default behavior writes uncompressed anyway).",
    )
    args = p.parse_args()

    in_path = Path(args.in_path)
    out_path = Path(args.out_path)

    m = Ra3Map(str(in_path))
    m.parse()
    ctx = m.get_context()

    rotate_context_right_angles(ctx, degrees=args.degrees, clockwise=(not args.ccw))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    m.save(str(out_path), compress=(not args.no_compress))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


