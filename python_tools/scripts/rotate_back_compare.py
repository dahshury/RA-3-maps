"""
Rotate a map, then rotate it back, and compare to the original using verify_map_rotation.py logic.

This is useful to test whether our rotation transform is truly invertible for all parsed assets.

Example:
  python scripts/rotate_back_compare.py \
    --orig "../RA3 Official maps/2 FI/map_mp_2_feasel8.map" \
    --degrees 180 \
    --work-dir "../RA3 Official maps/2 FI/test/rotate_back_compare"
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from map_processor.ra3map import Ra3Map
from map_processor.utils.map_rotation import rotate_context_right_angles


def main() -> int:
    p = argparse.ArgumentParser(description="Rotate -> rotate back -> compare.")
    p.add_argument("--orig", required=True, help="Original .map path")
    p.add_argument("--degrees", type=int, required=True, help="Rotation degrees (multiple of 90)")
    p.add_argument("--ccw", action="store_true", help="Interpret degrees as CCW (default CW)")
    p.add_argument("--work-dir", required=True, help="Directory to write intermediate maps into")
    args = p.parse_args()

    orig_path = Path(args.orig)
    work_dir = Path(args.work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    rot_path = work_dir / f"{orig_path.stem}_rot{args.degrees}.map"
    back_path = work_dir / f"{orig_path.stem}_rot{args.degrees}_back.map"

    # Rotate
    m = Ra3Map(str(orig_path))
    m.parse()
    rotate_context_right_angles(m.get_context(), degrees=args.degrees, clockwise=(not args.ccw))
    m.save(str(rot_path), compress=True)

    # Rotate back (inverse)
    inv_deg = (-args.degrees) % 360
    m2 = Ra3Map(str(rot_path))
    m2.parse()
    rotate_context_right_angles(m2.get_context(), degrees=inv_deg, clockwise=True)
    m2.save(str(back_path), compress=True)

    print(f"rotated:   {rot_path}")
    print(f"rot-back:  {back_path}")
    print("Now verify with:")
    print(
        f"  python scripts/verify_map_rotation.py --orig \"{orig_path}\" --rot \"{back_path}\" --degrees 0"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())










