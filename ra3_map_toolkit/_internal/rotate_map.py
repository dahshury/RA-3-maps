"""
Rotate / flip an RA3 .map file.

Modes:
  rot90cw   - rotate 90 degrees clockwise (right)
  rot90ccw  - rotate 90 degrees counter-clockwise (left)
  rot180    - rotate 180 degrees
  flipx     - rotate around X axis (mirror y / top<->bottom)
  flipy     - rotate around Y axis (mirror x / left<->right)

Usage:
  python rotate_map.py --in src.map --out out.map --mode rot90cw
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from map_processor.core.ra3map import Ra3Map
from map_processor.utils.map_rotation import (
    rotate_context_right_angles,
    flip_context_axis,
)


MODES = ("rot90cw", "rot90ccw", "rot180", "flipx", "flipy")


def apply_transform(ctx, mode: str) -> None:
    if mode == "rot90cw":
        rotate_context_right_angles(ctx, degrees=90, clockwise=True)
    elif mode == "rot90ccw":
        rotate_context_right_angles(ctx, degrees=90, clockwise=False)
    elif mode == "rot180":
        rotate_context_right_angles(ctx, degrees=180, clockwise=True)
    elif mode == "flipx":
        flip_context_axis(ctx, axis="x")
    elif mode == "flipy":
        flip_context_axis(ctx, axis="y")
    else:
        raise ValueError(f"unknown mode {mode!r}; valid: {MODES}")


def parse_args(argv):
    p = argparse.ArgumentParser(prog="rotate_map", description=__doc__)
    p.add_argument("--in", dest="input", required=True)
    p.add_argument("--out", dest="output", required=True)
    p.add_argument("--mode", required=True, choices=MODES)
    p.add_argument("--no-compress", action="store_true")
    p.add_argument("--copy-preview", action="store_true",
                   help="Copy <stem>.tga sidecar (game preview) next to output if present.")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv if argv is not None else sys.argv[1:])
    src = Path(args.input)
    dst = Path(args.output)
    dst.parent.mkdir(parents=True, exist_ok=True)

    m = Ra3Map(str(src))
    m.parse()
    apply_transform(m.get_context(), args.mode)
    m.save(str(dst), compress=(not args.no_compress))

    if args.copy_preview:
        preview = src.parent / f"{src.stem}.tga"
        if preview.exists():
            shutil.copy2(preview, dst.parent / f"{dst.stem}.tga")

    print(f"OK: {src.name} --{args.mode}--> {dst.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
