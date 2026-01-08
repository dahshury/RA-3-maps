"""
Create a *game-detectable* RA3 map folder from an input `.map`, optionally rotated.

RA3 expects maps in:
  <RA3 Maps Root>/<MapName>/<MapName>.map
and typically:
  <MapName>_art.tga

Important: the game only cares about the *filename*, not the actual image encoding.
MapCreatorCore writes PNG bytes with a `.tga` extension.

Example:
  python scripts/package_rotated_map_for_game.py \\
    --in "../RA3 Official maps/2 II/map_mp_2_rao1.map" \\
    --degrees 180 \\
    --name "map_mp_2_rao1_rot180" \\
    --out-root "C:/Users/<you>/AppData/Roaming/Red Alert 3/Maps"
"""

from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path
import sys

# Allow running directly
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from map_processor.ra3map import Ra3Map
from map_processor.utils.map_rotation import rotate_context_right_angles


def _default_ra3_maps_root() -> Path:
    # Typical on Windows: %APPDATA%/Red Alert 3/Maps
    appdata = os.environ.get("APPDATA")
    if appdata:
        return Path(appdata) / "Red Alert 3" / "Maps"
    # Fallback: current working directory
    return Path.cwd()


def _copy_preview_files(src_map_path: Path, dst_dir: Path, dst_name: str) -> None:
    """
    Copy/rename minimap preview files if present next to the source map.
    Preference order matches MapCreatorCore.modifyMap:
      1) <src>_art.tga
      2) <src>.tga
    """
    src_dir = src_map_path.parent
    src_stem = src_map_path.stem

    candidates = [
        src_dir / f"{src_stem}_art.tga",
        src_dir / f"{src_stem}.tga",
    ]
    for cand in candidates:
        if cand.exists():
            shutil.copy2(cand, dst_dir / f"{dst_name}_art.tga")
            return


def main() -> int:
    p = argparse.ArgumentParser(description="Package a rotated RA3 map into a game-detectable folder.")
    p.add_argument("--in", dest="in_path", required=True, help="Input .map path")
    p.add_argument("--degrees", type=int, default=0, help="Rotation degrees (0/90/180/270/360)")
    p.add_argument("--ccw", action="store_true", help="Interpret degrees as counter-clockwise (default clockwise).")
    p.add_argument("--name", required=True, help="Output map name (folder name + .map base name)")
    p.add_argument(
        "--out-root",
        default=str(_default_ra3_maps_root()),
        help="RA3 maps root folder (default: %APPDATA%/Red Alert 3/Maps if available)",
    )
    p.add_argument(
        "--no-preview-copy",
        action="store_true",
        help="Do not copy preview _art.tga/.tga from the source folder.",
    )
    args = p.parse_args()

    src_map = Path(args.in_path)
    out_root = Path(args.out_root)
    out_name = args.name

    out_dir = out_root / out_name
    out_dir.mkdir(parents=True, exist_ok=True)
    out_map_path = out_dir / f"{out_name}.map"

    m = Ra3Map(str(src_map))
    m.parse()
    ctx = m.get_context()

    deg = int(args.degrees) % 360
    if deg:
        rotate_context_right_angles(ctx, degrees=deg, clockwise=(not args.ccw))

    # Save compressed (RefPack) so it matches what RA3 expects for user maps.
    m.save(str(out_map_path), compress=True)

    if not args.no_preview_copy:
        _copy_preview_files(src_map, out_dir, out_name)

    print(f"Packaged map folder: {out_dir}")
    print(f"Map file: {out_map_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())










