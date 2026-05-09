"""
Compose RA3 `.map` files into a single output by placing each source map at a
chosen layout position. Composition is a strict superset of duplication: the
`duplicate` preset tiles a single source Nx*Ny times.

Constraints:
- Every source must have >= 2 players.
- Total players across all cells must be <= 6 (RA3 lobby cap).

Examples (run from python_tools/):

  # Duplicate (the original duplicate_map.py behavior):
  #   2-player -> 4-player by horizontal tile
  python scripts/compose_map.py --preset duplicate \
    --maps "../RA3 Official maps/2 II/map_mp_2_rao1.map" \
    --nx 2 --ny 1 \
    --out "out/2_II_X2.map"

  # Two different 2-player maps side by side -> 4-player composite (X axis)
  python scripts/compose_map.py --preset row \
    --maps "../RA3 Official maps/2 II/map_mp_2_rao1.map" \
           "../RA3 Official maps/2 IS/map_mp_2_feasel6.map" \
    --out "out/II_plus_IS.map"

  # Three 2-player maps in a triangle (2 top + 1 bottom-centered) -> 6-player
  python scripts/compose_map.py --preset triangle_top \
    --maps A.map B.map C.map \
    --out "out/triangle.map"

  # Free-form layout (e.g. 2x2 with one cell spanning):
  python scripts/compose_map.py \
    --cell A.map 0 0 --cell B.map 1 0 \
    --cell C.map 0 1 --cell-span C.map 2 1 \
    --out out/freeform.map

Slot ordering for --maps under each preset:
  row              : maps[0] left ... maps[N-1] right
  col              : maps[0] top  ... maps[N-1] bottom
  triangle_top     : A=top-left,         B=top-right, C=bottom-spanning
  triangle_bottom  : A=top-spanning,     B=bottom-left, C=bottom-right
  triangle_left    : A=top-left,         B=bottom-left, C=right-spanning
  triangle_right   : A=left-spanning,    B=top-right,   C=bottom-right
  duplicate        : maps[0] tiled into nx*ny grid (uses --nx --ny)
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Dict, List, Tuple

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from map_processor.ra3map import Ra3Map
from map_processor.utils.map_composition import (
    CompositionCell, CompositionSpec, compose_context, presets,
)


PRESET_NAMES = ("duplicate", "row", "col",
                "triangle_top", "triangle_bottom", "triangle_left", "triangle_right")


def _parse_cell_arg(values: List[str]) -> Tuple[str, int, int]:
    if len(values) != 3:
        raise argparse.ArgumentTypeError(
            "--cell expects 3 args: PATH COL ROW"
        )
    return (values[0], int(values[1]), int(values[2]))


def _parse_cell_span_arg(values: List[str]) -> Tuple[str, int, int]:
    if len(values) != 3:
        raise argparse.ArgumentTypeError(
            "--cell-span expects 3 args: PATH SPAN_COLS SPAN_ROWS"
        )
    return (values[0], int(values[1]), int(values[2]))


def _build_spec_from_preset(args, parsed_ctxs: Dict[str, object]) -> CompositionSpec:
    name = args.preset
    paths = list(args.maps or [])
    ctxs = [parsed_ctxs[p] for p in paths]
    pad_x = int(args.pad_x)
    pad_y = int(args.pad_y)

    if name == "duplicate":
        if len(ctxs) != 1:
            raise SystemExit("--preset duplicate requires exactly one --maps entry")
        return presets.duplicate(ctxs[0], int(args.nx), int(args.ny), pad_x, pad_y)
    if name == "row":
        if len(ctxs) < 2:
            raise SystemExit("--preset row requires >= 2 --maps entries")
        return presets.row(ctxs, pad_x)
    if name == "col":
        if len(ctxs) < 2:
            raise SystemExit("--preset col requires >= 2 --maps entries")
        return presets.col(ctxs, pad_y)
    if name == "triangle_top":
        if len(ctxs) != 3:
            raise SystemExit("--preset triangle_top requires exactly 3 --maps entries (A B C)")
        return presets.triangle_top(*ctxs, pad_x=pad_x, pad_y=pad_y)
    if name == "triangle_bottom":
        if len(ctxs) != 3:
            raise SystemExit("--preset triangle_bottom requires exactly 3 --maps entries (A B C)")
        return presets.triangle_bottom(*ctxs, pad_x=pad_x, pad_y=pad_y)
    if name == "triangle_left":
        if len(ctxs) != 3:
            raise SystemExit("--preset triangle_left requires exactly 3 --maps entries (A B C)")
        return presets.triangle_left(*ctxs, pad_x=pad_x, pad_y=pad_y)
    if name == "triangle_right":
        if len(ctxs) != 3:
            raise SystemExit("--preset triangle_right requires exactly 3 --maps entries (A B C)")
        return presets.triangle_right(*ctxs, pad_x=pad_x, pad_y=pad_y)
    raise SystemExit(f"unknown preset: {name}")


def _build_spec_from_cells(args, parsed_ctxs: Dict[str, object]) -> CompositionSpec:
    spans: Dict[str, Tuple[int, int]] = {}
    for path, sc, sr in (args.cell_span or []):
        spans[path] = (sc, sr)
    cell_objs: List[CompositionCell] = []
    for path, col, row in args.cell:
        sc, sr = spans.get(path, (1, 1))
        cell_objs.append(CompositionCell(
            source=parsed_ctxs[path], col=col, row=row,
            span_cols=sc, span_rows=sr,
        ))
    return CompositionSpec(
        cells=tuple(cell_objs),
        pad_x=int(args.pad_x), pad_y=int(args.pad_y),
    )


def main() -> int:
    p = argparse.ArgumentParser(
        description="Compose RA3 .map files (presets or free-form). "
                    "Duplication is just `--preset duplicate`."
    )
    p.add_argument("--out", dest="out_path", required=True, help="Output .map path")
    p.add_argument("--pad-x", type=int, default=0,
                   help="Tile-unit gap between adjacent columns (water-filled). 0 = none.")
    p.add_argument("--pad-y", type=int, default=0,
                   help="Tile-unit gap between adjacent rows (water-filled). 0 = none.")
    p.add_argument("--no-compress", action="store_true",
                   help="Write uncompressed output (compressor is not implemented; "
                        "default writes uncompressed anyway).")

    # Preset mode
    p.add_argument("--preset", choices=PRESET_NAMES,
                   help="Layout preset. See module docstring for slot ordering.")
    p.add_argument("--maps", nargs="+", default=None,
                   help="Source .map paths in slot order (preset mode).")
    p.add_argument("--nx", type=int, default=1,
                   help="(--preset duplicate) tile count along X.")
    p.add_argument("--ny", type=int, default=1,
                   help="(--preset duplicate) tile count along Y.")

    # Free-form mode
    p.add_argument("--cell", action="append", nargs=3, metavar=("PATH", "COL", "ROW"),
                   default=[], help="Free-form: place PATH at grid (COL, ROW). Repeat per cell.")
    p.add_argument("--cell-span", action="append", nargs=3,
                   metavar=("PATH", "SPAN_COLS", "SPAN_ROWS"), default=[],
                   help="Optional span override for a free-form cell. Repeat per cell.")

    args = p.parse_args()

    if not args.preset and not args.cell:
        p.error("either --preset or one or more --cell required")
    if args.preset and args.cell:
        p.error("--preset and --cell are mutually exclusive")

    # Convert int args
    args.cell = [(path, int(col), int(row)) for path, col, row in args.cell]
    args.cell_span = [(path, int(sc), int(sr)) for path, sc, sr in args.cell_span]

    # Parse all unique input maps. Note: in preset=duplicate, the same path
    # appears once but is referenced by every cell. In free-form, we parse
    # each unique path once and reuse.
    if args.preset:
        unique_paths = list(dict.fromkeys(args.maps or []))
    else:
        unique_paths = list(dict.fromkeys(path for path, *_ in args.cell))

    if not unique_paths:
        p.error("no source maps specified")

    parsed_maps: Dict[str, Ra3Map] = {}
    parsed_ctxs: Dict[str, object] = {}
    for path in unique_paths:
        m = Ra3Map(path)
        m.parse()
        parsed_maps[path] = m
        parsed_ctxs[path] = m.get_context()

    # The first unique path becomes the canvas (mutated in place + saved out).
    canvas_path = unique_paths[0]
    canvas_map = parsed_maps[canvas_path]
    canvas_ctx = parsed_ctxs[canvas_path]

    # The (col=0, row=0) cell must be the canvas. For presets this is the
    # convention. For free-form, validate the user passed --cell of canvas at (0,0).
    if args.preset:
        # presets always anchor maps[0] at (0, 0); make sure canvas_ctx is maps[0].
        if (args.maps or [None])[0] != canvas_path:
            # unique-path order matches maps order; this is just a sanity
            raise SystemExit("internal error: canvas path mismatch")
        spec = _build_spec_from_preset(args, parsed_ctxs)
    else:
        # Free-form: ensure exactly one cell is at (0, 0) and that it's the
        # first unique path.
        zero_cells = [c for c in args.cell if c[1] == 0 and c[2] == 0]
        if len(zero_cells) != 1:
            p.error("free-form mode needs exactly one --cell at (0, 0)")
        if zero_cells[0][0] != canvas_path:
            p.error(
                f"the --cell at (0, 0) must come first; got '{zero_cells[0][0]}' "
                f"but the first --cell was '{canvas_path}'"
            )
        spec = _build_spec_from_cells(args, parsed_ctxs)

    compose_context(canvas_ctx, spec)

    out_path = Path(args.out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas_map.save(str(out_path), compress=(not args.no_compress))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
