"""
Compose RA3 .map files into a single output by placing each source at a
chosen layout cell. Composition is a strict superset of duplication --
"duplicate" is just `--preset duplicate` with one source tiled Nx*Ny times.

Hard-capped at 6 resulting players. Each source must have >= 2 players.

Presets (slot ordering for `--maps A B C ...`):
  duplicate         maps[0] tiled into nx*ny grid (uses --nx --ny)
  row               1xN row, left-to-right
  col               Nx1 column, top-to-bottom
  triangle_top      A=top-left, B=top-right, C=bottom-spanning
  triangle_bottom   A=top-spanning, B=bottom-left, C=bottom-right
  triangle_left     A=top-left, B=bottom-left, C=right-spanning
  triangle_right    A=left-spanning, B=top-right, C=bottom-right

Emits one JSON line per progress event when --json-progress is set:
  {"event": "compose_start", "preset": str, "maps": [str, ...], "output": str, "total_steps": int}
  {"event": "compose_step", "step": "parse"|"layout"|"compose"|"save"|"tga", "detail"?: str}
  {"event": "compose_complete", "preset": str, "success": bool, "output"?: str, "error"?: str}
  {"event": "compose_done", "success": int, "fail": int, "total": int, "output": str}
  {"event": "fatal", "error": str}
"""
from __future__ import annotations

import argparse
import contextlib
import io
import json
import shutil
import sys
import traceback
from pathlib import Path
from typing import Dict, List, Optional, Tuple


def get_app_root() -> Path:
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parent


_ROOT = get_app_root()
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from map_processor.core.ra3map import Ra3Map  # noqa: E402
from map_processor.utils.map_composition import (  # noqa: E402
    CompositionCell,
    CompositionSpec,
    MAX_PLAYERS,
    compose_context,
    presets as compose_presets,
)


PRESET_NAMES = (
    "duplicate", "row", "col",
    "triangle_top", "triangle_bottom", "triangle_left", "triangle_right",
)


# ---------------------------------------------------------------------------
# Eligibility / player count.
# ---------------------------------------------------------------------------

PLAYER_START_IDS = {
    "Player_1_Start", "Player_2_Start", "Player_3_Start",
    "Player_4_Start", "Player_5_Start", "Player_6_Start",
}


def get_player_count(map_path: Path) -> Optional[int]:
    """Return the number of `Player_<n>_Start` waypoints in this map."""
    try:
        with _silence_stdout():
            m = Ra3Map(str(map_path))
            m.parse()
        ctx = m.get_context()
        from map_processor.assets.objects.objects_list import ObjectsList
        objs = ctx.get_asset_by_type(ObjectsList)
        if objs is None:
            return None
        n = 0
        for obj in objs.map_objects:
            if obj.unique_id in PLAYER_START_IDS:
                n += 1
        return n
    except Exception:
        return None


# ---------------------------------------------------------------------------
# JSON event emission.
# ---------------------------------------------------------------------------

JSON_MODE = False
_REAL_STDOUT = sys.stdout


@contextlib.contextmanager
def _silence_stdout():
    if not JSON_MODE:
        yield
        return
    sink = io.StringIO()
    old = sys.stdout
    sys.stdout = sink
    try:
        yield
    finally:
        sys.stdout = old


def emit(event: str, **fields) -> None:
    if JSON_MODE:
        _REAL_STDOUT.write(json.dumps({"event": event, **fields}) + "\n")
        _REAL_STDOUT.flush()
        return
    if event == "compose_start":
        print(f"Composing {fields.get('preset')} from {len(fields.get('maps', []))} "
              f"map(s) -> {fields.get('output')}")
    elif event == "compose_step":
        detail = fields.get("detail")
        suffix = f" ({detail})" if detail else ""
        print(f"  - {fields.get('step')}{suffix}")
    elif event == "compose_complete":
        if fields.get("success"):
            print(f"  [OK] -> {fields.get('output','')}")
        else:
            print(f"  [FAIL] {fields.get('error','')}")
    elif event == "compose_done":
        print(f"\nDone. success={fields.get('success',0)} "
              f"fail={fields.get('fail',0)}")
    elif event == "fatal":
        print(f"FATAL: {fields.get('error','')}", file=sys.stderr)


# ---------------------------------------------------------------------------
# Output naming.
# ---------------------------------------------------------------------------

def _label_for_preset(preset: str, maps: List[Path], nx: int, ny: int) -> str:
    if preset == "duplicate":
        parts = []
        if nx > 1: parts.append(f"x{nx}")
        if ny > 1: parts.append(f"y{ny}")
        return "_".join(parts) or "noop"
    short = {
        "row": "row",
        "col": "col",
        "triangle_top": "triTop",
        "triangle_bottom": "triBot",
        "triangle_left": "triLeft",
        "triangle_right": "triRight",
    }.get(preset, preset)
    return short


def _output_paths(
    canvas: Path,
    out_root: Path,
    preset: str,
    maps: List[Path],
    nx: int,
    ny: int,
) -> Tuple[Path, Path, str]:
    label = _label_for_preset(preset, maps, nx, ny)
    if preset == "duplicate":
        stem = f"{canvas.stem}_{label}"
    else:
        # Combine canvas stem + first 8 chars of every other map's stem.
        suffix_pieces = [label] + [m.stem[:10] for m in maps[1:]]
        stem = f"{canvas.stem}_{'_'.join(suffix_pieces)}"
    # Cap length so Windows path limits don't bite.
    if len(stem) > 80:
        stem = stem[:77] + "..."
    out_dir = out_root / stem
    out_map = out_dir / f"{stem}.map"
    return out_dir, out_map, label


def _copy_preview_tga(src: Path, dst_dir: Path, dst_stem: str) -> None:
    preview = src.parent / f"{src.stem}.tga"
    if preview.exists():
        try:
            shutil.copy2(preview, dst_dir / f"{dst_stem}.tga")
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Build a CompositionSpec from a preset + parsed contexts.
# ---------------------------------------------------------------------------

_VALID_ALIGN_X = ("left", "center", "right")
_VALID_ALIGN_Y = ("top", "center", "bottom")


def _normalize_aligns(
    aligns: Optional[List[str]],
    n: int,
    valid: tuple,
    default: str,
) -> List[str]:
    """Pad / truncate `aligns` to length `n`, validating each entry."""
    out = list(aligns or [])
    out = out[:n] + [default] * max(0, n - len(out))
    for v in out:
        if v not in valid:
            raise ValueError(f"invalid alignment {v!r}; expected one of {valid}")
    return out


def _build_spec(
    preset: str,
    ctxs: List[object],
    *,
    nx: int = 2,
    ny: int = 1,
    pad_x: int = 0,
    pad_y: int = 0,
    align_x: Optional[List[str]] = None,
    align_y: Optional[List[str]] = None,
) -> CompositionSpec:
    """Build a CompositionSpec. Per-slot align_x/align_y are applied to every
    cell (cell index = slot index, except for `duplicate` where every cell
    derives from a single source and shares the same alignment)."""
    if preset == "duplicate":
        if len(ctxs) != 1:
            raise ValueError("--preset duplicate requires exactly one --maps entry")
        spec = compose_presets.duplicate(ctxs[0], nx, ny, pad_x, pad_y)
    elif preset == "row":
        if len(ctxs) < 2:
            raise ValueError("--preset row requires >= 2 --maps entries")
        spec = compose_presets.row(ctxs, pad_x)
    elif preset == "col":
        if len(ctxs) < 2:
            raise ValueError("--preset col requires >= 2 --maps entries")
        spec = compose_presets.col(ctxs, pad_y)
    elif preset == "triangle_top":
        if len(ctxs) != 3:
            raise ValueError("--preset triangle_top requires exactly 3 --maps entries (A B C)")
        spec = compose_presets.triangle_top(*ctxs, pad_x=pad_x, pad_y=pad_y)
    elif preset == "triangle_bottom":
        if len(ctxs) != 3:
            raise ValueError("--preset triangle_bottom requires exactly 3 --maps entries (A B C)")
        spec = compose_presets.triangle_bottom(*ctxs, pad_x=pad_x, pad_y=pad_y)
    elif preset == "triangle_left":
        if len(ctxs) != 3:
            raise ValueError("--preset triangle_left requires exactly 3 --maps entries (A B C)")
        spec = compose_presets.triangle_left(*ctxs, pad_x=pad_x, pad_y=pad_y)
    elif preset == "triangle_right":
        if len(ctxs) != 3:
            raise ValueError("--preset triangle_right requires exactly 3 --maps entries (A B C)")
        spec = compose_presets.triangle_right(*ctxs, pad_x=pad_x, pad_y=pad_y)
    else:
        raise ValueError(f"unknown preset: {preset!r}")

    # Re-emit the cells with the requested per-slot alignment (only when any
    # non-default value was passed). For `duplicate`, slot 0's alignment is
    # broadcast to every tiled cell.
    if (align_x or align_y):
        n_slots = 1 if preset == "duplicate" else len(ctxs)
        ax = _normalize_aligns(align_x, n_slots, _VALID_ALIGN_X, "center")
        ay = _normalize_aligns(align_y, n_slots, _VALID_ALIGN_Y, "center")
        new_cells = []
        for cell in spec.cells:
            if preset == "duplicate":
                slot_idx = 0
            else:
                # Each preset's lower-cell-index matches input slot order
                # (compose_presets.* assigns cells in input order).
                slot_idx = next(
                    (i for i, c in enumerate(ctxs) if c is cell.source),
                    -1,
                )
                if slot_idx < 0:
                    new_cells.append(cell)
                    continue
            new_cells.append(CompositionCell(
                source=cell.source,
                col=cell.col, row=cell.row,
                span_cols=cell.span_cols, span_rows=cell.span_rows,
                align_x=ax[slot_idx],
                align_y=ay[slot_idx],
            ))
        spec = CompositionSpec(cells=tuple(new_cells), pad_x=spec.pad_x, pad_y=spec.pad_y)

    return spec


# ---------------------------------------------------------------------------
# Per-composition execution.
# ---------------------------------------------------------------------------

def compose_one(
    preset: str,
    maps: List[Path],
    out_root: Path,
    *,
    nx: int = 2,
    ny: int = 1,
    pad_x: int = 0,
    pad_y: int = 0,
    align_x: Optional[List[str]] = None,
    align_y: Optional[List[str]] = None,
    compress: bool = True,
    copy_tga: bool = True,
) -> bool:
    canvas = maps[0]
    out_dir, out_map, label = _output_paths(canvas, out_root, preset, maps, nx, ny)

    try:
        emit("compose_step", step="parse", detail=f"{len(maps)} map(s)")

        # Parse each unique source path once and reuse for any cell that
        # references the same path (duplicate preset).
        unique_paths: List[Path] = []
        for p in maps:
            if p not in unique_paths:
                unique_paths.append(p)

        parsed_maps: Dict[str, Ra3Map] = {}
        parsed_ctxs: Dict[str, object] = {}
        with _silence_stdout():
            for p in unique_paths:
                m = Ra3Map(str(p))
                m.parse()
                parsed_maps[str(p)] = m
                parsed_ctxs[str(p)] = m.get_context()

        # Player-count cap check up front so we can fail-fast with a clean error.
        ctxs_in_order = [parsed_ctxs[str(p)] for p in maps]
        total_players = 0
        for ctx in ctxs_in_order:
            from map_processor.assets.objects.objects_list import ObjectsList
            objs = ctx.get_asset_by_type(ObjectsList)
            count = sum(1 for o in (objs.map_objects if objs else []) if o.unique_id in PLAYER_START_IDS)
            if count < 2:
                raise ValueError(f"source has only {count} player(s); need >= 2")
            total_players += count

        # For duplicate preset, total_players is single-source's count * nx * ny.
        if preset == "duplicate":
            total_players = (total_players // len(maps)) * nx * ny if maps else total_players
        if total_players > MAX_PLAYERS:
            raise ValueError(
                f"would exceed {MAX_PLAYERS}-player cap: total={total_players}"
            )

        emit("compose_step", step="layout")

        spec = _build_spec(
            preset, ctxs_in_order,
            nx=nx, ny=ny, pad_x=pad_x, pad_y=pad_y,
            align_x=align_x, align_y=align_y,
        )

        emit("compose_step", step="compose")
        canvas_ctx = parsed_ctxs[str(canvas)]
        with _silence_stdout():
            compose_context(canvas_ctx, spec)

        emit("compose_step", step="save")
        out_dir.mkdir(parents=True, exist_ok=True)
        with _silence_stdout():
            parsed_maps[str(canvas)].save(str(out_map), compress=compress)

        if copy_tga:
            emit("compose_step", step="tga")
            _copy_preview_tga(canvas, out_dir, out_map.stem)

        emit("compose_complete", preset=preset, success=True, output=str(out_dir))
        return True
    except Exception as e:
        emit("compose_complete", preset=preset, success=False, error=str(e))
        return False


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv: List[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="compose_engine",
        description="Compose RA3 .map files into a single output. "
                    f"Total players capped at {MAX_PLAYERS}; sources need >= 2 players each.",
    )
    p.add_argument("--preset", choices=PRESET_NAMES, default="duplicate",
                   help="Layout preset.")
    p.add_argument("--maps", nargs="+", required=True,
                   help="Source .map paths in slot order. For 'duplicate', pass exactly one.")
    p.add_argument("--out-dir", default=None,
                   help="Output root folder. Defaults to ../converted_maps.")
    p.add_argument("--nx", type=int, default=2, help="(duplicate) tile count along X.")
    p.add_argument("--ny", type=int, default=1, help="(duplicate) tile count along Y.")
    p.add_argument("--pad-x", type=int, default=0,
                   help="Tile-unit gap inserted between adjacent X-axis cells (filled with water).")
    p.add_argument("--pad-y", type=int, default=0,
                   help="Tile-unit gap inserted between adjacent Y-axis cells (filled with water).")
    p.add_argument("--align-x", default=None,
                   help="Comma-separated per-slot horizontal alignment: left|center|right. "
                        "Only matters when a source's playable area is smaller than its allocated cell.")
    p.add_argument("--align-y", default=None,
                   help="Comma-separated per-slot vertical alignment: top|center|bottom.")
    p.add_argument("--no-compress", action="store_true")
    p.add_argument("--no-copy-tga", action="store_true",
                   help="Do not copy <stem>.tga preview next to outputs.")
    p.add_argument("--json-progress", action="store_true")
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    global JSON_MODE
    args = parse_args(argv if argv is not None else sys.argv[1:])
    JSON_MODE = args.json_progress

    maps = [Path(m) for m in args.maps]
    for m in maps:
        if not m.exists():
            emit("fatal", error=f"Source not found: {m}")
            return 1

    out_root = Path(args.out_dir) if args.out_dir else _ROOT.parent / "converted_maps"
    out_root.mkdir(parents=True, exist_ok=True)

    compress = not args.no_compress
    copy_tga = not args.no_copy_tga

    try:
        emit(
            "compose_start",
            preset=args.preset,
            maps=[str(m) for m in maps],
            output=str(out_root),
            total_steps=5,  # parse, layout, compose, save, tga
        )
        ax = [s.strip() for s in args.align_x.split(",")] if args.align_x else None
        ay = [s.strip() for s in args.align_y.split(",")] if args.align_y else None
        ok = compose_one(
            args.preset, maps, out_root,
            nx=int(args.nx), ny=int(args.ny),
            pad_x=int(args.pad_x), pad_y=int(args.pad_y),
            align_x=ax, align_y=ay,
            compress=compress, copy_tga=copy_tga,
        )
        emit(
            "compose_done",
            success=1 if ok else 0,
            fail=0 if ok else 1,
            total=1,
            output=str(out_root),
        )
        return 0 if ok else 1
    except Exception as e:
        emit("fatal", error=f"{e}\n{traceback.format_exc()}")
        return 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        emit("fatal", error="Cancelled by user")
        sys.exit(1)
