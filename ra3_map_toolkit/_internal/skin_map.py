"""
Decompose / "skin" an RA3 map into layered strips so the contribution of each
visual layer can be inspected independently.

For one source map, produces 4+4 stripped variants:

  ISOLATED (each removes exactly ONE layer):
    iso1_no_blends.map     - blends/single_edge_blends/cliff_blends zeroed
    iso2_no_textures.map   - texture palette collapsed to one neutral (Dirt_Yucatan01)
    iso3_no_objects.map    - ObjectsList emptied
    iso4_flat.map          - HeightMapData.elevations flattened to mean

  CUMULATIVE (each strip adds on top of the previous):
    cum1_blends_off.map
    cum2_blends_textures_off.map
    cum3_blends_textures_objects_off.map
    cum4_skeleton.map      - everything stripped (flat empty plane)

Each output is also rendered to a PNG minimap via the toolkit's minimap_generator.

Usage:
  python skin_map.py --src <path/to/map.map> [--out-dir <output_root>] [--no-render]

When invoked from the toolkit TUI, pass --json-progress for line-delimited JSON
events on stdout.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import sys
import traceback
from pathlib import Path
from typing import Callable, Iterable, Optional

import numpy as np


def get_app_root() -> Path:
    if getattr(sys, 'frozen', False):
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parent


_ROOT = get_app_root()
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from map_processor.core.ra3map import Ra3Map  # noqa: E402
from map_processor.assets.terrain.texture import Texture  # noqa: E402
from minimap_generator import generate_minimap  # noqa: E402


JSON_MODE = False
_REAL_STDOUT = sys.stdout


@contextlib.contextmanager
def _silence_stdout():
    """In JSON mode, route inner prints to a sink so they don't pollute the JSON stream."""
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
    """Emit a progress event. JSON line in --json-progress mode, plain text otherwise."""
    if JSON_MODE:
        payload = {"event": event, **fields}
        _REAL_STDOUT.write(json.dumps(payload) + "\n")
        _REAL_STDOUT.flush()
        return

    if event == "skin_start":
        print(f"Source: {fields.get('source', '')}")
        print(f"Output: {fields.get('output', '')}")
        print(f"Variants: {fields.get('total', 0)}\n")
    elif event == "skin_variant_start":
        idx = fields.get("index", 0)
        total = fields.get("total", 0)
        name = fields.get("name", "?")
        desc = fields.get("description", "")
        print(f"  [{idx}/{total}] {name}  {desc}")
    elif event == "skin_step":
        print(f"      - {fields.get('step', '')}")
    elif event == "skin_variant_complete":
        if fields.get("success"):
            print(f"      [OK] -> {fields.get('output', '')}")
        else:
            print(f"      [FAIL] {fields.get('error', '')}")
    elif event == "skin_done":
        print(f"\nDone. Success: {fields.get('success', 0)}  "
              f"Fail: {fields.get('fail', 0)}  Output: {fields.get('output', '')}")
    elif event == "fatal":
        print(f"FATAL: {fields.get('error', '')}", file=sys.stderr)


# ---------------------------------------------------------------------------
# Strip functions (ported verbatim from scripts/skin_map_layers.py).
# ---------------------------------------------------------------------------

def strip_blends(blend) -> None:
    """Zero the blends arrays - hard texture transitions remain."""
    blend.blends = np.zeros_like(blend.blends)
    blend.single_edge_blends = np.zeros_like(blend.single_edge_blends)
    blend.cliff_blends = np.zeros_like(blend.cliff_blends)


def strip_textures(blend) -> None:
    """Reduce palette to one neutral texture; all tiles point at it.

    Any official ground-like neutral works. Use Dirt_Yucatan01 if available
    in the source map; else fall back to the first existing texture.
    """
    NEUTRAL = "Dirt_Yucatan01"
    found_idx = None
    for i, t in enumerate(blend.textures):
        if t.name == NEUTRAL:
            found_idx = i
            break
    if found_idx is None:
        # Use first texture, but rename it to NEUTRAL
        if blend.textures:
            blend.textures[0].name = NEUTRAL
            found_idx = 0
        else:
            t = Texture()
            t.cell_start = 0
            t.cell_count = 16
            t.cell_size = 4
            t.magic_value = 0
            t.name = NEUTRAL
            blend.textures = [t]
            found_idx = 0
    # Collapse palette to that single texture
    blend.textures = [blend.textures[found_idx]]
    blend.textures[0].cell_start = 0  # only one entry now
    # Tiles all point at palette index 0, pattern 0
    blend.tiles = np.zeros_like(blend.tiles, dtype=np.uint16)


def strip_objects(objects_list) -> None:
    """Remove all entries from ObjectsList. Renderer's legend will be empty."""
    objects_list.map_objects = []


def flatten_heightmap(h_asset) -> None:
    """Set elevations to a constant. Use mean as the constant so seismic-zero
    issues don't surface. Also reset the raw uint16 elevations to match.
    """
    elev = h_asset.elevations.astype(np.float32)
    target = float(elev.mean())
    h_asset.elevations = np.full_like(elev, target)
    if hasattr(h_asset, "_elevations_raw") and h_asset._elevations_raw is not None:
        # Approximate: SageFloat16 stores at constant resolution; pick a
        # stable mid-range raw value.
        raw = h_asset._elevations_raw
        mid = int(np.median(raw))
        h_asset._elevations_raw = np.full_like(raw, mid)


# ---------------------------------------------------------------------------
# Variant pipeline
# ---------------------------------------------------------------------------

VariantFn = Callable[[object], None]


def _isolated_variants() -> list[tuple[str, str, list[VariantFn]]]:
    return [
        ("iso1_no_blends", "Strip BLENDS only",
         [lambda ctx: strip_blends(ctx.get_asset("BlendTileData"))]),
        ("iso2_no_textures", "Strip TEXTURES only",
         [lambda ctx: strip_textures(ctx.get_asset("BlendTileData"))]),
        ("iso3_no_objects", "Strip OBJECTS only",
         [lambda ctx: strip_objects(ctx.get_asset("ObjectsList"))]),
        ("iso4_flat", "Flatten ELEVATIONS only",
         [lambda ctx: flatten_heightmap(ctx.get_asset("HeightMapData"))]),
    ]


def _cumulative_variants() -> list[tuple[str, str, list[VariantFn]]]:
    return [
        ("cum1_blends_off", "Cumulative: blends off",
         [lambda ctx: strip_blends(ctx.get_asset("BlendTileData"))]),
        ("cum2_blends_textures_off", "Cumulative: blends + textures off",
         [lambda ctx: strip_blends(ctx.get_asset("BlendTileData")),
          lambda ctx: strip_textures(ctx.get_asset("BlendTileData"))]),
        ("cum3_blends_textures_objects_off",
         "Cumulative: blends + textures + objects off",
         [lambda ctx: strip_blends(ctx.get_asset("BlendTileData")),
          lambda ctx: strip_textures(ctx.get_asset("BlendTileData")),
          lambda ctx: strip_objects(ctx.get_asset("ObjectsList"))]),
        ("cum4_skeleton",
         "Cumulative: everything off (skeleton = flat empty plane)",
         [lambda ctx: strip_blends(ctx.get_asset("BlendTileData")),
          lambda ctx: strip_textures(ctx.get_asset("BlendTileData")),
          lambda ctx: strip_objects(ctx.get_asset("ObjectsList")),
          lambda ctx: flatten_heightmap(ctx.get_asset("HeightMapData"))]),
    ]


def all_variants() -> list[tuple[str, str, list[VariantFn]]]:
    return _isolated_variants() + _cumulative_variants()


def _save_variant(src_path: Path, fns: Iterable[VariantFn], out_path: Path,
                  compress: bool) -> "object":
    """Parse fresh, apply each strip fn, save. Returns the parsed Ra3Map's
    context so the caller can drive a minimap render without re-parsing."""
    m = Ra3Map(str(src_path))
    with _silence_stdout():
        m.parse()
    ctx = m.get_context()
    for fn in fns:
        fn(ctx)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with _silence_stdout():
        m.save(str(out_path), compress=compress)
    return ctx


def _render_minimap(ctx, out_png: Path) -> bool:
    """Render a PNG minimap using the toolkit's minimap_generator."""
    try:
        with _silence_stdout():
            img = generate_minimap(ctx)
        if img is None:
            return False
        out_png.parent.mkdir(parents=True, exist_ok=True)
        with _silence_stdout():
            img.save(str(out_png), format='PNG')
        return True
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def skin_one_map(
    src_path: Path,
    out_dir: Path,
    *,
    compress: bool = True,
    render: bool = True,
) -> tuple[int, int]:
    """
    Produce all 8 stripped variants for a single source map.

    Outputs are written under: out_dir / <src_stem>_skinned/ ...
    Each variant becomes <name>.map and (if render) <name>.png alongside it.
    The original is also rendered as 00_original.png for reference.

    Returns (success_count, fail_count).
    """
    src_path = Path(src_path)
    if not src_path.exists():
        raise FileNotFoundError(f"Source not found: {src_path}")

    out_root = out_dir / f"{src_path.stem}_skinned"
    out_root.mkdir(parents=True, exist_ok=True)

    variants = all_variants()
    total = len(variants)
    emit("skin_start", source=str(src_path), output=str(out_root), total=total)

    # Render the original first as a reference (best-effort, never fatal).
    if render:
        try:
            ref = Ra3Map(str(src_path))
            with _silence_stdout():
                ref.parse()
            _render_minimap(ref.get_context(), out_root / "00_original.png")
        except Exception:
            pass

    success = 0
    fail = 0
    for idx, (name, desc, fns) in enumerate(variants, start=1):
        out_map = out_root / f"{name}.map"
        emit("skin_variant_start", index=idx, total=total, name=name,
             description=desc)
        try:
            emit("skin_step", name=name, step="parse_and_strip")
            ctx = _save_variant(src_path, fns, out_map, compress=compress)

            if render:
                emit("skin_step", name=name, step="render")
                _render_minimap(ctx, out_root / f"{name}.png")

            emit("skin_variant_complete", index=idx, total=total,
                 name=name, success=True, output=str(out_map))
            success += 1
        except Exception as e:
            emit("skin_variant_complete", index=idx, total=total,
                 name=name, success=False, error=str(e))
            fail += 1

    emit("skin_done", success=success, fail=fail, total=total,
         output=str(out_root))
    return success, fail


def skin_directory(
    src_dir: Path,
    out_dir: Path,
    *,
    compress: bool = True,
    render: bool = True,
) -> tuple[int, int]:
    """Skin every .map under src_dir (recursive). Toolkit-output folders
    (already-converted / already-skinned) are skipped to avoid feedback loops.
    Returns (total_success, total_fail) summed across maps."""
    total_success = 0
    total_fail = 0
    for src in sorted(src_dir.rglob("*.map")):
        n = src.name.lower()
        if "[archon]" in n or "_skinned" in str(src.parent).lower():
            continue
        if any(n.endswith(f"_{s}.map") for s in (
            "nosw", "noair", "nosw_noair", "inf_only", "tanks_only")):
            continue
        s, f = skin_one_map(src, out_dir, compress=compress, render=render)
        total_success += s
        total_fail += f
    return total_success, total_fail


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="skin_map",
        description="Decompose an RA3 map into 4 isolated + 4 cumulative stripped variants.",
    )
    p.add_argument("--src", required=True,
                   help="Source .map file or directory containing .map files.")
    p.add_argument("--out-dir", default=None,
                   help="Output root folder. Defaults to ../converted_maps "
                        "relative to the toolkit _internal/ directory.")
    p.add_argument("--no-render", action="store_true",
                   help="Skip PNG minimap rendering.")
    p.add_argument("--no-compress", action="store_true",
                   help="Write uncompressed .map output.")
    p.add_argument("--json-progress", action="store_true",
                   help="Emit JSON line events on stdout (machine-readable).")
    return p.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    global JSON_MODE
    args = parse_args(argv if argv is not None else sys.argv[1:])
    JSON_MODE = args.json_progress

    src = Path(args.src)
    if not src.exists():
        emit("fatal", error=f"Source not found: {src}")
        return 1

    if args.out_dir:
        out_dir = Path(args.out_dir)
    else:
        out_dir = _ROOT.parent / "converted_maps"
    out_dir.mkdir(parents=True, exist_ok=True)

    compress = not args.no_compress
    render = not args.no_render

    try:
        if src.is_dir():
            success, fail = skin_directory(src, out_dir,
                                           compress=compress, render=render)
        else:
            success, fail = skin_one_map(src, out_dir,
                                         compress=compress, render=render)
        return 0 if fail == 0 else 1
    except Exception as e:
        emit("fatal", error=f"{e}\n{traceback.format_exc()}")
        return 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        emit("fatal", error="Cancelled by user")
        sys.exit(1)
