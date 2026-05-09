"""
RA3 Map Toolkit batch engine.

Two transforms — both can be applied to the same source independently or in combination:
  Archon transform        Pair builder/controller into 2-6 player Archon-style maps.
                          Sources must be 1-3 player.
  Match restrictions      Apply 1-of-5 unit-restriction script sets that ban
                          superweapons, aircraft, vehicles or infantry from
                          being built. Donor scripts originally from Jenkins's
                          OMV pack (jenkinsmedia.com.au).

When both are enabled, archon runs first (changes player layout) then the
restriction scripts are appended. Output filenames combine both:
`[Archon]<Stem>_<suffix>.map`.

Two execution modes:
  --scan-only   Discover maps and emit metadata; do not convert.
  (default)     Convert maps. Pass --maps to limit to a subset.

With --json-progress, every status is emitted as a JSON line on stdout. The
termcn TUI (`ra3_map_toolkit.exe`) drives this. Without --json-progress the
output is plain human-readable text.

Usage:
  ra3_engine.exe --apply-archon [INPUT] [OUTPUT]
  ra3_engine.exe --restrictions nosw,noair [INPUT] [OUTPUT]
  ra3_engine.exe --apply-archon --restrictions nosw [INPUT] [OUTPUT]
  ra3_engine.exe --scan-only [INPUT]

Common options:
  --no-compress                Write uncompressed output.
  --no-sidecars                Skip XML sidecars (map.xml + overrides.xml).
  --maps name1.map name2.map   Only convert listed map filenames (basename match).

Archon-only options (ignored unless --apply-archon):
  --wb-normalize-terrain       WorldBuilder terrain canonicalization.
  --offset N                   Controller spawn offset (default 800).

Restriction options:
  --restrictions CSV           Comma-separated keys: nosw,noair,nosw_noair,inf_only,tanks_only.
                               If empty/unset and --apply-archon is set, runs archon-only.
                               If empty/unset and --apply-archon is NOT set, errors.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import shutil
import sys
from pathlib import Path
from typing import Optional


def get_app_root() -> Path:
    if getattr(sys, 'frozen', False):
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parent


_ROOT = get_app_root()
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from map_processor.core.ra3map import Ra3Map
from transform_to_archon import (
    transform_to_archon,
    find_player_starts,
    are_maps_same_base,
    _is_paired_archon_3p_template,
    _generate_art_tga,
    _write_sidecars,
)
from transform_to_omv import (
    OMV_VARIATIONS,
    get_template_path as omv_template_path,
    transform_to_omv,
    derive_display_name,
    variant_filename,
)
# Imported so PyInstaller bundles skin_map.py into ra3_engine.exe.
# Skin mode is dispatched via `--mode skin` (see main()).
import skin_map  # noqa: E402, F401
# Same for rotate mode (--mode rotate).
import rotate_engine  # noqa: E402, F401
# Same for compose mode (--mode compose). Composition is a strict superset
# of duplication; duplicate is just `--compose-preset duplicate --compose-nx N --compose-ny M`.
import compose_engine  # noqa: E402, F401


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

    if event == "scan_start":
        print("Scanning for .map files...")
    elif event == "scan_progress":
        name = fields.get("name", "?")
        players = fields.get("players")
        reason = fields.get("reason", "")
        if players is None:
            print(f"  {name}: SKIPPED ({reason})")
        else:
            print(f"  {name}: {players} player(s)")
    elif event == "scan_complete":
        print(f"Found {fields.get('total', 0)} convertible map(s)")
    elif event == "convert_start":
        variation = fields.get("variation")
        suffix = f" [{variation}]" if variation else ""
        print(f"\n[{fields['index']}/{fields['total']}] {fields['name']}{suffix}")
    elif event == "convert_step":
        print(f"  - {fields['step']}")
    elif event == "convert_complete":
        if fields.get("success"):
            print(f"  [OK] {fields['name']} -> {fields.get('output', '')}")
        else:
            print(f"  [FAIL] {fields['name']}: {fields.get('error', '')}")
    elif event == "done":
        print(
            f"\nDone. Success: {fields.get('success', 0)}  "
            f"Fail: {fields.get('fail', 0)}  Skipped: {fields.get('skipped', 0)}"
        )
    elif event == "fatal":
        print(f"FATAL: {fields.get('error', '')}", file=sys.stderr)


def get_player_count(map_path: Path) -> Optional[int]:
    """Detect the number of builder players in a map. Returns None if unparseable."""
    try:
        with _silence_stdout():
            ra3map = Ra3Map(str(map_path))
            ra3map.parse()
            context = ra3map.get_context()

        builder_count = 0
        objs = context.get_asset("ObjectsList")
        if objs:
            for obj in objs.map_objects:
                unique_id = getattr(obj, 'unique_id', None)
                if unique_id and 'Player_' in unique_id and '_Start' in unique_id:
                    try:
                        num = int(unique_id.split('Player_')[1].split('_')[0])
                        if num > builder_count:
                            builder_count = num
                    except (ValueError, IndexError):
                        pass

        return builder_count if builder_count > 0 else None
    except Exception:
        return None


def find_maps(folder: Path) -> list[Path]:
    """Recursively find all convertible .map files in folder."""
    maps = []
    for path in folder.rglob("*.map"):
        n = path.name.lower()
        if "[archon]" in n:
            continue
        # Skip restriction-suffixed inputs so re-runs don't reprocess toolkit outputs.
        if any(n.endswith(f"_{s}.map") for s in (
            "nosw", "noair", "nosw_noair", "inf_only", "tanks_only")):
            continue
        if "template" in str(path).lower():
            continue
        maps.append(path)
    return maps


def get_map_display_name(map_path: Path) -> str:
    name = map_path.stem
    for prefix in ["map_mp_", "map_"]:
        if name.lower().startswith(prefix):
            name = name[len(prefix):]
    return name


def archon_name_for(map_path: Path) -> str:
    display_name = get_map_display_name(map_path)
    clean_name = "_".join(
        word.capitalize()
        for word in display_name.replace("-", "_").split("_")
        if word
    )
    return f"[Archon]{clean_name}"


def _output_stem(map_path: Path, apply_archon: bool, variation: Optional[str]) -> str:
    """
    Compose the output folder/file stem for the requested combination.

      archon-only          [Archon]Cabana_Republic
      restriction-only     map_mp_2_feasel1_nosw
      combined             [Archon]Cabana_Republic_nosw
    """
    if apply_archon and variation:
        suffix = OMV_VARIATIONS[variation]["filename_suffix"]
        return f"{archon_name_for(map_path)}_{suffix}"
    if apply_archon:
        return archon_name_for(map_path)
    if variation:
        return variant_filename(map_path, variation)
    raise ValueError("convert_map called with neither archon nor variation")


def convert_map(
    map_path: Path,
    output_folder: Path,
    index: int,
    total: int,
    *,
    apply_archon: bool,
    variation: Optional[str],
    template_2p: Optional[Path],
    template_3p: Optional[Path],
    player_count: Optional[int],
    compress: bool,
    write_sidecars: bool,
    wb_normalize_terrain: bool,
) -> tuple[bool, str, str]:
    """
    Apply any combination of {archon, match-restriction} to one source map.

    `player_count` is required (and respected for template selection) when
    apply_archon is True; otherwise it can be None.
    """
    if not apply_archon and not variation:
        return False, "", "convert_map called with neither archon nor variation"

    out_stem = _output_stem(map_path, apply_archon, variation)
    out_dir = output_folder / out_stem
    out_map = out_dir / f"{out_stem}.map"

    archon_template: Optional[Path] = None
    omv_donor: Optional[Path] = None

    emit(
        "convert_start",
        name=map_path.name,
        archon_name=out_stem,
        variation=variation or "",
        index=index,
        total=total,
        player_count=player_count or 0,
        apply_archon=apply_archon,
    )

    try:
        emit("convert_step", name=map_path.name, step="parse_source")
        with _silence_stdout():
            source_map = Ra3Map(str(map_path))
            source_map.parse()
            source_ctx = source_map.get_context()

        # ----- Archon transform -----
        if apply_archon:
            if player_count is None:
                player_count = len(find_player_starts(source_ctx)) or 2
            archon_template = template_2p if player_count <= 2 else template_3p
            if archon_template is None:
                raise RuntimeError("archon template missing for this player count")

            emit("convert_step", name=map_path.name, step="parse_archon_template")
            with _silence_stdout():
                template_map = Ra3Map(str(archon_template))
                template_map.parse()
                template_ctx = template_map.get_context()

                num_builders = len(find_player_starts(source_ctx))
                same_base = are_maps_same_base(source_ctx, template_ctx)
                paired_3p = num_builders == 3 and _is_paired_archon_3p_template(template_ctx)

            # Fast path: 3p source whose template IS the same map already paired —
            # only valid when no further restriction transform is requested.
            if same_base and paired_3p and not variation:
                out_dir.mkdir(parents=True, exist_ok=True)
                emit("convert_step", name=map_path.name, step="copy_template",
                     detail="same-base 3p paired")
                with _silence_stdout():
                    shutil.copy2(archon_template, out_map)
                emit("convert_step", name=map_path.name, step="minimap")
                with _silence_stdout():
                    _generate_art_tga(out_map, source_ctx, source_map_path=map_path)
                if write_sidecars:
                    emit("convert_step", name=map_path.name, step="sidecars")
                    with _silence_stdout():
                        _write_sidecars(out_map, source_ctx, archon_template,
                                        source_map_path=map_path)
                emit("convert_complete", name=map_path.name, archon_name=out_stem,
                     success=True, output=str(out_dir))
                return True, str(out_dir), ""

            emit("convert_step", name=map_path.name, step="archon_transform")
            with _silence_stdout():
                transform_to_archon(
                    source_ctx,
                    template_ctx,
                    wb_normalize_terrain=wb_normalize_terrain,
                )

        # ----- Restriction transform (runs on top of archon-modified context if both) -----
        if variation:
            omv_donor = omv_template_path(variation)
            if not omv_donor.exists():
                raise FileNotFoundError(f"Restriction donor template missing: {omv_donor}")

            emit("convert_step", name=map_path.name, step="parse_restriction_donor")
            with _silence_stdout():
                donor = Ra3Map(str(omv_donor))
                donor.parse()
                donor_ctx = donor.get_context()

            wi = source_ctx.get_asset("WorldInfo")
            existing = wi.properties.get_property("mapName") if wi else None
            base_display = (existing.data if existing else "") or derive_display_name(map_path)
            # If archon already ran, prefer the archon name as the display base so
            # the restriction suffix lands cleanly on top: "[Archon]Cabana ..." -> "[Archon]Cabana ... [NO SW]".
            if apply_archon:
                base_display = archon_name_for(map_path)

            emit("convert_step", name=map_path.name, step="restriction_transform")
            with _silence_stdout():
                transform_to_omv(source_ctx, donor_ctx, base_display, variation)

        # ----- Save + minimap + sidecars -----
        out_dir.mkdir(parents=True, exist_ok=True)

        emit("convert_step", name=map_path.name, step="save")
        with _silence_stdout():
            source_map.save(str(out_map), compress=compress)

        emit("convert_step", name=map_path.name, step="minimap")
        with _silence_stdout():
            _generate_art_tga(out_map, source_ctx, source_map_path=map_path)
            preview = map_path.parent / f"{map_path.stem}.tga"
            if preview.exists():
                shutil.copy2(preview, out_dir / f"{out_stem}.tga")

        if write_sidecars:
            sidecar_template = archon_template or omv_donor
            if sidecar_template is not None:
                emit("convert_step", name=map_path.name, step="sidecars")
                with _silence_stdout():
                    _write_sidecars(out_map, source_ctx, sidecar_template,
                                    source_map_path=map_path)

        emit("convert_complete", name=map_path.name, archon_name=out_stem,
             variation=variation or "", success=True, output=str(out_dir))
        return True, str(out_dir), ""

    except Exception as e:
        err = str(e)
        emit("convert_complete", name=map_path.name, archon_name=out_stem,
             variation=variation or "", success=False, error=err)
        return False, str(out_dir), err


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="ra3_engine",
        description="Batch transform RA3 maps (Archon and match-restriction transforms, independently or combined).",
    )
    parser.add_argument("input", nargs="?", default=None, help="Input folder (default: ../maps_to_convert)")
    parser.add_argument("output", nargs="?", default=None, help="Output folder (default: ../converted_maps)")
    parser.add_argument("--mode",
                        choices=["convert", "skin", "rotate", "compose"],
                        default="convert",
                        help="Pipeline mode. 'convert' (default) runs Archon/restriction "
                             "transforms; 'skin' decomposes a single source into "
                             "isolated + cumulative layer strips; 'rotate' applies one "
                             "or more right-angle rotations / flips; 'compose' stitches "
                             "1..3 source maps into a single output via a layout preset "
                             "(duplication is `--compose-preset duplicate`).")
    parser.add_argument("--skin-source", default=None,
                        help="(skin mode) Path to a single source .map to decompose. "
                             "If omitted, skins every map under the input folder.")
    parser.add_argument("--no-render", action="store_true",
                        help="(skin mode) Skip PNG minimap rendering.")
    parser.add_argument("--rotate-source", default=None,
                        help="(rotate mode) Path to a single source .map to rotate. "
                             "If omitted, rotates every map under the input folder.")
    parser.add_argument("--rotate-ops", default="rot90cw",
                        help="(rotate mode) Comma-separated ops: any of "
                             "rot90cw,rot90ccw,rot180,flipx,flipy.")
    parser.add_argument("--compose-preset", default="duplicate",
                        choices=("duplicate", "row", "col",
                                 "triangle_top", "triangle_bottom",
                                 "triangle_left", "triangle_right"),
                        help="(compose mode) Layout preset.")
    parser.add_argument("--compose-maps", nargs="+", default=None,
                        help="(compose mode) Source .map paths in slot order. "
                             "For 'duplicate' pass exactly one; for 'row'/'col' pass 2-3; "
                             "for any 'triangle_*' pass exactly 3.")
    parser.add_argument("--compose-nx", type=int, default=2,
                        help="(compose mode, duplicate preset) tile count along X.")
    parser.add_argument("--compose-ny", type=int, default=1,
                        help="(compose mode, duplicate preset) tile count along Y.")
    parser.add_argument("--compose-pad-x", type=int, default=0,
                        help="(compose mode) tile-unit gap between adjacent X cells.")
    parser.add_argument("--compose-pad-y", type=int, default=0,
                        help="(compose mode) tile-unit gap between adjacent Y cells.")
    parser.add_argument("--compose-align-x", default=None,
                        help="(compose mode) comma-separated per-slot horizontal alignment "
                             "(left|center|right). Affects placement when a source's playable "
                             "area is smaller than its allocated cell.")
    parser.add_argument("--compose-align-y", default=None,
                        help="(compose mode) comma-separated per-slot vertical alignment "
                             "(top|center|bottom).")
    parser.add_argument("--apply-archon", action="store_true",
                        help="Apply Archon paired-builder/controller transform.")
    parser.add_argument("--json-progress", action="store_true",
                        help="Emit JSON line events on stdout (machine-readable)")
    parser.add_argument("--scan-only", action="store_true",
                        help="Discover maps and exit; do not convert")
    parser.add_argument("--maps", nargs="*", default=None,
                        help="Subset of map basenames to convert")
    parser.add_argument("--no-compress", action="store_true",
                        help="Write uncompressed output")
    parser.add_argument("--no-sidecars", action="store_true",
                        help="Skip writing XML sidecars (map.xml + overrides.xml)")
    parser.add_argument("--wb-normalize-terrain", action="store_true",
                        help="(Archon only) WorldBuilder terrain canonicalization")
    parser.add_argument("--offset", type=float, default=800.0,
                        help="(Archon only) controller spawn offset in world units")
    parser.add_argument("--restrictions", default=None,
                        help="Comma-separated match-restriction keys: nosw,noair,nosw_noair,inf_only,tanks_only")
    return parser.parse_args(argv)


def scan(input_folder: Path, apply_archon: bool) -> tuple[list[tuple[Path, int]], list[dict]]:
    """
    Scan input_folder for convertible maps.

    Returns (accepted, skipped). `accepted` is [(path, player_count), ...].
    Archon-enabled scans drop maps with player_count > 3 (no template available).
    """
    emit("scan_start")
    accepted: list[tuple[Path, int]] = []
    skipped: list[dict] = []

    for map_path in find_maps(input_folder):
        player_count = get_player_count(map_path)
        if player_count is None:
            skipped.append({"name": map_path.name, "reason": "Could not parse"})
            emit("scan_progress", name=map_path.name, players=None, reason="Could not parse")
            continue

        if apply_archon and player_count > 3:
            skipped.append({"name": map_path.name,
                            "reason": f"{player_count} players (max 3 for archon)"})
            emit("scan_progress", name=map_path.name, players=player_count,
                 reason="too many players for archon")
            continue

        accepted.append((map_path, player_count))
        emit("scan_progress", name=map_path.name, path=str(map_path),
             players=player_count, reason="convertible")

    p2 = sum(1 for _, n in accepted if n <= 2)
    p3 = sum(1 for _, n in accepted if n == 3)
    emit("scan_complete", total=len(accepted), by_2p=p2, by_3p=p3, skipped=skipped)
    return accepted, skipped


def _resolve_variations(arg: Optional[str]) -> list[str]:
    if not arg:
        return []
    out: list[str] = []
    for v in arg.split(","):
        k = v.strip().lower()
        if not k:
            continue
        if k not in OMV_VARIATIONS:
            raise ValueError(f"Unknown match restriction {k!r}; valid: {list(OMV_VARIATIONS)}")
        if k not in out:
            out.append(k)
    return out


def main() -> int:
    global JSON_MODE
    args = parse_args(sys.argv[1:])
    JSON_MODE = args.json_progress

    user_root = _ROOT.parent

    input_folder = Path(args.input) if args.input else user_root / "maps_to_convert"
    output_folder = Path(args.output) if args.output else user_root / "converted_maps"

    input_folder.mkdir(parents=True, exist_ok=True)
    if not args.scan_only:
        output_folder.mkdir(parents=True, exist_ok=True)

    # ---- Skin mode: dispatch to skin_map and return early. ----
    # We share the JSON_MODE flag so events on stdout stream cleanly.
    if args.mode == "skin":
        skin_map.JSON_MODE = JSON_MODE
        render = not args.no_render
        compress = not args.no_compress
        try:
            if args.skin_source:
                src = Path(args.skin_source)
                if not src.exists():
                    emit("fatal", error=f"Skin source not found: {src}")
                    return 1
                success, fail = skin_map.skin_one_map(
                    src, output_folder, compress=compress, render=render)
            else:
                success, fail = skin_map.skin_directory(
                    input_folder, output_folder,
                    compress=compress, render=render)
            return 0 if fail == 0 else 1
        except Exception as e:
            import traceback as _tb
            emit("fatal", error=f"{e}\n{_tb.format_exc()}")
            return 1

    # ---- Rotate mode: dispatch to rotate_engine and return early. ----
    if args.mode == "rotate":
        rotate_engine.JSON_MODE = JSON_MODE
        compress = not args.no_compress
        try:
            ops = rotate_engine._resolve_ops(args.rotate_ops)
        except ValueError as e:
            emit("fatal", error=str(e))
            return 1
        try:
            if args.rotate_source:
                src = Path(args.rotate_source)
                if not src.exists():
                    emit("fatal", error=f"Rotate source not found: {src}")
                    return 1
                total = len(ops)
                rotate_engine.emit("rotate_start", source=str(src),
                                   output=str(output_folder), total_ops=total)
                success, fail = rotate_engine.rotate_one_source(
                    src, output_folder, ops, compress=compress)
            else:
                sources = rotate_engine.find_source_maps(input_folder)
                total = len(sources) * len(ops)
                rotate_engine.emit("rotate_start", source=str(input_folder),
                                   output=str(output_folder), total_ops=total)
                success, fail = rotate_engine.rotate_directory(
                    input_folder, output_folder, ops, compress=compress)
            rotate_engine.emit("rotate_done", success=success, fail=fail,
                               total=success + fail, output=str(output_folder))
            return 0 if fail == 0 else 1
        except Exception as e:
            import traceback as _tb
            emit("fatal", error=f"{e}\n{_tb.format_exc()}")
            return 1

    # ---- Compose mode: dispatch to compose_engine and return early. ----
    if args.mode == "compose":
        compose_engine.JSON_MODE = JSON_MODE
        compress = not args.no_compress
        try:
            if not args.compose_maps:
                raise ValueError("--compose-maps is required in compose mode")
            maps = [Path(m) for m in args.compose_maps]
            for m in maps:
                if not m.exists():
                    raise ValueError(f"compose source not found: {m}")
            compose_engine.emit(
                "compose_start",
                preset=args.compose_preset,
                maps=[str(m) for m in maps],
                output=str(output_folder),
                total_steps=5,
            )
            ax = [s.strip() for s in args.compose_align_x.split(",")] if args.compose_align_x else None
            ay = [s.strip() for s in args.compose_align_y.split(",")] if args.compose_align_y else None
            ok = compose_engine.compose_one(
                args.compose_preset, maps, output_folder,
                nx=int(args.compose_nx), ny=int(args.compose_ny),
                pad_x=int(args.compose_pad_x), pad_y=int(args.compose_pad_y),
                align_x=ax, align_y=ay,
                compress=compress, copy_tga=True,
            )
            compose_engine.emit(
                "compose_done", success=1 if ok else 0,
                fail=0 if ok else 1, total=1, output=str(output_folder))
            return 0 if ok else 1
        except Exception as e:
            import traceback as _tb
            emit("fatal", error=f"{e}\n{_tb.format_exc()}")
            return 1

    try:
        variations = _resolve_variations(args.restrictions)
    except ValueError as e:
        emit("fatal", error=str(e))
        return 1

    apply_archon: bool = args.apply_archon

    if not args.scan_only and not apply_archon and not variations:
        emit("fatal", error="Nothing to do: pass --apply-archon and/or --restrictions.")
        return 1

    emit("start",
         input=str(input_folder),
         output=str(output_folder),
         scan_only=args.scan_only,
         apply_archon=apply_archon,
         variations=variations)

    template_2p: Optional[Path] = None
    template_3p: Optional[Path] = None
    if apply_archon:
        template_2p = _ROOT / "templates" / "2p" / "Archon Fire Island [1.4].map"
        template_3p = _ROOT / "templates" / "3p" / "[Archon]Hidden_Fortress_1.2.map"
        if not template_2p.exists():
            emit("fatal", error=f"2-player archon template not found: {template_2p}")
            return 1
        if not template_3p.exists():
            emit("fatal", error=f"3-player archon template not found: {template_3p}")
            return 1

    if variations:
        omv_dir = _ROOT / "templates" / "omv"
        if not omv_dir.exists():
            emit("fatal", error=f"Restriction donor templates folder not found: {omv_dir}")
            return 1

    accepted, skipped = scan(input_folder, apply_archon)

    if args.scan_only:
        emit("done", success=0, fail=0, skipped=len(skipped),
             output=str(output_folder), scan_only=True)
        return 0

    if args.maps is not None:
        wanted = {m.lower() for m in args.maps}
        accepted = [(p, n) for p, n in accepted if p.name.lower() in wanted]

    # Each source produces 1 (archon-only or omv-only) or N variations (omv with
    # or without archon). When variations is empty but archon is on, run once per source.
    runs_per_source = max(1, len(variations))
    convertible = len(accepted) * runs_per_source
    if convertible == 0:
        emit("done", success=0, fail=0, skipped=len(skipped), output=str(output_folder))
        return 0

    convert_kwargs = {
        "compress": not args.no_compress,
        "write_sidecars": not args.no_sidecars,
        "wb_normalize_terrain": args.wb_normalize_terrain,
        "template_2p": template_2p,
        "template_3p": template_3p,
        "apply_archon": apply_archon,
    }

    iter_variations: list[Optional[str]] = [v for v in variations] if variations else [None]

    success_count = 0
    fail_count = 0
    idx = 0
    for src_path, players in accepted:
        for variation in iter_variations:
            idx += 1
            ok, _, _ = convert_map(
                src_path, output_folder, idx, convertible,
                variation=variation,
                player_count=players,
                **convert_kwargs,
            )
            if ok:
                success_count += 1
            else:
                fail_count += 1

    emit("done", success=success_count, fail=fail_count, skipped=len(skipped),
         output=str(output_folder))
    return 0 if fail_count == 0 else 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        emit("fatal", error="Cancelled by user")
        sys.exit(1)
    except Exception as e:
        import traceback
        emit("fatal", error=f"{e}\n{traceback.format_exc()}")
        sys.exit(1)
