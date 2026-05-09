"""
Rotate / flip an RA3 .map file (or a folder of them) and save each variant.

Pipeline modes (-m via batch_convert.py --mode rotate):
  rot90cw    rotate 90 degrees clockwise
  rot90ccw   rotate 90 degrees counter-clockwise
  rot180     rotate 180 degrees
  flipx      flip across X axis (top<->bottom mirror); cliff-wall meshes get
             a +180 deg fixup so the textured side faces the new cliff edge.
  flipy      flip across Y axis (left<->right mirror); same fixup.

Multiple ops can be requested in a single run (CSV list); each one writes its
own output map.

Emits one JSON line per progress event when --json-progress is set:

  {"event": "rotate_start", "source": <path>, "output": <dir>, "total_ops": int}
  {"event": "rotate_op_start", "index": int, "total": int, "op": str, "source": str}
  {"event": "rotate_step", "op": str, "step": "parse"|"rotate"|"save"|"tga"}
  {"event": "rotate_op_complete", "index": int, "total": int, "op": str,
                                  "success": bool, "output"?: str, "error"?: str}
  {"event": "rotate_done", "success": int, "fail": int, "total": int, "output": str}
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
from typing import Iterable, List, Optional


def get_app_root() -> Path:
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parent


_ROOT = get_app_root()
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from map_processor.core.ra3map import Ra3Map  # noqa: E402
from map_processor.utils.map_rotation import (  # noqa: E402
    rotate_context_right_angles,
    flip_context_axis,
)


VALID_OPS = ("rot90cw", "rot90ccw", "rot180", "flipx", "flipy")


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
    if event == "rotate_start":
        print(f"Rotating {fields.get('source')} -> {fields.get('output')} "
              f"({fields.get('total_ops')} op(s))")
    elif event == "rotate_op_start":
        print(f"\n[{fields['index']}/{fields['total']}] {fields['op']}  "
              f"({fields.get('source','')})")
    elif event == "rotate_step":
        print(f"  - {fields['step']}")
    elif event == "rotate_op_complete":
        if fields.get("success"):
            print(f"  [OK] -> {fields.get('output','')}")
        else:
            print(f"  [FAIL] {fields.get('error','')}")
    elif event == "rotate_done":
        print(f"\nDone. success={fields.get('success',0)} "
              f"fail={fields.get('fail',0)}")
    elif event == "fatal":
        print(f"FATAL: {fields.get('error','')}", file=sys.stderr)


def _apply(ctx, op: str) -> None:
    if op == "rot90cw":
        rotate_context_right_angles(ctx, degrees=90, clockwise=True)
    elif op == "rot90ccw":
        rotate_context_right_angles(ctx, degrees=90, clockwise=False)
    elif op == "rot180":
        rotate_context_right_angles(ctx, degrees=180, clockwise=True)
    elif op == "flipx":
        flip_context_axis(ctx, axis="x")
    elif op == "flipy":
        flip_context_axis(ctx, axis="y")
    else:
        raise ValueError(f"unknown op {op!r}; valid: {VALID_OPS}")


def _output_paths(src: Path, out_root: Path, op: str) -> tuple[Path, Path]:
    """
    Folder/file naming mirrors the convert mode: one folder per output map,
    `<stem>_<op>` for both the folder and the .map inside it.
    """
    stem = f"{src.stem}_{op}"
    out_dir = out_root / stem
    out_map = out_dir / f"{stem}.map"
    return out_dir, out_map


def _copy_preview_tga(src: Path, dst_dir: Path, dst_stem: str) -> None:
    preview = src.parent / f"{src.stem}.tga"
    if preview.exists():
        try:
            shutil.copy2(preview, dst_dir / f"{dst_stem}.tga")
        except Exception:
            pass


def rotate_one_op(src: Path, out_root: Path, op: str, *,
                  index: int, total: int,
                  compress: bool = True,
                  copy_tga: bool = True) -> bool:
    out_dir, out_map = _output_paths(src, out_root, op)
    emit("rotate_op_start", index=index, total=total, op=op, source=src.name)
    try:
        emit("rotate_step", op=op, step="parse")
        with _silence_stdout():
            m = Ra3Map(str(src))
            m.parse()

        emit("rotate_step", op=op, step="rotate")
        with _silence_stdout():
            _apply(m.get_context(), op)

        emit("rotate_step", op=op, step="save")
        out_dir.mkdir(parents=True, exist_ok=True)
        with _silence_stdout():
            m.save(str(out_map), compress=compress)

        if copy_tga:
            emit("rotate_step", op=op, step="tga")
            _copy_preview_tga(src, out_dir, out_map.stem)

        emit("rotate_op_complete", index=index, total=total, op=op,
             success=True, output=str(out_dir))
        return True
    except Exception as e:
        emit("rotate_op_complete", index=index, total=total, op=op,
             success=False, error=str(e))
        return False


def rotate_one_source(src: Path, out_root: Path, ops: List[str], *,
                      compress: bool = True, copy_tga: bool = True,
                      index_offset: int = 0,
                      total_override: Optional[int] = None) -> tuple[int, int]:
    success = 0
    fail = 0
    total = total_override if total_override is not None else len(ops)
    for i, op in enumerate(ops):
        ok = rotate_one_op(src, out_root, op,
                           index=index_offset + i + 1, total=total,
                           compress=compress, copy_tga=copy_tga)
        if ok:
            success += 1
        else:
            fail += 1
    return success, fail


def find_source_maps(folder: Path) -> List[Path]:
    """Find candidate source maps under folder. Skips toolkit outputs."""
    out: List[Path] = []
    for p in sorted(folder.rglob("*.map")):
        n = p.name.lower()
        if "[archon]" in n:
            continue
        if any(n.endswith(f"_{s}.map") for s in VALID_OPS):
            continue
        if any(n.endswith(f"_{s}.map") for s in (
                "nosw", "noair", "nosw_noair", "inf_only", "tanks_only")):
            continue
        if "template" in str(p).lower():
            continue
        out.append(p)
    return out


def rotate_directory(src_dir: Path, out_root: Path, ops: List[str], *,
                     compress: bool = True,
                     copy_tga: bool = True) -> tuple[int, int]:
    sources = find_source_maps(src_dir)
    total = len(sources) * len(ops)
    if total == 0:
        emit("rotate_done", success=0, fail=0, total=0, output=str(out_root))
        return 0, 0
    s_total = f_total = 0
    for i, src in enumerate(sources):
        s, f = rotate_one_source(
            src, out_root, ops,
            compress=compress, copy_tga=copy_tga,
            index_offset=i * len(ops),
            total_override=total,
        )
        s_total += s; f_total += f
    return s_total, f_total


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv: List[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="rotate_engine",
        description="Rotate / flip RA3 .map files. Multiple ops per source supported.",
    )
    p.add_argument("--src", required=True,
                   help="Source .map file or directory (recursive).")
    p.add_argument("--out-dir", default=None,
                   help="Output root folder. Defaults to ../converted_maps.")
    p.add_argument("--ops", default="rot90cw",
                   help="Comma-separated ops (any of: " + ",".join(VALID_OPS) + ")")
    p.add_argument("--no-compress", action="store_true")
    p.add_argument("--no-copy-tga", action="store_true",
                   help="Do not copy <stem>.tga preview next to outputs.")
    p.add_argument("--json-progress", action="store_true")
    return p.parse_args(argv)


def _resolve_ops(arg: str) -> List[str]:
    out: List[str] = []
    for v in arg.split(","):
        k = v.strip().lower()
        if not k:
            continue
        if k not in VALID_OPS:
            raise ValueError(f"unknown op {k!r}; valid: {VALID_OPS}")
        if k not in out:
            out.append(k)
    if not out:
        raise ValueError("no ops requested")
    return out


def main(argv: Optional[List[str]] = None) -> int:
    global JSON_MODE
    args = parse_args(argv if argv is not None else sys.argv[1:])
    JSON_MODE = args.json_progress

    try:
        ops = _resolve_ops(args.ops)
    except ValueError as e:
        emit("fatal", error=str(e))
        return 1

    src = Path(args.src)
    if not src.exists():
        emit("fatal", error=f"Source not found: {src}")
        return 1

    out_root = Path(args.out_dir) if args.out_dir else _ROOT.parent / "converted_maps"
    out_root.mkdir(parents=True, exist_ok=True)

    compress = not args.no_compress
    copy_tga = not args.no_copy_tga

    try:
        if src.is_dir():
            sources = find_source_maps(src)
            total = len(sources) * len(ops)
            emit("rotate_start", source=str(src), output=str(out_root),
                 total_ops=total)
            success, fail = rotate_directory(
                src, out_root, ops, compress=compress, copy_tga=copy_tga)
        else:
            total = len(ops)
            emit("rotate_start", source=str(src), output=str(out_root),
                 total_ops=total)
            success, fail = rotate_one_source(
                src, out_root, ops, compress=compress, copy_tga=copy_tga)
        emit("rotate_done", success=success, fail=fail, total=success + fail,
             output=str(out_root))
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
