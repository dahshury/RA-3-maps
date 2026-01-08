"""
Verify that a rotated RA3 map matches the expected right-angle rotation of an original map.

This is a diagnostic tool: it does NOT modify maps.

Usage:
  python scripts/verify_map_rotation.py --orig "<orig.map>" --rot "<rot.map>" --degrees 180

What it checks (when assets are implemented in Python):
- HeightMapData elevations (and raw elevations)
- BlendTileData: tiles (via GetTexture/GetTile semantics), blends + singleEdgeBlends (via blend_info remap),
  passability + bool grids
- ObjectsList: per-object (by unique_id) position + angle
- Water/river/wave/trigger polygons: point-by-point rotation
- ScriptArgument (type 16) positions if scripts are present

What it cannot fully check:
- DefaultMajorAsset blocks (raw binary) such as NamedCameras/CameraAnimationList/WaypointsList in this repo.
  It will report them as "unverified" and whether the bytes are unchanged vs original.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# Allow running this file directly (python scripts/verify_map_rotation.py)
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from map_processor.ra3map import Ra3Map
from map_processor.core.default_major_asset import DefaultMajorAsset
from map_processor.assets.objects.objects_list import ObjectsList
from map_processor.assets.terrain.height_map_data import HeightMapData
from map_processor.assets.terrain.blend_tile_data import BlendTileData
from map_processor.assets.terrain.blend_direction import BlendDirection
from map_processor.assets.water.standing_water_areas import StandingWaterAreas
from map_processor.assets.water.river_areas import RiverAreas
from map_processor.assets.water.standing_wave_areas import StandingWaveAreas
from map_processor.assets.triggers.trigger_areas import TriggerAreas
from map_processor.assets.scripts.player_scripts_list import PlayerScriptsList
from map_processor.assets.scripts.script_argument import ScriptArgument


WORLD_UNITS_PER_TILE = 10.0


@dataclass(frozen=True)
class RotationSpec:
    degrees: int
    clockwise: bool = True

    @property
    def turns_cw(self) -> int:
        deg = int(self.degrees) % 360
        if deg % 90 != 0:
            raise ValueError(f"Rotation must be a multiple of 90 degrees, got {self.degrees}")
        turns = (deg // 90) % 4
        return turns if self.clockwise else (-turns) % 4


def _rotate_xy_index(old_x: int, old_y: int, old_w: int, old_h: int, turns_cw: int) -> Tuple[int, int]:
    turns_cw = turns_cw % 4
    if turns_cw == 0:
        return old_x, old_y
    if turns_cw == 1:
        return old_y, old_w - 1 - old_x
    if turns_cw == 2:
        return old_w - 1 - old_x, old_h - 1 - old_y
    return old_h - 1 - old_y, old_x


def _rotate_grid_xy(arr: np.ndarray, turns_cw: int) -> np.ndarray:
    turns_cw = turns_cw % 4
    if turns_cw == 0:
        return np.asarray(arr)
    mat_yx = np.asarray(arr).T
    rotated_yx = np.rot90(mat_yx, k=-turns_cw)
    return rotated_yx.T


def _get_tile_value(x: int, y: int, texture: int) -> int:
    row_first = (y % 8) // 2 * 16 + (y % 2) * 2
    current = (x % 8) // 2 * 4 + (x % 2) + row_first
    return current + 64 * texture


def _get_texture_from_tile(x: int, y: int, tile_value: int) -> int:
    row_first = (y % 8) // 2 * 16 + (y % 2) * 2
    current = (x % 8) // 2 * 4 + (x % 2) + row_first
    return (int(tile_value) - current) // 64


_ROT_180 = {
    1: 17,
    17: 1,
    2: 18,
    18: 2,
    4: 24,
    24: 4,
    8: 20,
    20: 8,
    36: 56,
    56: 36,
    40: 52,
    52: 40,
    33: 49,
    49: 33,
}

_ROT_90 = {
    1: 18,
    18: 17,
    17: 2,
    2: 1,
    4: 8,
    8: 20,
    20: 24,
    24: 4,
    36: 52,
    52: 56,
    56: 40,
    40: 36,
    33: 50,
    50: 49,
    49: 34,
    34: 33,
}


def _rotate_blend_dir_value(v: int, turns_cw: int) -> int:
    turns_cw = turns_cw % 4
    if turns_cw == 0:
        return v
    if turns_cw == 2:
        return _ROT_180.get(v, v)
    out = v
    for _ in range(turns_cw):
        out = _ROT_90.get(out, out)
    return out


def _rotate_world_xy(
    x: float,
    y: float,
    playable_w_tiles: int,
    playable_h_tiles: int,
    turns_cw: int,
) -> Tuple[float, float]:
    turns_cw = turns_cw % 4
    if turns_cw == 0:
        return x, y
    max_x = float(playable_w_tiles) * WORLD_UNITS_PER_TILE
    max_y = float(playable_h_tiles) * WORLD_UNITS_PER_TILE
    if turns_cw == 1:
        return y, max_x - x
    if turns_cw == 2:
        return max_x - x, max_y - y
    return max_y - y, x


def _angle_close(a: float, b: float, eps: float) -> bool:
    # circular diff
    d = (a - b) % 360.0
    d = min(d, 360.0 - d)
    return d <= eps


def _float_close(a: float, b: float, eps: float) -> bool:
    return abs(a - b) <= eps


def _vec_close(a: Tuple[float, float, float], b: Tuple[float, float, float], eps: float) -> bool:
    return _float_close(a[0], b[0], eps) and _float_close(a[1], b[1], eps) and _float_close(a[2], b[2], eps)


def _rotate_angle_deg(angle_deg: float, turns_cw: int) -> float:
    return (angle_deg - 90.0 * (turns_cw % 4)) % 360.0


def _collect_script_positions(ps: PlayerScriptsList) -> List[Tuple[str, Tuple[float, float, float]]]:
    out: List[Tuple[str, Tuple[float, float, float]]] = []
    for sl_i, sl in enumerate(ps.script_lists):
        for sc_i, script in enumerate(getattr(sl, "scripts", []) or []):
            # actions
            for branch_name, actions in [
                ("true", getattr(script, "script_action_on_true", []) or []),
                ("false", getattr(script, "script_action_on_false", []) or []),
            ]:
                for a_i, action in enumerate(actions):
                    for arg_i, arg in enumerate(getattr(action, "arguments", []) or []):
                        if isinstance(arg, ScriptArgument) and arg.argument_type == 16:
                            out.append((f"sl{sl_i}.sc{sc_i}.{branch_name}.a{a_i}.arg{arg_i}", arg.position))
            # conditions
            for or_i, orc in enumerate(getattr(script, "script_or_conditions", []) or []):
                for c_i, cond in enumerate(getattr(orc, "conditions", []) or []):
                    sc = getattr(cond, "script_content", None)
                    for arg_i, arg in enumerate(getattr(sc, "arguments", []) or []):
                        if isinstance(arg, ScriptArgument) and arg.argument_type == 16:
                            out.append((f"sl{sl_i}.sc{sc_i}.or{or_i}.c{c_i}.arg{arg_i}", arg.position))
    return out


def main() -> int:
    p = argparse.ArgumentParser(description="Verify an RA3 rotated map against expected rotation.")
    p.add_argument("--orig", required=True, help="Original .map path")
    p.add_argument("--rot", required=True, help="Rotated .map path")
    p.add_argument("--degrees", type=int, required=True, help="Rotation in degrees (multiple of 90)")
    p.add_argument("--ccw", action="store_true", help="Interpret degrees as counter-clockwise")
    p.add_argument("--eps-pos", type=float, default=1e-3, help="Position epsilon for object comparisons")
    p.add_argument("--eps-angle", type=float, default=1e-3, help="Angle epsilon for object comparisons")
    p.add_argument("--max-violations", type=int, default=50, help="Max violations to print per category")
    p.add_argument("--json-out", default="", help="Optional path to write JSON report")
    args = p.parse_args()

    spec = RotationSpec(degrees=args.degrees, clockwise=(not args.ccw))
    turns = spec.turns_cw

    orig = Ra3Map(args.orig)
    orig.parse()
    rot = Ra3Map(args.rot)
    rot.parse()
    octx = orig.get_context()
    rctx = rot.get_context()

    report: Dict[str, Any] = {"turns_cw": turns, "violations": {}, "unverified_assets": []}

    def add_violation(cat: str, msg: str) -> None:
        report["violations"].setdefault(cat, []).append(msg)

    # Determine playable dims for world-space assets (objects/polygons/scripts)
    oh = octx.get_asset_by_type(HeightMapData)
    if oh is not None:
        playable_w = int(oh.playable_width)
        playable_h = int(oh.playable_height)
    else:
        border = int(getattr(octx, "border", 0) or 0)
        playable_w = int(octx.map_width) - 2 * border
        playable_h = int(octx.map_height) - 2 * border
        if playable_w <= 0 or playable_h <= 0:
            playable_w, playable_h = int(octx.map_width), int(octx.map_height)

    # --- HeightMapData ---
    if oh is not None:
        rh = rctx.get_asset_by_type(HeightMapData)
        if rh is None:
            add_violation("HeightMapData", "Missing HeightMapData in rotated map")
        else:
            exp = _rotate_grid_xy(oh.elevations, turns)
            if exp.shape != rh.elevations.shape:
                add_violation("HeightMapData", f"elevations shape mismatch {exp.shape} vs {rh.elevations.shape}")
            else:
                diff = np.max(np.abs(exp - rh.elevations))
                if diff > 1e-6:
                    add_violation("HeightMapData", f"elevations max abs diff {diff}")

    # --- BlendTileData ---
    ob = octx.get_asset_by_type(BlendTileData)
    rb = rctx.get_asset_by_type(BlendTileData)
    if ob is None or rb is None:
        add_violation("BlendTileData", f"Missing BlendTileData (orig={ob is not None}, rot={rb is not None})")
    else:
        w, h = ob.map_width, ob.map_height
        if (w, h) != (rb.map_width, rb.map_height):
            add_violation("BlendTileData", f"map dims mismatch orig=({w},{h}) rot=({rb.map_width},{rb.map_height})")
        else:
            # tiles: verify texture indices match at rotated locations
            bad_tiles = 0
            for ox in range(w):
                for oy in range(h):
                    nx, ny = _rotate_xy_index(ox, oy, w, h, turns)
                    otex = _get_texture_from_tile(ox, oy, int(ob.tiles[ox, oy]))
                    rtex = _get_texture_from_tile(nx, ny, int(rb.tiles[nx, ny]))
                    if otex != rtex:
                        if bad_tiles < args.max_violations:
                            add_violation("BlendTileData.tiles", f"texture mismatch at orig({ox},{oy}) -> rot({nx},{ny}): {otex} != {rtex}")
                        bad_tiles += 1
            if bad_tiles:
                add_violation("BlendTileData.tiles", f"TOTAL tile-texture mismatches: {bad_tiles}")

            # blends and single_edge_blends: verify per-cell secondary texture + dir rotated and tile-value recomputed
            def check_blend_layer(layer_name: str, ogrid: np.ndarray, rgrid: np.ndarray) -> None:
                bad = 0
                for ox in range(w):
                    for oy in range(h):
                        oidx = int(ogrid[ox, oy])
                        if oidx <= 0:
                            continue
                        nx, ny = _rotate_xy_index(ox, oy, w, h, turns)
                        ridx = int(rgrid[nx, ny])
                        if ridx <= 0:
                            if bad < args.max_violations:
                                add_violation(layer_name, f"missing blend at rot({nx},{ny}) from orig({ox},{oy}) idx={oidx}")
                            bad += 1
                            continue
                        if not (1 <= oidx <= len(ob.blend_info)) or not (1 <= ridx <= len(rb.blend_info)):
                            if bad < args.max_violations:
                                add_violation(layer_name, f"index out of range orig idx={oidx} rot idx={ridx}")
                            bad += 1
                            continue
                        obi = ob.blend_info[oidx - 1]
                        rbi = rb.blend_info[ridx - 1]

                        sec_tex = _get_texture_from_tile(ox, oy, int(obi.secondary_texture_tile))
                        exp_sec_tile = _get_tile_value(nx, ny, sec_tex)
                        exp_dir = _rotate_blend_dir_value(int(obi.blend_direction), turns)

                        got_sec_tile = int(rbi.secondary_texture_tile)
                        got_dir = int(rbi.blend_direction)
                        if got_sec_tile != exp_sec_tile or got_dir != exp_dir:
                            if bad < args.max_violations:
                                add_violation(
                                    layer_name,
                                    f"mismatch orig({ox},{oy}) -> rot({nx},{ny}): "
                                    f"exp(sec_tile={exp_sec_tile},dir={exp_dir}) got(sec_tile={got_sec_tile},dir={got_dir})",
                                )
                            bad += 1
                if bad:
                    add_violation(layer_name, f"TOTAL mismatches: {bad}")

            check_blend_layer("BlendTileData.blends", ob.blends, rb.blends)
            check_blend_layer("BlendTileData.single_edge_blends", ob.single_edge_blends, rb.single_edge_blends)

            # passability + bool grids (rotation should match)
            for name in ["passability", "visibility", "buildability", "tiberium_growability", "passage_width"]:
                oa = getattr(ob, name, None)
                ra = getattr(rb, name, None)
                if oa is None or ra is None:
                    continue
                exp = _rotate_grid_xy(oa, turns)
                if exp.shape != ra.shape:
                    add_violation("BlendTileData.flags", f"{name} shape mismatch exp={exp.shape} got={ra.shape}")
                else:
                    if np.any(exp != ra):
                        # count mismatches
                        mism = int(np.sum(exp != ra))
                        add_violation("BlendTileData.flags", f"{name} mismatched cells: {mism}")

    # --- ObjectsList ---
    oo = octx.get_asset_by_type(ObjectsList)
    ro = rctx.get_asset_by_type(ObjectsList)
    if oo is None or ro is None:
        add_violation("ObjectsList", f"Missing ObjectsList (orig={oo is not None}, rot={ro is not None})")
    else:
        rot_by_uid: Dict[str, Any] = {}
        for obj in ro.map_objects:
            uid = obj.unique_id or f"__no_uid__:{obj.type_name}:{id(obj)}"
            rot_by_uid[uid] = obj

        missing = 0
        bad = 0
        for obj in oo.map_objects:
            uid = obj.unique_id or f"__no_uid__:{obj.type_name}:{id(obj)}"
            robj = rot_by_uid.get(uid)
            if robj is None:
                if missing < args.max_violations:
                    add_violation("ObjectsList", f"missing object uid={uid} type={obj.type_name}")
                missing += 1
                continue
            ex, ey = _rotate_world_xy(obj.position[0], obj.position[1], playable_w, playable_h, turns)
            ez = obj.position[2]
            exp_pos = (float(ex), float(ey), float(ez))
            exp_ang = _rotate_angle_deg(obj.angle, turns)

            if not _vec_close(exp_pos, robj.position, args.eps_pos) or not _angle_close(exp_ang, robj.angle, args.eps_angle):
                if bad < args.max_violations:
                    add_violation(
                        "ObjectsList",
                        f"uid={uid} type={obj.type_name}: exp_pos={exp_pos} got_pos={robj.position}, "
                        f"exp_ang={exp_ang} got_ang={robj.angle}",
                    )
                bad += 1
        if missing:
            add_violation("ObjectsList", f"TOTAL missing objects: {missing}")
        if bad:
            add_violation("ObjectsList", f"TOTAL mismatched objects: {bad}")

    # --- Polygons ---
    def check_points(asset_name: str, o_points_list: List[List[Tuple[float, float]]], r_points_list: List[List[Tuple[float, float]]]) -> None:
        if len(o_points_list) != len(r_points_list):
            add_violation(asset_name, f"area count mismatch orig={len(o_points_list)} rot={len(r_points_list)}")
            return
        bad = 0
        for i, (op, rp) in enumerate(zip(o_points_list, r_points_list)):
            if len(op) != len(rp):
                if bad < args.max_violations:
                    add_violation(asset_name, f"area[{i}] point count mismatch orig={len(op)} rot={len(rp)}")
                bad += 1
                continue
            for j, (p, q) in enumerate(zip(op, rp)):
                ex, ey = _rotate_world_xy(p[0], p[1], playable_w, playable_h, turns)
                if not (_float_close(ex, q[0], args.eps_pos) and _float_close(ey, q[1], args.eps_pos)):
                    if bad < args.max_violations:
                        add_violation(asset_name, f"area[{i}] pt[{j}] exp=({ex},{ey}) got={q}")
                    bad += 1
                    break
        if bad:
            add_violation(asset_name, f"TOTAL mismatched areas: {bad}")

    oswa = octx.get_asset_by_type(StandingWaterAreas)
    rswa = rctx.get_asset_by_type(StandingWaterAreas)
    if oswa and rswa:
        check_points("StandingWaterAreas", [a.points for a in oswa.water_areas], [a.points for a in rswa.water_areas])

    oriv = octx.get_asset_by_type(RiverAreas)
    rriv = rctx.get_asset_by_type(RiverAreas)
    if oriv and rriv:
        check_points("RiverAreas", [a.points for a in oriv.areas], [a.points for a in rriv.areas])

    oswv = octx.get_asset_by_type(StandingWaveAreas)
    rswv = rctx.get_asset_by_type(StandingWaveAreas)
    if oswv and rswv:
        check_points("StandingWaveAreas", [a.points for a in oswv.areas], [a.points for a in rswv.areas])

    otr = octx.get_asset_by_type(TriggerAreas)
    rtr = rctx.get_asset_by_type(TriggerAreas)
    if otr and rtr:
        check_points("TriggerAreas", [a.points for a in otr.areas], [a.points for a in rtr.areas])

    # --- ScriptArgument positions (if any scripts exist) ---
    ops = octx.get_asset_by_type(PlayerScriptsList)
    rps = rctx.get_asset_by_type(PlayerScriptsList)
    if ops and rps:
        o_pos = _collect_script_positions(ops)
        r_pos = dict(_collect_script_positions(rps))
        bad = 0
        missing = 0
        for k, pos in o_pos:
            if k not in r_pos:
                if missing < args.max_violations:
                    add_violation("Scripts", f"missing pos arg {k}")
                missing += 1
                continue
            ex, ey = _rotate_world_xy(pos[0], pos[1], playable_w, playable_h, turns)
            ez = pos[2]
            got = r_pos[k]
            if not _vec_close((ex, ey, ez), got, args.eps_pos):
                if bad < args.max_violations:
                    add_violation("Scripts", f"{k}: exp={(ex,ey,ez)} got={got}")
                bad += 1
        if missing:
            add_violation("Scripts", f"TOTAL missing script positions: {missing}")
        if bad:
            add_violation("Scripts", f"TOTAL mismatched script positions: {bad}")

    # --- Unverified assets (DefaultMajorAsset) ---
    for o_asset in octx.map_struct.assets:
        if isinstance(o_asset, DefaultMajorAsset):
            name = o_asset.get_asset_name()
            r_asset = rctx.map_struct.get_asset_by_name(name)
            same = False
            if isinstance(r_asset, DefaultMajorAsset):
                same = o_asset.data == r_asset.data
            report["unverified_assets"].append({"name": name, "bytes_equal_to_orig": same, "size": len(o_asset.data)})

    # Print report
    violations_total = sum(len(v) for v in report["violations"].values())
    print(f"turns_cw={turns} total_violation_lines={violations_total}")
    for cat, msgs in report["violations"].items():
        print(f"\n[{cat}]")
        for m in msgs[: args.max_violations]:
            print("-", m)
        if len(msgs) > args.max_violations:
            print(f"... ({len(msgs) - args.max_violations} more)")

    if report["unverified_assets"]:
        print("\n[UNVERIFIED DefaultMajorAsset blocks]")
        for a in report["unverified_assets"]:
            print(f"- {a['name']}: size={a['size']} bytes_equal_to_orig={a['bytes_equal_to_orig']}")

    if args.json_out:
        Path(args.json_out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json_out).write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"\nWrote JSON report to: {args.json_out}")

    # exit code: 0 if no violations in checked assets
    return 0 if not report["violations"] else 2


if __name__ == "__main__":
    raise SystemExit(main())


