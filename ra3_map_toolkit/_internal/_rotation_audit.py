"""
Comprehensive mutation tester for the rotation/flip subsystem.

Categories:
  A) Algebraic round-trip identities (composition laws)
  B) Per-object position/angle invariants (predicted vs actual)
  C) Grid bit-identity for identity-equivalent compositions
  D) Object-vs-terrain alignment (distance-to-nearest-impassable preservation)

Run:
  python _rotation_audit.py [--map PATH]
"""
from __future__ import annotations
import argparse
import copy
import math
import sys
from pathlib import Path
from typing import Any, Callable, List, Optional, Tuple

import numpy as np

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from map_processor.core.ra3map import Ra3Map
from map_processor.assets.terrain.height_map_data import HeightMapData
from map_processor.assets.terrain.blend_tile_data import BlendTileData
from map_processor.assets.objects.objects_list import ObjectsList
from map_processor.utils.map_rotation import (
    rotate_context_right_angles,
    flip_context_axis,
    WORLD_UNITS_PER_TILE,
)

DEFAULT_MAP = r"e:/DL/Projects/Ra3 texture gen/RA 3 maps/RA3 Official maps/2 II/map_mp_2_rao1.map"


# ----- Transform DSL -----
# A "step" is one of: ('rot', degrees, clockwise) or ('flip', axis)

def step_rot_cw(deg: int): return ("rot", int(deg), True)
def step_rot_ccw(deg: int): return ("rot", int(deg), False)
def step_flip(axis: str): return ("flip", axis)

def apply_steps(ctx, steps, mesh_fixup=None):
    """mesh_fixup={} disables flip mesh-fixup (use for pure-algebra tests)."""
    for s in steps:
        if s[0] == "rot":
            rotate_context_right_angles(ctx, degrees=s[1], clockwise=s[2])
        elif s[0] == "flip":
            flip_context_axis(ctx, axis=s[1], mesh_fixup=mesh_fixup)
        else:
            raise ValueError(f"unknown step: {s}")


# ----- Equality predicates -----

def _arr_eq(a, b) -> bool:
    if a is None and b is None: return True
    if a is None or b is None: return False
    a, b = np.asarray(a), np.asarray(b)
    if a.shape != b.shape: return False
    return bool(np.array_equal(a, b))


def heights_equal(c1, c2) -> Tuple[bool, str]:
    h1 = c1.get_asset_by_type(HeightMapData)
    h2 = c2.get_asset_by_type(HeightMapData)
    if not _arr_eq(h1.elevations, h2.elevations):
        return False, "elevations differ"
    if (h1.map_width, h1.map_height) != (h2.map_width, h2.map_height):
        return False, f"dims differ {h1.map_width}x{h1.map_height} vs {h2.map_width}x{h2.map_height}"
    return True, ""


def blends_equal(c1, c2) -> Tuple[bool, str]:
    b1 = c1.get_asset_by_type(BlendTileData)
    b2 = c2.get_asset_by_type(BlendTileData)
    if not _arr_eq(b1.tiles, b2.tiles): return False, "tiles differ"
    # blends/single_edge_blends use *index into blend_info* — to compare semantically,
    # resolve each cell to (sec_tile, dir) tuples.
    def resolve(b, layer):
        info = b.blend_info or []
        out = np.zeros(layer.shape, dtype=np.int64)
        # encode (sec_tile << 16) | dir for each cell; 0 for empty
        for x in range(layer.shape[0]):
            for y in range(layer.shape[1]):
                idx = int(layer[x, y])
                if idx > 0 and idx <= len(info):
                    bi = info[idx - 1]
                    out[x, y] = (int(bi.secondary_texture_tile) << 32) | int(bi.blend_direction)
        return out
    if not _arr_eq(resolve(b1, b1.blends), resolve(b2, b2.blends)):
        return False, "blends layer differs (semantic)"
    if not _arr_eq(resolve(b1, b1.single_edge_blends), resolve(b2, b2.single_edge_blends)):
        return False, "single_edge_blends layer differs (semantic)"
    if not _arr_eq(b1.passability, b2.passability):
        return False, "passability differs"
    return True, ""


def objects_equal(c1, c2, eps_pos=1e-3, eps_ang=1e-3) -> Tuple[bool, str]:
    o1 = c1.get_asset_by_type(ObjectsList)
    o2 = c2.get_asset_by_type(ObjectsList)
    if len(o1.map_objects) != len(o2.map_objects):
        return False, f"object count {len(o1.map_objects)} vs {len(o2.map_objects)}"
    by_id_2 = {}
    for o in o2.map_objects:
        key = (o.unique_id or "", o.type_name)
        by_id_2.setdefault(key, []).append(o)
    for o in o1.map_objects:
        key = (o.unique_id or "", o.type_name)
        cands = by_id_2.get(key, [])
        if not cands:
            return False, f"object {key} missing"
        # closest match (handles non-unique ids)
        best = min(cands, key=lambda b: sum((b.position[i] - o.position[i]) ** 2 for i in range(3)))
        if any(abs(best.position[i] - o.position[i]) > eps_pos for i in range(3)):
            return False, f"object {key} pos {o.position} vs {best.position}"
        d = (best.angle - o.angle) % 360.0
        d = min(d, 360.0 - d)
        if d > eps_ang:
            return False, f"object {key} angle {o.angle:.4f} vs {best.angle:.4f}"
    return True, ""


def context_equal(c1, c2) -> Tuple[bool, List[str]]:
    fails = []
    ok, m = heights_equal(c1, c2)
    if not ok: fails.append(f"heights: {m}")
    ok, m = blends_equal(c1, c2)
    if not ok: fails.append(f"blends: {m}")
    ok, m = objects_equal(c1, c2)
    if not ok: fails.append(f"objects: {m}")
    return (not fails), fails


# ----- Object position predictor (oracle) -----
# Replays the per-object math from map_rotation.py to predict where each object
# *should* end up under a given step sequence.

def predict_object_state(x, y, angle, play_w_units, play_h_units, step) -> Tuple[float, float, float, float, float]:
    """Returns (new_x, new_y, new_angle, new_play_w_units, new_play_h_units)."""
    if step[0] == "rot":
        deg, cw = step[1], step[2]
        turns = ((deg // 90) % 4)
        if not cw:
            turns = (-turns) % 4
        new_play_w, new_play_h = (play_h_units, play_w_units) if turns % 2 == 1 else (play_w_units, play_h_units)
        if turns == 0:
            return (x, y, angle, new_play_w, new_play_h)
        if turns == 1:
            nx, ny = (y, play_w_units - x)
        elif turns == 2:
            nx, ny = (play_w_units - x, play_h_units - y)
        else:
            nx, ny = (play_h_units - y, x)
        new_angle = (angle - 90.0 * turns) % 360.0
        return (nx, ny, new_angle, new_play_w, new_play_h)
    elif step[0] == "flip":
        axis = step[1]
        if axis == "y":
            return (play_w_units - x, y, (180.0 - angle) % 360.0, play_w_units, play_h_units)
        elif axis == "x":
            return (x, play_h_units - y, (360.0 - angle) % 360.0, play_w_units, play_h_units)
    raise ValueError(f"bad step {step}")


def predict_object_after(x, y, angle, play_w_units, play_h_units, steps):
    cx, cy, ca = x, y, angle
    pw, ph = play_w_units, play_h_units
    for s in steps:
        cx, cy, ca, pw, ph = predict_object_state(cx, cy, ca, pw, ph, s)
    return cx, cy, ca


# ----- Bound check -----

def all_objects_in_bounds(ctx, eps=1.0) -> Tuple[bool, str]:
    h = ctx.get_asset_by_type(HeightMapData)
    if h is None: return True, ""
    pw = h.playable_width * WORLD_UNITS_PER_TILE
    ph = h.playable_height * WORLD_UNITS_PER_TILE
    bad = 0
    first = ""
    for o in ctx.get_asset_by_type(ObjectsList).map_objects:
        x, y, _ = o.position
        if x < -eps or y < -eps or x > pw + eps or y > ph + eps:
            bad += 1
            if not first:
                first = f"{o.type_name} pos=({x:.1f},{y:.1f}) bounds=({pw:.0f},{ph:.0f})"
    if bad:
        return False, f"{bad} OOB; first: {first}"
    return True, ""


# ----- Object-vs-terrain alignment -----

def object_terrain_distances(ctx, sample=200) -> List[Tuple[str, float]]:
    """For a sampled subset of objects, return (type_name, dist_to_nearest_impassable_in_world_units)."""
    h = ctx.get_asset_by_type(HeightMapData)
    b = ctx.get_asset_by_type(BlendTileData)
    if h is None or b is None or b.impassable is None:
        return []
    # impassable grid: shape (W, H) of bool, indexed by tile incl. border
    imp = np.asarray(b.impassable, dtype=np.bool_)
    if not imp.any():
        return []
    # Collect impassable tile centers in world coords (PLAYABLE-RELATIVE).
    border = h.border_width
    ys, xs = np.where(imp.T)  # ys rows (full y), xs cols (full x)
    # convert full-map tile -> playable-relative world units
    pxs = (xs - border + 0.5) * WORLD_UNITS_PER_TILE
    pys = (ys - border + 0.5) * WORLD_UNITS_PER_TILE
    pts = np.stack([pxs, pys], axis=1)  # (N,2)

    objs = ctx.get_asset_by_type(ObjectsList).map_objects
    if sample and len(objs) > sample:
        idxs = np.linspace(0, len(objs) - 1, sample).astype(int)
        objs = [objs[i] for i in idxs]
    out = []
    for o in objs:
        ox, oy = float(o.position[0]), float(o.position[1])
        d2 = (pts[:, 0] - ox) ** 2 + (pts[:, 1] - oy) ** 2
        out.append((o.type_name, float(np.sqrt(d2.min()))))
    return out


# ----- Test runner -----

def run_audit(map_path: str) -> int:
    print(f"audit: {map_path}")
    m = Ra3Map(map_path)
    m.parse()
    base = m.get_context()

    def fresh():
        return copy.deepcopy(base)

    def run(steps, mesh_fixup=None) -> Any:
        c = fresh()
        apply_steps(c, steps, mesh_fixup=mesh_fixup)
        return c

    # Algebra tests must use pure mirror (no mesh fixup heuristic), otherwise
    # `flipx o flipx = id` etc. will fail (each flip adds 180° to cliff walls).
    def run_pure(steps) -> Any:
        return run(steps, mesh_fixup={})

    failures = 0
    def report(cat: str, name: str, ok: bool, msg: str = ""):
        nonlocal failures
        if not ok:
            failures += 1
        flag = "PASS" if ok else "FAIL"
        msg_str = f" -- {msg}" if msg else ""
        print(f"  [{cat}] {flag}  {name}{msg_str}")

    # ---- A) Algebraic round-trips ----
    print("\n[A] Algebraic round-trip identities")
    cases = [
        ("rot90cw o rot90ccw = id", [step_rot_cw(90), step_rot_ccw(90)], []),
        ("rot90ccw o rot90cw = id", [step_rot_ccw(90), step_rot_cw(90)], []),
        ("rot90cw x4 = id",              [step_rot_cw(90)] * 4, []),
        ("rot90ccw x4 = id",             [step_rot_ccw(90)] * 4, []),
        ("rot180 o rot180 = id",    [step_rot_cw(180)] * 2, []),
        ("rot90cw x2 = rot180",          [step_rot_cw(90)] * 2, [step_rot_cw(180)]),
        ("flipx o flipx = id",      [step_flip("x")] * 2, []),
        ("flipy o flipy = id",      [step_flip("y")] * 2, []),
        ("flipx o flipy = rot180",  [step_flip("x"), step_flip("y")], [step_rot_cw(180)]),
        ("flipy o flipx = rot180",  [step_flip("y"), step_flip("x")], [step_rot_cw(180)]),
    ]
    for name, lhs, rhs in cases:
        c1 = run_pure(lhs)
        c2 = run_pure(rhs)
        ok, fails = context_equal(c1, c2)
        report("A", name, ok, "; ".join(fails))

    # ---- B) Per-object pos/angle predicted vs actual ----
    print("\n[B] Per-object position/angle (predicted vs actual)")
    base_h = base.get_asset_by_type(HeightMapData)
    play_w_u = base_h.playable_width * WORLD_UNITS_PER_TILE
    play_h_u = base_h.playable_height * WORLD_UNITS_PER_TILE
    base_objs = list(base.get_asset_by_type(ObjectsList).map_objects)

    transforms = [
        ("rot90cw",  [step_rot_cw(90)]),
        ("rot90ccw", [step_rot_ccw(90)]),
        ("rot180",   [step_rot_cw(180)]),
        ("flipx",    [step_flip("x")]),
        ("flipy",    [step_flip("y")]),
    ]
    for tname, steps in transforms:
        c = run_pure(steps)
        new_objs = c.get_asset_by_type(ObjectsList).map_objects
        bad_pos = bad_ang = 0
        first = ""
        # match by index since transform preserves ordering
        for o_orig, o_new in zip(base_objs, new_objs):
            px, py, pa = predict_object_after(
                o_orig.position[0], o_orig.position[1], o_orig.angle, play_w_u, play_h_u, steps
            )
            if abs(o_new.position[0] - px) > 1e-2 or abs(o_new.position[1] - py) > 1e-2:
                bad_pos += 1
                if not first:
                    first = f"{o_orig.type_name}: pred=({px:.2f},{py:.2f}) got={o_new.position[:2]}"
            d = (o_new.angle - pa) % 360.0; d = min(d, 360.0 - d)
            if d > 1e-2:
                bad_ang += 1
        ok = (bad_pos == 0 and bad_ang == 0)
        report("B", f"{tname} pos+angle", ok, f"bad_pos={bad_pos} bad_ang={bad_ang}{(' first: ' + first) if first else ''}")

    # B2) Bounds check
    print("\n[B2] All objects within new playable bounds")
    for tname, steps in transforms:
        c = run(steps)
        ok, msg = all_objects_in_bounds(c)
        report("B2", f"{tname} bounds", ok, msg)

    # ---- C) Grid bit-identity for identity-equivalent compositions ----
    print("\n[C] Grid bit-identity round-trips")
    id_cases = [
        ("rot90cw x4",          [step_rot_cw(90)] * 4),
        ("rot90ccw x4",         [step_rot_ccw(90)] * 4),
        ("rot180 x2",           [step_rot_cw(180)] * 2),
        ("rot90cw + rot90ccw",  [step_rot_cw(90), step_rot_ccw(90)]),
        ("flipx x2",            [step_flip("x")] * 2),
        ("flipy x2",            [step_flip("y")] * 2),
    ]
    for name, steps in id_cases:
        c = run(steps)
        h_ok, h_msg = heights_equal(base, c)
        b_ok, b_msg = blends_equal(base, c)
        ok = h_ok and b_ok
        msg = "; ".join(x for x in [h_msg if not h_ok else "", b_msg if not b_ok else ""] if x)
        report("C", name, ok, msg)

    # ---- D) Object-vs-terrain distance preservation ----
    print("\n[D] Object-vs-impassable distance preservation (sample 100)")
    base_d = sorted([d for _, d in object_terrain_distances(base, sample=100)])
    for tname, steps in transforms:
        c = run(steps)
        cd = sorted([d for _, d in object_terrain_distances(c, sample=100)])
        if len(base_d) != len(cd):
            report("D", tname, False, f"sample size diff {len(base_d)} vs {len(cd)}")
            continue
        # Distances are invariant under rotation/flip; small drift OK due to sampling indices.
        diffs = np.abs(np.array(base_d) - np.array(cd))
        max_diff = float(diffs.max()) if len(diffs) else 0.0
        ok = max_diff <= 1.0  # 1 world unit slack
        report("D", tname, ok, f"max_dist_diff={max_diff:.2f}u")

    # ---- E) Flip mesh-fixup heuristic ----
    print("\n[E] Flip mesh-fixup applied to asymmetric mesh objects (default fixup ON)")
    from map_processor.utils.map_rotation import DEFAULT_FLIP_MESH_FIXUP
    for tname, steps in [("flipx", [step_flip("x")]), ("flipy", [step_flip("y")])]:
        c = run(steps)  # default fixup
        new_objs = c.get_asset_by_type(ObjectsList).map_objects
        bad = 0
        for o_orig, o_new in zip(base_objs, new_objs):
            # expected = pure mirror angle + fixup if name matches
            _, _, pure_ang = predict_object_after(
                o_orig.position[0], o_orig.position[1], o_orig.angle, play_w_u, play_h_u, steps
            )
            extra = 0.0
            for substr, deg in DEFAULT_FLIP_MESH_FIXUP.items():
                if substr.upper() in (o_orig.type_name or "").upper():
                    extra += deg
                    break
            expected_ang = (pure_ang + extra) % 360.0
            d = (o_new.angle - expected_ang) % 360.0; d = min(d, 360.0 - d)
            if d > 1e-2:
                bad += 1
        ok = (bad == 0)
        report("E", f"{tname} fixup", ok, f"bad_ang={bad}")

    # E2) Symmetric objects (grass/trees/etc.) should NOT be touched by fixup
    print("\n[E2] Symmetric objects unaffected by fixup")
    for tname, steps in [("flipx", [step_flip("x")]), ("flipy", [step_flip("y")])]:
        c_pure = run_pure(steps)
        c_default = run(steps)
        pure_objs = c_pure.get_asset_by_type(ObjectsList).map_objects
        def_objs = c_default.get_asset_by_type(ObjectsList).map_objects
        diff_count = 0
        for o_orig, op, od in zip(base_objs, pure_objs, def_objs):
            if abs(op.angle - od.angle) > 1e-2:
                # angle differs => fixup applied; should be a known-asymmetric type
                if not any(s.upper() in (o_orig.type_name or "").upper() for s in DEFAULT_FLIP_MESH_FIXUP):
                    diff_count += 1
        ok = (diff_count == 0)
        report("E2", f"{tname} fixup scope", ok, f"unexpected_changes={diff_count}")

    print(f"\n=== TOTAL FAILURES: {failures} ===")
    return 0 if failures == 0 else 1


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--map", default=DEFAULT_MAP)
    args = p.parse_args()
    return run_audit(args.map)


if __name__ == "__main__":
    sys.exit(main())
