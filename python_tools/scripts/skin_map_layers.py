#!/usr/bin/env python3
"""Decompose a map into layered strips, one per parsable asset, so each layer's
contribution can be visualised in isolation and as a cumulative skeleton.

For a single source map this produces N isolated outputs (one layer removed)
and a cumulative chain culminating in the bare skeleton.

ISOLATED layers (each removes exactly ONE layer / asset):
  Terrain/blends:
    iso_no_blends         BlendTileData.{blends, single_edge_blends, cliff_blends} -> 0
    iso_no_textures       Palette collapsed to a single neutral; tiles -> 0
    iso_flat              HeightMapData elevations -> mean
  Masks (within BlendTileData):
    iso_passability_open  passability = Passable everywhere
    iso_buildable_all     buildability = True everywhere
    iso_visible_all       visibility   = True everywhere
    iso_no_shrubs         dynamic_shrubbery = 0
    iso_no_tib_growth     tiberium_growability = False
  Objects / gameplay:
    iso_no_objects        ObjectsList emptied
    iso_no_mp_restrict    MPPositionList side_restriction cleared
    iso_no_teams          Teams emptied
    iso_no_scripts        PlayerScriptsList emptied
    iso_no_triggers       TriggerAreas emptied
    iso_no_missions       MissionObjectives + MissionHotSpots emptied
    iso_no_build_lists    BuildLists emptied
    iso_no_library        LibraryMaps + LibraryMapLists emptied
  Water / world:
    iso_no_water_areas    StandingWaterAreas + RiverAreas + StandingWaveAreas emptied
    iso_no_post_effects   PostEffectsChunk emptied
    iso_no_fog            FogSettings disabled

CUMULATIVE chain (each step adds the previous strips):
  cum1_logic_off          scripts + triggers + missions + build_lists + library + post_effects + fog
  cum2_water_off          + water areas
  cum3_blends_off         + blends
  cum4_textures_off       + textures (palette neutralised)
  cum5_objects_off        + objects + teams + mp_restrict
  cum6_masks_off          + passability/buildability/visibility/shrubs/tib normalised
  cum7_skeleton           + flat heightmap (the bare plane)

Each output is rendered to a PNG so the contribution of each layer is visible.

Usage:
  python scripts/skin_map_layers.py --src "../RA3 Official maps/2 II/map_mp_2_rao1.map"
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np


def _python_tools_root() -> Path:
    return Path(__file__).resolve().parents[1]


sys.path.insert(0, str(_python_tools_root()))
from map_processor import Ra3Map  # noqa: E402


# -------------------------- BlendTileData layers --------------------------

def strip_blends(blend) -> None:
    """Zero the blends arrays - hard texture transitions remain."""
    if blend is None:
        return
    blend.blends = np.zeros_like(blend.blends)
    blend.single_edge_blends = np.zeros_like(blend.single_edge_blends)
    if blend.cliff_blends is not None:
        blend.cliff_blends = np.zeros_like(blend.cliff_blends)


def strip_textures(blend) -> None:
    """Reduce palette to one neutral texture; all tiles point at it."""
    if blend is None:
        return
    from map_processor.assets.terrain.texture import Texture

    NEUTRAL = "Dirt_Yucatan01"
    found_idx = None
    for i, t in enumerate(blend.textures):
        if t.name == NEUTRAL:
            found_idx = i
            break
    if found_idx is None:
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
    blend.textures = [blend.textures[found_idx]]
    blend.textures[0].cell_start = 0
    blend.tiles = np.zeros_like(blend.tiles, dtype=np.uint16)


def reset_passability(blend) -> None:
    """Set passability to fully Passable."""
    if blend is None or blend.passability is None:
        return
    from map_processor.assets.terrain.passability import Passability
    blend.passability = np.full_like(blend.passability, int(Passability.Passable))


def reset_buildability(blend) -> None:
    if blend is None or blend.buildability is None:
        return
    blend.buildability = np.ones_like(blend.buildability, dtype=np.bool_)


def reset_visibility(blend) -> None:
    if blend is None or blend.visibility is None:
        return
    blend.visibility = np.ones_like(blend.visibility, dtype=np.bool_)


def reset_dynamic_shrubbery(blend) -> None:
    if blend is None or blend.dynamic_shrubbery is None:
        return
    blend.dynamic_shrubbery = np.zeros_like(blend.dynamic_shrubbery, dtype=np.uint8)


def reset_tib_growability(blend) -> None:
    if blend is None or blend.tiberium_growability is None:
        return
    blend.tiberium_growability = np.zeros_like(blend.tiberium_growability, dtype=np.bool_)


# -------------------------- Heightmap --------------------------

def flatten_heightmap(h_asset) -> None:
    """Set elevations to a constant (mean)."""
    if h_asset is None:
        return
    elev = h_asset.elevations.astype(np.float32)
    target = float(elev.mean())
    h_asset.elevations = np.full_like(elev, target)
    if hasattr(h_asset, "_elevations_raw") and h_asset._elevations_raw is not None:
        raw = h_asset._elevations_raw
        mid = int(np.median(raw))
        h_asset._elevations_raw = np.full_like(raw, mid)


# -------------------------- Object / gameplay layers --------------------------

def strip_objects(objects_list) -> None:
    if objects_list is None:
        return
    objects_list.map_objects = []


def strip_mp_restrictions(mp_list) -> None:
    """Clear side_restriction on each MPPositionInfo (positions are fixed-size 6)."""
    if mp_list is None:
        return
    for p in mp_list.positions:
        p.side_restriction = []


def strip_teams(teams) -> None:
    if teams is None:
        return
    teams.teams = []


def strip_scripts(scripts) -> None:
    if scripts is None:
        return
    scripts.script_lists = []


def strip_triggers(triggers) -> None:
    if triggers is None:
        return
    triggers.areas = []


def strip_missions(objectives, hot_spots) -> None:
    if objectives is not None:
        objectives.objectives = {}
    if hot_spots is not None:
        hot_spots.spots = []


def strip_build_lists(build_lists) -> None:
    if build_lists is None:
        return
    build_lists.build_list = []


def strip_library(lib_maps, lib_map_lists) -> None:
    if lib_maps is not None:
        lib_maps.library_maps = []
    if lib_map_lists is not None:
        lib_map_lists.library_maps = []


# -------------------------- Water / world layers --------------------------

def strip_water_areas(standing, rivers, waves) -> None:
    if standing is not None:
        standing.water_areas = []
    if rivers is not None:
        rivers.areas = []
    if waves is not None:
        waves.areas = []


def strip_post_effects(post) -> None:
    if post is None:
        return
    post.effects = []


def strip_fog(fog) -> None:
    if fog is None:
        return
    fog.enabled = False


# -------------------------- Pipeline --------------------------

def save_copy(src_path: Path, modify_fn, out_path: Path) -> None:
    m = Ra3Map(str(src_path))
    m.parse()
    ctx = m.get_context()
    modify_fn(ctx)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    m.save(str(out_path), compress=True)


def render(map_path: Path, out_dir: Path) -> Path | None:
    import subprocess
    cmd = [sys.executable, str(_python_tools_root() / "scripts" / "generate_map_image.py"),
           str(map_path), str(out_dir)]
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True, timeout=120)
        for png in out_dir.glob(f"{map_path.stem}_terrain_comprehensive.png"):
            return png
    except Exception as e:  # noqa: BLE001
        print(f"  [render-warn] {map_path.name}: {e}")
    return None


# Helper: wrap each strip so it pulls assets by name from the context.

def _blend(ctx):
    return ctx.get_asset("BlendTileData")


def _heightmap(ctx):
    return ctx.get_asset("HeightMapData")


def _objects(ctx):
    return ctx.get_asset("ObjectsList")


# Each entry: (slug, description, mutator(ctx))
ISOLATED_LAYERS = [
    # Terrain
    ("iso01_no_blends",       "Strip BLENDS only",
     lambda c: strip_blends(_blend(c))),
    ("iso02_no_textures",     "Strip TEXTURES only",
     lambda c: strip_textures(_blend(c))),
    ("iso03_flat",            "Flatten ELEVATIONS only",
     lambda c: flatten_heightmap(_heightmap(c))),
    # Masks
    ("iso04_passability_open", "Force PASSABILITY = Passable",
     lambda c: reset_passability(_blend(c))),
    ("iso05_buildable_all",   "Force BUILDABILITY = True",
     lambda c: reset_buildability(_blend(c))),
    ("iso06_visible_all",     "Force VISIBILITY = True",
     lambda c: reset_visibility(_blend(c))),
    ("iso07_no_shrubs",       "Clear DYNAMIC SHRUBBERY",
     lambda c: reset_dynamic_shrubbery(_blend(c))),
    ("iso08_no_tib_growth",   "Clear TIBERIUM GROWABILITY",
     lambda c: reset_tib_growability(_blend(c))),
    # Objects / gameplay
    ("iso09_no_objects",      "Empty ObjectsList",
     lambda c: strip_objects(_objects(c))),
    ("iso10_no_mp_restrict",  "Clear MP position side_restriction",
     lambda c: strip_mp_restrictions(c.get_asset("MPPositionList"))),
    ("iso11_no_teams",        "Empty Teams",
     lambda c: strip_teams(c.get_asset("Teams"))),
    ("iso12_no_scripts",      "Empty PlayerScriptsList",
     lambda c: strip_scripts(c.get_asset("PlayerScriptsList"))),
    ("iso13_no_triggers",     "Empty TriggerAreas",
     lambda c: strip_triggers(c.get_asset("TriggerAreas"))),
    ("iso14_no_missions",     "Empty MissionObjectives + MissionHotSpots",
     lambda c: strip_missions(c.get_asset("MissionObjectives"),
                              c.get_asset("MissionHotSpots"))),
    ("iso15_no_build_lists",  "Empty BuildLists",
     lambda c: strip_build_lists(c.get_asset("BuildLists"))),
    ("iso16_no_library",      "Empty LibraryMaps + LibraryMapLists",
     lambda c: strip_library(c.get_asset("LibraryMaps"),
                             c.get_asset("LibraryMapLists"))),
    # Water / world
    ("iso17_no_water_areas",  "Empty Standing/River/Wave water areas",
     lambda c: strip_water_areas(c.get_asset("StandingWaterAreas"),
                                 c.get_asset("RiverAreas"),
                                 c.get_asset("StandingWaveAreas"))),
    ("iso18_no_post_effects", "Empty PostEffectsChunk",
     lambda c: strip_post_effects(c.get_asset("PostEffectsChunk"))),
    ("iso19_no_fog",          "Disable FogSettings",
     lambda c: strip_fog(c.get_asset("FogSettings"))),
]


# Cumulative chain: progressively peel layers from the outside in.
# Order: gameplay-logic -> world-effects -> water -> blends -> textures
#        -> objects/teams/mp -> masks -> heightmap.
def _cum_logic(c):
    strip_scripts(c.get_asset("PlayerScriptsList"))
    strip_triggers(c.get_asset("TriggerAreas"))
    strip_missions(c.get_asset("MissionObjectives"), c.get_asset("MissionHotSpots"))
    strip_build_lists(c.get_asset("BuildLists"))
    strip_library(c.get_asset("LibraryMaps"), c.get_asset("LibraryMapLists"))
    strip_post_effects(c.get_asset("PostEffectsChunk"))
    strip_fog(c.get_asset("FogSettings"))


def _cum_water(c):
    _cum_logic(c)
    strip_water_areas(c.get_asset("StandingWaterAreas"),
                      c.get_asset("RiverAreas"),
                      c.get_asset("StandingWaveAreas"))


def _cum_blends(c):
    _cum_water(c)
    strip_blends(_blend(c))


def _cum_textures(c):
    _cum_blends(c)
    strip_textures(_blend(c))


def _cum_objects(c):
    _cum_textures(c)
    strip_objects(_objects(c))
    strip_teams(c.get_asset("Teams"))
    strip_mp_restrictions(c.get_asset("MPPositionList"))


def _cum_masks(c):
    _cum_objects(c)
    reset_passability(_blend(c))
    reset_buildability(_blend(c))
    reset_visibility(_blend(c))
    reset_dynamic_shrubbery(_blend(c))
    reset_tib_growability(_blend(c))


def _cum_skeleton(c):
    _cum_masks(c)
    flatten_heightmap(_heightmap(c))


CUMULATIVE_LAYERS = [
    ("cum1_logic_off",     "Cumulative: scripts/triggers/missions/builds/library/post-fx/fog off",
     _cum_logic),
    ("cum2_water_off",     "+ water areas off",     _cum_water),
    ("cum3_blends_off",    "+ blends off",          _cum_blends),
    ("cum4_textures_off",  "+ textures off",        _cum_textures),
    ("cum5_objects_off",   "+ objects/teams/mp off", _cum_objects),
    ("cum6_masks_off",     "+ masks normalised",    _cum_masks),
    ("cum7_skeleton",      "+ heightmap flat (skeleton)", _cum_skeleton),
]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", type=Path, required=True)
    ap.add_argument("--out_dir", type=Path, default=None,
                    help="Output dir; default: <src_dir>/skinned/")
    ap.add_argument("--no_render", action="store_true", help="Skip rendering PNGs")
    ap.add_argument("--only", choices=["iso", "cum"], default=None,
                    help="Limit to isolated or cumulative outputs")
    args = ap.parse_args()

    if not args.src.exists():
        raise SystemExit(f"Source not found: {args.src}")
    out_dir = args.out_dir or (args.src.parent / "skinned")
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Source: {args.src}")
    print(f"Output: {out_dir}\n")

    if not args.no_render:
        render(args.src, out_dir / "_renders" / "00_original")

    do_iso = args.only in (None, "iso")
    do_cum = args.only in (None, "cum")

    if do_iso:
        print("=== ISOLATED single-layer strips ===")
        for name, desc, fn in ISOLATED_LAYERS:
            out_path = out_dir / f"{name}.map"
            print(f"  [{name}]  {desc}")
            try:
                save_copy(args.src, fn, out_path)
            except Exception as e:  # noqa: BLE001
                print(f"    [error] {e}")
                continue
            if not args.no_render:
                render(out_path, out_dir / "_renders" / name)

    if do_cum:
        print("\n=== CUMULATIVE progressive strips ===")
        for name, desc, fn in CUMULATIVE_LAYERS:
            out_path = out_dir / f"{name}.map"
            print(f"  [{name}]  {desc}")
            try:
                save_copy(args.src, fn, out_path)
            except Exception as e:  # noqa: BLE001
                print(f"    [error] {e}")
                continue
            if not args.no_render:
                render(out_path, out_dir / "_renders" / name)

    total = (len(ISOLATED_LAYERS) if do_iso else 0) + (len(CUMULATIVE_LAYERS) if do_cum else 0)
    print(f"\nDone. {total} stripped maps written to {out_dir}")
    if not args.no_render:
        print(f"Renders under {out_dir / '_renders'}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
