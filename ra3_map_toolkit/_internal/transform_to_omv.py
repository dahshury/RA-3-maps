"""
Transform a normal RA3 multiplayer map into one of the 5 OMV (Official Map Variations) modes:

  NO_SW          - Disables all faction superweapons.
  NO_AIR         - Disables aircraft, anti-air units, and AA structures.
  NO_SW_NO_AIR   - Combines NO_SW + NO_AIR + No Upgrades.
  INF_ONLY       - Only infantry units; war factories / airfields / heavy navy disabled.
  TANKS_ONLY     - Only ground vehicles; infantry and most navy disabled.

OMV scripts originally by Jenkins (jenkinsmedia.com.au), preserved here as donor map
templates so the resulting maps are bit-compatible with the original OMV pack.

The transform is straightforward:
  1. Parse source map.
  2. Parse donor template (one of the 5 variants).
  3. Copy the donor's `_neutral_` ScriptList scripts (the OMV restriction script set)
     into the source's `_neutral_` ScriptList.
  4. Update WorldInfo `mapName` and `mapDescription` to add the variation suffix /
     attribution.
  5. Save the transformed map alongside copied/generated sidecar art.

Usage:
  python transform_to_omv.py --in map.map --out variant_map.map --variation nosw
"""

from __future__ import annotations

import argparse
import copy
import shutil
import sys
from pathlib import Path
from typing import Optional

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from map_processor.core.ra3map import Ra3Map
from map_processor.core.ra3map_struct import MapDataContext
from transform_to_archon import remap_script, _generate_art_tga


# OMV variation metadata. Order matches the user-facing list (NO_SW first).
# The display name suffix and description are taken from the original OMV pack
# (verified against donor maps in side/_ULTIMATE RA3 OMV PACK).
OMV_VARIATIONS = {
    "nosw": {
        "label": "NO SW",
        "display_suffix": "[NO SW]",
        "filename_suffix": "nosw",
        "description": "No Superweapons script by Jenkins\r\nwww.jenkinsmedia.com.au",
        "template": "nosw.map",
    },
    "noair": {
        "label": "NO AIR",
        "display_suffix": "[NO AIR]",
        "filename_suffix": "noair",
        "description": "No Air script by Jenkins\r\nwww.jenkinsmedia.com.au",
        "template": "noair.map",
    },
    "nosw_noair": {
        "label": "NO SW/NO AIR",
        "display_suffix": "[NO SW/NO AIR]",
        "filename_suffix": "nosw_noair",
        "description": "No Superweapons, No Upgrades & No Air scripts\r\nby Jenkins\r\nwww.jenkinsmedia.com.au",
        "template": "nosw_noair.map",
    },
    "inf_only": {
        "label": "INF ONLY",
        "display_suffix": "[INF ONLY]",
        "filename_suffix": "inf_only",
        "description": "No Tanks, No Navy & No Air scripts by Jenkins\r\nwww.jenkinsmedia.com.au",
        "template": "inf_only.map",
    },
    "tanks_only": {
        "label": "TANKS ONLY",
        "display_suffix": "[TANKS ONLY]",
        "filename_suffix": "tanks_only",
        "description": "No Infantry, No Navy & No Air scripts by Jenkins\r\nwww.jenkinsmedia.com.au",
        "template": "tanks_only.map",
    },
}


# Asset-type strings the script structures need in the target map's string pool.
_SCRIPT_ASSET_STRINGS = (
    "PlayerScriptsList",
    "ScriptList",
    "ScriptGroup",
    "Script",
    "OrCondition",
    "Condition",
    "ScriptAction",
    "ScriptActionFalse",
)


def get_variation(key: str) -> dict:
    if key not in OMV_VARIATIONS:
        raise ValueError(f"unknown OMV variation {key!r}; valid keys: {list(OMV_VARIATIONS)}")
    return OMV_VARIATIONS[key]


def get_template_path(key: str, internal_dir: Optional[Path] = None) -> Path:
    """Locate the donor template .map for a variation (bundled in templates/omv/)."""
    base = internal_dir if internal_dir else _ROOT
    return base / "templates" / "omv" / OMV_VARIATIONS[key]["template"]


def copy_neutral_scripts(source_ctx: MapDataContext, donor_ctx: MapDataContext) -> int:
    """
    Copy the donor's `script_lists[0]` (the `_neutral_` GameScriptList) scripts
    into the source map's `_neutral_` ScriptList.

    OMV scripts target `<All Players>` with no per-player references, so no
    player-offset remapping is needed.

    Returns the number of scripts copied.
    """
    for s in _SCRIPT_ASSET_STRINGS:
        source_ctx.map_struct.register_string(s)

    src_psl = source_ctx.get_asset("PlayerScriptsList")
    donor_psl = donor_ctx.get_asset("PlayerScriptsList")
    if not src_psl or not donor_psl:
        raise RuntimeError("PlayerScriptsList missing from source or donor map")
    if not src_psl.script_lists or not donor_psl.script_lists:
        raise RuntimeError("PlayerScriptsList is empty in source or donor map")

    donor_neutral = donor_psl.script_lists[0]
    src_neutral = src_psl.script_lists[0]

    count = 0
    for s in donor_neutral.scripts:
        copied = copy.deepcopy(s)
        remap_script(copied, source_ctx, player_offset=0)
        src_neutral.scripts.append(copied)
        count += 1
    return count


def update_map_metadata(source_ctx: MapDataContext, base_display_name: str, variation_key: str) -> None:
    """
    Update WorldInfo `mapName` and `mapDescription` to add the OMV variation
    suffix and Jenkins attribution.
    """
    var = get_variation(variation_key)
    wi = source_ctx.get_asset("WorldInfo")
    if wi is None:
        raise RuntimeError("WorldInfo asset missing from source map")

    new_name = f"{base_display_name} {var['display_suffix']}"
    wi.properties.set_property("mapName", new_name)
    wi.properties.set_property("mapDescription", var["description"])


def transform_to_omv(source_ctx: MapDataContext,
                     donor_ctx: MapDataContext,
                     base_display_name: str,
                     variation_key: str) -> int:
    """High-level transform. Returns number of scripts copied."""
    n = copy_neutral_scripts(source_ctx, donor_ctx)
    update_map_metadata(source_ctx, base_display_name, variation_key)
    return n


def derive_display_name(map_path: Path) -> str:
    """
    Heuristic display name from an official map's filename. Tries to mirror what
    the OMV pack uses (e.g. `map_mp_2_feasel1` -> `Cabana Republic`-style mapping
    isn't recoverable from filename alone; we fall back to a Title-Cased stem).

    For the toolkit we accept that maps with empty WorldInfo.mapName get a
    derived name. Callers can override.
    """
    stem = map_path.stem
    for prefix in ("map_mp_", "map_"):
        if stem.lower().startswith(prefix):
            stem = stem[len(prefix):]
            break
    parts = [p for p in stem.replace("-", "_").split("_") if p]
    return " ".join(p.capitalize() for p in parts) or stem


def variant_filename(source_path: Path, variation_key: str) -> str:
    """
    Build the OMV-style filename for a variant: `<source_stem>_<suffix>`.
    Matches the original OMV pack convention (e.g. `map_mp_2_feasel1_nosw`).
    """
    return f"{source_path.stem}_{OMV_VARIATIONS[variation_key]['filename_suffix']}"


def parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="transform_to_omv",
        description="Apply an OMV variation (NO_SW, NO_AIR, etc.) to an RA3 map.",
    )
    p.add_argument("--in", dest="input", required=True, help="Source .map file")
    p.add_argument("--out", dest="output", required=True, help="Output .map file")
    p.add_argument("--variation", required=True, choices=list(OMV_VARIATIONS),
                   help="Which OMV variation to apply")
    p.add_argument("--display-name", default=None,
                   help="Base display name (override). Defaults to source map's WorldInfo.mapName or filename.")
    p.add_argument("--no-compress", action="store_true", help="Save uncompressed")
    return p.parse_args(argv)


def main(argv: list[str]) -> int:
    args = parse_args(argv)

    src_path = Path(args.input)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    src = Ra3Map(str(src_path))
    src.parse()
    src_ctx = src.get_context()

    template = get_template_path(args.variation)
    if not template.exists():
        print(f"ERROR: donor template missing: {template}", file=sys.stderr)
        return 1

    donor = Ra3Map(str(template))
    donor.parse()
    donor_ctx = donor.get_context()

    if args.display_name:
        base_name = args.display_name
    else:
        wi = src_ctx.get_asset("WorldInfo")
        existing = wi.properties.get_property("mapName") if wi else None
        base_name = (existing.data if existing else "") or derive_display_name(src_path)

    n = transform_to_omv(src_ctx, donor_ctx, base_name, args.variation)
    src.save(str(out_path), compress=not args.no_compress)
    _generate_art_tga(out_path, src_ctx, source_map_path=src_path)
    # Copy preview .tga next to the source if present.
    preview = src_path.parent / f"{src_path.stem}.tga"
    if preview.exists():
        shutil.copy2(preview, out_path.parent / f"{out_path.stem}.tga")

    print(f"OK: {src_path.name} + {args.variation} -> {out_path.name} ({n} scripts copied)")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
