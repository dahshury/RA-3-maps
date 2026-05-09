"""
Transform a normal RA3 multiplayer map into an Archon mode map.

Archon mode allows two players to share a base:
- Builder: Handles base building, unit production, and secret protocols
- Controller: Controls combat units in battle

This script transforms a 1-3 player map into a 2-6 player archon map by:
1. Adding controller player slots (paired with each original player)
2. Adding controller start positions (offset from builder positions)
3. Adding required teams for the archon system
4. Copying archon scripts from a template map

Usage:
  python scripts/transform_to_archon.py --in map.map --out archon_map.map --template archon_template.map
"""

from __future__ import annotations

import argparse
import copy
import math
from pathlib import Path
import sys
from typing import Dict, List, Tuple, Optional, Any, Set
import re
import shutil
import random

# Add parent to path
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from map_processor.core.ra3map import Ra3Map
from map_processor.core.ra3map_struct import MapDataContext
from map_processor.assets.multiplayer.mp_position_list import MPPositionList
from map_processor.assets.multiplayer.mp_position_info import MPPositionInfo
from map_processor.assets.sides.sides_list import SidesList
from map_processor.assets.sides.player import Player
from map_processor.assets.objects.objects_list import ObjectsList
from map_processor.assets.objects.map_object import MapObject
from map_processor.assets.teams.teams import Teams
from map_processor.assets.teams.team import Team
from map_processor.assets.scripts.player_scripts_list import PlayerScriptsList
from map_processor.assets.scripts.script_list import ScriptList
from map_processor.assets.scripts.script_group import ScriptGroup
from map_processor.assets.scripts.script import Script
from map_processor.assets.scripts.or_condition import OrCondition
from map_processor.assets.scripts.script_action import ScriptAction, ScriptActionFalse
from map_processor.assets.scripts.condition import Condition
from map_processor.assets.scripts.script_content import ScriptContent
from map_processor.assets.scripts.script_argument import ScriptArgument
from map_processor.assets.assets.asset_property import AssetProperty, AssetPropertyCollection, AssetPropertyType
from map_processor.core.default_major_asset import DefaultMajorAsset
from map_processor.assets.assets.asset_list import AssetBlock
from map_processor.assets.library.library_maps import LibraryMaps
from map_processor.assets.build.build_list import BuildList

# Import minimap generator for TGA generation
from minimap_generator import generate_minimap, save_minimap_tga

_TEXTURE_ATLAS_LOOKUP: Optional[Dict[str, Tuple[str, int]]] = None
_TEXTURE_NAME_MAP: Optional[Dict[str, Tuple[str, str]]] = None


def _build_texture_atlas_lookup() -> Dict[str, Tuple[str, int]]:
    """
    Build a lookup table: `BaseTexture filename (no ext)` -> `(NormalTexture filename (no ext), TextureID)`
    as seen in WB/official `map.xml`.

    We avoid reverse-engineering the TextureID hash by mining the existing `map.xml` files
    already present in this repo (official maps + prior WB exports). In this repo the mapping
    is consistent and conflict-free.
    """
    global _TEXTURE_ATLAS_LOOKUP
    if _TEXTURE_ATLAS_LOOKUP is not None:
        return _TEXTURE_ATLAS_LOOKUP

    official_root = _ROOT.parent / "RA3 Official maps"
    lookup: Dict[str, Tuple[str, int]] = {}
    if not official_root.exists():
        _TEXTURE_ATLAS_LOOKUP = lookup
        return lookup

    # Match a WB `<Tile ... />` regardless of line endings/indentation.
    pat = re.compile(
        r'BaseTexture="ART:Terrain/([^"]+)\.tga"\s+'
        r'NormalTexture="ART:Terrain/([^"]+)\.tga"\s+'
        r'TextureID="(\d+)"',
        flags=re.MULTILINE,
    )

    for fp in official_root.rglob("map.xml"):
        try:
            txt = fp.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue
        for base, normal, tid in pat.findall(txt):
            if base not in lookup:
                lookup[base] = (normal, int(tid))

    _TEXTURE_ATLAS_LOOKUP = lookup
    return lookup


def _build_texture_name_map() -> Dict[str, Tuple[str, str]]:
    """
    Build mapping: internal BlendTileData texture name -> (BaseTexture(no ext), NormalTexture(no ext)).

    This is needed for universal correctness because some internal names map to non-obvious
    filenames (e.g. `Cliff_Iceland03` -> `TClif_Iceland03`).
    """
    global _TEXTURE_NAME_MAP
    if _TEXTURE_NAME_MAP is not None:
        return _TEXTURE_NAME_MAP

    # Comprehensive hardcoded fallback from C# WorldInfo.cs and Resources.cs
    # This ensures the script works without needing the C# source files present
    _EMBEDDED_TEXTURE_MAP: Dict[str, Tuple[str, str]] = {
        # Romania
        "Pavement_Romania01": ("Pavement_Romania01", "Pavement_Romania01_NRM"),
        "Dirt_Romania01": ("Dirt_Romania01", "Dirt_Romania01_NRM"),
        "Dock_Romania01": ("Dock_Romania01", "Dock_Romania01_NRM"),
        # Hot_Springs
        "Dirt_HotSprings01": ("TDirt_HotSprings01", "TDirt_HotSprings01_NRM"),
        "Dirt_HotSprings02": ("TDirt_HotSprings02", "TDirt_HotSprings02_NRM"),
        "Dirt_HotSprings03": ("TDirt_HotSprings03", "TDirt_HotSprings03_NRM"),
        "Dirt_HotSprings04": ("TDirt_HotSprings04", "TDirt_HotSprings04_NRM"),
        "Dirt_HotSprings05": ("TDirt_HotSprings05", "TDirt_HotSprings05_NRM"),
        "Dirt_HotSprings06": ("TDirt_HotSprings06", "TDirt_HotSprings06_NRM"),
        "Dirt_HotSprings07": ("TDirt_HotSprings07", "TDirt_HotSprings07_NRM"),
        "Dirt_HotSprings08": ("TDirt_HotSprings08", "TDirt_HotSprings08_NRM"),
        "Grass_HotSprings01": ("TGrass_HotSprings01", "TGrass_HotSprings01_NRM"),
        "Grass_HotSprings02": ("TGrass_HotSprings02", "TGrass_HotSprings02_NRM"),
        "Grass_HotSprings03": ("TGrass_HotSprings03", "TGrass_HotSprings03_NRM"),
        "Grass_HotSprings04": ("TGrass_HotSprings04", "TGrass_HotSprings04_NRM"),
        "Grass_HotSprings05": ("TGrass_HotSprings05", "TGrass_HotSprings05_NRM"),
        "Grass_HotSprings06": ("TGrass_HotSprings06", "TGrass_HotSprings06_NRM"),
        "Grass_HotSprings07": ("TGrass_HotSprings07", "TGrass_HotSprings07_NRM"),
        "Grass_HotSprings08": ("TGrass_HotSprings08", "TGrass_HotSprings08_NRM"),
        "Transition_HotSprings01": ("Transition_HotSprings01", "Transition_HotSprings01_NRM"),
        "Transition_HotSprings02": ("Transition_HotSprings02", "Transition_HotSprings02_NRM"),
        "Transition_HotSprings03": ("Transition_HotSprings03", "Transition_HotSprings03_NRM"),
        "Transition_HotSprings04": ("Transition_HotSprings04", "Transition_HotSprings04_NRM"),
        # Island_Fortress / Misc
        "SteelDeck01": ("TMisc_SteelDeck01", "TMisc_SteelDeck01_nrm"),
        "SteelDeck02": ("TMisc_SteelDeck02", "TMisc_SteelDeck02_nrm"),
        "SteelDeck03": ("TMisc_SteelDeck03", "TMisc_SteelDeck03_nrm"),
        "SteelDeck04": ("TMisc_SteelDeck04", "TMisc_SteelDeck04_nrm"),
        "SteelDeck05": ("TMisc_SteelDeck05", "TMisc_SteelDeck05_nrm"),
        "SteelDeck06": ("TMisc_SteelDeck06", "TMisc_SteelDeck06_nrm"),
        "Asphalt01": ("TMisc_Asphalt01", "TMisc_Asphalt01_nrm"),
        "Asphalt02": ("TMisc_Asphalt02", "TMisc_Asphalt02_nrm"),
        "Asphalt03": ("TMisc_Asphalt03", "TMisc_Asphalt03_nrm"),
        "Asphalt04": ("TMisc_Asphalt04", "TMisc_Asphalt04_nrm"),
        "Asphalt05": ("TMisc_Asphalt05", "TMisc_Asphalt05_nrm"),
        "FortressBlackEdge": ("TMisc_BlackEdge", "TMisc_BlackEdge_nrm"),
        "Mud_Fortress01": ("Mud_Fortress01", "Mud_Fortress01_NRM"),
        "Mud_Fortress02": ("Mud_Fortress02", "Mud_Fortress02_NRM"),
        # Yucatan
        "Grass_Yucatan01": ("TGrass_Yucatan01", "TGrass_Yucatan01_nrm"),
        "Grass_Yucatan02": ("TGrass_Yucatan02", "TGrass_Yucatan02_nrm"),
        "Grass_Yucatan03": ("TGrass_Yucatan03", "TGrass_Yucatan03_nrm"),
        "Grass_Yucatan04": ("TGrass_Yucatan04", "TGrass_Yucatan04_nrm"),
        "Grass_Yucatan05": ("TGrass_Yucatan05", "TGrass_Yucatan05_nrm"),
        "Grass_Yucatan06": ("TGrass_Yucatan06", "TGrass_Yucatan06_nrm"),
        "Grass_Yucatan07": ("TGrass_Yucatan07", "TGrass_Yucatan07_nrm"),
        "Grass_Yucatan08": ("TGrass_Yucatan08", "TGrass_Yucatan08_nrm"),
        "Grass_Yucatan09": ("TGrass_Yucatan09", "TGrass_Yucatan09_nrm"),
        "Rock_Yucatan01": ("TRock_Yucatan01", "TRock_Yucatan01_nrm"),
        "Rock_Yucatan02": ("TRock_Yucatan02", "TRock_Yucatan02_nrm"),
        "Rock_Yucatan03": ("TRock_Yucatan03", "TRock_Yucatan03_nrm"),
        "Rock_Yucatan04": ("TRock_Yucatan04", "TRock_Yucatan04_nrm"),
        "Rock_Yucatan05": ("TRock_Yucatan05", "TRock_Yucatan05_nrm"),
        "Gravel_Yucatan01": ("TGravel_Yucatan01", "TGravel_Yucatan01_nrm"),
        "Dirt_Yucatan01": ("TDirt_Yucatan01", "TDirt_Yucatan01_nrm"),
        "Dirt_Yucatan02": ("TDirt_Yucatan02", "TDirt_Yucatan02_nrm"),
        "Dirt_Yucatan03": ("TDirt_Yucatan03", "TDirt_Yucatan03_nrm"),
        "Dirt_Yucatan04": ("TDirt_Yucatan04", "TDirt_Yucatan04_nrm"),
        "Dirt_Yucatan05": ("TDirt_Yucatan05", "TDirt_Yucatan05_nrm"),
        "Dirt_Yucatan06": ("TDirt_Yucatan06", "TDirt_Yucatan06_nrm"),
        "Reef_Yucatan01": ("TReef_Yucatan01", "TReef_Yucatan01_nrm"),
        "Reef_Yucatan02": ("TReef_Yucatan02", "TReef_Yucatan02_nrm"),
        "Pavement_Yucatan01": ("TPavement_Yucatan01", "TPavement_Yucatan01_nrm"),
        "Transition_Yucatan01": ("TTransition_Yucatan01", "TTransition_Yucatan01_nrm"),
        "Transition_Yucatan02": ("TTransition_Yucatan02", "TTransition_Yucatan02_nrm"),
        "Transition_Yucatan03": ("TTransition_Yucatan03", "TTransition_Yucatan03_nrm"),
        "Transition_Yucatan04": ("TTransition_Yucatan04", "TTransition_Yucatan04_nrm"),
        "Transition_Yucatan05": ("TTransition_Yucatan05", "TTransition_Yucatan05_nrm"),
        "Transition_Yucatan06": ("TTransition_Yucatan06", "TTransition_Yucatan06_nrm"),
        "Transition_Yucatan07": ("TTransition_Yucatan07", "TTransition_Yucatan07_nrm"),
        "Transition_Yucatan08": ("TTransition_Yucatan08", "TTransition_Yucatan08_nrm"),
        "Transition_Yucatan09": ("TTransition_Yucatan09", "TTransition_Yucatan09_nrm"),
        "Transition_Yucatan10": ("TTransition_Yucatan10", "TTransition_Yucatan10_nrm"),
        "Transition_Yucatan11": ("TTransition_Yucatan11", "TTransition_Yucatan11_nrm"),
        "Transition_Yucatan12": ("TTransition_Yucatan12", "TTransition_Yucatan12_nrm"),
        "Transition_Yucatan13": ("TTransition_Yucatan13", "TTransition_Yucatan13_nrm"),
        "Transition_Yucatan14": ("TTransition_Yucatan14", "TTransition_Yucatan14_nrm"),
        "Transition_Yucatan15": ("TTransition_Yucatan15", "TTransition_Yucatan15_nrm"),
        "Transition_Yucatan16": ("TTransition_Yucatan16", "TTransition_Yucatan16_nrm"),
        "Transition_Yucatan17": ("TTransition_Yucatan17", "TTransition_Yucatan17_nrm"),
        "Transition_Yucatan18": ("TTransition_Yucatan18", "TTransition_Yucatan18_nrm"),
        "Transition_Yucatan19": ("TTransition_Yucatan19", "TTransition_Yucatan19_nrm"),
        "Transition_Yucatan20": ("TTransition_Yucatan20", "TTransition_Yucatan20_nrm"),
        "Transition_Yucatan21": ("TTransition_Yucatan21", "TTransition_Yucatan21_nrm"),
        # Golf_Course
        "Pave_Golf01": ("TPave_Golf01", "TPave_Golf01_NRM"),
        "Pave_Golf02": ("TPave_Golf02", "TPave_Golf02_NRM"),
        "Pave_Golf03": ("TPave_Golf03", "TPave_Golf03_NRM"),
        "Pave_Golf04": ("TPave_Golf04", "TPave_Golf04_NRM"),
        "Pave_Golf05": ("TPave_Golf05", "TPave_Golf05_NRM"),
        "Pave_Golf06": ("TPave_Golf06", "TPave_Golf06_NRM"),
        "Pave_Golf07": ("TPave_Golf07", "TPave_Golf07_NRM"),
        "Grass_Golf01": ("TGrass_Golf01", "TGrass_Golf01_NRM"),
        "Grass_Golf02": ("TGrass_Golf02", "TGrass_Golf02_NRM"),
        "Grass_Golf03": ("TGrass_Golf03", "TGrass_Golf03_NRM"),
        "Grass_Golf04": ("TGrass_Golf04", "TGrass_Golf04_NRM"),
        "Grass_Golf05": ("TGrass_Golf05", "TGrass_Golf05_NRM"),
        "Grass_Golf06": ("TGrass_Golf06", "TGrass_Golf06_NRM"),
        "Grass_Golf07": ("TGrass_Golf07", "TGrass_Golf07_NRM"),
        "Grass_Golf08": ("TGrass_Golf08", "TGrass_Golf08_NRM"),
        # Solvang
        "Snow_Solvang01": ("TSnow_Solvang01", "TSnow_Solvang01_nrm"),
        "Snow_Solvang02": ("TSnow_Solvang02", "TSnow_Solvang02_nrm"),
        "Snow_Solvang03": ("TSnow_Solvang03", "TSnow_Solvang03_nrm"),
        "Snow_Solvang04": ("TSnow_Solvang04", "TSnow_Solvang04_nrm"),
        "Snow_Solvang05": ("TSnow_Solvang05", "TSnow_Solvang05_nrm"),
        "Snow_Solvang06": ("TSnow_Solvang06", "TSnow_Solvang06_nrm"),
        "Snow_Solvang07": ("TSnow_Solvang07", "TSnow_Solvang07_nrm"),
        "Snow_Solvang08": ("TSnow_Solvang08", "TSnow_Solvang08_NRM"),
        "Snow_Solvang09": ("TSnow_Solvang09", "TSnow_Solvang09_NRM"),
        "Snow_Solvang10": ("TSnow_Solvang10", "Temp_NRM"),
        "Cliff_Solvang01": ("TClif_Solvang01", "TClif_Solvang01_NRM"),
        "Cliff_Solvang02": ("TClif_Solvang02", "TClif_Solvang02_NRM"),
        "Cliff_Solvang03": ("TClif_Solvang03", "TClif_Solvang03_NRM"),
        "Cliff_Solvang04": ("TClif_Solvang04", "TClif_Solvang04_NRM"),
        "Rock_Solvang01": ("TRock_Solvang01", "Temp_NRM"),
        "Rock_Solvang02": ("TRock_Solvang02", "Temp_NRM"),
        "Pavement_Solvang01": ("TPave_Solvang01", "TPave_Solvang01_nrm"),
        # Heidelberg
        "Grass_Heidelberg01": ("Grass_Heidelberg01", "Grass_Heidelberg01_nrm"),
        "Grass_Heidelberg02": ("Grass_Heidelberg02", "Grass_Heidelberg02_nrm"),
        "Grass_Heidelberg03": ("Grass_Heidelberg03", "Grass_Heidelberg03_nrm"),
        "Grass_Heidelberg04": ("Grass_Heidelberg04", "Grass_Heidelberg04_nrm"),
        "Grass_Heidelberg05": ("Grass_Heidelberg05", "Grass_Heidelberg05_nrm"),
        "Grass_Heidelberg06": ("Grass_Heidelberg06", "Grass_Heidelberg06_nrm"),
        "Grass_Heidelberg07": ("Grass_Heidelberg07", "Grass_Heidelberg07_nrm"),
        "Grass_Heidelberg08": ("Grass_Heidelberg08", "Grass_Heidelberg08_nrm"),
        "Grass_Heidelberg09": ("Grass_Heidelberg09", "Grass_Heidelberg09_nrm"),
        "Grass_Heidelberg10": ("Grass_Heidelberg10", "Grass_Heidelberg10_nrm"),
        "Grass_Heidelberg11": ("Grass_Heidelberg11", "Grass_Heidelberg11_nrm"),
        "Transition_Heidelberg01": ("Transition_Heidelberg01", "Transition_Heidelberg01_nrm"),
        "Transition_Heidelberg02": ("Transition_Heidelberg02", "Transition_Heidelberg02_nrm"),
        "Transition_Heidelberg03": ("Transition_Heidelberg03", "Transition_Heidelberg03_nrm"),
        "Transition_Heidelberg04": ("Transition_Heidelberg04", "Transition_Heidelberg04_nrm"),
        "Transition_Heidelberg05": ("Transition_Heidelberg05", "Transition_Heidelberg05_nrm"),
        "Transition_Heidelberg06": ("Transition_Heidelberg06", "Transition_Heidelberg06_nrm"),
        "Transition_Heidelberg07": ("Transition_Heidelberg07", "Transition_Heidelberg07_nrm"),
        "Transition_Heidelberg08": ("Transition_Heidelberg08", "Transition_Heidelberg08_nrm"),
        "Pavement_Heidel01": ("TPave_Heidel01", "TPave_Heidel01_nrm"),
        "Pavement_Heidel02": ("TPave_Heidel02", "TPave_Heidel02_nrm"),
        "Pavement_Heidel03": ("TPave_Heidel03", "TPave_Heidel03_nrm"),
        "Pavement_Heidel04": ("TPave_Heidel04", "TPave_Heidel04_nrm"),
        "Pavement_Heidel05": ("TPave_Heidel05", "TPave_Heidel05_nrm"),
        "Pavement_Heidel06": ("TPave_Heidel06", "TPave_Heidel06_nrm"),
        "Pavement_Heidel07": ("TPave_Heidel07", "TPave_Heidel07_nrm"),
        "Pavement_Heidel08": ("TPave_Heidel08", "TPave_Heidel08_nrm"),
        "Pavement_Heidel09": ("TPave_Heidel09", "TPave_Heidel09_nrm"),
        "Pavement_Heidelberg10": ("Pavement_Heidelberg10", "Pavement_Heidelberg10_nrm"),
        "Pavement_Heidelberg11": ("Pavement_Heidelberg11", "Pavement_Heidelberg11_nrm"),
        "Pavement_Heidelberg12": ("Pavement_Heidelberg12", "Pavement_Heidelberg12_nrm"),
        "Gravel_Heidelberg01": ("Gravel_Heidelberg01", "Gravel_Heidelberg01_NRM"),
        "Dirt_Heidelberg01": ("Dirt_Heidelberg01", "Dirt_Heidelberg01_NRM"),
        # Geneva
        "Grass_Geneva01": ("TGrass_Geneva01", "TGrass_Geneva01_NRM"),
        "Grass_Geneva02": ("TGrass_Geneva02", "TGrass_Geneva02_NRM"),
        "Grass_Geneva03": ("TGrass_Geneva03", "TGrass_Geneva03_NRM"),
        "Grass_Geneva04": ("TGrass_Geneva04", "TGrass_Geneva04_NRM"),
        "Grass_Geneva05": ("TGrass_Geneva05", "TGrass_Geneva05_NRM"),
        "Grass_GenevaClockA": ("TGrass_GenevaClockA", "TGrass_GenevaClockA_NRM"),
        "Grass_GenevaClockB": ("TGrass_GenevaClockB", "TGrass_GenevaClockB_NRM"),
        "Grass_GenevaClockC": ("TGrass_GenevaClockC", "TGrass_GenevaClockC_NRM"),
        "Grass_GenevaClockD": ("TGrass_GenevaClockD", "TGrass_GenevaClockD_NRM"),
        "Pavement_Geneva01": ("TPave_Geneva01", "TPave_Geneva01_NRM"),
        "Pavement_Geneva02": ("TPave_Geneva02", "TPave_Geneva02_NRM"),
        "Pavement_Geneva03": ("TPave_Geneva03", "TPave_Geneva03_NRM"),
        "Pavement_Geneva04": ("TPave_Geneva04", "TPave_Geneva04_NRM"),
        "Pavement_Geneva05": ("TPave_Geneva05", "TPave_Geneva05_NRM"),
        "Pavement_Geneva06": ("TPave_Geneva06", "TPave_Geneva06_NRM"),
        # Cannes
        "Sand_Cannes01": ("TSand_Cannes01", "TSand_Cannes01_NRM"),
        "Sand_Cannes02": ("TSand_Cannes02", "TSand_Cannes02_NRM"),
        "Sand_Cannes03": ("TSand_Cannes03", "TSand_Cannes03_NRM"),
        "Sand_Cannes04": ("TSand_Cannes04", "TSand_Cannes04_NRM"),
        "Sand_Cannes05": ("TSand_Cannes05", "TSand_Cannes05_NRM"),
        "Sand_Cannes06": ("TSand_Cannes06", "TSand_Cannes06_NRM"),
        "Sand_Cannes07": ("TSand_Cannes07", "TSand_Cannes07_NRM"),
        "Sand_Cannes08": ("TSand_Cannes08", "TSand_Cannes08_NRM"),
        "Sand_Cannes09": ("TSand_Cannes09", "TSand_Cannes09_NRM"),
        "Sand_Cannes10": ("TSand_Cannes10", "TSand_Cannes10_NRM"),
        "Pave_Cannes01": ("TPave_Cannes01", "TPave_Cannes01_NRM"),
        "Pave_Cannes02": ("TPave_Cannes02", "TPave_Cannes02_NRM"),
        "Pave_Cannes03": ("TPave_Cannes03", "TPave_Cannes03_NRM"),
        # Havana
        "Mud_Havana01": ("Mud_Havana01", "Mud_Havana01_NRM"),
        "Mud_Havana02": ("Mud_Havana02", "Mud_Havana02_NRM"),
        "Pavement_Havana01": ("Pavement_Havana01", "Pavement_Havana01_NRM"),
        "Pavement_Havana02": ("Pavement_Havana02", "Pavement_Havana02_NRM"),
        "Pavement_Havana03": ("Pavement_Havana03", "Pavement_Havana03_NRM"),
        "Pavement_Havana04": ("Pavement_Havana04", "Pavement_Havana04_NRM"),
        "Pavement_Havana05": ("Pavement_Havana05", "Pavement_Havana05_NRM"),
        "Reef_Havana01": ("Reef_Havana01", "Reef_Havana01_NRM"),
        "Reef_Havana02": ("Reef_Havana02", "Reef_Havana02_NRM"),
        # Mykonos
        "Pavement_Mykonos01": ("TPave_Mykonos01", "TPave_Mykonos01_NRM"),
        "Pavement_Mykonos02": ("TPave_Mykonos02", "TPave_Mykonos02_NRM"),
        "Pavement_Mykonos03": ("TPave_Mykonos03", "TPave_Mykonos03_NRM"),
        "Pavement_Mykonos04": ("TPave_Mykonos04", "TPave_Mykonos04_NRM"),
        "Pavement_Mykonos05": ("TPave_Mykonos05", "TPave_Mykonos05_NRM"),
        "Dirt_Mykonos01": ("TDirt_Mykonos01", "TDirt_Mykonos01_NRM"),
        "Dirt_Mykonos02": ("TDirt_Mykonos02", "TDirt_Mykonos02_NRM"),
        "Dirt_Mykonos03": ("TDirt_Mykonos03", "TDirt_Mykonos03_NRM"),
        "Dirt_Mykonos04": ("TDirt_Mykonos04", "TDirt_Mykonos04_NRM"),
        "Dirt_Mykonos05": ("TDirt_Mykonos05", "TDirt_Mykonos05_NRM"),
        "Dirt_Mykonos06": ("TDirt_Mykonos06", "TDirt_Mykonos06_NRM"),
        "Dirt_Mykonos07": ("TDirt_Mykonos07", "TDirt_Mykonos07_NRM"),
        "Grass_Mykonos01": ("TGrass_Mykonos01", "TGrass_Mykonos01_NRM"),
        "Grass_Mykonos02": ("TGrass_Mykonos02", "TGrass_Mykonos02_NRM"),
        # Kremlin
        "Pavement_Kremlin01": ("Pavement_Kremlin01", "Pavement_Kremlin01_NRM"),
        "Pavement_Kremlin02": ("Pavement_Kremlin02", "Pavement_Kremlin02_NRM"),
        "Pavement_Kremlin03": ("Pavement_Kremlin03", "Pavement_Kremlin03_NRM"),
        "Pavement_Kremlin04": ("Pavement_Kremlin04", "Pavement_Kremlin04_NRM"),
        "Pavement_Kremlin05": ("Pavement_Kremlin05", "Pavement_Kremlin05_NRM"),
        "Pavement_Kremlin06": ("Pavement_Kremlin06", "Pavement_Kremlin06_NRM"),
        "Transition_Kremlin01": ("Transition_Kremlin01", "Transition_Kremlin01_NRM"),
        "Transition_Kremlin02": ("Transition_Kremlin02", "Transition_Kremlin02_NRM"),
        "Transition_Kremlin03": ("Transition_Kremlin03", "Transition_Kremlin03_NRM"),
        "Transition_Kremlin04": ("Transition_Kremlin04", "Transition_Kremlin04_NRM"),
        "Transition_Kremlin05": ("Transition_Kremlin05", "Transition_Kremlin05_NRM"),
        "Transition_Kremlin06": ("Transition_Kremlin06", "Transition_Kremlin06_NRM"),
        "Transition_Kremlin07": ("Transition_Kremlin07", "Transition_Kremlin07_NRM"),
        "Transition_Kremlin08": ("Transition_Kremlin08", "Transition_Kremlin08_NRM"),
        # Odessa
        "Pavement_Odessa01": ("Pavement_Odessa01", "Pavement_Odessa01_NRM"),
        "Rock_Odessa01": ("Rock_Odessa01", "Rock_Odessa01_NRM"),
        # Santa_Monica
        "Pave_SantaMonica01": ("TPave_SantaMonica01", "TPave_SantaMonica01_NRM"),
        "Pave_SantaMonica02": ("TPave_SantaMonica02", "TPave_SantaMonica02_NRM"),
        "Pave_SantaMonica03": ("TPave_SantaMonica03", "TPave_SantaMonica03_NRM"),
        "Pave_SantaMonica04": ("TPave_SantaMonica04", "TPave_SantaMonica04_NRM"),
        "Pave_SantaMonica05": ("TPave_SantaMonica05", "TPave_SantaMonica05_NRM"),
        "Sand_SantaMonica01": ("TSand_SantaMonica01", "TSand_SantaMonica01_NRM"),
        "Sand_SantaMonica02": ("TSand_SantaMonica02", "TSand_SantaMonica02_NRM"),
        "Grass_SantaMonica01": ("Grass_SantaMonica01", "Grass_SantaMonica01_NRM"),
        # Saint_Petersburg
        "Pavement_SaintPetersburg01": ("Pavement_SaintPetersburg01", "Pavement_SaintPetersburg01_NRM"),
        "Pavement_SaintPetersburg02": ("Pavement_SaintPetersburg02", "Pavement_SaintPetersburg02_NRM"),
        "Pavement_SaintPetersburg03": ("Pavement_SaintPetersburg03", "Pavement_SaintPetersburg03_NRM"),
        "Pavement_SaintPetersburg04": ("Pavement_SaintPetersburg04", "Pavement_SaintPetersburg04_NRM"),
        "Pavement_SaintPetersburg05": ("Pavement_SaintPetersburg05", "Pavement_SaintPetersburg05_NRM"),
        "Pavement_SaintPetersburg06": ("Pavement_SaintPetersburg06", "Pavement_SaintPetersburg06_NRM"),
        "Pavement_SaintPetersburg07": ("Pavement_SaintPetersburg07", "Pavement_SaintPetersburg07_NRM"),
        "Pavement_SaintPetersburg08": ("Pavement_SaintPetersburg08", "Pavement_SaintPetersburg08_NRM"),
        "Grass_SaintPetersburg01": ("Grass_SaintPetersburg01", "Grass_SaintPetersburg01_NRM"),
        # Easter_Island
        "Dirt_Easter01": ("TDirt_Easter01", "TDirt_Easter01_NRM"),
        "Dirt_Easter02": ("TDirt_Easter02", "TDirt_Easter02_NRM"),
        "Dirt_Easter03": ("TDirt_Easter03", "TDirt_Easter03_NRM"),
        "Dirt_Easter04": ("TDirt_Easter04", "TDirt_Easter04_NRM"),
        "Dirt_Easter05": ("TDirt_Easter05", "TDirt_Easter05_NRM"),
        "Dirt_Easter06": ("TDirt_Easter06", "TDirt_Easter06_NRM"),
        "Dirt_Easter07": ("TDirt_Easter07", "TDirt_Easter07_NRM"),
        "Dirt_Easter08": ("TDirt_Easter08", "TDirt_Easter08_NRM"),
        "Dirt_Easter09": ("TDirt_Easter09", "TDirt_Easter09_NRM"),
        "Dirt_Easter10": ("TDirt_Easter10", "TDirt_Easter10_NRM"),
        "Dirt_Easter11": ("TDirt_Easter11", "TDirt_Easter11_NRM"),
        "Dirt_Easter12": ("TDirt_Easter12", "TDirt_Easter12_NRM"),
        "Grass_Easter01": ("TGrass_Easter01", "TGrass_Easter01_NRM"),
        "Grass_Easter02": ("TGrass_Easter02", "TGrass_Easter02_NRM"),
        "Grass_Easter03": ("TGrass_Easter03", "TGrass_Easter03_NRM"),
        "Grass_Easter04": ("TGrass_Easter04", "TGrass_Easter04_NRM"),
        "Grass_Easter05": ("TGrass_Easter05", "TGrass_Easter05_NRM"),
        "Grass_Easter06": ("TGrass_Easter06", "TGrass_Easter06_NRM"),
        "Grass_Easter07": ("TGrass_Easter07", "TGrass_Easter07_NRM"),
        "Cliff_Easter01": ("TCliff_Easter01", "TCliff_Easter01_NRM"),
        "Cliff_Easter02": ("TCliff_Easter02", "TCliff_Easter02_NRM"),
        "Cliff_Easter03": ("TCliff_Easter03", "TCliff_Easter03_NRM"),
        "Cliff_Easter04": ("TCliff_Easter04", "TCliff_Easter04_NRM"),
        # Cape_Cod
        "Grass_CapeCod01": ("TGrass_CapeCod01", "TGrass_CapeCod01_nrm"),
        "Grass_CapeCod02": ("TGrass_CapeCod02", "TGrass_CapeCod02_nrm"),
        "Grass_CapeCod03": ("TGrass_CapeCod03", "TGrass_CapeCod03_nrm"),
        "Grass_CapeCod04": ("TGrass_CapeCod04", "TGrass_CapeCod04_nrm"),
        "Grass_CapeCod05": ("TGrass_CapeCod05", "TGrass_CapeCod05_nrm"),
        "Grass_CapeCod06": ("TGrass_CapeCod06", "TGrass_CapeCod06_nrm"),
        "Grass_CapeCod07": ("TGrass_CapeCod07", "TGrass_CapeCod07_nrm"),
        "Grass_CapeCod08": ("TGrass_CapeCod08", "TGrass_CapeCod08_nrm"),
        "Grass_CapeCod09": ("TGrass_CapeCod09", "TGrass_CapeCod09_nrm"),
        "Grass_CapeCod10": ("TGrass_CapeCod10", "TGrass_CapeCod10_nrm"),
        "Grass_CapeCod11": ("TGrass_CapeCod11", "TGrass_CapeCod11_nrm"),
        "Grass_CapeCod12": ("TGrass_CapeCod12", "TGrass_CapeCod12_nrm"),
        "Grass_CapeCod13": ("TGrass_CapeCod13", "TGrass_CapeCod13_nrm"),
        "Grass_CapeCod14": ("TGrass_CapeCod14", "TGrass_CapeCod14_nrm"),
        "Grass_CapeCod15": ("TGrass_CapeCod15", "TGrass_CapeCod15_nrm"),
        "Grass_CapeCod16": ("TGrass_CapeCod16", "TGrass_CapeCod16_nrm"),
        "Grass_CapeCod17": ("TGrass_CapeCod17", "TGrass_CapeCod17_nrm"),
        "Grass_CapeCod18": ("TGrass_CapeCod18", "TGrass_CapeCod18_nrm"),
        "Grass_CapeCod19": ("TGrass_CapeCod19", "TGrass_CapeCod19_nrm"),
        "Grass_CapeCod20": ("TGrass_CapeCod20", "TGrass_CapeCod20_nrm"),
        "Grass_CapeCod21": ("TGrass_CapeCod21", "TGrass_CapeCod21_nrm"),
        "Grass_CapeCod22": ("TGrass_CapeCod22", "TGrass_CapeCod22_nrm"),
        "Grass_CapeCod23": ("TGrass_CapeCod23", "TGrass_CapeCod23_nrm"),
        "Grass_CapeCod24": ("TGrass_CapeCod24", "TGrass_CapeCod24_nrm"),
        "Pavement_CapeCod01": ("TPave_CapeCod01", "TPave_CapeCod01_nrm"),
        "Pavement_CapeCod02": ("TPave_CapeCod02", "TPave_CapeCod02_nrm"),
        "Dirt_CapeCod01": ("TDirt_CapeCod01", "TDirt_CapeCod01_nrm"),
        "Dirt_CapeCod02": ("TDirt_CapeCod02", "TDirt_CapeCod02_nrm"),
        "Dirt_CapeCod03": ("TDirt_CapeCod03", "TDirt_CapeCod03_nrm"),
        "Dirt_CapeCod04": ("TDirt_CapeCod04", "TDirt_CapeCod04_nrm"),
        "Dirt_CapeCod05": ("TDirt_CapeCod05", "TDirt_CapeCod05_nrm"),
        "Dirt_CapeCod06": ("TDirt_CapeCod06", "TDirt_CapeCod06_nrm"),
        "Dirt_CapeCod07": ("TDirt_CapeCod07", "TDirt_CapeCod07_nrm"),
        "Dirt_CapeCod08": ("TDirt_CapeCod08", "TDirt_CapeCod08_nrm"),
        "Cliff_CapeCod01": ("TCliff_CapeCod01", "TCliff_CapeCod01_nrm"),
        "Cliff_CapeCod02": ("TCliff_CapeCod02", "TCliff_CapeCod02_nrm"),
        "Cliff_CapeCod03": ("TCliff_CapeCod03", "TCliff_CapeCod03_nrm"),
        "Cliff_CapeCod04": ("TCliff_CapeCod04", "TCliff_CapeCod04_nrm"),
        "Cliff_CapeCod05": ("TCliff_CapeCod05", "TCliff_CapeCod05_nrm"),
        # New_York
        "Pavement_NewYork01": ("TPave_NewYork01", "TPave_NewYork01_NRM"),
        "Pavement_NewYork02": ("TPave_NewYork02", "TPave_NewYork02_NRM"),
        "Pavement_NewYork03": ("TPave_NewYork03", "TPave_NewYork03_NRM"),
        "Grass_NewYork01": ("TGrass_NewYork01", "TGrass_NewYork01_NRM"),
        # Mount_Rushmore
        "Pavement_MtRush01": ("TPave_MtRush01", "TPave_MtRush01_NRM"),
        "Grass_MtRush01": ("Grass_MtRush01", "Grass_MtRush01_NRM"),
        "Grass_MtRush02": ("Grass_MtRush02", "Grass_MtRush02_NRM"),
        "Grass_MtRush03": ("Grass_MtRush03", "Grass_MtRush03_NRM"),
        "Grass_MtRush04": ("Grass_MtRush04", "Grass_MtRush04_NRM"),
        "Grass_MtRush05": ("Grass_MtRush05", "Grass_MtRush05_NRM"),
        "Grass_MtRush06": ("Grass_MtRush06", "Grass_MtRush06_NRM"),
        "Grass_MtRush07": ("Grass_MtRush07", "Grass_MtRush07_NRM"),
        "Grass_MtRush08": ("Grass_MtRush08", "Grass_MtRush08_NRM"),
        "Grass_MtRush09": ("Grass_MtRush09", "Grass_MtRush09_NRM"),
        "Grass_MtRush10": ("Grass_MtRush10", "Grass_MtRush10_NRM"),
        "Grass_MtRush11": ("Grass_MtRush11", "Grass_MtRush11_NRM"),
        "Grass_MtRush12": ("Grass_MtRush12", "Grass_MtRush12_NRM"),
        "Grass_MtRush13": ("Grass_MtRush13", "Grass_MtRush13_NRM"),
        "Cliff_MtRush01": ("Cliff_MtRush01", "Cliff_MtRush01_NRM"),
        "Cliff_MtRush02": ("Cliff_MtRush02", "Cliff_MtRush02_NRM"),
        # Amsterdam
        "Pavement_Amsterdam01": ("TPave_Amsterdam01", "TPave_Amsterdam01_NRM"),
        "Pavement_Amsterdam02": ("TPave_Amsterdam02", "TPave_Amsterdam02_NRM"),
        "Pavement_Amsterdam03": ("TPave_Amsterdam03", "TPave_Amsterdam03_NRM"),
        "Pavement_Amsterdam04": ("TPave_Amsterdam04", "TPave_Amsterdam04_NRM"),
        "Pavement_Amsterdam05": ("TPave_Amsterdam05", "TPave_Amsterdam05_NRM"),
        # Iceland (note: Clif not Cliff)
        "Cliff_Iceland01": ("TClif_Iceland01", "TClif_Iceland01_NRM"),
        "Cliff_Iceland02": ("TClif_Iceland02", "TClif_Iceland02_NRM"),
        "Cliff_Iceland03": ("TClif_Iceland03", "TClif_Iceland03_NRM"),
        "Cliff_Iceland04": ("TClif_Iceland04", "TClif_Iceland04_NRM"),
        "Cliff_Iceland05": ("TClif_Iceland05", "TClif_Iceland05_NRM"),
        "Cliff_Iceland06": ("TClif_Iceland06", "TClif_Iceland06_NRM"),
        "Cliff_Iceland07": ("TClif_Iceland07", "TClif_Iceland07_NRM"),
        "Snow_Iceland02": ("TSnow_Iceland02", "TSnow_Iceland02_NRM"),
        "Snow_Iceland03": ("TSnow_Iceland03", "TSnow_Iceland03_NRM"),
        "Snow_Iceland04": ("TSnow_Iceland04", "TSnow_Iceland04_NRM"),
        "Snow_Iceland05": ("TSnow_Iceland05", "TSnow_Iceland05_NRM"),
        "Rock_Iceland01": ("TRock_Iceland01", "TRock_Iceland01_NRM"),
        "Rock_Iceland02": ("TRock_Iceland02", "TRock_Iceland02_NRM"),
        "Rock_Iceland03": ("TRock_Iceland03", "TRock_Iceland03_NRM"),
        "Rock_Iceland04": ("TRock_Iceland04", "TRock_Iceland04_NRM"),
        "Dirt_Iceland02": ("TDirt_Iceland02", "TDirt_Iceland02_NRM"),
        "Dirt_Iceland03": ("TDirt_Iceland03", "TDirt_Iceland03_NRM"),
        "Dirt_Iceland04": ("TDirt_Iceland04", "TDirt_Iceland04_NRM"),
        "Dirt_Iceland05": ("TDirt_Iceland05", "TDirt_Iceland05_NRM"),
        "Dirt_Iceland06": ("TDirt_Iceland06", "TDirt_Iceland06_NRM"),
        "Transition_Iceland01": ("Transition_Iceland01", "Transition_Iceland01_NRM"),
        "Transition_Iceland02": ("Transition_Iceland02", "Transition_Iceland02_NRM"),
        "Transition_Iceland03": ("Transition_Iceland03", "Transition_Iceland03_NRM"),
        "Transition_Iceland04": ("Transition_Iceland04", "Transition_Iceland04_NRM"),
        # Tokyo_Harbor
        "Pavement_TokyoHarbor01": ("Pavement_TokyoHarbor01", "Pavement_TokyoHarbor01_NRM"),
        "Pavement_TokyoHarbor02": ("Pavement_TokyoHarbor02", "Pavement_TokyoHarbor02_NRM"),
        "Pavement_TokyoHarbor03": ("Pavement_TokyoHarbor03", "Pavement_TokyoHarbor03_NRM"),
        "Pavement_TokyoHarbor04": ("Pavement_TokyoHarbor04", "Pavement_TokyoHarbor04_NRM"),
        "Pavement_TokyoHarbor05": ("TPave_TokyoHarbor05", "TPave_TokyoHarbor05_NRM"),
        "Pavement_TokyoHarbor06": ("TPave_TokyoHarbor06", "TPave_TokyoHarbor06_NRM"),
        "Pavement_TokyoHarbor07": ("TPave_TokyoHarbor07", "TPave_TokyoHarbor07_NRM"),
        "Pavement_TokyoHarbor08": ("TPave_TokyoHarbor08", "TPave_TokyoHarbor08_NRM"),
        "Pavement_TokyoHarbor09": ("Pavement_TokyoHarbor09", "Pavement_TokyoHarbor09_NRM"),
        "Pavement_TokyoHarbor10": ("TPave_TokyoHarbor10", "TPave_TokyoHarbor10_NRM"),
        # RA3 Base
        "RA3_DeepOcean": ("RA3_DeepOcean", "RA3_DeepOcean_NRM"),
        "RA3_ShallowSeaFloor": ("RA3_ShallowSeaFloor", "RA3_ShallowSeaFloor_NRM"),
        "RA3_Elevation0": ("RA3_Elevation0", "RA3_Elevation0_NRM"),
        "RA3_Elevation1": ("RA3_Elevation1", "RA3_Elevation1_NRM"),
        "RA3_Elevation2": ("RA3_Elevation2", "RA3_Elevation2_NRM"),
        "RA3Grid1": ("RA3Grid1", "RA3Grid1_NRM"),
        # Gibraltar (note: Gibralter typo in original)
        "Cliff_Gibraltar1": ("Cliff_Gibralter1", "Cliff_Gibralter1_NRM"),
        "Cliff_Gibraltar2": ("Cliff_Gibralter2", "Cliff_Gibralter2_NRM"),
        "Pavement_Gibraltar1": ("Pavement_Gibraltar01", "Pavement_Gibraltar01_NRM"),
        "Pavement_Gibraltar2": ("Pavement_Gibraltar02", "Pavement_Gibraltar02_NRM"),
        "Pavement_Gibraltar3": ("Pavement_Gibraltar03", "Pavement_Gibraltar03_NRM"),
        "Pavement_Gibraltar5": ("Pavement_GibraltarBoardwalk", "Pavement_GibraltarBoardwalk_NRM"),
        "Pavement_Gibraltar6": ("Pavement_Gibraltar04", "Pavement_Gibraltar04_NRM"),
        "Grass_Gibraltar1": ("Grass_Gibraltar01", "Grass_Gibraltar01_NRM"),
        "Grass_Gibraltar2": ("Grass_Gibraltar02", "Grass_Gibraltar02_NRM"),
        "Grass_Gibraltar3": ("Grass_Gibraltar03", "Grass_Gibraltar03_NRM"),
        # Gypsy_Village
        "Grass_Gypsy01": ("TGrass_Gypsy01", "TGrass_Gypsy01_NRM"),
        "Grass_Gypsy02": ("TGrass_Gypsy02", "TGrass_Gypsy02_NRM"),
        "Grass_Gypsy03": ("TGrass_Gypsy03", "TGrass_Gypsy03_NRM"),
        "Grass_Gypsy04": ("TGrass_Gypsy04", "TGrass_Gypsy04_NRM"),
        "Dirt_Gypsy01": ("TDirt_Gypsy01", "TDirt_Gypsy01_NRM"),
        "Dirt_Gypsy02": ("TDirt_Gypsy02", "TDirt_Gypsy02_NRM"),
        "Dirt_Gypsy03": ("TDirt_Gypsy03", "TDirt_Gypsy03_NRM"),
        # Hawaii
        "Grass_Hawaii01": ("Grass_Hawaii01", "Grass_Hawaii01_NRM"),
        "Grass_Hawaii02": ("Grass_Hawaii02", "Grass_Hawaii02_NRM"),
        "Grass_Hawaii03": ("Grass_Hawaii03", "Grass_Hawaii03_NRM"),
        "Grass_Hawaii04": ("Grass_Hawaii04", "Grass_Hawaii04_NRM"),
        "Grass_Hawaii05": ("Grass_Hawaii05", "Grass_Hawaii05_NRM"),
        "Grass_Hawaii06": ("Grass_Hawaii06", "Grass_Hawaii06_NRM"),
        "Grass_Hawaii07": ("Grass_Hawaii07", "Grass_Hawaii07_NRM"),
        "Grass_Hawaii08": ("Grass_Hawaii08", "Grass_Hawaii08_NRM"),
        "Grass_Hawaii09": ("Grass_Hawaii09", "Grass_Hawaii09_NRM"),
        "Sand_Hawaii01": ("TSand_Hawaii01", "TSand_Hawaii01_NRM"),
        "Sand_Hawaii02": ("TSand_Hawaii02", "TSand_Hawaii02_NRM"),
        "Sand_Hawaii03": ("TSand_Hawaii03", "TSand_Hawaii03_NRM"),
        "Sand_Hawaii04": ("TSand_Hawaii04", "TSand_Hawaii04_NRM"),
        # Vladivostok
        "Pavement_Vlad01": ("Pavement_Vlad01", "Pavement_Vlad01_NRM"),
        "Pavement_Vlad02": ("Pavement_Vlad02", "Pavement_Vlad02_NRM"),
        "Pavement_Vlad03": ("Pavement_Vlad03", "Pavement_Vlad03_NRM"),
        "Pavement_Vlad04": ("Pavement_Vlad04", "Pavement_Vlad04_NRM"),
        "Pavement_Vlad05": ("Pavement_Vlad05", "Pavement_Vlad05_NRM"),
        "Pavement_Vlad06": ("Pavement_Vlad06", "Pavement_Vlad06_NRM"),
        "Pavement_Vlad07": ("Pavement_Vlad07", "Pavement_Vlad07_NRM"),
        "Pavement_Vlad08": ("Pavement_Vlad08", "Pavement_Vlad08_NRM"),
        "Pavement_Vlad09": ("Pavement_Vlad09", "Pavement_Vlad09_NRM"),
        "Pavement_Vlad10": ("Pavement_Vlad10", "Pavement_Vlad10_NRM"),
        "Pavement_Vlad11": ("Pavement_Vlad11", "Pavement_Vlad11_NRM"),
        "Pavement_Vlad12": ("Pavement_Vlad12", "Pavement_Vlad12_NRM"),
        # Brighton_Beach
        "BB_Gravel01": ("BB_Gravel01", "BB_Gravel01_NRM"),
        "BB_Gravel02": ("BB_Gravel02", "BB_Gravel02_NRM"),
        "BB_Dirt01": ("BB_Dirt01", "BB_Dirt01_NRM"),
        "BB_Dirt02": ("BB_Dirt02", "BB_Dirt02_NRM"),
        "BB_Pavement01": ("BB_Pavement01", "BB_Pavement01_NRM"),
        "BB_Pavement02": ("BB_Pavement02", "BB_Pavement02_NRM"),
    }

    # Start with the embedded fallback
    mapping: Dict[str, Tuple[str, str]] = dict(_EMBEDDED_TEXTURE_MAP)

    # Try to load additional mappings from C# sources if available
    workspace_root = _ROOT.parent.parent
    sources = [
        workspace_root / "Ra3Solution" / "MapCoreLib" / "Core" / "Asset" / "WorldInfo.cs",
        workspace_root / "MapCreatorCore" / "MapCreatorCoreLib" / "Core" / "Utility" / "Resources.cs",
    ]

    entry_pat = re.compile(r'\{\s*"([^"]+)"\s*,\s*"([^"]+)"\s*\}')
    for fp in sources:
        if not fp.exists():
            continue
        try:
            txt = fp.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue
        for key, value in entry_pat.findall(txt):
            parts = [p for p in value.split(";") if p]
            if len(parts) >= 2:
                mapping.setdefault(key, (parts[0], parts[1]))

    _TEXTURE_NAME_MAP = mapping
    return mapping


def _fast_hash_id(s: str) -> int:
    """
    Port of C# `FashHash.GetHashCode` (lowercased ASCII, seed=length, FastHashFn).

    We use this to compute TerrainTextureAtlas TextureID universally (no lookup gaps).
    """
    b = s.lower().encode("ascii", errors="ignore")
    if not b:
        return 0

    h = len(b) & 0xFFFFFFFF
    n = len(b)
    i = 0
    while n >= 4:
        h = (h + int.from_bytes(b[i:i+2], "little")) & 0xFFFFFFFF
        tmp = (int.from_bytes(b[i+2:i+4], "little") ^ ((h << 5) & 0xFFFFFFFF)) & 0xFFFFFFFF
        h ^= (tmp << 11) & 0xFFFFFFFF
        h = (h + (h >> 11)) & 0xFFFFFFFF
        i += 4
        n -= 4

    if n == 1:
        h = (h + b[i]) & 0xFFFFFFFF
        h ^= (h << 10) & 0xFFFFFFFF
        h = (h + (h >> 1)) & 0xFFFFFFFF
    elif n == 2:
        h = (h + int.from_bytes(b[i:i+2], "little")) & 0xFFFFFFFF
        h ^= (h << 11) & 0xFFFFFFFF
        h = (h + (h >> 17)) & 0xFFFFFFFF
    elif n == 3:
        h = (h + int.from_bytes(b[i:i+2], "little")) & 0xFFFFFFFF
        h ^= (h << 16) & 0xFFFFFFFF
        h ^= (b[i+2] << 18) & 0xFFFFFFFF
        h = (h + (h >> 11)) & 0xFFFFFFFF

    h ^= (h << 3) & 0xFFFFFFFF
    h = (h + (h >> 5)) & 0xFFFFFFFF
    h ^= (h << 2) & 0xFFFFFFFF
    h = (h + (h >> 15)) & 0xFFFFFFFF
    h ^= (h << 10) & 0xFFFFFFFF
    return h & 0xFFFFFFFF


def _compute_texture_id(base_name: str) -> int:
    """
    Compute TextureID the way official map.xml files do:
    - If base starts with `TMisc_`, hash the suffix (e.g. TMisc_SteelDeck01 -> SteelDeck01)
    - Else if base starts with `T`, hash without the leading `T` (e.g. TDirt_Yucatan01 -> Dirt_Yucatan01)
    - Else hash the base name as-is (e.g. Pavement_Heidelberg12)
    """
    if base_name.startswith("TMisc_"):
        key = base_name[len("TMisc_"):]
    elif base_name.startswith("T"):
        key = base_name[1:]
    else:
        key = base_name
    return _fast_hash_id(key)


def _resolve_texture_entry(texture_name: str, atlas_lookup: Dict[str, Tuple[str, int]]) -> Tuple[str, str, int]:
    """
    Resolve internal texture name -> (base, normal, texture_id).

    Prefers explicit name mappings from the C# resources, then falls back to heuristic.
    TextureID is computed via the FastHash algorithm if lookup is missing/0.
    """
    name_map = _build_texture_name_map()
    if texture_name in name_map:
        base, normal = name_map[texture_name]
    else:
        base = _resolve_wb_base_name(texture_name, atlas_lookup) or ("T" + texture_name)
        normal = atlas_lookup.get(base, (base + "_NRM", 0))[0]

    tid = atlas_lookup.get(base, (normal, 0))[1]
    if tid == 0:
        tid = _compute_texture_id(base)
    return base, normal, tid

def _resolve_wb_base_name(texture_name: str, atlas_lookup: Dict[str, Tuple[str, int]]) -> Optional[str]:
    """
    Resolve an internal BlendTileData texture name (e.g. 'SteelDeck01') into the WB/engine
    base texture filename (e.g. 'TMisc_SteelDeck01') using a lookup mined from existing map.xmls.
    """
    if texture_name in atlas_lookup and atlas_lookup[texture_name][1] != 0:
        return texture_name

    candidates: List[str] = []
    # Common patterns:
    candidates.append("T" + texture_name)
    candidates.append("TMisc_" + texture_name)
    # Some families sometimes omit T:
    candidates.append(texture_name)

    # Family-specific tweaks (cover common official oddities)
    if texture_name.startswith("RA3_"):
        candidates.insert(0, texture_name)
    if texture_name.startswith("Gravel_"):
        candidates.insert(0, texture_name)
        candidates.insert(1, "T" + texture_name)
    if texture_name.startswith("Pavement_"):
        candidates.insert(0, "T" + texture_name)

    matches = [c for c in candidates if c in atlas_lookup]
    if not matches:
        return None

    # Prefer candidates with a non-zero TextureID (0 is commonly used as a placeholder
    # for legacy/unsupported names in some map.xmls).
    matches.sort(key=lambda c: (atlas_lookup[c][1] == 0, candidates.index(c)))
    return matches[0]


def _extract_road_ids_from_map_file(map_path: Path) -> List[str]:
    """
    Extract road IDs by scanning the raw (decompressed) `.map` bytes.

    These IDs often exist as plain strings in the file but are not necessarily parsed into
    our current asset model (so scanning the raw bytes is the most reliable, general approach).
    """
    try:
        data = map_path.read_bytes()
    except Exception:
        return []

    # Decompress if needed (RefPack)
    if len(data) >= 4:
        flag = int.from_bytes(data[:4], "little", signed=False)
        # Import lazily to avoid import-order issues when running as a script.
        from map_processor.utils.constants import UNCOMPRESSED_FLAG, COMPRESSED_FLAG
        from map_processor.utils.refpack import RefPackDecompressor

        if flag == COMPRESSED_FLAG and len(data) >= 8:
            from io import BytesIO
            src = BytesIO(data)
            src.seek(8)
            out = BytesIO()
            RefPackDecompressor.decompress(src, out)
            data = out.getvalue()
        elif flag == UNCOMPRESSED_FLAG:
            pass

    text = data.decode("latin-1", errors="ignore")
    pat = re.compile(r'([A-Za-z]+(?:Road|Sidewalk|Footpath|ParkingLines)\d{2})')
    found = sorted(set(pat.findall(text)))
    return found


def _write_minimal_map_xml(out_path: Path, context: MapDataContext, map_stem: str, 
                           source_map_path: Optional[Path] = None,
                           is_multiplayer: bool = True) -> None:
    """
    Write a minimal WB-style `map.xml` that is sufficient for in-game terrain textures.
    
    Args:
        is_multiplayer: Whether this is a multiplayer map (default True for archon maps)
    """
    blend = context.get_asset("BlendTileData")
    world_info = context.get_asset("WorldInfo")
    height = context.get_asset("HeightMapData")
    if not blend or not world_info or not height:
        return

    atlas_lookup = _build_texture_atlas_lookup()

    # Start positions: include InitialCameraPosition + all Player_*_Start we can find.
    starts = find_player_starts(context)
    starts_sorted = sorted(starts, key=lambda s: s.unique_id)
    num_players = len(starts_sorted)

    # Properties
    border = int(getattr(height, "border_width", context.border if context.border != -1 else 0))
    w = int(context.map_width)
    h = int(context.map_height)

    tts_prop = world_info.properties.get_property("terrainTextureStrings")
    tts = tts_prop.data if tts_prop else ""

    # Minimal include set (enough for WB/game to resolve basic instances)
    includes = [
        ('DATA:static.xml', 'reference'),
        ('DATA:global.xml', 'reference'),
        ('DATA:audio.xml', 'reference'),
        ('ART:EVDefault.xml', 'instance'),
        ('ART:LUSaturateColors_Vol.xml', 'instance'),
        ('ART:TSCloudMed.xml', 'instance'),
        ('ART:TSNoiseUrb.xml', 'instance'),
        ('DATA:GlobalData/roads.xml', 'instance'),
    ]

    def esc(s: str) -> str:
        return (s.replace("&", "&amp;")
                 .replace("<", "&lt;")
                 .replace(">", "&gt;")
                 .replace('"', "&quot;"))

    lines: List[str] = []
    lines.append('<?xml version="1.0" encoding="UTF-8"?>')
    lines.append('<AssetDeclaration xmlns="uri:ea.com:eala:asset" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">')
    lines.append('\t<Tags/>')
    lines.append('\t<Includes>')
    for src, typ in includes:
        lines.append('\t\t<Include')
        lines.append(f'\t\t\tsource="{esc(src)}"')
        lines.append(f'\t\t\ttype="{esc(typ)}"/>')
    lines.append('\t</Includes>')

    # GameMap + MapMetaData
    lines.append('\t<GameMap>')

    # Emit Road IDs (these affect road overlays/markings). We can derive them by scanning
    # string property values in the map (they are not necessarily in the string pool).
    road_ids = _extract_road_ids_from_map_file(source_map_path) if source_map_path else []
    for rid in road_ids:
        lines.append(f'\t\t<Road id="{esc(rid)}"/>')

    lines.append('\t\t<MapMetaData')
    lines.append(f'\t\t\tBorderSize="{border}"')
    lines.append('\t\t\tCRC="0"')
    lines.append(f'\t\t\tDescription="Map:{esc(map_stem)}/Desc"')
    lines.append(f'\t\t\tDisplayName="{esc(map_stem)}"')
    lines.append('\t\t\tFileName="data"')
    lines.append(f'\t\t\tHeight="{h}"')
    lines.append(f'\t\t\tIsMultiplayer="{str(is_multiplayer).lower()}"')
    lines.append('\t\t\tIsOfficial="false"')
    lines.append(f'\t\t\tNumPlayers="{num_players}"')
    lines.append(f'\t\t\tWidth="{w}">')

    # Initial camera + start positions
    lines.append('\t\t\t<StartPosition Name="InitialCameraPosition">')
    lines.append('\t\t\t\t<Position x="0" y="0" z="0"/>')
    lines.append('\t\t\t</StartPosition>')
    for s in starts_sorted:
        x0, y0, z0 = s.position
        lines.append(f'\t\t\t<StartPosition Name="{esc(s.unique_id)}">')
        lines.append(f'\t\t\t\t<Position x="{x0}" y="{y0}" z="{z0}"/>')
        lines.append('\t\t\t</StartPosition>')
    lines.append('\t\t</MapMetaData>')

    # Environment + WorldDict (only what we need for terrain textures)
    lines.append('\t\t<EnvironmentData Cloud="TSCloudMed" Environment="EVDefault" Macro="TSNoiseUrb"/>')
    lines.append('\t\t<WorldDict>')
    if world_info.properties.get_property("musicZone"):
        lines.append('\t\t\t<AssetIdProperty Key="musicZone" Value="MusicPalette_NotSet"/>')
    if world_info.properties.get_property("weather"):
        lines.append('\t\t\t<IntProperty Key="weather" Value="0"/>')
    if tts:
        lines.append(f'\t\t\t<StringProperty Key="terrainTextureStrings" Value="{esc(str(tts))}"/>')
    lines.append('\t\t</WorldDict>')
    lines.append('\t\t<PostEffect Effect="LUSaturateColors_Vol"/>')
    lines.append('\t</GameMap>')

    # TerrainTextureAtlas
    lines.append(f'\t<TerrainTextureAtlas AllowLossyCompression="true" AtlasSize="2048" id="{esc(map_stem)}">')

    tiles: List[Tuple[str, str, int]] = []
    for t in blend.textures:
        base, normal, tid = _resolve_texture_entry(t.name, atlas_lookup)
        tiles.append((base, normal, tid))

    # Official map.xml files sort atlas tiles by TextureID (ascending), with 0s last.
    tiles.sort(key=lambda x: (x[2] == 0, x[2], x[0]))

    for base, normal, tid in tiles:
        lines.append('\t\t<Tile')
        lines.append(f'\t\t\tBaseTexture="ART:Terrain/{esc(base)}.tga"')
        lines.append(f'\t\t\tNormalTexture="ART:Terrain/{esc(normal)}.tga"')
        lines.append(f'\t\t\tTextureID="{tid}"/>')
    lines.append('\t</TerrainTextureAtlas>')
    lines.append('</AssetDeclaration>')

    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _extract_road_ids_from_context(context: MapDataContext) -> List[str]:
    """
    Extract `<Road id="..."/>` IDs by scanning string-valued properties across the map.

    These IDs often appear as raw strings in the `.map` payload (not in the string pool),
    e.g. `HavanaSidewalk03`, `YucatanDirtRoad01`.
    """
    pat = re.compile(r'^[A-Za-z]+(?:Road|Sidewalk|Footpath|ParkingLines)\d{2}$')
    found: set[str] = set()

    objs = context.get_asset("ObjectsList")
    if objs:
        for o in objs.map_objects:
            pc = o.asset_property_collection
            for prop in pc.property_map.values():
                if prop.property_type != AssetPropertyType.string_type:
                    continue
                if isinstance(prop.data, str) and pat.match(prop.data):
                    found.add(prop.data)

    # Deterministic ordering
    return sorted(found)


def _generate_art_tga(out_map_path: Path, context: MapDataContext, source_map_path: Optional[Path] = None) -> None:
    """
    Generate the _art.tga minimap from the source map's internal data.
    If the source map has an existing _art.tga, copy it. Otherwise generate from height/blend data.
    Never copies from template - always uses source map's own data.
    """
    out_dir = out_map_path.parent
    map_stem = out_map_path.stem
    out_art = out_dir / f"{map_stem}_art.tga"
    
    # Try source map's existing art first (if it has one)
    if source_map_path:
        src_art = source_map_path.parent / f"{source_map_path.stem}_art.tga"
        if src_art.exists():
            shutil.copy2(src_art, out_art)
            return
    
    # Generate minimap from map data
    if save_minimap_tga(context, out_art):
        print(f"  Generated minimap: {out_art.name}")
    else:
        print(f"  Warning: Could not generate minimap (missing height/blend data)")


def _write_sidecars(out_map_path: Path, context: MapDataContext, template_map_path: Optional[Path], source_map_path: Optional[Path]) -> None:
    """
    Write common RA3 sidecar files into the output folder:
    - `map.xml` (generated, minimal)
    - `overrides.xml` (minimal)
    - `map.str` + preview `.tga` files (copied from template folder when available)
    """
    out_dir = out_map_path.parent
    map_stem = out_map_path.stem

    # Minimal overrides.xml (matches what we've been using)
    overrides = out_dir / "overrides.xml"
    overrides.write_text(
        "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n"
        "<AssetDeclaration xmlns=\"uri:ea.com:eala:asset\" xmlns:xsi=\"http://www.w3.org/2001/XMLSchema-instance\">\n"
        "\t<Tags/>\n"
        "\t<Includes/>\n"
        "</AssetDeclaration>\n",
        encoding="utf-8",
    )

    # map.xml (generated)
    _write_minimal_map_xml(out_dir / "map.xml", context, map_stem, source_map_path=source_map_path)

    # map.str (localization): preserve SOURCE metadata, never blindly take it from template
    # because that can overwrite the converted map's displayed name/description in-game.
    if source_map_path:
        src_map_str = source_map_path.parent / "map.str"
        if src_map_str.exists():
            shutil.copy2(src_map_str, out_dir / "map.str")
    
    # Note: TGA files are generated from source map data, not copied from template


def _ensure_worldinfo_terrain_texture_strings(context: MapDataContext) -> None:
    """
    Ensure `WorldInfo` contains the `terrainTextureStrings` property, as WB writes it on save.
    """
    world_info = context.get_asset("WorldInfo")
    blend = context.get_asset("BlendTileData")
    if not world_info or not blend or not getattr(blend, "textures", None):
        return

    atlas_lookup = _build_texture_atlas_lookup()
    parts: List[str] = []
    for t in blend.textures:
        base, normal, _ = _resolve_texture_entry(t.name, atlas_lookup)
        parts.append(base)
        parts.append(normal)

    # WB's string ends with a trailing ';'
    s = ";".join(parts) + ";"

    existing = world_info.properties.get_property("terrainTextureStrings")
    if existing:
        existing.data = s
    else:
        world_info.properties.add_property("terrainTextureStrings", s, context)


def _next_unique_suffix(context: MapDataContext, prefix: str, default_start: int = 1000) -> int:
    """
    Find a numeric suffix for `uniqueID` strings of the form `"{prefix} <n>"`.
    Returns the next available integer.
    """
    objs = context.get_asset("ObjectsList")
    if not objs:
        return default_start
    best = None
    for o in objs.map_objects:
        uid = o.asset_property_collection.get_property("uniqueID")
        if not uid or not isinstance(uid.data, str):
            continue
        s = uid.data
        if not s.startswith(prefix + " "):
            continue
        try:
            n = int(s.split(" ", 1)[1])
        except Exception:
            continue
        best = n if best is None else max(best, n)
    return (best + 1) if best is not None else default_start


def _playable_bounds_world(context: MapDataContext) -> Tuple[float, float, float, float]:
    """
    Returns the playable-area bounds in world units (x_min, x_max, y_min, y_max).

    RA3 maps have a `border` (in tiles) which is not playable. Using border-derived bounds
    gives us a reliable "off-play" area to place archon helper objects (keepalive buildings,
    apron/airfield waypoints) without interfering with gameplay.
    """
    world_w = context.map_width * 10
    world_h = context.map_height * 10
    border_tiles = context.border if getattr(context, "border", None) not in (None, -1) else 20
    border_w = max(0, border_tiles) * 10
    x_min = border_w
    y_min = border_w
    x_max = world_w - border_w
    y_max = world_h - border_w
    return x_min, x_max, y_min, y_max


def _clamp(v: float, lo: float, hi: float) -> float:
    return lo if v < lo else (hi if v > hi else v)


def _offplay_anchor_near_builder(context: MapDataContext,
                                 builder_pos: Tuple[float, float, float],
                                 margin: float = 700.0) -> Tuple[float, float, float]:
    """
    Pick a deterministic off-play anchor position for a builder, by pushing just outside the
    *closest* playable-edge.

    This matches how the official/community archon maps place controller helper objects near edges
    (often within the border region rather than requiring extreme coordinates).
    """
    x_min, x_max, y_min, y_max = _playable_bounds_world(context)
    bx, by, _ = builder_pos

    # Clamp the "along-edge" coordinate to playable bounds so anchors don't fly off into extremes.
    cx = _clamp(bx, x_min, x_max)
    cy = _clamp(by, y_min, y_max)

    d_left = abs(bx - x_min)
    d_right = abs(x_max - bx)
    d_bottom = abs(by - y_min)
    d_top = abs(y_max - by)

    edge = min(
        [(d_left, "left"), (d_right, "right"), (d_bottom, "bottom"), (d_top, "top")],
        key=lambda t: t[0],
    )[1]

    if edge == "left":
        return (x_min - margin, cy, 0.0)
    if edge == "right":
        return (x_max + margin, cy, 0.0)
    if edge == "bottom":
        return (cx, y_min - margin, 0.0)
    # top
    return (cx, y_max + margin, 0.0)


def _create_controller_keepalive_building(controller_number: int,
                                         unique_suffix: int,
                                         position: Tuple[float, float, float],
                                         context: MapDataContext,
                                         *,
                                         type_name: str,
                                         object_name: str,
                                         object_enabled: bool,
                                         object_indestructible: bool) -> MapObject:
    """
    Create the off-map, untargetable 'keep-alive' building used by official Archon maps
    to prevent controller AIs from resigning (they would otherwise start with no units/buildings).

    This is **scheme-dependent**:
    - 1v1/2v2 consecutive-archon templates typically use `SovietPowerPlantAdvanced` and keep it enabled.
    - 3v3 paired-archon templates (HF/Caldera) use `EI_EasterIslandHeadDefense` and keep it disabled+indestructible.
    """
    o = MapObject()
    o.id = context.map_struct.register_string("Object")
    o.version = 3
    o.name = "Object"

    o.type_name = type_name
    o.position = position
    o.angle = -45.0
    o.road_option = 0

    pc = AssetPropertyCollection()

    def add(name: str, prop_type: AssetPropertyType, data: object) -> None:
        ap = AssetProperty()
        ap.property_type = prop_type
        ap.name = name
        ap.data = data
        ap.id = context.map_struct.register_string(name)
        pc.property_map[name] = ap

    # IMPORTANT: Preserve the property order used by official paired-3p Archon maps (HF/CoC).
    # Some SAGE engine subsystems can be surprisingly order-sensitive for Object properties.
    add("objectInitialHealth", AssetPropertyType.int_type, 100)
    add("objectEnabled", AssetPropertyType.bool_type, bool(object_enabled))
    add("objectIndestructible", AssetPropertyType.bool_type, bool(object_indestructible))
    add("objectUnsellable", AssetPropertyType.bool_type, True)
    add("objectPowered", AssetPropertyType.bool_type, True)
    add("objectRecruitableAI", AssetPropertyType.bool_type, False)
    add("objectTargetable", AssetPropertyType.bool_type, False)  # Cannot be targeted/attacked
    add("objectSleeping", AssetPropertyType.bool_type, False)
    add("objectBasePriority", AssetPropertyType.int_type, 40)
    add("objectBasePhase", AssetPropertyType.int_type, 1)
    add("originalOwner", AssetPropertyType.string_type, f"Player_{controller_number}/teamPlayer_{controller_number}")
    add("uniqueID", AssetPropertyType.string_type, f"{type_name} {unique_suffix}")
    add("objectLayer", AssetPropertyType.string_type, "")
    add("objectName", AssetPropertyType.string_type, object_name)

    o.asset_property_collection = pc
    return o


def _ensure_controller_keepalive_buildings(
    context: MapDataContext,
    num_builders: int,
    controller_numbers: Optional[List[int]] = None,
) -> None:
    """
    Ensure controller players own an off-map building so AI controllers do not resign.
    Skip if the controller already owns any building (e.g., from template).
    
    Places buildings in the non-playable border area, derived from builder start positions.
    """
    print("\nCreating controller keepalive buildings...")
    objs = context.get_asset("ObjectsList")
    if not objs:
        return

    if controller_numbers is None:
        # Consecutive scheme (used by 1v1/2v2 templates): controllers are num_builders+1..2*num_builders
        controller_numbers = [(i + 1) + num_builders for i in range(num_builders)]

    # Detect paired-3p scheme by explicit controller numbers [2,4,6]
    is_paired_3p = (num_builders == 3 and sorted(controller_numbers) == [2, 4, 6])
    builder_numbers = [1, 3, 5] if is_paired_3p else list(range(1, num_builders + 1))

    # Choose keepalive building type/properties based on scheme
    if is_paired_3p:
        keepalive_type = "EI_EasterIslandHeadDefense"
        object_enabled = False
        object_indestructible = True
        object_name_fmt = "Controller_Player_{n}"
        suffix = _next_unique_suffix(context, keepalive_type, default_start=4000)
    else:
        keepalive_type = "SovietPowerPlantAdvanced"
        object_enabled = True
        object_indestructible = False
        object_name_fmt = "P{n}P"
        suffix = _next_unique_suffix(context, keepalive_type, default_start=2000)

    # Map builder -> off-play anchor
    builder_start_positions: Dict[int, Tuple[float, float, float]] = {}
    for o in objs.map_objects:
        if getattr(o, "unique_id", None) in {f"Player_{n}_Start" for n in builder_numbers}:
            try:
                n = int(o.unique_id.split("_")[1])
            except Exception:
                continue
            builder_start_positions[n] = o.position

    # Pair up builders with controllers in a stable order
    pairs: List[Tuple[int, int]] = []
    if is_paired_3p:
        # Explicit pair mapping: 1-2, 3-4, 5-6
        pairs = [(1, 2), (3, 4), (5, 6)]
    else:
        pairs = [(b, c) for b, c in zip(builder_numbers, controller_numbers)]

    for builder_number, controller_number in pairs:
        owner_prefix = f"Player_{controller_number}/"
        # Check if controller already owns ANY object
        already_owns_something = False
        for o in objs.map_objects:
            prop_owner = o.asset_property_collection.get_property("originalOwner")
            if prop_owner and isinstance(prop_owner.data, str) and prop_owner.data.startswith(owner_prefix):
                already_owns_something = True
                break
        if already_owns_something:
            continue

        builder_pos = builder_start_positions.get(builder_number)
        if builder_pos is None:
            # Fallback to map center if a start is missing (should be rare)
            world_w = context.map_width * 10
            world_h = context.map_height * 10
            builder_pos = (world_w / 2, world_h / 2, 0.0)

        pos = _offplay_anchor_near_builder(context, builder_pos, margin=700.0)

        objs.map_objects.append(
            _create_controller_keepalive_building(
                controller_number,
                suffix,
                pos,
                context,
                type_name=keepalive_type,
                object_name=object_name_fmt.format(n=controller_number),
                object_enabled=object_enabled,
                object_indestructible=object_indestructible,
            )
        )
        print(f"  Created keepalive for Player_{controller_number} at ({pos[0]:.1f}, {pos[1]:.1f}, {pos[2]:.1f})")
        suffix += 1


def _ensure_standing_wave_areas_for_keepalives(context: MapDataContext) -> None:
    """
    Create StandingWaveAreas (water areas) underneath each controller keepalive building.
    
    These water areas prevent the keepalive buildings from being destroyed by terrain
    when placed outside the map boundaries. Each keepalive building needs a small water
    area directly underneath it.
    """
    from map_processor.assets.water.standing_wave_areas import StandingWaveAreas
    from map_processor.assets.water.standing_wave_area import StandingWaveArea
    
    objects = context.get_asset("ObjectsList")
    if not objects:
        return
    
    # Find all keepalive buildings
    keepalives = []
    for o in objects.map_objects:
        uid = o.asset_property_collection.get_property('uniqueID')
        if uid and 'EasterIslandHead' in uid.data:
            owner = o.asset_property_collection.get_property('originalOwner')
            pos = o.position
            if owner:
                keepalives.append((owner.data, pos))
    
    if not keepalives:
        return
    
    # Get or create StandingWaveAreas asset
    swa = context.get_asset("StandingWaveAreas")
    if not swa:
        swa = StandingWaveAreas()
        swa.id = context.map_struct.register_string("StandingWaveAreas")
        swa.version = swa.get_version()
        swa.name = "StandingWaveAreas"
        context.map_struct.assets.append(swa)
    
    # Find the next available area ID
    next_id = max([area.id for area in swa.areas], default=0) + 1 if swa.areas else 1
    
    # Create a water area for each keepalive
    # Use a small square (4 points) centered on the keepalive position
    water_size = 200.0  # 200 units radius (400x400 square)
    
    for owner, (kx, ky, kz) in keepalives:
        # Create a square water area centered on the keepalive
        area = StandingWaveArea()
        area.id = next_id
        next_id += 1
        area.name = "Keepalive Water Area"
        area.particle_effect = ""
        area.uv_scroll_speed = 0.0
        area.additive_blending = False
        
        # Create 4-point square: bottom-left, bottom-right, top-right, top-left
        area.points = [
            (kx - water_size, ky - water_size),  # Bottom-left
            (kx + water_size, ky - water_size),  # Bottom-right
            (kx + water_size, ky + water_size),  # Top-right
            (kx - water_size, ky + water_size),  # Top-left
        ]
        
        swa.areas.append(area)
    
    if keepalives:
        print(f"  Created {len(keepalives)} StandingWaveAreas for keepalive buildings")


def _generate_named_cameras_for_archon(
    context: MapDataContext,
    num_builders: int,
    template_ctx: Optional[MapDataContext] = None,
) -> int:
    """
    Generate NamedCameras asset with camera entries for controller players.
    
    For paired 3p Archon mode:
    - PLAYER_2_SET: Camera at Player_1_Start position (builder 1's partner is controller 2)
    - PLAYER_4_SET: Camera at Player_3_Start position (builder 3's partner is controller 4)
    - PLAYER_6_SET: Camera at Player_5_Start position (builder 5's partner is controller 6)
    
    The camera positions should be at the BUILDER start locations (where the controller will view).
    Camera settings (zoom, pitch, etc.) are taken from the template if available.
    
    Returns number of cameras created.
    """
    import struct
    
    # Find NamedCameras asset
    named_cameras = None
    for asset in context.map_struct.assets:
        if asset.get_asset_name() == 'NamedCameras':
            named_cameras = asset
            break
    
    if named_cameras is None:
        print("  Warning: NamedCameras asset not found")
        return 0
    
    # Get builder start positions (builders are 1, 3, 5 for paired 3p)
    objs = context.get_asset('ObjectsList')
    if not objs:
        return 0
    
    builder_positions = {}  # builder_num -> (x, y)
    if num_builders == 3:
        builder_nums = [1, 3, 5]
    else:
        builder_nums = list(range(1, num_builders + 1))
    
    for obj in objs.map_objects:
        uid = getattr(obj, 'unique_id', None)
        if uid and 'Start' in str(uid) and 'Player_' in str(uid):
            try:
                pn = int(uid.split('_')[1])
                pos = obj.position
                builder_positions[pn] = (pos[0], pos[1])
            except (ValueError, IndexError, TypeError):
                pass
    
    # Camera settings should match the map being converted, not the template
    # Use default settings observed in Caldera GT which work universally:
    # These settings define camera zoom, pitch, and height for controller view
    # We use Caldera-style settings as they're common for 3p archon maps
    # [short=3, floats: 1.0, 0.0, 0.0, 1.158, 0.873], height=280.0
    camera_settings = bytes.fromhex('0300803f00000000000000009338943ff3665f3f00008c43')
    
    # Build camera entries
    # For paired 3p: controller 2 at P1 start, controller 4 at P3 start, controller 6 at P5 start
    if num_builders == 3:
        camera_pairs = [(2, 1), (4, 3), (6, 5)]  # (controller, builder)
    else:
        camera_pairs = [(b + num_builders, b) for b in builder_nums]  # consecutive scheme
    
    cameras_data = []
    for controller_num, builder_num in camera_pairs:
        if builder_num not in builder_positions:
            print(f"  Warning: Builder {builder_num} start position not found for camera")
            continue
        
        x, y = builder_positions[builder_num]
        camera_name = f"PLAYER_{controller_num}_SET"
        name_bytes = camera_name.encode('ascii')
        
        # Build camera entry: x(4) + y(4) + unk(4) + name_len(2) + name + settings(24)
        entry = struct.pack('<ff', x, y)  # x, y
        entry += struct.pack('<I', 0)  # unknown (always 0)
        entry += struct.pack('<H', len(name_bytes))  # name length
        entry += name_bytes  # name
        entry += camera_settings  # camera settings (24 bytes)
        
        cameras_data.append(entry)
    
    # Build final NamedCameras data
    count = len(cameras_data)
    final_data = struct.pack('<I', count) + b''.join(cameras_data)
    
    # Update the asset
    named_cameras.data = final_data
    named_cameras.data_size = len(final_data)
    
    return count


def _is_paired_archon_3p_template(template_ctx: MapDataContext) -> bool:
    """
    Detect the 3p archon template style used by community 3v3 archon maps.

    Characteristics (observed in [Archon]Caldera_of_Chaos_1.2):
    - Player starts 1..6 exist.
    - Scripts are primarily in list 0 (group 'Archon Mode Package') and lists 11..16 (Macro/Micro per pair).
    - Special waypoints exist: 'Apron Ocuppier Player_{1,3,5}' and 'Linked Airfield Player_{2,4,6}'.
    """
    try:
        objs = template_ctx.get_asset("ObjectsList")
        if not objs:
            return False

        start_ids = set()
        waypoint_names = set()
        for o in objs.map_objects:
            if isinstance(getattr(o, "unique_id", None), str) and o.unique_id.startswith("Player_") and o.unique_id.endswith("_Start"):
                start_ids.add(o.unique_id)
            if "Waypoint" in (getattr(o, "type_name", "") or ""):
                pn = o.asset_property_collection.get_property("waypointName")
                if pn and isinstance(pn.data, str):
                    waypoint_names.add(pn.data)

        required_starts = {f"Player_{i}_Start" for i in range(1, 7)}
        if not required_starts.issubset(start_ids):
            return False

        # Waypoint signature
        if not any(n.startswith("Apron Ocuppier Player_") for n in waypoint_names):
            return False
        if not any(n.startswith("Linked Airfield Player_") for n in waypoint_names):
            return False

        psl = template_ctx.get_asset("PlayerScriptsList")
        if not psl or not psl.script_lists:
            return False

        list0 = psl.script_lists[0]
        if any((g.name or "").strip().lower() == "archon mode package" for g in list0.script_groups):
            return True

        # Fallback: macro/micro group names
        for idx in range(min(17, len(psl.script_lists))):
            for g in psl.script_lists[idx].script_groups:
                nm = (g.name or "").lower()
                if "archon p1-2 macro control" in nm or "archon p1-2 micro control" in nm:
                    return True

        return False
    except Exception:
        return False


def _ensure_script_lists_len(psl: PlayerScriptsList, target_ctx: MapDataContext, n: int) -> None:
    """Ensure `psl.script_lists` has at least `n` ScriptList entries."""
    while len(psl.script_lists) < n:
        new_list = ScriptList()
        new_list.id = target_ctx.map_struct.register_string("ScriptList")
        new_list.version = 1
        new_list.name = "ScriptList"
        new_list.scripts = []
        new_list.script_groups = []
        psl.script_lists.append(new_list)


def _copy_full_player_scripts_list_from_template(source_ctx: MapDataContext, template_ctx: MapDataContext) -> None:
    """
    Copy the entire PlayerScriptsList structure from template to source, remapping string pools.
    This is required for paired 3p templates, where scripts are not in the same per-player lists as 1v1 templates.
    """
    template_psl = template_ctx.get_asset("PlayerScriptsList")
    source_psl = source_ctx.get_asset("PlayerScriptsList")
    if not template_psl or not source_psl:
        print("  Warning: Could not find PlayerScriptsList in one of the maps")
        return

    # Ensure target context has the core asset type strings.
    for s in ['PlayerScriptsList', 'ScriptList', 'ScriptGroup', 'Script', 'OrCondition', 'Condition', 'ScriptAction', 'ScriptActionFalse']:
        source_ctx.map_struct.register_string(s)

    # Replace all script lists with deep-copied + remapped ones
    source_psl.script_lists = []
    for sl in template_psl.script_lists:
        copied = copy.deepcopy(sl)
        remap_script_list(copied, source_ctx, player_offset=0)
        source_psl.script_lists.append(copied)

    print(f"  Copied full PlayerScriptsList from template: {len(source_psl.script_lists)} lists")


def _copy_paired_3p_scripts_from_template(source_ctx: MapDataContext, template_ctx: MapDataContext, num_builders: int = 3) -> None:
    """
    Copy paired-3p archon scripts (global + per-player macro/micro) from a template, but
    **align ScriptList indices to the source map's Player_1 base index**.

    Why: different maps can have different counts of system players (e.g. `PlyrNeutral`),
    shifting the index where Player_1 begins. The Archon macro/micro scripts are stored in
    per-player ScriptLists and must line up with Player_1..Player_6.
    """
    template_psl = template_ctx.get_asset("PlayerScriptsList")
    source_psl = source_ctx.get_asset("PlayerScriptsList")
    if not template_psl or not source_psl:
        print("  Warning: Could not find PlayerScriptsList")
        return

    # Ensure core strings exist
    for s in ['PlayerScriptsList', 'ScriptList', 'ScriptGroup', 'Script', 'OrCondition', 'Condition', 'ScriptAction', 'ScriptActionFalse']:
        source_ctx.map_struct.register_string(s)

    tpl_base = find_first_player_index(template_ctx)
    src_base = find_first_player_index(source_ctx)
    num_humans = num_builders * 2  # 6 for 3p paired

    # Ensure we have enough script lists for the archon player slots
    # We need src_base + num_humans lists (e.g., 13 + 6 = 19 for Fried River)
    required_lists = src_base + num_humans
    _ensure_script_lists_len(source_psl, source_ctx, required_lists)

    # Merge list 0: copy ALL scripts/groups from template EXCEPT template-specific ones
    if template_psl.script_lists:
        src_list0 = source_psl.script_lists[0]
        tpl_list0 = template_psl.script_lists[0]
        
        # Template-specific scripts to EXCLUDE (not part of core archon)
        template_specific_scripts = {'Music'}
        
        # Get existing script/group names in source
        existing_script_names = {s.name for s in src_list0.scripts}
        existing_group_names = {g.name for g in src_list0.script_groups}
        
        # Add all scripts from template (except template-specific and already in source)
        for s in tpl_list0.scripts:
            if s.name not in template_specific_scripts and s.name not in existing_script_names:
                copied = copy.deepcopy(s)
                remap_script(copied, source_ctx)
                src_list0.scripts.append(copied)
        
        # Add all script groups from template (not already in source)
        for g in tpl_list0.script_groups:
            if g.name not in existing_group_names:
                copied = copy.deepcopy(g)
                remap_script_group(copied, source_ctx)
                src_list0.script_groups.append(copied)

    # Copy per-player lists for Player_1..Player_6 into the correct base indices
    copied_lists = 0
    for p in range(num_humans):
        tpl_idx = tpl_base + p
        src_idx = src_base + p
        if tpl_idx >= len(template_psl.script_lists) or src_idx >= len(source_psl.script_lists):
            continue
        copied = copy.deepcopy(template_psl.script_lists[tpl_idx])
        remap_script_list(copied, source_ctx, player_offset=0)
        source_psl.script_lists[src_idx] = copied
        copied_lists += 1

    print(f"  Paired-3p scripts: template Player_1 base={tpl_base}, source Player_1 base={src_base}, copied per-player lists={copied_lists}")


def _apply_paired_3p_player_start_numbering_for_different_map(context: MapDataContext) -> None:
    """
    For non-same-base maps in paired 3p scheme:
    - Rename existing builder starts Player_1_Start, Player_2_Start, Player_3_Start -> Player_1_Start, Player_3_Start, Player_5_Start
      (preserving their positions).
    
    The mapping is:
      - Player_1 stays as Player_1 (builder 1)
      - Player_2 becomes Player_5 (builder 5) - P2 moves to last builder slot
      - Player_3 stays as Player_3 (builder 3)
    
    This matches the official Caldera archon conversion pattern.
    """
    objs = context.get_asset("ObjectsList")
    if not objs:
        return

    # Map existing starts by player number
    starts = {}  # player_num -> object
    for o in objs.map_objects:
        if isinstance(getattr(o, "unique_id", None), str) and o.unique_id.startswith("Player_") and o.unique_id.endswith("_Start"):
            try:
                n = int(o.unique_id.split("_")[1])
            except Exception:
                continue
            # Only the original 1..3 builders
            if 1 <= n <= 3:
                starts[n] = o
    
    if len(starts) != 3:
        return

    # Mapping: original player number -> new builder number
    # P1 -> P1, P2 -> P5, P3 -> P3
    rename_map = {1: 1, 2: 5, 3: 3}
    
    for src_n, tgt_n in rename_map.items():
        if src_n == tgt_n:
            continue  # No change needed
        o = starts.get(src_n)
        if not o:
            continue
        new_uid = f"Player_{tgt_n}_Start"
        # MapObject.unique_id is derived from properties; update the properties instead of assigning unique_id.
        uid_prop = o.asset_property_collection.get_property("uniqueID")
        if uid_prop:
            uid_prop.data = new_uid
        wpn_prop = o.asset_property_collection.get_property("waypointName")
        if wpn_prop:
            wpn_prop.data = new_uid
        # NOTE: Do NOT register unique_id VALUE in string pool - it's stored as raw string data


def _replace_player_starts_from_template(context: MapDataContext, template_ctx: MapDataContext) -> None:
    """
    Replace Player_{1..6}_Start objects in `context` with those from `template_ctx`.
    Only safe when maps share the same base layout.
    """
    objs = context.get_asset("ObjectsList")
    t_objs = template_ctx.get_asset("ObjectsList")
    if not objs or not t_objs:
        return

    start_ids = {f"Player_{i}_Start" for i in range(1, 7)}

    # Remove any existing start objects in source
    kept = []
    for o in objs.map_objects:
        if getattr(o, "unique_id", None) in start_ids:
            continue
        kept.append(o)
    objs.map_objects = kept

    # Add template start objects
    for o in t_objs.map_objects:
        if getattr(o, "unique_id", None) in start_ids:
            new_obj = copy.deepcopy(o)
            for name, prop in new_obj.asset_property_collection.property_map.items():
                prop.id = context.map_struct.register_string(name)
            objs.map_objects.append(new_obj)


def _remap_waypoint_ids_from_template(context: MapDataContext, template_ctx: MapDataContext) -> int:
    """
    For paired-3p Archon maps, waypointID numbering is *not arbitrary* in official maps:
    the 12 Archon waypoints occupy IDs 1..12 (Apron/Linked = 1..6, Player_*_Start = 7..12),
    but the exact mapping from name -> id varies per template.

    Our converter may inherit old waypointIDs from the base map (e.g., 1..3 on Player_1..3_Start),
    and then allocate new IDs above that for Archon waypoints, which breaks initialization and
    can cause controllers to be defeated immediately.

    This function copies the template's waypointName->waypointID mapping onto the current map
    for any waypoint objects with matching waypointName.

    Returns number of waypointID values changed.
    """
    objs = context.get_asset("ObjectsList")
    t_objs = template_ctx.get_asset("ObjectsList")
    if not objs or not t_objs:
        return 0

    # Build template mapping waypointName -> waypointID
    tpl: Dict[str, int] = {}
    for o in t_objs.map_objects:
        if (o.type_name or "") != "*Waypoints/Waypoint":
            continue
        n_prop = o.asset_property_collection.get_property("waypointName")
        i_prop = o.asset_property_collection.get_property("waypointID")
        if not n_prop or not isinstance(n_prop.data, str):
            continue
        if not i_prop or not isinstance(i_prop.data, int):
            continue
        tpl[n_prop.data] = i_prop.data

    if not tpl:
        return 0

    changed = 0
    for o in objs.map_objects:
        if (o.type_name or "") != "*Waypoints/Waypoint":
            continue
        n_prop = o.asset_property_collection.get_property("waypointName")
        if not n_prop or not isinstance(n_prop.data, str):
            continue
        desired = tpl.get(n_prop.data)
        if desired is None:
            continue

        i_prop = o.asset_property_collection.get_property("waypointID")
        if i_prop and isinstance(i_prop.data, int):
            if i_prop.data != desired:
                i_prop.data = desired
                changed += 1
        else:
            new_prop = AssetProperty()
            new_prop.property_type = AssetPropertyType.int_type
            new_prop.name = "waypointID"
            new_prop.data = desired
            new_prop.id = context.map_struct.register_string("waypointID")
            o.asset_property_collection.property_map["waypointID"] = new_prop
            changed += 1

    return changed


def _ensure_assetlist_has_archon_blocks(context: MapDataContext, extra_blocks: int) -> None:
    """
    Ensure `AssetList` contains enough extra AssetBlocks for newly-introduced Archon entities.

    Empirically (diffing HF/CoC base -> Archon GT), Archon conversion increases `AssetList` by
    adding additional blocks of `type_id == 2486173485` (0x942FFF2D). The instance_id values
    appear only inside the AssetList itself (not referenced elsewhere in the file), so we
    can safely generate unique instance_id values as long as we:
    - Keep `(type_id, instance_id)` pairs unique within the list
    - Use the same `type_id` as official maps for these extra blocks

    NOTE: We only apply this in different-base-map conversions. Same-base-map conversions
    copy the template AssetList wholesale and therefore already have correct blocks.
    """
    asset_list = context.get_asset("AssetList")
    if not asset_list:
        return

    ARCHON_TYPE_ID = 2486173485

    existing: Set[Tuple[int, int]] = {(b.type_id, b.instance_id) for b in asset_list.asset_blocks}

    # Deterministic RNG so repeated runs on the same inputs are stable.
    rng = random.Random(0xA3C5F1)

    added = 0
    while added < extra_blocks:
        inst = rng.getrandbits(32)
        if inst == 0:
            continue
        key = (ARCHON_TYPE_ID, inst)
        if key in existing:
            continue

        blk = AssetBlock()
        blk.type_id = ARCHON_TYPE_ID
        blk.instance_id = inst
        asset_list.asset_blocks.append(blk)
        existing.add(key)
        added += 1


def _normalize_blend_tile_data_for_wb(context: MapDataContext) -> None:
    """
    Normalize `BlendTileData` the way WB does on save:
    - Canonicalize `textures` order based on *first usage in the tile grid* (x-major scan: x then y)
      and re-assign `cell_start` slots sequentially (0, 16, 32, ...).
    - Remap `tiles` so each cell resolves to the same texture name after the reorder.

    This matches the behavior we observed by diffing a WB-saved `.map`:
    WB will reassign `cell_start` between texture names (not just reorder the list),
    and it does so deterministically based on the terrain paint layout.
    """
    blend = context.get_asset("BlendTileData")
    if not blend or not getattr(blend, "textures", None) or blend.tiles is None:
        return

    import numpy as np

    textures = blend.textures
    w = int(blend.map_width)
    h = int(blend.map_height)

    # Compute per-cell "current" offset used by the RA3 tile encoding.
    x = np.arange(w, dtype=np.int32).reshape((w, 1))
    y = np.arange(h, dtype=np.int32).reshape((1, h))
    row_first = ((y % 8) // 2) * 16 + (y % 2) * 2
    current = ((x % 8) // 2) * 4 + (x % 2) + row_first

    # Decode current texture index per cell
    tiles_i32 = np.asarray(blend.tiles, dtype=np.int32)
    old_tex = ((tiles_i32 - current) // 64).astype(np.int32)  # shape (w,h), values [0..len(textures))

    # Find first occurrence for each texture using x-major scan (x then y).
    big = (10**9, 10**9)
    first = [None] * len(textures)
    for x0 in range(w):
        col = old_tex[x0, :]
        for idx in range(len(textures)):
            if first[idx] is None:
                ys = np.where(col == idx)[0]
                if ys.size:
                    first[idx] = (int(x0), int(ys[0]))
    first2 = [v if v is not None else big for v in first]

    order = sorted(range(len(textures)), key=lambda i: first2[i])
    if order == list(range(len(textures))):
        # Still ensure cell_start is canonical (sequential) like WB writes
        for i, t in enumerate(textures):
            t.cell_start = i * 16
        return

    # old_index -> new_index
    old_to_new = [0] * len(textures)
    for new_i, old_i in enumerate(order):
        old_to_new[old_i] = new_i

    # Reorder textures and canonicalize cell_start slots (0,16,32,...)
    new_textures = [textures[i] for i in order]
    for i, t in enumerate(new_textures):
        t.cell_start = i * 16
    blend.textures = new_textures

    # Remap tiles so painted texture names remain unchanged under new indices.
    remap = np.asarray(old_to_new, dtype=np.int32)
    new_tex = remap[old_tex]
    blend.tiles = (new_tex * 64 + current).astype(np.uint16)



def remap_script_content(content: ScriptContent, 
                         target_ctx: MapDataContext,
                         is_subclass: bool = False) -> None:
    """
    Remap string indices in a ScriptContent to target context.
    Updates all string references to use target's string pool.
    
    Args:
        content: The ScriptContent to remap
        target_ctx: Target context with string pool
        is_subclass: If True, this is a subclass (ScriptAction) that can be registered
    """
    # Only register if this is a proper subclass (ScriptAction, ScriptActionFalse)
    # ScriptContent used directly in ScriptCondition doesn't need registration
    if is_subclass:
        content.register_self(target_ctx)
    else:
        # For bare ScriptContent in conditions, the asset name is "Condition"
        # We need to update the id to point to the correct string pool index
        content.id = target_ctx.map_struct.register_string("Condition")
    
    # Update content_name index
    if content.content_name:
        content.name_index = target_ctx.map_struct.register_string(content.content_name)


def remap_script_condition(condition: Condition,
                           target_ctx: MapDataContext,
                           player_offset: int = 0) -> None:
    """Remap string indices in a Condition asset."""
    # Condition now contains ScriptContent-like data directly (not wrapped in script_content)
    # Update the asset id and name_index
    update_asset_id(condition, target_ctx)
    condition.name_index = target_ctx.map_struct.register_string(condition.content_name)
    
    # Remap player references in condition arguments
    if player_offset != 0:
        for arg in condition.arguments:
            if arg.string_value:
                arg.string_value = _offset_player_reference(arg.string_value, player_offset)


def update_asset_id(asset, target_ctx: MapDataContext) -> None:
    """
    Update a MajorAsset's id to use the target context's string pool.
    Does NOT modify the asset's name attribute (which some assets use for data).
    """
    asset_name = asset.get_asset_name()
    asset.id = target_ctx.map_struct.register_string(asset_name)
    asset.version = asset.get_version()


def remap_or_condition(or_cond: OrCondition,
                       target_ctx: MapDataContext,
                       player_offset: int = 0) -> None:
    """Remap string indices in an OrCondition."""
    update_asset_id(or_cond, target_ctx)
    
    for condition in or_cond.conditions:
        remap_script_condition(condition, target_ctx, player_offset)


def _offset_player_reference(value: str, player_offset: int) -> str:
    """
    Offset player number references in a string value.
    
    Examples with offset=1:
        "Player_3" -> "Player_4"
        "Player_3/P3s team" -> "Player_4/P4s team"
        "P3jc8" -> "P4jc8"
        "teamPlayer_3" -> "teamPlayer_4"
    """
    if player_offset == 0:
        return value
    
    import re
    
    result = value
    
    # Match Player_N patterns
    def replace_player(m):
        num = int(m.group(1))
        # Only offset controller players (3+)
        if num >= 3:
            return f"Player_{num + player_offset}"
        return m.group(0)
    result = re.sub(r'Player_(\d+)', replace_player, result)
    
    # Match teamPlayer_N patterns
    def replace_team_player(m):
        num = int(m.group(1))
        if num >= 3:
            return f"teamPlayer_{num + player_offset}"
        return m.group(0)
    result = re.sub(r'teamPlayer_(\d+)', replace_team_player, result)
    
    # Match P3, P4, etc. patterns (for team names like P3s team, P4s jc, P3jc8)
    # Use a single comprehensive pattern to avoid double-processing
    def replace_p_num(m):
        prefix = m.group(1) or ""
        num = int(m.group(2))
        suffix = m.group(3) or ""
        if num >= 3:
            return f"{prefix}P{num + player_offset}{suffix}"
        return m.group(0)
    # Match P followed by digit, at start of string or after /
    # Suffix can be space, lowercase letter, or end of string
    result = re.sub(r'(^|/)P(\d+)(\s|[a-z]|$)', replace_p_num, result)
    
    return result


def remap_script_action(action: ScriptAction,
                        target_ctx: MapDataContext,
                        player_offset: int = 0) -> None:
    """Remap string indices in a ScriptAction (which is a ScriptContent subclass)."""
    update_asset_id(action, target_ctx)
    remap_script_content(action, target_ctx, is_subclass=True)
    
    # Remap player references in arguments
    if player_offset != 0:
        for arg in action.arguments:
            if arg.string_value:
                arg.string_value = _offset_player_reference(arg.string_value, player_offset)


def remap_script(script: Script,
                 target_ctx: MapDataContext,
                 player_offset: int = 0) -> None:
    """Remap all string indices in a Script."""
    # Update asset id without overwriting the script's name
    update_asset_id(script, target_ctx)
    
    # Also remap script name if it contains player references
    if player_offset != 0 and script.name:
        script.name = _offset_player_reference(script.name, player_offset)
    
    # Remap conditions
    for or_cond in script.script_or_conditions:
        remap_or_condition(or_cond, target_ctx, player_offset)
    
    # Remap actions on true
    for action in script.script_action_on_true:
        remap_script_action(action, target_ctx, player_offset)
    
    # Remap actions on false
    for action in script.script_action_on_false:
        remap_script_action(action, target_ctx, player_offset)


def remap_script_group(group: ScriptGroup,
                       target_ctx: MapDataContext,
                       player_offset: int = 0) -> None:
    """Remap all string indices in a ScriptGroup."""
    # Update asset id without overwriting the group's name
    update_asset_id(group, target_ctx)
    
    # Also remap group name if it contains player references
    if player_offset != 0 and group.name:
        group.name = _offset_player_reference(group.name, player_offset)
    
    for script in group.scripts:
        remap_script(script, target_ctx, player_offset)
    
    for subgroup in group.script_groups:
        remap_script_group(subgroup, target_ctx, player_offset)


def remap_script_list(script_list: ScriptList,
                      target_ctx: MapDataContext,
                      player_offset: int = 0) -> None:
    """Remap all string indices in a ScriptList."""
    update_asset_id(script_list, target_ctx)
    
    for script in script_list.scripts:
        remap_script(script, target_ctx, player_offset)
    
    for group in script_list.script_groups:
        remap_script_group(group, target_ctx, player_offset)


def find_first_player_index(ctx: MapDataContext) -> int:
    """Find the index of the first actual player (Player_1) in SidesList."""
    sides = ctx.get_asset('SidesList')
    if not sides:
        return 12  # Default fallback
    
    for i, p in enumerate(sides.players):
        prop = p.asset_property_collection.get_property('playerName')
        if prop and prop.data == 'Player_1':
            return i
    
    return 12  # Default if not found


def _transform_player_reference(value: str, 
                                 src_builder: int, tgt_builder: int,
                                 src_controller: int, tgt_controller: int) -> str:
    """
    Transform player references in a script string from source builder/controller 
    to target builder/controller.
    
    Examples (src_builder=1, tgt_builder=3, src_controller=3, tgt_controller=6):
        "Player_1" -> "Player_3"
        "Player_3" -> "Player_6" 
        "P1 APL" -> "P3 APL"
        "P3jc8" -> "P6jc8"
        "Player_3/P3s jc" -> "Player_6/P6s jc"
        "teamPlayer_1" -> "teamPlayer_3"
    """
    import re
    result = value
    
    # Transform teamPlayer_N patterns FIRST (before Player_N to avoid substring match)
    def replace_team_player(m):
        num = int(m.group(1))
        if num == src_builder:
            return f"teamPlayer_{tgt_builder}"
        elif num == src_controller:
            return f"teamPlayer_{tgt_controller}"
        return m.group(0)
    result = re.sub(r'teamPlayer_(\d+)', replace_team_player, result)
    
    # Transform Player_N patterns (use negative lookbehind to avoid matching inside teamPlayer_)
    def replace_player(m):
        num = int(m.group(1))
        if num == src_builder:
            return f"Player_{tgt_builder}"
        elif num == src_controller:
            return f"Player_{tgt_controller}"
        return m.group(0)
    result = re.sub(r'(?<!team)Player_(\d+)', replace_player, result)
    
    # Transform P{N} patterns at start or after /
    def replace_p_num(m):
        prefix = m.group(1) or ""
        num = int(m.group(2))
        suffix = m.group(3) or ""
        if num == src_builder:
            return f"{prefix}P{tgt_builder}{suffix}"
        elif num == src_controller:
            return f"{prefix}P{tgt_controller}{suffix}"
        return m.group(0)
    result = re.sub(r'(^|/)P(\d+)(\s|[a-z]|$)', replace_p_num, result)
    
    return result


def _transform_script_for_new_builder(script: Script, target_ctx: MapDataContext,
                                       src_builder: int, tgt_builder: int,
                                       src_controller: int, tgt_controller: int) -> None:
    """Transform a script's player references for a new builder/controller pair."""
    update_asset_id(script, target_ctx)
    
    # Transform script name
    if script.name:
        script.name = _transform_player_reference(
            script.name, src_builder, tgt_builder, src_controller, tgt_controller
        )
    
    # Transform conditions
    for or_cond in script.script_or_conditions:
        update_asset_id(or_cond, target_ctx)
        for condition in or_cond.conditions:
            remap_script_content(condition.script_content, target_ctx, is_subclass=False)
            if hasattr(condition.script_content, 'arguments'):
                for arg in condition.script_content.arguments:
                    if arg.string_value:
                        arg.string_value = _transform_player_reference(
                            arg.string_value, src_builder, tgt_builder, 
                            src_controller, tgt_controller
                        )
    
    # Transform actions
    for action in script.script_action_on_true + script.script_action_on_false:
        update_asset_id(action, target_ctx)
        remap_script_content(action, target_ctx, is_subclass=True)
        for arg in action.arguments:
            if arg.string_value:
                arg.string_value = _transform_player_reference(
                    arg.string_value, src_builder, tgt_builder,
                    src_controller, tgt_controller
                )


def _transform_group_for_new_builder(group: ScriptGroup, target_ctx: MapDataContext,
                                      src_builder: int, tgt_builder: int,
                                      src_controller: int, tgt_controller: int) -> None:
    """Transform a script group's player references for a new builder/controller pair."""
    update_asset_id(group, target_ctx)
    
    # Transform group name
    if group.name:
        group.name = _transform_player_reference(
            group.name, src_builder, tgt_builder, src_controller, tgt_controller
        )
    
    # Transform all scripts in the group
    for script in group.scripts:
        _transform_script_for_new_builder(
            script, target_ctx, src_builder, tgt_builder, src_controller, tgt_controller
        )
    
    # Transform subgroups recursively
    for subgroup in group.script_groups:
        _transform_group_for_new_builder(
            subgroup, target_ctx, src_builder, tgt_builder, src_controller, tgt_controller
        )


def copy_scripts_from_template(source_ctx: MapDataContext,
                                template_ctx: MapDataContext,
                                num_builders: int = 2) -> None:
    """
    Copy archon scripts from template to source context.
    
    This ADDS scripts to the correct script list indices, preserving the original structure.
    Dynamically finds where actual players start in each map since different maps have
    different system player counts.
    
    Args:
        source_ctx: Source map context to add scripts to
        template_ctx: Template archon map to copy scripts from
        num_builders: Number of builder players (1-3)
    """
    # Paired 3p templates require ScriptList alignment to Player_1 base index.
    if num_builders == 3 and _is_paired_archon_3p_template(template_ctx):
        _copy_paired_3p_scripts_from_template(source_ctx, template_ctx, num_builders=num_builders)
        return

    template_scripts = template_ctx.get_asset('PlayerScriptsList')
    source_scripts = source_ctx.get_asset('PlayerScriptsList')
    
    if not template_scripts or not source_scripts:
        print("  Warning: Could not find PlayerScriptsList in one of the maps")
        return
    
    # Register all necessary asset type strings in the target context
    asset_strings = [
        'PlayerScriptsList', 'ScriptList', 'ScriptGroup', 'Script',
        'OrCondition', 'Condition', 'ScriptAction', 'ScriptActionFalse'
    ]
    for s in asset_strings:
        source_ctx.map_struct.register_string(s)
    
    original_count = len(source_scripts.script_lists)
    
    # Add empty script lists for the new controller players
    for i in range(num_builders):
        new_list = ScriptList()
        # Set MajorAsset fields (required for WorldBuilder to recognize)
        new_list.id = source_ctx.map_struct.register_string("ScriptList")
        new_list.version = 1
        new_list.name = "ScriptList"
        new_list.scripts = []
        new_list.script_groups = []
        source_scripts.script_lists.append(new_list)
    
    print(f"  Script lists: {original_count} -> {len(source_scripts.script_lists)}")
    
    # Find where actual players start in each map
    template_base = find_first_player_index(template_ctx)
    source_base = find_first_player_index(source_ctx)
    print(f"  Template player base: {template_base}, Source player base: {source_base}")
    
    # Calculate player offset for script remapping
    # Template has 2 builders (controllers are Player_3, Player_4)
    # For 3 builders, controllers should be Player_4, Player_5, Player_6
    template_num_builders = 2  # Template always has 2 builders
    player_offset = num_builders - template_num_builders
    if player_offset != 0:
        print(f"  Player offset: {player_offset} (remapping Player_3->Player_{3+player_offset}, etc.)")
    
    # Copy global scripts from template list 0 to source list 0
    if len(template_scripts.script_lists) > 0:
        template_global = template_scripts.script_lists[0]
        if template_global.scripts or template_global.script_groups:
            for script in template_global.scripts:
                copied = copy.deepcopy(script)
                remap_script(copied, source_ctx, player_offset)
                source_scripts.script_lists[0].scripts.append(copied)
            for group in template_global.script_groups:
                copied = copy.deepcopy(group)
                remap_script_group(copied, source_ctx, player_offset)
                source_scripts.script_lists[0].script_groups.append(copied)
            print(f"  Copied {len(template_global.scripts)} global scripts, {len(template_global.script_groups)} groups to list 0")
    
    # Copy builder scripts from template to source at correct indices
    # Builder scripts also reference controller players, so they need offset too
    for b in range(min(num_builders, template_num_builders)):
        template_idx = template_base + b  # e.g., 11, 12 in template
        source_idx = source_base + b      # e.g., 12, 13 in source
        if template_idx < len(template_scripts.script_lists) and source_idx < len(source_scripts.script_lists):
            template_list = template_scripts.script_lists[template_idx]
            if template_list.scripts or template_list.script_groups:
                for script in template_list.scripts:
                    copied = copy.deepcopy(script)
                    remap_script(copied, source_ctx, player_offset)
                    source_scripts.script_lists[source_idx].scripts.append(copied)
                for group in template_list.script_groups:
                    copied = copy.deepcopy(group)
                    remap_script_group(copied, source_ctx, player_offset)
                    source_scripts.script_lists[source_idx].script_groups.append(copied)
                print(f"  Copied builder {b+1} scripts from list {template_idx} to list {source_idx}")
    
    # Copy controller scripts from template to source at correct indices
    for c in range(min(num_builders, template_num_builders)):
        template_idx = template_base + template_num_builders + c  # e.g., 13, 14 in template
        source_idx = source_base + num_builders + c      # e.g., 14, 15 in source for 3 builders
        if template_idx < len(template_scripts.script_lists) and source_idx < len(source_scripts.script_lists):
            template_list = template_scripts.script_lists[template_idx]
            if template_list.scripts or template_list.script_groups:
                for script in template_list.scripts:
                    copied = copy.deepcopy(script)
                    remap_script(copied, source_ctx, player_offset)
                    source_scripts.script_lists[source_idx].scripts.append(copied)
                for group in template_list.script_groups:
                    copied = copy.deepcopy(group)
                    remap_script_group(copied, source_ctx, player_offset)
                    source_scripts.script_lists[source_idx].script_groups.append(copied)
                print(f"  Copied controller {c+1} scripts from list {template_idx} to list {source_idx}")
    
    # Generate scripts for additional builders (beyond template capacity) by interpolation
    # For a 3-player map with 2-player template, we generate builder 3 & controller 3 scripts
    # by copying builder 1 & controller 1 and transforming player references
    if num_builders > template_num_builders:
        print(f"  Generating scripts for {num_builders - template_num_builders} additional builder(s)...")
        
        # Use builder 1 and controller 1 as the source template
        template_builder_idx = template_base  # Builder 1 in template
        template_controller_idx = template_base + template_num_builders  # Controller 1 in template
        
        for extra_b in range(template_num_builders, num_builders):
            target_builder_num = extra_b + 1  # e.g., 3 for the 3rd builder
            target_controller_num = target_builder_num + num_builders  # e.g., 6 for controller of builder 3
            
            # Target list indices
            builder_list_idx = source_base + extra_b
            controller_list_idx = source_base + num_builders + extra_b
            
            if builder_list_idx >= len(source_scripts.script_lists):
                continue
            if controller_list_idx >= len(source_scripts.script_lists):
                continue
            
            # Copy and transform builder scripts
            if template_builder_idx < len(template_scripts.script_lists):
                template_list = template_scripts.script_lists[template_builder_idx]
                for script in template_list.scripts:
                    copied = copy.deepcopy(script)
                    _transform_script_for_new_builder(copied, source_ctx, 1, target_builder_num, 3, target_controller_num)
                    source_scripts.script_lists[builder_list_idx].scripts.append(copied)
                for group in template_list.script_groups:
                    copied = copy.deepcopy(group)
                    _transform_group_for_new_builder(copied, source_ctx, 1, target_builder_num, 3, target_controller_num)
                    source_scripts.script_lists[builder_list_idx].script_groups.append(copied)
                print(f"  Generated builder {target_builder_num} scripts (interpolated from builder 1)")
            
            # Copy and transform controller scripts
            if template_controller_idx < len(template_scripts.script_lists):
                template_list = template_scripts.script_lists[template_controller_idx]
                for script in template_list.scripts:
                    copied = copy.deepcopy(script)
                    _transform_script_for_new_builder(copied, source_ctx, 1, target_builder_num, 3, target_controller_num)
                    source_scripts.script_lists[controller_list_idx].scripts.append(copied)
                for group in template_list.script_groups:
                    copied = copy.deepcopy(group)
                    _transform_group_for_new_builder(copied, source_ctx, 1, target_builder_num, 3, target_controller_num)
                    source_scripts.script_lists[controller_list_idx].script_groups.append(copied)
                print(f"  Generated controller {target_controller_num - num_builders} scripts (interpolated from controller 1)")
    
    # Count total scripts for reporting
    total_scripts = 0
    for sl in source_scripts.script_lists:
        total_scripts += len(sl.scripts)
        for group in sl.script_groups:
            total_scripts += count_scripts_in_group_recursive(group)
    
    print(f"  Total scripts after copy: {total_scripts}")


def count_scripts_in_group_recursive(group: ScriptGroup) -> int:
    """Count all scripts in a group and its subgroups."""
    count = len(group.scripts)
    for subgroup in group.script_groups:
        count += count_scripts_in_group_recursive(subgroup)
    return count


def find_player_starts(context: MapDataContext) -> List[MapObject]:
    """Find all player start waypoints in the map."""
    objects_list = context.get_asset('ObjectsList')
    if not objects_list or not objects_list.map_objects:
        return []
    
    player_start_ids = {'Player_1_Start', 'Player_2_Start', 'Player_3_Start',
                        'Player_4_Start', 'Player_5_Start', 'Player_6_Start'}
    
    starts = []
    for obj in objects_list.map_objects:
        if obj.unique_id in player_start_ids:
            starts.append(obj)
    
    # Sort by player number
    starts.sort(key=lambda x: int(x.unique_id.split('_')[1]))
    return starts


def count_actual_players(context: MapDataContext) -> int:
    """Count the actual number of player slots used in the map."""
    starts = find_player_starts(context)
    return len(starts)


def calculate_controller_offset(builder_pos: Tuple[float, float, float], 
                                 map_width: int, map_height: int,
                                 offset_distance: float = 800.0) -> Tuple[float, float, float]:
    """
    Calculate controller spawn position offset from builder position.
    
    The controller position should be slightly offset from the builder's base,
    placed in a direction AWAY from the map center (behind the base, towards
    the edge of the map - and CAN be outside map bounds).
    
    In the original archon maps:
    - Controller is ~800-1000 units away from builder
    - Direction is away from map center (controller is behind builder's base)
    - Controllers CAN be outside map bounds (negative coords) - this is intentional!
    """
    bx, by, bz = builder_pos
    
    # Map dimensions in world coordinates (10 units per tile)
    map_max_x = map_width * 10
    map_max_y = map_height * 10
    center_x = map_max_x / 2
    center_y = map_max_y / 2
    
    # Direction from center to builder position (this points away from center)
    dx = bx - center_x
    dy = by - center_y
    
    # Normalize and apply offset distance
    distance = math.sqrt(dx * dx + dy * dy)
    if distance > 0:
        # Unit vector pointing away from center
        offset_x = (dx / distance) * offset_distance
        offset_y = (dy / distance) * offset_distance
    else:
        # Builder is at center, offset in a default direction
        offset_x = -offset_distance
        offset_y = 0
    
    # Controller position: builder position + offset (away from center)
    # NOTE: Controllers CAN be outside map bounds - do NOT clamp!
    new_x = bx + offset_x
    new_y = by + offset_y
    
    return (new_x, new_y, bz)


def create_controller_start_waypoint(builder_start: MapObject, 
                                      controller_number: int,
                                      controller_pos: Tuple[float, float, float],
                                      context: MapDataContext) -> MapObject:
    """Create a controller start waypoint based on the builder's start."""
    controller = MapObject()
    
    # Set MajorAsset fields (critical for WorldBuilder/game to recognize the object)
    controller.id = context.map_struct.register_string("Object")
    controller.version = 3  # MapObject version
    controller.name = "Object"
    
    controller.position = controller_pos
    controller.angle = builder_start.angle
    controller.road_option = builder_start.road_option
    controller.type_name = builder_start.type_name  # *Waypoints/Waypoint
    
    # Copy and modify properties
    controller.asset_property_collection = AssetPropertyCollection()
    
    # Copy basic properties from builder
    for name, prop in builder_start.asset_property_collection.property_map.items():
        if name in ('uniqueID', 'waypointID', 'waypointName'):
            continue  # We'll set these specifically
        
        # Create new property with same values
        new_prop = AssetProperty()
        new_prop.property_type = prop.property_type
        new_prop.name = name
        new_prop.data = prop.data
        new_prop.id = context.map_struct.register_string(name)
        controller.asset_property_collection.property_map[name] = new_prop
    
    # Set controller-specific properties
    unique_id = f'Player_{controller_number}_Start'
    
    # Add uniqueID
    uid_prop = AssetProperty()
    uid_prop.property_type = AssetPropertyType.string_type
    uid_prop.name = 'uniqueID'
    uid_prop.data = unique_id
    uid_prop.id = context.map_struct.register_string('uniqueID')
    controller.asset_property_collection.property_map['uniqueID'] = uid_prop
    
    # Add waypointName
    wpn_prop = AssetProperty()
    wpn_prop.property_type = AssetPropertyType.string_type
    wpn_prop.name = 'waypointName'
    wpn_prop.data = unique_id
    wpn_prop.id = context.map_struct.register_string('waypointName')
    controller.asset_property_collection.property_map['waypointName'] = wpn_prop
    
    # Add waypointID (use controller_number + 5 to avoid conflicts)
    wpid_prop = AssetProperty()
    wpid_prop.property_type = AssetPropertyType.int_type
    wpid_prop.name = 'waypointID'
    wpid_prop.data = controller_number + 5
    wpid_prop.id = context.map_struct.register_string('waypointID')
    controller.asset_property_collection.property_map['waypointID'] = wpid_prop
    
    # Add waypointType (archon maps use this to distinguish controller spawns)
    wpt_prop = AssetProperty()
    wpt_prop.property_type = AssetPropertyType.int_type
    wpt_prop.name = 'waypointType'
    wpt_prop.data = 1  # All controller spawns use waypointType=1
    wpt_prop.id = context.map_struct.register_string('waypointType')
    controller.asset_property_collection.property_map['waypointType'] = wpt_prop
    
    return controller


def _reorder_string_pool_by_serialization(context: MapDataContext) -> None:
    """
    Reorder string pool by simulating a fresh parse of the map.
    
    The idea is to create a new string pool by "re-registering" all strings
    in the order they would be encountered during a fresh parse of the assets.
    This ensures the string pool order matches what the game engine expects.
    """
    from io import BytesIO
    
    # Save all assets to a buffer to determine the natural string order
    # This is the order strings would appear if we parsed this map fresh
    
    # We'll rebuild the string pool from scratch
    old_pool = context.map_struct.string_pool.copy()
    
    # Build new pool by walking assets in order
    new_order = []
    seen = set()
    
    def add_str(s):
        if s and s not in seen and s in old_pool:
            new_order.append(s)
            seen.add(s)
    
    # Walk assets in order, collecting strings
    for asset in context.map_struct.assets:
        # Asset name first
        add_str(asset.get_asset_name())
        
        # Then recursively collect all strings from this asset
        _collect_all_strings_from_asset(asset, add_str)
    
    # Add any remaining strings
    for s in sorted(old_pool.keys()):
        add_str(s)
    
    # Build new pool with 1-based indexing
    new_pool = {s: i+1 for i, s in enumerate(new_order)}
    
    # Create mapping from old to new indices
    old_to_new = {old_pool[s]: new_pool[s] for s in old_pool}
    
    # Update all references
    for asset in context.map_struct.assets:
        if asset.id in old_to_new:
            asset.id = old_to_new[asset.id]
        _update_asset_string_refs(asset, old_to_new)
    
    # Replace string pool
    context.map_struct.string_pool = new_pool


def _collect_all_strings_from_asset(asset, add_str) -> None:
    """Recursively collect all strings from an asset in parse order."""
    # Properties from various collection types
    for attr in ['properties', 'property_collection', 'asset_property_collection']:
        pc = getattr(asset, attr, None)
        if pc and hasattr(pc, 'property_map'):
            for prop_name, prop in pc.property_map.items():
                add_str(prop_name)
                # Some properties have string data
                if hasattr(prop, 'data') and isinstance(prop.data, str):
                    # Don't add property VALUES to string pool - only names
                    pass
    
    # Nested lists
    if hasattr(asset, 'players'):
        for item in asset.players:
            _collect_all_strings_from_asset(item, add_str)
    
    if hasattr(asset, 'teams'):
        for item in asset.teams:
            _collect_all_strings_from_asset(item, add_str)
    
    if hasattr(asset, 'map_objects'):
        for item in asset.map_objects:
            _collect_all_strings_from_asset(item, add_str)
    
    if hasattr(asset, 'script_lists'):
        add_str('ScriptList')
        for sl in asset.script_lists:
            _collect_script_list_strings_deep(sl, add_str)


def _collect_script_list_strings_deep(sl, add_str) -> None:
    """Collect all strings from a script list."""
    for script in getattr(sl, 'scripts', []):
        add_str('Script')
        _collect_script_strings_deep(script, add_str)
    
    for group in getattr(sl, 'script_groups', []):
        add_str('ScriptGroup')
        _collect_script_group_strings_deep(group, add_str)


def _collect_script_strings_deep(script, add_str) -> None:
    """Collect all strings from a script."""
    for cond in getattr(script, 'script_or_conditions', []):
        add_str('OrCondition')
        for c in getattr(cond, 'conditions', []):
            add_str('Condition')
            # Script content internal name
            if hasattr(c, 'internal_name'):
                add_str(c.internal_name)
    
    for action in getattr(script, 'script_action_on_true', []):
        add_str('ScriptAction')
        if hasattr(action, 'internal_name'):
            add_str(action.internal_name)
    
    for action in getattr(script, 'script_action_on_false', []):
        add_str('ScriptActionFalse')
        if hasattr(action, 'internal_name'):
            add_str(action.internal_name)


def _collect_script_group_strings_deep(group, add_str) -> None:
    """Collect all strings from a script group."""
    for script in getattr(group, 'scripts', []):
        add_str('Script')
        _collect_script_strings_deep(script, add_str)
    
    for subgroup in getattr(group, 'script_groups', []):
        add_str('ScriptGroup')
        _collect_script_group_strings_deep(subgroup, add_str)


def _reorder_string_pool_for_archon(context: MapDataContext) -> None:
    """
    Reorder the string pool to match the expected archon map order.
    
    The GT archon maps have a specific string pool order where strings appear
    in the order they're encountered when parsing assets sequentially:
    - Asset name, then all its property names, then next asset, etc.
    
    This function rebuilds the string pool in that order and updates all references.
    """
    old_pool = context.map_struct.string_pool.copy()
    
    # Build new order by "re-parsing" assets in order
    new_order = []
    seen = set()
    
    def add_string(s):
        if s and s not in seen and s in old_pool:
            new_order.append(s)
            seen.add(s)
    
    # Process each asset in order
    for asset in context.map_struct.assets:
        asset_name = asset.get_asset_name()
        add_string(asset_name)
        
        # Add property names from this asset
        if hasattr(asset, 'properties') and hasattr(asset.properties, 'property_map'):
            for prop_name in asset.properties.property_map.keys():
                add_string(prop_name)
        
        if hasattr(asset, 'asset_property_collection') and asset.asset_property_collection:
            for prop_name in asset.asset_property_collection.property_map.keys():
                add_string(prop_name)
        
        if hasattr(asset, 'property_collection') and asset.property_collection:
            for prop_name in asset.property_collection.property_map.keys():
                add_string(prop_name)
        
        # Handle complex assets
        if hasattr(asset, 'players'):
            for player in asset.players:
                if hasattr(player, 'asset_property_collection'):
                    for prop_name in player.asset_property_collection.property_map.keys():
                        add_string(prop_name)
        
        if hasattr(asset, 'teams'):
            for team in asset.teams:
                if hasattr(team, 'property_collection'):
                    for prop_name in team.property_collection.property_map.keys():
                        add_string(prop_name)
        
        if hasattr(asset, 'map_objects'):
            for obj in asset.map_objects:
                if hasattr(obj, 'asset_property_collection'):
                    for prop_name in obj.asset_property_collection.property_map.keys():
                        add_string(prop_name)
        
        if hasattr(asset, 'script_lists'):
            add_string('ScriptList')
            for sl in asset.script_lists:
                _collect_script_list_strings(sl, add_string)
    
    # Add any remaining strings that weren't encountered
    for s, _ in sorted(old_pool.items(), key=lambda x: x[1]):
        add_string(s)
    
    # Build new pool (1-indexed)
    new_pool = {s: i+1 for i, s in enumerate(new_order)}
    
    # Create old-to-new index mapping
    old_to_new = {old_pool[s]: new_pool[s] for s in old_pool}
    
    # Update all references in assets
    for asset in context.map_struct.assets:
        if asset.id in old_to_new:
            asset.id = old_to_new[asset.id]
        _update_asset_string_refs(asset, old_to_new)
    
    # Replace the string pool
    context.map_struct.string_pool = new_pool


def _collect_script_list_strings(sl, add_string):
    """Collect strings from a script list."""
    add_string('Script')
    add_string('ScriptGroup')
    add_string('ScriptAction')
    add_string('ScriptActionFalse')
    add_string('OrCondition')
    add_string('Condition')
    
    for script in getattr(sl, 'scripts', []):
        _collect_script_strings(script, add_string)
    for group in getattr(sl, 'script_groups', []):
        _collect_script_group_strings(group, add_string)


def _collect_script_strings(script, add_string):
    """Collect strings from a script."""
    for cond in getattr(script, 'script_or_conditions', []):
        for c in getattr(cond, 'conditions', []):
            if hasattr(c, 'internal_name'):
                add_string(c.internal_name)
            for arg in getattr(c, 'arguments', []):
                if hasattr(arg, 'string_argument'):
                    add_string(arg.string_argument)
    for action in getattr(script, 'script_action_on_true', []) + getattr(script, 'script_action_on_false', []):
        if hasattr(action, 'internal_name'):
            add_string(action.internal_name)
        for arg in getattr(action, 'arguments', []):
            if hasattr(arg, 'string_argument'):
                add_string(arg.string_argument)


def _collect_script_group_strings(group, add_string):
    """Collect strings from a script group."""
    for script in getattr(group, 'scripts', []):
        _collect_script_strings(script, add_string)
    for subgroup in getattr(group, 'script_groups', []):
        _collect_script_group_strings(subgroup, add_string)


def _update_asset_string_refs(asset, old_to_new: dict) -> None:
    """Update string references in an asset based on the old-to-new mapping."""
    # Handle property collections
    if hasattr(asset, 'property_collection') and asset.property_collection:
        _update_property_collection_refs(asset.property_collection, old_to_new)
    
    if hasattr(asset, 'asset_property_collection') and asset.asset_property_collection:
        _update_property_collection_refs(asset.asset_property_collection, old_to_new)
    
    # Handle lists of items
    if hasattr(asset, 'players'):
        for player in asset.players:
            _update_asset_string_refs(player, old_to_new)
    
    if hasattr(asset, 'teams'):
        for team in asset.teams:
            if hasattr(team, 'property_collection'):
                _update_property_collection_refs(team.property_collection, old_to_new)
    
    if hasattr(asset, 'map_objects'):
        for obj in asset.map_objects:
            _update_asset_string_refs(obj, old_to_new)
    
    if hasattr(asset, 'script_lists'):
        for sl in asset.script_lists:
            _update_script_list_refs(sl, old_to_new)
    
    if hasattr(asset, 'properties') and hasattr(asset.properties, 'property_map'):
        _update_property_collection_refs(asset.properties, old_to_new)


def _update_property_collection_refs(pc, old_to_new: dict) -> None:
    """Update string references in a property collection."""
    for prop in pc.property_map.values():
        if prop.id in old_to_new:
            prop.id = old_to_new[prop.id]


def _update_script_list_refs(sl, old_to_new: dict) -> None:
    """Update string references in a script list."""
    if sl.id in old_to_new:
        sl.id = old_to_new[sl.id]
    
    for script in getattr(sl, 'scripts', []):
        _update_script_refs(script, old_to_new)
    
    for group in getattr(sl, 'script_groups', []):
        _update_script_group_refs(group, old_to_new)


def _update_script_refs(script, old_to_new: dict) -> None:
    """Update string references in a script."""
    if script.id in old_to_new:
        script.id = old_to_new[script.id]
    
    for cond in getattr(script, 'script_or_conditions', []):
        if cond.id in old_to_new:
            cond.id = old_to_new[cond.id]
        for c in getattr(cond, 'conditions', []):
            if c.id in old_to_new:
                c.id = old_to_new[c.id]
    
    for action in getattr(script, 'script_action_on_true', []) + getattr(script, 'script_action_on_false', []):
        if action.id in old_to_new:
            action.id = old_to_new[action.id]


def _update_script_group_refs(group, old_to_new: dict) -> None:
    """Update string references in a script group."""
    if group.id in old_to_new:
        group.id = old_to_new[group.id]
    
    for script in getattr(group, 'scripts', []):
        _update_script_refs(script, old_to_new)
    
    for subgroup in getattr(group, 'script_groups', []):
        _update_script_group_refs(subgroup, old_to_new)


def add_global_version(context: MapDataContext) -> None:
    """
    Add GlobalVersion asset if not present.
    GlobalVersion is required by WorldBuilder and should be inserted after AssetList.
    """
    # Check if already present
    if context.get_asset('GlobalVersion') is not None:
        return
    
    # Create GlobalVersion asset
    gv = DefaultMajorAsset('GlobalVersion')
    gv.id = context.map_struct.register_string('GlobalVersion')
    gv.version = 1
    gv.name = 'GlobalVersion'
    gv.data = b''  # Empty data
    gv.data_size = 0
    
    # Insert after AssetList (position 1)
    # AssetList is always first, GlobalVersion should be second
    context.map_struct.assets.insert(1, gv)
    print(f"  Added GlobalVersion asset")


def create_controller_player(controller_number: int, context: MapDataContext) -> Player:
    """Create a new controller player."""
    player = Player()
    player.build_list_items = []
    player.asset_property_collection = AssetPropertyCollection()
    
    name = f'Player_{controller_number}'
    
    # Add required properties
    # AI properties are int_type with value 255 to disable AI behaviors (prevents resignation)
    properties = {
        'playerName': (AssetPropertyType.string_type, name),
        'playerIsHuman': (AssetPropertyType.bool_type, False),
        'playerDisplayName': (AssetPropertyType.string_unicode_type, name),
        'playerFaction': (AssetPropertyType.string_name_value_type, 'PlayerTemplate:FactionCivilian'),
        'playerAllies': (AssetPropertyType.string_type, ''),
        'playerEnemies': (AssetPropertyType.string_type, ''),
        'aiBaseBuilder': (AssetPropertyType.int_type, 255),
        'aiUnitBuilder': (AssetPropertyType.int_type, 255),
        'aiTeamBuilder': (AssetPropertyType.int_type, 255),
        'aiEconomyBuilder': (AssetPropertyType.int_type, 255),
        'aiWallBuilder': (AssetPropertyType.int_type, 255),
        'aiUnitUpgrader': (AssetPropertyType.int_type, 255),
        'aiScienceUpgrader': (AssetPropertyType.int_type, 255),
        'aiTactical': (AssetPropertyType.int_type, 255),
        'aiOpeningMover': (AssetPropertyType.int_type, 255),
        # Note: aiPersonality is NOT present in ground truth Fire Island, so we omit it
    }
    
    for prop_name, (prop_type, prop_data) in properties.items():
        prop = AssetProperty()
        prop.property_type = prop_type
        prop.name = prop_name
        prop.data = prop_data
        prop.id = context.map_struct.register_string(prop_name)
        player.asset_property_collection.property_map[prop_name] = prop
    
    return player


def create_controller_team(controller_number: int, context: MapDataContext) -> Team:
    """Create a basic team for the controller player."""
    team = Team()
    team.property_collection = AssetPropertyCollection()
    
    player_name = f'Player_{controller_number}'
    team_name = f'teamPlayer_{controller_number}'
    
    properties = {
        'teamName': (AssetPropertyType.string_type, team_name),
        'teamOwner': (AssetPropertyType.string_type, player_name),
        # Ground truth + archon templates expect controller player teams to be singleton
        'teamIsSingleton': (AssetPropertyType.bool_type, True),
    }
    
    for prop_name, (prop_type, prop_data) in properties.items():
        prop = AssetProperty()
        prop.property_type = prop_type
        prop.name = prop_name
        prop.data = prop_data
        prop.id = context.map_struct.register_string(prop_name)
        team.property_collection.property_map[prop_name] = prop
    
    return team


def create_archon_team(team_name: str, owner: str, context: MapDataContext) -> Team:
    """Create an archon-specific team with full properties."""
    team = Team()
    team.property_collection = AssetPropertyCollection()
    
    properties = {
        # Some archon templates only set this on a subset of teams, but ground truth
        # includes it broadly. Keeping it ensures teams are script-visible in-game.
        'exportWithScript': (AssetPropertyType.bool_type, True),
        'teamName': (AssetPropertyType.string_type, team_name),
        'teamOwner': (AssetPropertyType.string_type, owner),
        'teamIsSingleton': (AssetPropertyType.bool_type, False),
        # Archon templates use 0 here
        'teamProductionPriority': (AssetPropertyType.int_type, 0),
        'teamUnitMaxCount1': (AssetPropertyType.int_type, 0),
        'teamUnitMinCount1': (AssetPropertyType.int_type, 0),
        'teamUnitMaxCount2': (AssetPropertyType.int_type, 0),
        'teamUnitMinCount2': (AssetPropertyType.int_type, 0),
        'teamUnitMaxCount3': (AssetPropertyType.int_type, 0),
        'teamUnitMinCount3': (AssetPropertyType.int_type, 0),
        'teamUnitMaxCount4': (AssetPropertyType.int_type, 0),
        'teamUnitMinCount4': (AssetPropertyType.int_type, 0),
        'teamUnitMaxCount5': (AssetPropertyType.int_type, 0),
        'teamUnitMinCount5': (AssetPropertyType.int_type, 0),
        'teamUnitMaxCount6': (AssetPropertyType.int_type, 0),
        'teamUnitMinCount6': (AssetPropertyType.int_type, 0),
        'teamUnitMaxCount7': (AssetPropertyType.int_type, 0),
        'teamUnitMinCount7': (AssetPropertyType.int_type, 0),
        'teamDescription': (AssetPropertyType.string_type, ''),
        'teamMaxInstances': (AssetPropertyType.int_type, 1),
        'teamProductionPrioritySuccessIncrease': (AssetPropertyType.int_type, 0),
        'teamProductionPriorityFailureDecrease': (AssetPropertyType.int_type, 0),
        # In templates this is stored as an int (not float) and is 0
        'teamInitialIdleSeconds': (AssetPropertyType.int_type, 0),
        'teamExecutesActionsOnCreate': (AssetPropertyType.bool_type, False),
    }
    
    for prop_name, (prop_type, prop_data) in properties.items():
        prop = AssetProperty()
        prop.property_type = prop_type
        prop.name = prop_name
        prop.data = prop_data
        prop.id = context.map_struct.register_string(prop_name)
        team.property_collection.property_map[prop_name] = prop
    
    return team


def create_multiplayer_beacons(context: MapDataContext) -> List[str]:
    """
    Create the 5 MultiplayerBeacon objects required for Archon maps:
    - 4 corner beacons (owned by PlyrCivilian/Corner Beacon)
    - 1 center beacon (owned by PlyrCivilian/Center Beacon)
    
    These beacons are positioned at map corners and center based on map dimensions.
    
    Returns list of created beacon unique IDs.
    """
    objects_list = context.get_asset('ObjectsList')
    if not objects_list:
        return []
    
    # Place beacons like official Archon maps:
    # Use the *inner playable* rectangle after subtracting 2*border tiles.
    # Example (HF): 640 tiles, border 20 => inner 600 tiles => 6000 world units.
    border_tiles = context.border if getattr(context, "border", None) not in (None, -1) else 20
    inner_w_tiles = max(0, context.map_width - 2 * border_tiles)
    inner_h_tiles = max(0, context.map_height - 2 * border_tiles)
    world_width = inner_w_tiles * 10
    world_height = inner_h_tiles * 10
    
    # Calculate positions (corners and center) in world coordinates
    corner_positions = [
        (0.0, 0.0, 0.0),                                    # Bottom-left
        (float(world_width), 0.0, 0.0),                    # Bottom-right  
        (0.0, float(world_height), 0.0),                   # Top-left
        (float(world_width), float(world_height), 0.0),    # Top-right
    ]
    center_position = (float(world_width) / 2, float(world_height) / 2, 0.0)
    
    # Find the max existing unique ID number to avoid conflicts
    max_id_num = 0
    for obj in objects_list.map_objects:
        uid = obj.unique_id
        if uid and ' ' in uid:
            try:
                num = int(uid.split()[-1])
                max_id_num = max(max_id_num, num)
            except ValueError:
                pass
    
    created = []
    
    # Create center beacon first (this matches the ground truth order where center has lower ID)
    center_id_num = max_id_num + 1
    center_beacon = _create_multiplayer_beacon(
        context, center_id_num, center_position, 'PlyrCivilian/Center Beacon'
    )
    objects_list.map_objects.append(center_beacon)
    created.append(center_beacon.unique_id)
    
    # Create corner beacons
    for i, pos in enumerate(corner_positions):
        corner_id_num = max_id_num + 2 + i
        corner_beacon = _create_multiplayer_beacon(
            context, corner_id_num, pos, 'PlyrCivilian/Corner Beacon'
        )
        objects_list.map_objects.append(corner_beacon)
        created.append(corner_beacon.unique_id)
    
    return created


def _create_multiplayer_beacon(context: MapDataContext, id_num: int, 
                                position: Tuple[float, float, float],
                                owner: str) -> MapObject:
    """Create a single MultiplayerBeacon object."""
    beacon = MapObject()
    
    # Set MajorAsset fields
    beacon.id = context.map_struct.register_string("Object")
    beacon.version = 3
    beacon.name = "Object"
    
    # Set object fields
    beacon.type_name = "MultiplayerBeacon"
    beacon.position = position
    beacon.angle = 0.0
    beacon.road_option = 0
    
    # Create property collection
    beacon.asset_property_collection = AssetPropertyCollection()
    
    unique_id = f"MultiplayerBeacon {id_num}"
    
    # Match property set/order from official Archon maps (HF GT).
    properties = {
        'objectInitialHealth': (AssetPropertyType.int_type, 100),
        'objectEnabled': (AssetPropertyType.bool_type, True),
        'objectIndestructible': (AssetPropertyType.bool_type, False),
        'objectUnsellable': (AssetPropertyType.bool_type, False),
        'objectPowered': (AssetPropertyType.bool_type, True),
        'objectRecruitableAI': (AssetPropertyType.bool_type, True),
        'objectTargetable': (AssetPropertyType.bool_type, False),
        'objectSleeping': (AssetPropertyType.bool_type, False),
        'objectBasePriority': (AssetPropertyType.int_type, 40),
        'objectBasePhase': (AssetPropertyType.int_type, 1),
        'originalOwner': (AssetPropertyType.string_type, owner),
        'uniqueID': (AssetPropertyType.string_type, unique_id),
        'objectLayer': (AssetPropertyType.string_type, ''),
    }
    
    for prop_name, (prop_type, prop_data) in properties.items():
        prop = AssetProperty()
        prop.property_type = prop_type
        prop.name = prop_name
        prop.data = prop_data
        prop.id = context.map_struct.register_string(prop_name)
        beacon.asset_property_collection.property_map[prop_name] = prop
    
    return beacon


def copy_all_assets_from_template(source_ctx: MapDataContext,
                                   template_ctx: MapDataContext) -> str:
    """
    Replace all assets in source with assets from template for bit-perfect reproduction.
    This is used when source and template share the exact same base map.
    """
    # Copy string pool entirely from template
    source_ctx.map_struct.string_pool = copy.deepcopy(template_ctx.map_struct.string_pool)
    source_ctx.map_struct.index_to_string = copy.deepcopy(template_ctx.map_struct.index_to_string)
    
    # Copy all assets from template
    source_ctx.map_struct.assets = copy.deepcopy(template_ctx.map_struct.assets)
    
    return f"Copied {len(source_ctx.map_struct.assets)} assets and {len(source_ctx.map_struct.string_pool)} strings"


def sync_assets_with_template(source_ctx: MapDataContext, 
                               template_ctx: MapDataContext) -> str:
    """
    Sync missing assets and strings from template to source.
    This ensures the output has the same structure as the template.
    """
    added_assets = []
    added_strings = []
    
    # Copy ONLY archon-required strings from template's string pool
    # DO NOT copy template-specific strings like Music-related or Player_*_Start
    template_specific_strings = {
        'PATH_MUSIC_ENABLE_SPECIFIC_DYNAMIC_SYSTEM', 'PATH_MUSIC_PLAY_EVENT',
        'Player_1_Start', 'Player_2_Start', 'Player_3_Start',
        'Player_4_Start', 'Player_5_Start', 'Player_6_Start',
    }
    for s, idx in template_ctx.map_struct.string_pool.items():
        if s in template_specific_strings:
            continue
        if s not in source_ctx.map_struct.string_pool:
            source_ctx.map_struct.register_string(s)
            added_strings.append(s)
    
    # Check for missing assets (by type)
    source_asset_types = {a.get_asset_name() for a in source_ctx.map_struct.assets}
    
    # Assets that should NOT be copied from template (they're structural and optional)
    skip_assets = {'GlobalVersion'}
    
    for template_asset in template_ctx.map_struct.assets:
        asset_type = template_asset.get_asset_name()
        
        # Skip if source already has this asset type
        if asset_type in source_asset_types:
            continue
        
        # Skip structural assets that cause issues when added
        if asset_type in skip_assets:
            continue
        
        # Deep copy the asset
        new_asset = copy.deepcopy(template_asset)
        
        # Update the asset's id to use source's string pool
        new_asset.id = source_ctx.map_struct.register_string(asset_type)
        
        # Insert at the correct position to maintain order
        # Find where it should go based on template order
        template_idx = template_ctx.map_struct.assets.index(template_asset)
        
        # Find the best insert position
        insert_pos = 0
        for i, source_asset in enumerate(source_ctx.map_struct.assets):
            source_type = source_asset.get_asset_name()
            # Find this asset's position in template
            for j, t_asset in enumerate(template_ctx.map_struct.assets):
                if t_asset.get_asset_name() == source_type:
                    if j < template_idx:
                        insert_pos = i + 1
                    break
        
        source_ctx.map_struct.assets.insert(insert_pos, new_asset)
        source_asset_types.add(asset_type)
        added_assets.append(asset_type)
    
    return f"Added {len(added_assets)} assets ({', '.join(added_assets) if added_assets else 'none'}), {len(added_strings)} strings"


def copy_additional_objects_from_template(source_ctx: MapDataContext, 
                                          template_ctx: MapDataContext,
                                          num_builders: int,
                                          same_base_map: bool = False) -> Tuple[int, List[str]]:
    """
    Copy additional objects from template that aren't in source.
    
    For same_base_map=True:
        Copies all objects from template that have unique_ids not in source.
        This includes decorations that may have been added during archon conversion.
        
    For same_base_map=False (different maps):
        Only copies waypoints that are specifically used by archon scripts.
        Does NOT copy decorations or terrain objects from a different map.
    
    Returns (count, list of copied object names).
    """
    source_objects = source_ctx.get_asset('ObjectsList')
    template_objects = template_ctx.get_asset('ObjectsList')
    
    # Build set of existing unique_ids in source
    source_ids = set()
    for obj in source_objects.map_objects:
        if obj.unique_id:
            source_ids.add(obj.unique_id)
    
    # Player starts are handled separately
    player_start_pattern = {'Player_1_Start', 'Player_2_Start', 'Player_3_Start',
                            'Player_4_Start', 'Player_5_Start', 'Player_6_Start'}
    
    # Archon-specific waypoints that need to be copied (these are referenced by scripts)
    # Note: mpw1, mpw2 have template-specific positions and should NOT be copied to different maps
    # For same base maps, copy all archon waypoints
    # For different maps, don't copy any position-dependent waypoints
    archon_waypoint_ids_same_base = {'map mid', 'mpw1', 'mpw2', 'P1sp', 'P2sp', 'P3sp'}
    # For different maps, we don't copy position-dependent waypoints 
    # since they'd be at wrong positions
    archon_waypoint_ids_different = set()  # Don't copy any for different maps
    
    archon_waypoint_ids = archon_waypoint_ids_same_base if same_base_map else archon_waypoint_ids_different
    
    copied = 0
    copied_names = []
    
    for obj in template_objects.map_objects:
        unique_id = obj.unique_id
        
        # Skip player starts (handled separately)
        if unique_id in player_start_pattern:
            continue
        
        # Skip if already in source
        if unique_id and unique_id in source_ids:
            continue
        
        # For different maps, skip all position-dependent objects
        if not same_base_map:
            continue  # Don't copy anything for different maps
        
        # Deep copy and add to source
        new_obj = copy.deepcopy(obj)
        
        # Re-register any property strings
        for name, prop in new_obj.asset_property_collection.property_map.items():
            prop.id = source_ctx.map_struct.register_string(name)
        
        source_objects.map_objects.append(new_obj)
        copied += 1
        if unique_id:
            copied_names.append(unique_id)
    
    return copied, copied_names


def replace_objects_from_template(source_ctx: MapDataContext,
                                   template_ctx: MapDataContext) -> str:
    """
    Replace all objects in source with objects from template.
    Use this when source and template are based on the same map
    to get bit-perfect object reproduction.
    """
    source_objects = source_ctx.get_asset('ObjectsList')
    template_objects = template_ctx.get_asset('ObjectsList')
    
    old_count = len(source_objects.map_objects)
    
    # Deep copy all objects from template
    source_objects.map_objects = []
    for obj in template_objects.map_objects:
        new_obj = copy.deepcopy(obj)
        
        # Re-register property strings to source's string pool
        for name, prop in new_obj.asset_property_collection.property_map.items():
            prop.id = source_ctx.map_struct.register_string(name)
        
        source_objects.map_objects.append(new_obj)
    
    new_count = len(source_objects.map_objects)
    return f"Replaced {old_count} objects with {new_count} from template"


def create_waypoint_from_template(template_waypoint: MapObject,
                                   new_unique_id: str,
                                   new_position: Tuple[float, float, float],
                                   context: MapDataContext,
                                   waypoint_id: int = 100) -> MapObject:
    """
    Create a new waypoint by copying all properties from a template waypoint.
    This ensures all required object properties (originalOwner, etc.) are present.
    """
    waypoint = MapObject()
    
    # Set MajorAsset fields
    waypoint.id = context.map_struct.register_string("Object")
    waypoint.version = 3
    waypoint.name = "Object"
    
    # Copy basic MapObject fields
    waypoint.position = new_position
    waypoint.angle = template_waypoint.angle
    waypoint.road_option = template_waypoint.road_option
    waypoint.type_name = template_waypoint.type_name
    
    # Copy all properties from template
    waypoint.asset_property_collection = AssetPropertyCollection()
    for name, prop in template_waypoint.asset_property_collection.property_map.items():
        if name in ('uniqueID', 'waypointID', 'waypointName'):
            continue  # We'll set these specifically
        
        new_prop = AssetProperty()
        new_prop.property_type = prop.property_type
        new_prop.name = name
        new_prop.data = prop.data
        new_prop.id = context.map_struct.register_string(name)
        waypoint.asset_property_collection.property_map[name] = new_prop
    
    # Set unique properties
    uid_prop = AssetProperty()
    uid_prop.property_type = AssetPropertyType.string_type
    uid_prop.name = 'uniqueID'
    uid_prop.data = new_unique_id
    uid_prop.id = context.map_struct.register_string('uniqueID')
    waypoint.asset_property_collection.property_map['uniqueID'] = uid_prop
    
    wpn_prop = AssetProperty()
    wpn_prop.property_type = AssetPropertyType.string_type
    wpn_prop.name = 'waypointName'
    wpn_prop.data = new_unique_id
    wpn_prop.id = context.map_struct.register_string('waypointName')
    waypoint.asset_property_collection.property_map['waypointName'] = wpn_prop
    
    wpid_prop = AssetProperty()
    wpid_prop.property_type = AssetPropertyType.int_type
    wpid_prop.name = 'waypointID'
    wpid_prop.data = waypoint_id
    wpid_prop.id = context.map_struct.register_string('waypointID')
    waypoint.asset_property_collection.property_map['waypointID'] = wpid_prop
    
    return waypoint


def create_archon_waypoints_for_new_map(context: MapDataContext, 
                                         player_starts: List[MapObject]) -> List[str]:
    """
    Create archon-specific waypoints for a new map (when no matching template exists).
    These waypoints are used by archon scripts for various purposes.
    
    Returns list of created waypoint names.
    """
    objects_list = context.get_asset('ObjectsList')
    created = []
    
    # Use an existing waypoint as template for properties (originalOwner, etc.)
    template_waypoint = player_starts[0] if player_starts else None
    if not template_waypoint:
        print("  Warning: No template waypoint found for archon waypoints")
        return created
    
    # Avoid duplicates if this function is called multiple times
    existing_wp_names = set()
    existing_wp_ids = set()
    max_wp_id = -1
    for obj in objects_list.map_objects:
        name_prop = obj.asset_property_collection.get_property('waypointName')
        if name_prop and isinstance(name_prop.data, str):
            existing_wp_names.add(name_prop.data)
        id_prop = obj.asset_property_collection.get_property('waypointID')
        if id_prop and isinstance(id_prop.data, int):
            existing_wp_ids.add(id_prop.data)
            max_wp_id = max(max_wp_id, id_prop.data)
    
    def alloc_id() -> int:
        nonlocal max_wp_id
        candidate = max_wp_id + 1
        while candidate in existing_wp_ids:
            candidate += 1
        existing_wp_ids.add(candidate)
        max_wp_id = candidate
        return candidate
    
    # Map dimensions in world coordinates
    map_w = context.map_width * 10
    map_h = context.map_height * 10
    
    # Create 'map mid' waypoint near the center of the *players*, not the map.
    # This matches how official archon templates place it.
    if len(player_starts) >= 2:
        mid_x = (player_starts[0].position[0] + player_starts[1].position[0]) / 2
        mid_y = (player_starts[0].position[1] + player_starts[1].position[1]) / 2
    else:
        mid_x = map_w / 2
        mid_y = map_h / 2
    
    if 'map mid' not in existing_wp_names:
        map_mid = create_waypoint_from_template(
            template_waypoint, 'map mid', (mid_x, mid_y, 0), context, waypoint_id=alloc_id()
        )
        objects_list.map_objects.append(map_mid)
        created.append('map mid')
    
    # Off-map archon waypoints used by scripts to spawn/teleport helper units out of play.
    # Place them safely outside the playable area with moderate offsets (avoid extreme coords).
    p1sp_pos = (-0.255 * map_w, -0.085 * map_h, 0.0)
    mpw1_pos = (-0.302 * map_w, -0.083 * map_h, 0.0)
    p2sp_pos = (1.125 * map_w, 1.133 * map_h, 0.0)
    mpw2_pos = (1.175 * map_w, 1.130 * map_h, 0.0)
    
    # Create P{b}sp markers (only first two are used by shipped archon scripts)
    for i, builder_start in enumerate(player_starts[:2]):
        builder_num = i + 1
        sp_name = f'P{builder_num}sp'
        if sp_name in existing_wp_names:
            continue
        new_pos = p1sp_pos if builder_num == 1 else p2sp_pos
        sp_waypoint = create_waypoint_from_template(
            template_waypoint, sp_name, new_pos, context, waypoint_id=alloc_id()
        )
        objects_list.map_objects.append(sp_waypoint)
        created.append(sp_name)
    
    # Create mpw1/mpw2 spawn points (only meaningful for 2-player archon)
    if len(player_starts) >= 2:
        if 'mpw1' not in existing_wp_names:
            mpw1 = create_waypoint_from_template(
                template_waypoint, 'mpw1', mpw1_pos, context, waypoint_id=alloc_id()
            )
            objects_list.map_objects.append(mpw1)
            created.append('mpw1')
        if 'mpw2' not in existing_wp_names:
            mpw2 = create_waypoint_from_template(
                template_waypoint, 'mpw2', mpw2_pos, context, waypoint_id=alloc_id()
            )
            objects_list.map_objects.append(mpw2)
            created.append('mpw2')
    
    return created


def create_paired_3p_archon_waypoints(context: MapDataContext,
                                       player_starts: List[MapObject],
                                       template_ctx: Optional[MapDataContext] = None) -> List[str]:
    """
    Create the special waypoints needed for paired 3p archon maps:
    - 'Apron Ocuppier Player_{1,3,5}' - for builders
    - 'Linked Airfield Player_{2,4,6}' - for controllers
    
    These are placed in the *off-play* border region (outside the playable rectangle).
    Their relative offsets are derived from the paired-3p template when available.
    """
    objects_list = context.get_asset('ObjectsList')
    created = []
    
    template_waypoint = player_starts[0] if player_starts else None
    if not template_waypoint:
        return created
    
    # Collect existing waypoint info
    existing_wp_names = set()
    existing_wp_ids = set()
    max_wp_id = -1
    for obj in objects_list.map_objects:
        name_prop = obj.asset_property_collection.get_property('waypointName')
        if name_prop and isinstance(name_prop.data, str):
            existing_wp_names.add(name_prop.data)
        id_prop = obj.asset_property_collection.get_property('waypointID')
        if id_prop and isinstance(id_prop.data, int):
            existing_wp_ids.add(id_prop.data)
            max_wp_id = max(max_wp_id, id_prop.data)
    
    def alloc_id() -> int:
        nonlocal max_wp_id
        candidate = max_wp_id + 1
        while candidate in existing_wp_ids:
            candidate += 1
        existing_wp_ids.add(candidate)
        max_wp_id = candidate
        return candidate
    
    map_w = context.map_width * 10
    map_h = context.map_height * 10
    # Playable bounds (world coords)
    x_min, x_max, y_min, y_max = _playable_bounds_world(context)
    
    # Get builder positions (assuming paired scheme: builders are 1, 3, 5)
    # Player starts might be in any order, so we need to identify them by unique_id
    builder_positions = {}
    for obj in objects_list.map_objects:
        if isinstance(obj.unique_id, str) and obj.unique_id.endswith('_Start'):
            try:
                num = int(obj.unique_id.split('_')[1])
                if num in (1, 3, 5):
                    builder_positions[num] = obj.position
            except:
                pass

    # Compute per-pair anchor points in off-play area (near closest playable edge to each builder)
    anchors: Dict[int, Tuple[float, float, float]] = {}
    for b, pos in builder_positions.items():
        anchors[b] = _offplay_anchor_near_builder(context, pos, margin=700.0)

    # Derive waypoint offsets from template (relative to each controller's keepalive building),
    # so we preserve the intended local layout even on different-base maps.
    # Returns mapping builder_num -> (apron_dxdy, linked_dxdy)
    deltas: Dict[int, Tuple[Tuple[float, float], Tuple[float, float]]] = {}
    if template_ctx:
        try:
            t_objs = template_ctx.get_asset("ObjectsList")
            if t_objs:
                # Controller building position by controller number
                t_controller_building: Dict[int, Tuple[float, float, float]] = {}
                for cn in (2, 4, 6):
                    for o in t_objs.map_objects:
                        owner = o.asset_property_collection.get_property("originalOwner")
                        if not owner or not isinstance(owner.data, str):
                            continue
                        if not owner.data.startswith(f"Player_{cn}/"):
                            continue
                        if "Waypoint" in (o.type_name or ""):
                            continue
                        t_controller_building[cn] = o.position
                        break

                # Waypoint position by waypointName
                t_wp_pos: Dict[str, Tuple[float, float, float]] = {}
                for o in t_objs.map_objects:
                    if "Waypoint" not in (o.type_name or ""):
                        continue
                    pn = o.asset_property_collection.get_property("waypointName")
                    if pn and isinstance(pn.data, str):
                        t_wp_pos[pn.data] = o.position

                for b, c in [(1, 2), (3, 4), (5, 6)]:
                    base = t_controller_building.get(c)
                    apron = t_wp_pos.get(f"Apron Ocuppier Player_{b}")
                    linked = t_wp_pos.get(f"Linked Airfield Player_{c}")
                    if not base or not apron or not linked:
                        continue
                    deltas[b] = (
                        (apron[0] - base[0], apron[1] - base[1]),
                        (linked[0] - base[0], linked[1] - base[1]),
                    )
        except Exception:
            deltas = {}

    # Reasonable defaults based on paired-3p templates (Caldera)
    default_apron_delta = (80.0, 35.0)
    default_linked_delta = (-120.0, 135.0)
    
    # For each builder/controller pair, create Apron and Linked Airfield waypoints in off-play area.
    pairs = [(1, 2), (3, 4), (5, 6)]
    
    for builder_num, controller_num in pairs:
        if builder_num not in builder_positions or builder_num not in anchors:
            continue

        ax, ay, _ = anchors[builder_num]
        apron_delta, linked_delta = deltas.get(builder_num, (default_apron_delta, default_linked_delta))
        
        # Apron Ocuppier for builder (placed outside, in direction away from center)
        apron_name = f"Apron Ocuppier Player_{builder_num}"
        if apron_name not in existing_wp_names:
            apron_pos = (ax + apron_delta[0], ay + apron_delta[1], 0.0)
            apron = create_waypoint_from_template(
                template_waypoint, apron_name, apron_pos, context, waypoint_id=alloc_id()
            )
            objects_list.map_objects.append(apron)
            created.append(apron_name)
        
        # Linked Airfield for controller (placed near Apron, slightly offset)
        linked_name = f"Linked Airfield Player_{controller_num}"
        if linked_name not in existing_wp_names:
            linked_pos = (ax + linked_delta[0], ay + linked_delta[1], 0.0)
            linked = create_waypoint_from_template(
                template_waypoint, linked_name, linked_pos, context, waypoint_id=alloc_id()
            )
            objects_list.map_objects.append(linked)
            created.append(linked_name)
    
    return created


def are_maps_same_base(source_ctx: MapDataContext, template_ctx: MapDataContext) -> bool:
    """
    Check if source and template maps share the same base terrain/layout.
    This is determined by comparing map dimensions and builder start positions.
    """
    # Check dimensions
    if source_ctx.map_width != template_ctx.map_width or source_ctx.map_height != template_ctx.map_height:
        return False
    
    # Get builder positions from both
    source_starts = find_player_starts(source_ctx)
    template_starts = find_player_starts(template_ctx)
    
    # Special handling for paired 3p archon templates (builders are Player_1, Player_3, Player_5)
    if _is_paired_archon_3p_template(template_ctx):
        # Collect source builder positions (whatever starts exist)
        src_pos = []
        for start in source_starts:
            try:
                n = int(start.unique_id.split('_')[1])
            except (ValueError, IndexError):
                continue
            # Source has 3 builders
            if 1 <= n <= 3:
                src_pos.append(start.position[:2])

        tpl_pos = []
        for start in template_starts:
            try:
                n = int(start.unique_id.split('_')[1])
            except (ValueError, IndexError):
                continue
            if n in (1, 3, 5):
                tpl_pos.append(start.position[:2])

        if len(src_pos) != 3 or len(tpl_pos) != 3:
            return False

        # Match positions ignoring numbering: each source start must be close to a unique template builder start.
        tolerance = 10.0
        used = [False, False, False]
        for sx, sy in src_pos:
            best_j = None
            best_d = 1e18
            for j, (tx, ty) in enumerate(tpl_pos):
                if used[j]:
                    continue
                d = abs(sx - tx) + abs(sy - ty)
                if d < best_d:
                    best_d = d
                    best_j = j
            if best_j is None:
                return False
            tx, ty = tpl_pos[best_j]
            if abs(sx - tx) > tolerance or abs(sy - ty) > tolerance:
                return False
            used[best_j] = True

        return True

    # Get positions of builders (Player_1, Player_2, etc.) - not controllers
    source_builder_positions = {}
    for start in source_starts:
        try:
            num = int(start.unique_id.split('_')[1])
            source_builder_positions[num] = start.position[:2]
        except (ValueError, IndexError):
            pass
    
    template_builder_positions = {}
    for start in template_starts:
        try:
            num = int(start.unique_id.split('_')[1])
            # Only compare original builders (1, 2, 3) - not controllers (4, 5, 6)
            if num <= 3:
                template_builder_positions[num] = start.position[:2]
        except (ValueError, IndexError):
            pass
    
    # Check if builder positions match (within tolerance)
    tolerance = 10.0  # World units
    for num, (sx, sy) in source_builder_positions.items():
        if num in template_builder_positions:
            tx, ty = template_builder_positions[num]
            if abs(sx - tx) > tolerance or abs(sy - ty) > tolerance:
                return False
        else:
            return False  # Missing builder in template
    
    return len(source_builder_positions) > 0


def get_template_controller_positions(template_ctx: MapDataContext, num_builders: int) -> Dict[int, Tuple[float, float, float]]:
    """
    Extract controller positions from template map.
    Returns dict mapping controller number to position.
    """
    positions = {}
    template_starts = find_player_starts(template_ctx)
    
    for start in template_starts:
        # Extract player number from unique_id
        try:
            num = int(start.unique_id.split('_')[1])
            if num > num_builders:  # This is a controller
                positions[num] = start.position
        except (ValueError, IndexError):
            pass
    
    return positions


def transform_to_archon(source_context: MapDataContext, 
                        template_context: Optional[MapDataContext] = None,
                        bit_perfect: bool = False,
                        wb_normalize_terrain: bool = False) -> MapDataContext:
    """
    Transform a normal map context into an archon map context.
    
    Args:
        source_context: The source map's MapDataContext
        template_context: Optional archon template map's context (for scripts and positions)
        bit_perfect: If True, replace objects/assets from template for exact reproduction
    
    Returns:
        Modified MapDataContext with archon support
    """
    context = source_context  # Modify in place
    
    # Ensure GlobalVersion exists (official archon maps include it)
    add_global_version(context)
    
    # Find existing player starts
    player_starts = find_player_starts(context)
    num_builders = len(player_starts)
    
    if num_builders > 3:
        raise ValueError(f"Map has {num_builders} players. Archon mode only supports maps with 1-3 players "
                        "(max 6 total players with controllers).")
    
    if num_builders == 0:
        raise ValueError("Map has no player start waypoints.")
    
    print(f"Found {num_builders} player start(s), will add {num_builders} controller(s)")
    
    # Detect if source and template share the same base map
    same_base_map = False
    if template_context:
        same_base_map = are_maps_same_base(context, template_context)
        if same_base_map:
            print("  Maps share same base - will use template positions and objects")
        else:
            print("  Different base map - will calculate positions and copy only archon waypoints")
    
    # Detect paired 3p template scheme (1-2,3-4,5-6) used by ground-truth 3v3 archon maps.
    paired_3p = bool(template_context) and num_builders == 3 and _is_paired_archon_3p_template(template_context)
    if paired_3p:
        print("  Detected paired 3p archon template (pairs: 1-2, 3-4, 5-6)")

    # === FAST PATH: Same base map - just copy all assets from template ===
    # This produces bit-perfect output without any manual modifications
    if same_base_map and template_context:
        print("\n=== Same base map: copying all assets from template ===")
        result = copy_all_assets_from_template(context, template_context)
        print(f"  {result}")
        
        # Apply terrain/sidecar compatibility
        if wb_normalize_terrain:
            _normalize_blend_tile_data_for_wb(context)
        _ensure_worldinfo_terrain_texture_strings(context)
        
        return context

    # If paired 3p and same base map, replace Player_{1..6}_Start objects from template to avoid numbering conflicts
    if paired_3p and template_context and same_base_map:
        _replace_player_starts_from_template(context, template_context)
        # Do NOT refresh num_builders; it represents builder count (3). We keep it stable.
        player_starts = find_player_starts(context)

    # Get template controller positions ONLY if same base map (for consecutive scheme)
    template_positions = {}
    if template_context and same_base_map and not paired_3p:
        template_positions = get_template_controller_positions(template_context, num_builders)

    # For paired 3p templates on different maps, rename builder starts to odd indices (1,3,5).
    if paired_3p and not same_base_map:
        _apply_paired_3p_player_start_numbering_for_different_map(context)
        player_starts = find_player_starts(context)  # refresh after rename
    
    # === Step 1: Add controller start waypoints ===
    objects_list = context.get_asset('ObjectsList')

    if paired_3p:
        # Paired scheme uses controllers 2,4,6 and builders 1,3,5.
        # If same base map, rely on template objects copy for precise placement; otherwise compute offsets.
        # Always ensure controller starts exist.
        desired_controller_starts = {f"Player_{i}_Start" for i in (2, 4, 6)}
        existing_ids = {o.unique_id for o in objects_list.map_objects if getattr(o, "unique_id", None)}

        if same_base_map and template_context:
            # Copy the controller start objects directly from template (exact parity)
            t_objs = template_context.get_asset("ObjectsList")
            if t_objs:
                for o in t_objs.map_objects:
                    if o.unique_id in desired_controller_starts and o.unique_id not in existing_ids:
                        new_obj = copy.deepcopy(o)
                        for name, prop in new_obj.asset_property_collection.property_map.items():
                            prop.id = context.map_struct.register_string(name)
                        objects_list.map_objects.append(new_obj)
                        existing_ids.add(new_obj.unique_id)
                        print(f"  Added {new_obj.unique_id} [from template]")
        else:
            # Compute controller starts by offsetting each builder start (odd numbers 1,3,5)
            # Builders are whatever starts we currently have; assign them in increasing player-number order.
            builder_nums = [1, 3, 5]
            for i, builder_start in enumerate(player_starts):
                builder_num = builder_nums[i] if i < len(builder_nums) else (i * 2 + 1)
                controller_num = builder_num + 1  # 1->2, 3->4, 5->6
                controller_uid = f"Player_{controller_num}_Start"
                if controller_uid in existing_ids:
                    continue
                controller_pos = calculate_controller_offset(
                    builder_start.position,
                    context.map_width,
                    context.map_height
                )
                controller_start = create_controller_start_waypoint(
                    builder_start, controller_num, controller_pos, context
                )
                objects_list.map_objects.append(controller_start)
                existing_ids.add(controller_uid)
                print(f"  Added Player_{controller_num}_Start at ({controller_pos[0]:.1f}, {controller_pos[1]:.1f}) [calculated]")
    else:
        for i, builder_start in enumerate(player_starts):
            builder_num = i + 1
            controller_num = builder_num + num_builders  # e.g., Builder 1 -> Controller 3

            # Use template position ONLY if same base map, otherwise calculate
            if same_base_map and controller_num in template_positions:
                controller_pos = template_positions[controller_num]
                pos_source = "from template"
            else:
                controller_pos = calculate_controller_offset(
                    builder_start.position,
                    context.map_width,
                    context.map_height
                )
                pos_source = "calculated"

            controller_start = create_controller_start_waypoint(
                builder_start, controller_num, controller_pos, context
            )

            objects_list.map_objects.append(controller_start)
            print(f"  Added Player_{controller_num}_Start at ({controller_pos[0]:.1f}, {controller_pos[1]:.1f}) [{pos_source}]")
    
    # === Step 2: Add controller players ===
    sides_list = context.get_asset('SidesList')

    if paired_3p and template_context:
        # For paired 3p archon, we need to:
        # 1. Keep source players 
        # 2. ADD archon-required properties from template (aiPersonality, playerColor, etc.)
        # 3. Add Player_4/5/6 if missing
        
        t_sides = template_context.get_asset("SidesList")
        
        # Build template player lookup
        template_players = {}
        if t_sides:
            for pl in t_sides.players:
                pn = pl.asset_property_collection.get_property("playerName")
                if pn:
                    template_players[pn.data] = pl
        
        # Update ALL source players with properties from template (add missing props)
        archon_props = ['aiPersonality', 'playerColor', 'playerRadarColor', 'exportWithScript']
        for pl in sides_list.players:
            pn = pl.asset_property_collection.get_property("playerName")
            if not pn:
                continue
            player_name = pn.data
            
            # Find matching template player
            tpl_player = template_players.get(player_name)
            if not tpl_player:
                continue
            
            # Copy archon-required properties from template if not in source
            for prop_name in archon_props:
                if pl.asset_property_collection.get_property(prop_name) is None:
                    tpl_prop = tpl_player.asset_property_collection.get_property(prop_name)
                    if tpl_prop:
                        # Create a copy and register string
                        new_prop = copy.deepcopy(tpl_prop)
                        new_prop.id = context.map_struct.register_string(prop_name)
                        pl.asset_property_collection.property_map[prop_name] = new_prop
        
        # Find which Player_N already exist
        existing_players = set()
        for pl in sides_list.players:
            pn = pl.asset_property_collection.get_property("playerName")
            if pn and isinstance(pn.data, str) and pn.data.startswith("Player_"):
                try:
                    n = int(pn.data.split("_")[1])
                    existing_players.add(n)
                except Exception:
                    pass
        
        # Add missing players (typically 4, 5, 6 for controllers)
        for n in [1, 2, 3, 4, 5, 6]:
            if n not in existing_players:
                # Copy from template if available
                tpl_player = template_players.get(f"Player_{n}")
                if tpl_player:
                    new_player = copy.deepcopy(tpl_player)
                    for name, prop in new_player.asset_property_collection.property_map.items():
                        prop.id = context.map_struct.register_string(name)
                else:
                    new_player = create_controller_player(n, context)
                sides_list.players.append(new_player)
                print(f"  Added Player_{n} to SidesList")
        
        print(f"  Updated SidesList with archon properties from template")
    else:
        for i in range(num_builders):
            controller_num = i + 1 + num_builders
            controller_player = create_controller_player(controller_num, context)
            sides_list.players.append(controller_player)
            print(f"  Added Player_{controller_num} to sides list")
    
    # === Step 2a: Expand LibraryMapLists to match player count ===
    # LibraryMapLists must have one entry per player slot (18 for 6-player archon)
    # This is critical - mismatch causes immediate player defeat!
    library_map_lists = context.get_asset('LibraryMapLists')
    if library_map_lists:
        target_count = len(sides_list.players)  # Should be 18 for archon
        current_count = len(library_map_lists.library_maps)
        if current_count < target_count:
            entries_to_add = target_count - current_count
            for _ in range(entries_to_add):
                new_entry = LibraryMaps()
                new_entry.library_maps = []  # Empty list, same as GT
                new_entry.version = 1
                new_entry.id = library_map_lists.library_maps[0].id if library_map_lists.library_maps else 0
                library_map_lists.library_maps.append(new_entry)
            print(f"  Expanded LibraryMapLists from {current_count} to {target_count} entries")
    
    # === Step 2b: Expand BuildLists to match player count ===
    # BuildLists must also have one entry per player slot (18 for 6-player archon).
    # For Player_* entries, official maps use 'PlayerTemplate:Civilian' (NOT 'FactionCivilian').
    build_lists = context.get_asset('BuildLists')
    if build_lists:
        target_count = len(sides_list.players)  # Should be 18 for archon
        current_count = len(build_lists.build_list)
        if current_count < target_count:
            entries_to_add = target_count - current_count
            for _ in range(entries_to_add):
                new_entry = BuildList()
                new_entry.faction = "PlayerTemplate:Civilian"
                new_entry.count = 0
                build_lists.build_list.append(new_entry)
            print(f"  Expanded BuildLists from {current_count} to {target_count} entries")
    
    # === Step 3: Add teams ===
    teams = context.get_asset('Teams')

    if paired_3p and template_context:
        # For paired 3p:
        # 1. Keep source teams but ADD archon-required properties from template
        # 2. ADD archon-specific teams from template that don't exist in source
        t_teams = template_context.get_asset("Teams")
        if t_teams:
            def key_for(team_obj: Team) -> Tuple[str, str]:
                n = team_obj.property_collection.get_property("teamName")
                o = team_obj.property_collection.get_property("teamOwner")
                return (n.data if n else "", o.data if o else "")
            
            def get_name(team_obj: Team) -> str:
                n = team_obj.property_collection.get_property("teamName")
                return n.data if n else ""

            # Build template team lookup
            template_teams = {key_for(tm): tm for tm in t_teams.teams}
            
            # Set exportWithScript=True on all teams (archon requirement).
            # IMPORTANT: This must be TRUE (not just present) for controller survival on paired-3p maps.
            # Skip first team (empty team) and teamPlyrCivilian.
            for i, tm in enumerate(teams.teams):
                team_name = get_name(tm)
                # Skip first empty team and teamPlyrCivilian
                if i == 0 or team_name == 'teamPlyrCivilian':
                    continue
                exp = tm.property_collection.get_property('exportWithScript')
                if exp is None:
                    prop = AssetProperty()
                    prop.id = context.map_struct.register_string('exportWithScript')
                    prop.property_type = AssetPropertyType.bool_type
                    prop.name = 'exportWithScript'
                    prop.data = True
                    tm.property_collection.property_map['exportWithScript'] = prop
                else:
                    # If present but wrong (older outputs, or templates that store it oddly), force-correct.
                    exp.property_type = AssetPropertyType.bool_type
                    exp.data = True
            
            source_keys = {key_for(tm) for tm in teams.teams}
            
            # Archon-specific team prefixes (only add these from template if not in source)
            archon_team_prefixes = ['teamPlayer_4', 'teamPlayer_5', 'teamPlayer_6', 
                                   'Apron Occupier', 'Linked Airfield', 
                                   'Center Beacon', 'Corner Beacon', 'Reference P']
            
            added_archon = 0
            
            # Add ONLY Archon-specific teams from template that aren't in source
            for tm in t_teams.teams:
                k = key_for(tm)
                team_name = get_name(tm)
                
                # Skip if already in source
                if k in source_keys:
                    continue
                
                # Only add Archon-related teams
                is_archon_team = any(team_name.startswith(prefix) for prefix in archon_team_prefixes)
                if is_archon_team:
                    copied = copy.deepcopy(tm)
                    for name, prop in copied.property_collection.property_map.items():
                        prop.id = context.map_struct.register_string(name)
                    # Ensure exportWithScript=True on added archon teams
                    exp = copied.property_collection.get_property('exportWithScript')
                    if exp is None:
                        prop = AssetProperty()
                        prop.id = context.map_struct.register_string('exportWithScript')
                        prop.property_type = AssetPropertyType.bool_type
                        prop.name = 'exportWithScript'
                        prop.data = True
                        copied.property_collection.property_map['exportWithScript'] = prop
                    else:
                        exp.property_type = AssetPropertyType.bool_type
                        exp.data = True
                    teams.teams.append(copied)
                    added_archon += 1
            
            print(f"  Teams: updated with archon properties, added {added_archon} archon teams, total={len(teams.teams)}")
    else:
        # Add teams for each builder/controller pair
        # IMPORTANT: Order matters! Add controller team, then archon teams for each pair
        archon_team_templates = [
            ('P{c}s team', 'Player_{c}'),     # Controller's team
            ('P{b}splt', 'Player_{b}'),       # Builder's split team
            ('p{b}hw', 'Player_{b}'),         # Builder's hardware team
            ('P{c}s jc', 'Player_{c}'),       # Controller's jet control team
        ]

        for i in range(num_builders):
            builder_num = i + 1
            controller_num = i + 1 + num_builders

            # First add the controller's base team
            controller_team = create_controller_team(controller_num, context)
            teams.teams.append(controller_team)
            print(f"  Added teamPlayer_{controller_num}")

            # Then add the archon-specific teams for this builder/controller pair
            for template, owner_template in archon_team_templates:
                team_name = template.format(b=builder_num, c=controller_num)
                owner = owner_template.format(b=builder_num, c=controller_num)

                archon_team = create_archon_team(team_name, owner, context)
                teams.teams.append(archon_team)
                print(f"  Added team '{team_name}' owned by '{owner}'")
    
    # === Step 4: Copy objects from template OR generate for new map ===
    if template_context:
        if bit_perfect or same_base_map:
            # For same base map, replace all objects to maintain exact order from template
            # This ensures bit-perfect parity with the ground truth
            print("\nReplacing all objects from template (same base map)...")
            result = replace_objects_from_template(context, template_context)
            print(f"  {result}")
        else:
            # Different map - generate archon waypoints instead of copying from template
            if paired_3p:
                # Paired 3p scripts only require Apron/Linked Airfield waypoints (and the 6 Player_*_Start).
                # Avoid adding unrelated 2-player archon waypoints (map mid/P1sp/P2sp/mpw1/mpw2) that can
                # shift waypoint IDs and confuse scripts.
                print("\nCreating paired 3p archon waypoints for new map...")
                paired_created = create_paired_3p_archon_waypoints(context, player_starts, template_ctx=template_context)
                print(f"  Created {len(paired_created)} paired waypoints: {', '.join(paired_created)}")
            else:
                print("\nGenerating archon waypoints for new map...")
                created = create_archon_waypoints_for_new_map(context, player_starts)
                print(f"  Created {len(created)} waypoints: {', '.join(created)}")
            
            # Also copy any archon-specific waypoints from template that might be needed
            print("  Checking template for additional archon waypoints...")
            copied, copied_names = copy_additional_objects_from_template(
                context, template_context, num_builders, same_base_map=False
            )
            if copied > 0:
                print(f"  Copied {copied} archon waypoints: {', '.join(copied_names)}")

            # Paired-3p templates rely on a specific waypointID numbering scheme (IDs 1..12) which
            # varies per template. Remap our waypointIDs to match the template after all waypoints exist.
            if paired_3p:
                changed = _remap_waypoint_ids_from_template(context, template_context)
                if changed:
                    print(f"  Remapped waypointIDs from template: updated {changed} waypoint(s)")
    else:
        # No template - generate basic archon waypoints
        print("\nGenerating archon waypoints (no template)...")
        created = create_archon_waypoints_for_new_map(context, player_starts)
        print(f"  Created {len(created)} waypoints: {', '.join(created)}")
    
    # === Step 4a: Create MultiplayerBeacon objects ===
    # These are required for Archon maps - 4 corner beacons + 1 center beacon
    # Only create if not already present (e.g., from bit_perfect copy or same_base_map)
    objects_list = context.get_asset('ObjectsList')
    existing_beacons = sum(1 for o in objects_list.map_objects if o.type_name == 'MultiplayerBeacon')
    if existing_beacons == 0:
        print("\nCreating MultiplayerBeacon objects...")
        beacons_created = create_multiplayer_beacons(context)
        print(f"  Created {len(beacons_created)} beacons: {', '.join(beacons_created)}")
    else:
        print(f"\n  {existing_beacons} MultiplayerBeacon objects already exist")
    
    # === Step 4b: Sync/copy assets and strings from template ===
    if template_context:
        if bit_perfect or same_base_map:
            # For same base map, copy all assets to ensure bit-perfect parity
            print("\nCopying all assets from template (same base map)...")
            result = copy_all_assets_from_template(context, template_context)
            print(f"  {result}")
        else:
            print("\nSyncing assets and strings with template...")
            synced = sync_assets_with_template(context, template_context)
            print(f"  {synced}")
    
    # === Step 5: Copy scripts from template ===
    # For same_base_map, scripts were already copied via copy_all_assets_from_template
    if template_context and not same_base_map:
        print("\nCopying archon scripts from template...")
        copy_scripts_from_template(context, template_context, num_builders)
    elif not template_context:
        print("\nWarning: No template provided. Archon scripts not added.")
        print("  The map will have the structure for archon mode but won't function correctly")
        print("  without the control transfer scripts.")
    else:
        print("\n  Scripts already copied via same-base-map asset copy")
    
    # === Step 6: Terrain/sidecar compatibility ===
    # Most official Archon maps keep BlendTileData identical to the base map.
    # We only apply WB's canonicalization when explicitly requested (useful for some maps
    # where textures don't render without a WB save).
    if wb_normalize_terrain:
        _normalize_blend_tile_data_for_wb(context)
    _ensure_worldinfo_terrain_texture_strings(context)
    if paired_3p:
        _ensure_controller_keepalive_buildings(context, num_builders, controller_numbers=[2, 4, 6])
    else:
        _ensure_controller_keepalive_buildings(context, num_builders)

    # === Step 6b: Generate NamedCameras for controller players ===
    # Archon mode requires camera entries for controllers (PLAYER_2_SET, PLAYER_4_SET, PLAYER_6_SET)
    # These are positioned at the builder start locations
    # Only generate if not already done (e.g., via same_base_map copy)
    named_cameras_generated = False
    for asset in context.map_struct.assets:
        if asset.get_asset_name() == 'NamedCameras':
            if hasattr(asset, 'data') and len(asset.data) > 4:
                import struct
                count = struct.unpack('<I', asset.data[:4])[0]
                if count >= num_builders:
                    named_cameras_generated = True
            break
    
    if not named_cameras_generated and not same_base_map:
        print("\nGenerating NamedCameras for controller players...")
        num_cams = _generate_named_cameras_for_archon(context, num_builders, template_context)
        print(f"  Created {num_cams} controller camera entries")

    # === Step 7: AssetList Archon block augmentation (paired 3p, different-base only) ===
    # Official paired-3p Archon conversions add additional AssetList blocks. Without them, we have
    # seen controller slots get defeated immediately on some converted 3p maps.
    if template_context and not same_base_map and paired_3p:
        _ensure_assetlist_has_archon_blocks(context, extra_blocks=4 * num_builders)  # 12 for 3p

    # === Step 8: String pool ordering ===
    # IMPORTANT: Do NOT reorder the string pool! The game expects strings to be
    # at the IDs they were originally assigned when the base map was loaded.
    # Reordering breaks map loading (v14+ bug was caused by this).
    # if template_context and not same_base_map:
    #     _reorder_string_pool_by_serialization(context)

    return context


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Transform a normal RA3 map into Archon mode.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Transform using an archon template for scripts:
  python scripts/transform_to_archon.py --in map.map --out archon_map.map --template archon_infinity.map
  
  # Transform without template (structure only, no scripts):
  python scripts/transform_to_archon.py --in map.map --out archon_map.map

Notes:
  - Only works on maps with 1-3 players (max 6 players total with controllers)
  - Controller spawn positions are automatically calculated
  - Scripts require a template archon map to copy from
"""
    )
    parser.add_argument("--in", dest="in_path", required=True, help="Input .map path")
    parser.add_argument("--out", dest="out_path", required=True, help="Output .map path")
    parser.add_argument(
        "--template",
        dest="template_path",
        help="Optional archon template .map path (for copying scripts)"
    )
    parser.add_argument(
        "--offset",
        type=float,
        default=800.0,
        help="Controller spawn offset distance from builder (default: 800 world units)"
    )
    parser.add_argument(
        "--no-compress",
        action="store_true",
        help="Write uncompressed output"
    )
    parser.add_argument(
        "--bit-perfect",
        action="store_true",
        help="Replace all objects/assets from template (for exact reproduction when maps share same base)"
    )
    parser.add_argument(
        "--no-sidecars",
        action="store_true",
        help="Do not write XML sidecar files (map.xml, map.str, overrides.xml) - the _art.tga is always copied"
    )
    parser.add_argument(
        "--wb-normalize-terrain",
        action="store_true",
        help="Apply WorldBuilder-style BlendTileData canonicalization (reorders texture slots + remaps tiles). "
             "Off by default because it can change blending vs official maps."
    )
    
    args = parser.parse_args()
    
    in_path = Path(args.in_path)
    out_path = Path(args.out_path)
    
    if not in_path.exists():
        print(f"Error: Input file not found: {in_path}")
        return 1
    
    print(f"Loading source map: {in_path}")
    source_map = Ra3Map(str(in_path))
    source_map.parse()
    source_ctx = source_map.get_context()
    
    template_ctx = None
    template_path: Optional[Path] = None
    if args.template_path:
        template_path = Path(args.template_path)
        if not template_path.exists():
            print(f"Error: Template file not found: {template_path}")
            return 1
        
        print(f"Loading template map: {template_path}")
        template_map = Ra3Map(str(template_path))
        template_map.parse()
        template_ctx = template_map.get_context()
    
    print(f"\nSource map: {source_ctx.map_width}x{source_ctx.map_height} tiles")
    
    # Check if we should just copy template bytes directly (for same-base bit-perfect)
    num_builders = len(find_player_starts(source_ctx))
    same_base = template_ctx and are_maps_same_base(source_ctx, template_ctx)
    paired_3p = template_ctx and num_builders == 3 and _is_paired_archon_3p_template(template_ctx)
    
    if args.bit_perfect and same_base and paired_3p and template_path:
        # For same-base 3p paired maps in bit-perfect mode, copy template bytes directly
        # This ensures true bit-perfect parity, avoiding any serialization differences
        print("\n=== Copying template file directly (same base, bit-perfect 3p mode) ===")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(template_path, out_path)
        _generate_art_tga(out_path, source_ctx, source_map_path=in_path)
        if not args.no_sidecars:
            _write_sidecars(out_path, source_ctx, template_path, source_map_path=in_path)
        print(f"\n=== Archon map saved to: {out_path} (copied from template) ===")
        return 0
    
    # Transform the map
    print("\n=== Transforming to Archon Mode ===")
    transform_to_archon(
        source_ctx,
        template_ctx,
        bit_perfect=args.bit_perfect,
        wb_normalize_terrain=args.wb_normalize_terrain,
    )
    
    # Save the output
    out_path.parent.mkdir(parents=True, exist_ok=True)
    source_map.save(str(out_path), compress=(not args.no_compress))
    
    # Generate minimap from source map data (or copy source's existing _art.tga)
    _generate_art_tga(out_path, source_ctx, source_map_path=in_path)
    
    # Optionally write XML sidecars (map.xml, overrides.xml)
    if not args.no_sidecars:
        _write_sidecars(out_path, source_ctx, template_path, source_map_path=in_path)
    
    print(f"\n=== Archon map saved to: {out_path} ===")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

