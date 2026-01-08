"""
Parser Configuration - Controls what assets to include/exclude during parsing and saving
"""
from typing import Set, Optional, List
from dataclasses import dataclass, field


@dataclass
class ParserConfig:
    """
    Configuration for map parsing and reconstruction.
    Controls which assets are included/excluded.
    """
    # Asset names to include (if None, includes all except excluded)
    included_assets: Optional[Set[str]] = None
    
    # Asset names to exclude (if None, excludes nothing)
    excluded_assets: Optional[Set[str]] = None
    
    # Object categories to include (if None, includes all except excluded)
    included_object_categories: Optional[Set[str]] = None
    
    # Object categories to exclude (if None, excludes nothing)
    excluded_object_categories: Optional[Set[str]] = None
    
    def __post_init__(self):
        """Validate configuration"""
        if self.included_assets is not None and self.excluded_assets is not None:
            # Check for conflicts
            conflict = self.included_assets & self.excluded_assets
            if conflict:
                raise ValueError(f"Assets cannot be both included and excluded: {conflict}")
        
        if self.included_object_categories is not None and self.excluded_object_categories is not None:
            conflict = self.included_object_categories & self.excluded_object_categories
            if conflict:
                raise ValueError(f"Object categories cannot be both included and excluded: {conflict}")
    
    def is_asset_included(self, asset_name: str) -> bool:
        """
        Check if an asset should be included.
        
        Args:
            asset_name: Name of the asset
            
        Returns:
            True if asset should be included, False otherwise
        """
        # If excluded list exists and asset is in it, exclude
        if self.excluded_assets is not None:
            if asset_name in self.excluded_assets:
                return False
        
        # If included list exists, only include if in list
        if self.included_assets is not None:
            return asset_name in self.included_assets
        
        # Default: include everything (unless excluded)
        return True
    
    def should_parse_asset(self, asset_name: str) -> bool:
        """
        Check if an asset should be parsed (read from file).
        Same as is_asset_included for now, but could be different in future.
        """
        return self.is_asset_included(asset_name)
    
    def should_save_asset(self, asset_name: str) -> bool:
        """
        Check if an asset should be saved (written to file).
        Same as is_asset_included for now, but could be different in future.
        """
        return self.is_asset_included(asset_name)
    
    @classmethod
    def default(cls) -> 'ParserConfig':
        """Create default config (includes everything)"""
        return cls()
    
    @classmethod
    def training_config(cls) -> 'ParserConfig':
        """
        Create config for AI training (based on AI_TRAINING_DATA_CATALOG.md).
        Includes only assets needed for training.
        
        Included Assets (18):
        - HeightMapData, BlendTileData, ObjectsList, WorldInfo
        - StandingWaterAreas, RiverAreas, StandingWaveAreas
        - SidesList, MPPositionList, Teams, BuildLists
        - GlobalWaterSettings, FogSettings, NamedCameras
        - GlobalLighting, PostEffectsChunk, EnvironmentData, AssetList
        
        Excluded Assets (6):
        - PlayerScriptsList, TriggerAreas, MissionHotSpots
        - MissionObjectives, LibraryMaps, LibraryMapLists
        
        Included Object Categories (5):
        - ore_node, oil_derrick, garrison, building, player_start
        """
        from ..utils.constants import (
            ASSET_HeightMapData, ASSET_BlendTileData, ASSET_ObjectsList,
            ASSET_WorldInfo, ASSET_StandingWaterAreas, ASSET_RiverAreas,
            ASSET_StandingWaveAreas, ASSET_SidesList, ASSET_MPPositionList,
            ASSET_Teams, ASSET_BuildLists, ASSET_GlobalWaterSettings,
            ASSET_FogSettings, ASSET_NamedCameras, ASSET_GlobalLighting,
            ASSET_PostEffectsChunk, ASSET_EnvironmentData, ASSET_AssetList,
            # Excluded assets
            ASSET_PlayerScriptsList, ASSET_TriggerAreas, ASSET_MissionHotSpots,
            ASSET_MissionObjectives, ASSET_LibraryMaps, ASSET_LibraryMapLists
        )
        
        included = {
            ASSET_HeightMapData, ASSET_BlendTileData, ASSET_ObjectsList,
            ASSET_WorldInfo, ASSET_StandingWaterAreas, ASSET_RiverAreas,
            ASSET_StandingWaveAreas, ASSET_SidesList, ASSET_MPPositionList,
            ASSET_Teams, ASSET_BuildLists, ASSET_GlobalWaterSettings,
            ASSET_FogSettings, ASSET_NamedCameras, ASSET_GlobalLighting,
            ASSET_PostEffectsChunk, ASSET_EnvironmentData, ASSET_AssetList
        }
        
        excluded = {
            ASSET_PlayerScriptsList, ASSET_TriggerAreas, ASSET_MissionHotSpots,
            ASSET_MissionObjectives, ASSET_LibraryMaps, ASSET_LibraryMapLists
        }
        
        # Object categories for training (canonical top-level categories).
        # NOTE: The ObjectsList parser expands these to include sub-categories
        # like `garrison_house` and `building_port_structure`.
        included_categories = {
            'ore_node',
            'oil_derrick',
            'garrison',
            'building',
            'player_start',
        }
        
        return cls(
            included_assets=included,
            excluded_assets=excluded,
            included_object_categories=included_categories
        )

