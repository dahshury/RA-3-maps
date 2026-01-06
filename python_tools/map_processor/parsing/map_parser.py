"""
RA3 Map Parser - Main parsing implementation
Based on Ra3MapParseImpl.cs
"""
import struct
from io import BytesIO
from typing import BinaryIO, Optional

from ..core.ra3map_struct import Ra3MapStruct, MapDataContext
from ..core.major_asset import MajorAsset
from ..core.default_major_asset import DefaultMajorAsset
from ..assets.terrain.height_map_data import HeightMapData
from ..utils.constants import (
    UNCOMPRESSED_FLAG, COMPRESSED_FLAG,
    ASSET_HeightMapData, ASSET_ObjectsList, ASSET_AssetList,
    ASSET_SidesList, ASSET_PlayerScriptsList, ASSET_BlendTileData,
    ASSET_WorldInfo, ASSET_Teams, ASSET_MPPositionList,
    ASSET_StandingWaterAreas, ASSET_GlobalLighting,
    ASSET_PostEffectsChunk, ASSET_NamedCameras,
    ASSET_LibraryMaps, ASSET_LibraryMapLists,
    ASSET_GlobalWaterSettings, ASSET_FogSettings,
    ASSET_MissionHotSpots, ASSET_RiverAreas,
    ASSET_StandingWaveAreas, ASSET_EnvironmentData,
    ASSET_BuildLists, ASSET_TriggerAreas, ASSET_MissionObjectives
)
from ..utils.refpack import RefPackDecompressor


class Ra3MapParser:
    """
    Parser for RA3 map files.
    Based on Ra3MapParseImpl.cs
    """
    
    def parse(self, map_path: str) -> MapDataContext:
        """
        Parse a RA3 map file.
        Based on parse method in Ra3MapParseImpl.cs
        
        Args:
            map_path: Path to the .map file
            
        Returns:
            MapDataContext with parsed map data
        """
        with open(map_path, 'rb') as fs:
            # Read file format flag
            flag_bytes = fs.read(4)
            if len(flag_bytes) < 4:
                raise ValueError("Map file too short")
            
            flag = struct.unpack('<I', flag_bytes)[0]
            
            if flag == UNCOMPRESSED_FLAG:
                # Map is uncompressed - reset to start (after flag)
                fs.seek(4)
                br = fs
            elif flag == COMPRESSED_FLAG:
                # Map is compressed - decompress it
                # C# code: br.BaseStream.Position = 8L; (after 4-byte flag + 4-byte size)
                fs.seek(8)
                
                decompressed = BytesIO()
                RefPackDecompressor.decompress(fs, decompressed)
                
                # C# code: br2.BaseStream.Position = 4L; (skip uncompressed flag in output)
                decompressed.seek(4)
                br = decompressed
            else:
                raise ValueError(f"Unknown map format flag: {flag}")
            
            return self._do_parse_map(br)
    
    def _do_parse_map(self, br: BinaryIO) -> MapDataContext:
        """
        Parse map structure from binary stream.
        Based on doParseMap in Ra3MapParseImpl.cs
        """
        map_struct = Ra3MapStruct()
        context = MapDataContext(map_struct)
        
        # Parse string pool
        self._parse_string_pool(br, map_struct)
        
        # Parse assets
        while br.tell() < self._get_stream_size(br):
            # Peek at next asset ID
            asset_id_pos = br.tell()
            asset_id = struct.unpack('<i', br.read(4))[0]
            br.seek(asset_id_pos)  # Go back
            
            # Get asset name from string pool
            asset_name = map_struct.find_string_by_index(asset_id)
            
            # Parse asset based on type
            asset = self._parse_asset(br, asset_name, context)
            if asset:
                map_struct.add_asset(asset)
            else:
                # Unknown asset - use default
                default_asset = DefaultMajorAsset(asset_name or f"Asset_{asset_id}")
                default_asset.from_stream(br, context)
                map_struct.add_asset(default_asset)
        
        return context
    
    def _parse_string_pool(self, br: BinaryIO, map_struct: Ra3MapStruct) -> None:
        """
        Parse string pool.
        Based on parseStringPool in Ra3MapParseImpl.cs
        C# code uses br.ReadString() which uses 7-bit encoded length, not readDefaultString()
        """
        from ..utils.binary_utils import BinaryUtils
        
        string_pool_size = struct.unpack('<i', br.read(4))[0]
        for i in range(string_pool_size):
            string_value = BinaryUtils.read_string_csharp(br)  # Use C# ReadString format
            string_index = struct.unpack('<i', br.read(4))[0]
            map_struct.register_string(string_value, string_index)
    
    def _parse_asset(self, br: BinaryIO, asset_name: Optional[str], 
                     context: MapDataContext) -> Optional[MajorAsset]:
        """
        Parse a specific asset type.
        Based on switch statement in doParseMap.
        """
        if asset_name == ASSET_HeightMapData:
            asset = HeightMapData()
            asset.from_stream(br, context)
            return asset
        elif asset_name == ASSET_ObjectsList:
            from ..assets.objects.objects_list import ObjectsList
            asset = ObjectsList()
            asset.from_stream(br, context)
            return asset
        elif asset_name == ASSET_AssetList:
            from ..assets.assets.asset_list import AssetList
            asset = AssetList()
            asset.from_stream(br, context)
            return asset
        elif asset_name == ASSET_SidesList:
            from ..assets.sides.sides_list import SidesList
            asset = SidesList()
            asset.from_stream(br, context)
            return asset
        elif asset_name == ASSET_PlayerScriptsList:
            from ..assets.scripts.player_scripts_list import PlayerScriptsList
            asset = PlayerScriptsList()
            asset.from_stream(br, context)
            return asset
        elif asset_name == ASSET_BlendTileData:
            from ..assets.terrain.blend_tile_data import BlendTileData
            asset = BlendTileData()
            asset.from_stream(br, context)
            return asset
        elif asset_name == ASSET_WorldInfo:
            from ..assets.world.world_info import WorldInfo
            asset = WorldInfo()
            asset.from_stream(br, context)
            return asset
        elif asset_name == ASSET_Teams:
            from ..assets.teams.teams import Teams
            asset = Teams()
            asset.from_stream(br, context)
            return asset
        elif asset_name == ASSET_MPPositionList:
            from ..assets.multiplayer.mp_position_list import MPPositionList
            asset = MPPositionList()
            asset.from_stream(br, context)
            return asset
        elif asset_name == ASSET_StandingWaterAreas:
            from ..assets.water.standing_water_areas import StandingWaterAreas
            asset = StandingWaterAreas()
            asset.from_stream(br, context)
            return asset
        elif asset_name == ASSET_GlobalLighting:
            from ..assets.world.global_lighting import GlobalLighting
            asset = GlobalLighting()
            asset.from_stream(br, context)
            return asset
        elif asset_name == ASSET_PostEffectsChunk:
            from ..assets.effects.post_effects_chunk import PostEffectsChunk
            asset = PostEffectsChunk()
            asset.from_stream(br, context)
            return asset
        elif asset_name == ASSET_NamedCameras:
            # NamedCameras has empty parseData/saveData in C#, but file contains data
            # Use DefaultMajorAsset to preserve raw bytes for bit-perfect reconstruction
            from ..core.default_major_asset import DefaultMajorAsset
            asset = DefaultMajorAsset(asset_name)
            asset.from_stream(br, context)
            return asset
        elif asset_name == ASSET_LibraryMaps:
            from ..assets.library.library_maps import LibraryMaps
            asset = LibraryMaps()
            asset.from_stream(br, context)
            return asset
        elif asset_name == ASSET_LibraryMapLists:
            from ..assets.library.library_map_lists import LibraryMapLists
            asset = LibraryMapLists()
            asset.from_stream(br, context)
            return asset
        elif asset_name == ASSET_GlobalWaterSettings:
            from ..assets.world.global_water_settings import GlobalWaterSettings
            asset = GlobalWaterSettings()
            asset.from_stream(br, context)
            return asset
        elif asset_name == ASSET_FogSettings:
            from ..assets.world.fog_settings import FogSettings
            asset = FogSettings()
            asset.from_stream(br, context)
            return asset
        elif asset_name == ASSET_EnvironmentData:
            from ..assets.world.environment_data import EnvironmentData
            asset = EnvironmentData()
            asset.from_stream(br, context)
            return asset
        elif asset_name == ASSET_MissionHotSpots:
            from ..assets.mission.mission_hot_spots import MissionHotSpots
            asset = MissionHotSpots()
            asset.from_stream(br, context)
            return asset
        elif asset_name == ASSET_RiverAreas:
            from ..assets.water.river_areas import RiverAreas
            asset = RiverAreas()
            asset.from_stream(br, context)
            return asset
        elif asset_name == ASSET_StandingWaveAreas:
            from ..assets.water.standing_wave_areas import StandingWaveAreas
            asset = StandingWaveAreas()
            asset.from_stream(br, context)
            return asset
        elif asset_name == ASSET_BuildLists:
            from ..assets.build.build_lists import BuildLists
            asset = BuildLists()
            asset.from_stream(br, context)
            return asset
        elif asset_name == ASSET_TriggerAreas:
            from ..assets.triggers.trigger_areas import TriggerAreas
            asset = TriggerAreas()
            asset.from_stream(br, context)
            return asset
        elif asset_name == ASSET_MissionObjectives:
            # MissionObjectives parsing is commented out in C# (just skips data)
            # Use DefaultMajorAsset to preserve raw bytes for bit-perfect reconstruction
            from ..core.default_major_asset import DefaultMajorAsset
            asset = DefaultMajorAsset(asset_name)
            asset.from_stream(br, context)
            return asset
        
        return None  # Unknown asset type
    
    def _get_stream_size(self, br: BinaryIO) -> int:
        """Get stream size (handles BytesIO and file objects)"""
        if hasattr(br, 'getbuffer'):
            # BytesIO
            return len(br.getbuffer())
        else:
            # File object
            current_pos = br.tell()
            br.seek(0, 2)  # Seek to end
            size = br.tell()
            br.seek(current_pos)
            return size

