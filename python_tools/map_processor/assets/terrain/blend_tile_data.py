"""
BlendTileData asset
Based on BlendTileData.cs
"""
import struct
import numpy as np
from typing import BinaryIO, List, TYPE_CHECKING

from ...core.major_asset import MajorAsset
from ...utils.constants import ASSET_BlendTileData
from ...utils.binary_utils import BinaryUtils
from ..terrain.passability import Passability
from ..terrain.texture import Texture
from ..terrain.blend_info import BlendInfo

if TYPE_CHECKING:
    from ...core.ra3map_struct import MapDataContext


class BlendTileData(MajorAsset):
    """
    Blend tile data asset (textures, tiles, passability, etc.).
    Based on BlendTileData.cs
    """
    
    def __init__(self):
        super().__init__()
        self.map_width: int = 0
        self.map_height: int = 0
        self.area: int = 0
        self.tiles: np.ndarray = None  # ushort[mapWidth, mapHeight]
        self.blends: np.ndarray = None  # ushort[mapWidth, mapHeight]
        self.single_edge_blends: np.ndarray = None  # ushort[mapWidth, mapHeight]
        self.cliff_blends: np.ndarray = None  # ushort[mapWidth, mapHeight]
        self.passability: np.ndarray = None  # Passability[mapWidth, mapHeight]
        self.impassable: np.ndarray = None  # bool[mapWidth, mapHeight]
        self.passage_width: np.ndarray = None  # bool[mapWidth, mapHeight]
        self.visibility: np.ndarray = None  # bool[mapWidth, mapHeight]
        self.buildability: np.ndarray = None  # bool[mapWidth, mapHeight]
        self.tiberium_growability: np.ndarray = None  # bool[mapWidth, mapHeight]
        self.dynamic_shrubbery: np.ndarray = None  # byte[mapWidth, mapHeight]
        self.texture_cell_count: int = 0
        self.textures: List[Texture] = []
        self.blend_info: List[BlendInfo] = []
        self.blends_count: int = 0
        self.cliff_blends_count: int = 0
        self.magic1: int = 0
        self.magic2: int = 0
    
    def get_asset_name(self) -> str:
        return ASSET_BlendTileData
    
    def get_version(self) -> int:
        return 27
    
    def parse_data(self, br: BinaryIO, context: 'MapDataContext') -> None:
        """
        Parse blend tile data.
        Based on parseData in BlendTileData.cs
        """
        if context.map_width == -2147483648 or context.map_height == -2147483648:  # Int32.MinValue
            raise ValueError("illegal mapWidth or mapHeight")
        
        self.map_width = context.map_width
        self.map_height = context.map_height
        
        area = struct.unpack('<i', br.read(4))[0]
        self.area = area
        
        # Read arrays
        self.tiles = BinaryUtils.read_array_2d(br, self.map_width, self.map_height, np.uint16)
        self.blends = BinaryUtils.read_array_2d(br, self.map_width, self.map_height, np.uint16)
        self.single_edge_blends = BinaryUtils.read_array_2d(br, self.map_width, self.map_height, np.uint16)
        self.cliff_blends = BinaryUtils.read_array_2d(br, self.map_width, self.map_height, np.uint16)
        
        # Initialize passability array
        self.passability = np.zeros((self.map_width, self.map_height), dtype=np.int32)
        
        # Read boolean arrays (returns tuple of (array, raw_bytes) for bit-perfect reconstruction)
        self.impassable, self._impassable_raw = BinaryUtils.read_array_2d(br, self.map_width, self.map_height, np.bool_)
        impassable_to_players, self._impassable_to_players_raw = BinaryUtils.read_array_2d(br, self.map_width, self.map_height, np.bool_)
        self.passage_width, self._passage_width_raw = BinaryUtils.read_array_2d(br, self.map_width, self.map_height, np.bool_)
        extra_passable, self._extra_passable_raw = BinaryUtils.read_array_2d(br, self.map_width, self.map_height, np.bool_)
        self.visibility, self._visibility_raw = BinaryUtils.read_array_2d(br, self.map_width, self.map_height, np.bool_)
        self.buildability, self._buildability_raw = BinaryUtils.read_array_2d(br, self.map_width, self.map_height, np.bool_)
        impassable_to_air_units, self._impassable_to_air_units_raw = BinaryUtils.read_array_2d(br, self.map_width, self.map_height, np.bool_)
        self.tiberium_growability, self._tiberium_growability_raw = BinaryUtils.read_array_2d(br, self.map_width, self.map_height, np.bool_)
        
        # Build passability array from boolean flags
        for y in range(self.map_height):
            for x in range(self.map_width):
                if self.impassable[x, y]:
                    self.passability[x, y] = Passability.Impassable
                elif impassable_to_players[x, y]:
                    self.passability[x, y] = Passability.ImpassableToPlayers
                elif impassable_to_air_units[x, y]:
                    self.passability[x, y] = Passability.ImpassableToAirUnits
                elif extra_passable[x, y]:
                    self.passability[x, y] = Passability.ExtraPassable
                else:
                    self.passability[x, y] = Passability.Passable
        
        # Read dynamic shrubbery (byte array)
        self.dynamic_shrubbery = BinaryUtils.read_array_2d(br, self.map_width, self.map_height, np.uint8)
        
        # Read texture info
        self.texture_cell_count = struct.unpack('<i', br.read(4))[0]
        self.blends_count = struct.unpack('<i', br.read(4))[0] - 1
        self.cliff_blends_count = struct.unpack('<i', br.read(4))[0] - 1
        texture_count = struct.unpack('<i', br.read(4))[0]
        
        # Read textures
        self.textures = []
        for j in range(texture_count):
            texture = Texture()
            texture.from_stream(br, context)
            self.textures.append(texture)
        
        # Read magic values
        self.magic1 = struct.unpack('<I', br.read(4))[0]
        self.magic2 = struct.unpack('<i', br.read(4))[0]
        
        # Read blend info
        self.blend_info = []
        for i in range(self.blends_count):
            blend_info = BlendInfo()
            blend_info.from_stream(br, context)
            self.blend_info.append(blend_info)
    
    def save_data(self, bw: BinaryIO, context: 'MapDataContext') -> None:
        """
        Save blend tile data.
        Based on saveData in BlendTileData.cs
        """
        # Write area
        bw.write(struct.pack('<i', self.map_height * self.map_width))
        
        # Write arrays
        BinaryUtils.write_array_2d(bw, self.tiles, np.uint16)
        BinaryUtils.write_array_2d(bw, self.blends, np.uint16)
        BinaryUtils.write_array_2d(bw, self.single_edge_blends, np.uint16)
        
        # Write cliff_blends placeholder (2 bytes per element)
        # Based on C#: bw.Write(new byte[mapHeight * mapWidth * 2]);
        placeholder = bytes(self.map_height * self.map_width * 2)
        bw.write(placeholder)
        
        # Convert passability back to boolean arrays
        impassable = np.zeros((self.map_width, self.map_height), dtype=np.bool_)
        impassable_to_players = np.zeros((self.map_width, self.map_height), dtype=np.bool_)
        impassable_to_air_units = np.zeros((self.map_width, self.map_height), dtype=np.bool_)
        extra_passable = np.zeros((self.map_width, self.map_height), dtype=np.bool_)
        
        for y in range(self.map_height):
            for x in range(self.map_width):
                p = Passability(self.passability[x, y])
                if p == Passability.Impassable:
                    impassable[x, y] = True
                elif p == Passability.ImpassableToPlayers:
                    impassable_to_players[x, y] = True
                elif p == Passability.ImpassableToAirUnits:
                    impassable_to_air_units[x, y] = True
                elif p == Passability.ExtraPassable:
                    extra_passable[x, y] = True
        
        # Write boolean arrays (use raw bytes for bit-perfect reconstruction)
        # Note: impassable, impassable_to_players, extra_passable, impassable_to_air_units
        # are reconstructed from passability, so we need to use original raw bytes
        BinaryUtils.write_array_2d(bw, (impassable, self._impassable_raw), np.bool_)
        BinaryUtils.write_array_2d(bw, (impassable_to_players, self._impassable_to_players_raw), np.bool_)
        BinaryUtils.write_array_2d(bw, (extra_passable, self._extra_passable_raw), np.bool_)
        BinaryUtils.write_array_2d(bw, (self.passage_width, self._passage_width_raw), np.bool_)
        BinaryUtils.write_array_2d(bw, (self.visibility, self._visibility_raw), np.bool_)
        BinaryUtils.write_array_2d(bw, (self.buildability, self._buildability_raw), np.bool_)
        BinaryUtils.write_array_2d(bw, (impassable_to_air_units, self._impassable_to_air_units_raw), np.bool_)
        BinaryUtils.write_array_2d(bw, (self.tiberium_growability, self._tiberium_growability_raw), np.bool_)
        BinaryUtils.write_array_2d(bw, self.dynamic_shrubbery, np.uint8)
        
        # Write texture info
        bw.write(struct.pack('<i', self.texture_cell_count))
        bw.write(struct.pack('<i', self.blends_count + 1))
        bw.write(struct.pack('<i', self.cliff_blends_count + 1))
        bw.write(struct.pack('<i', len(self.textures)))
        
        # Write textures
        for texture in self.textures:
            texture.save_data(bw, context)
        
        # Write magic values
        bw.write(struct.pack('<I', self.magic1))
        bw.write(struct.pack('<i', 0))  # magic2 is written as 0
        
        # Write blend info
        for blend_info in self.blend_info:
            blend_info.save_data(bw, context)
    
    def get_texture(self, x: int, y: int) -> int:
        """
        Get texture index at position (x, y).
        Based on GetTexture in BlendTileData.cs
        """
        row_first = (y % 8 // 2) * 16 + (y % 2) * 2
        current = (x % 8 // 2) * 4 + (x % 2) + row_first
        return int((self.tiles[x, y] - current) // 64)
    
    def get_texture_name(self, x: int, y: int) -> str:
        """
        Get texture name at position (x, y).
        Based on GetTextureName in BlendTileData.cs
        """
        texture_index = self.get_texture(x, y)
        if 0 <= texture_index < len(self.textures):
            return self.textures[texture_index].name
        return ""

