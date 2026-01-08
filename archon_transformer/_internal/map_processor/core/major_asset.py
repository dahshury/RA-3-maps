"""
Major Asset base class
Based on MajorAsset.cs
"""
from abc import ABC, abstractmethod
from typing import BinaryIO, TYPE_CHECKING

if TYPE_CHECKING:
    from .ra3map_struct import MapDataContext


class MajorAsset(ABC):
    """
    Base class for all major assets in RA3 maps.
    Based on MajorAsset.cs
    """
    
    def __init__(self):
        self.id: int = 0
        self.version: int = 0
        self.name: str = ""
        self.data_size: int = 0
        self.data_start_pos: int = 0  # Position where data starts (after header)
    
    @abstractmethod
    def get_asset_name(self) -> str:
        """Return the asset name (e.g., "HeightMapData")"""
        pass
    
    def get_version(self) -> int:
        """Return the asset version"""
        return 0
    
    def register_self(self, context: 'MapDataContext') -> None:
        """Register this asset in the context"""
        self.name = self.get_asset_name()
        self.id = context.map_struct.register_string(self.name)
        self.version = self.get_version()
    
    def from_stream(self, br: BinaryIO, context: 'MapDataContext') -> 'MajorAsset':
        """
        Parse asset from binary stream.
        Based on fromStream in MajorAsset.cs
        """
        import struct
        
        # Read header
        self.id = struct.unpack('<i', br.read(4))[0]
        self.version = struct.unpack('<h', br.read(2))[0]
        self.data_size = struct.unpack('<i', br.read(4))[0]
        self.data_start_pos = br.tell()  # Store as instance variable (like C# dataStartPos field)
        
        # Get asset name from string pool
        self.name = context.map_struct.find_string_by_index(self.id) or ""
        
        # Parse asset-specific data
        self.parse_data(br, context)
        
        # Verify we read the correct amount (optional check)
        bytes_read = br.tell() - self.data_start_pos
        if bytes_read != self.data_size:
            # Some assets may have variable sizes, so this is a warning, not error
            pass
        
        return self
    
    def parse_data(self, br: BinaryIO, context: 'MapDataContext') -> None:
        """
        Parse asset-specific data. Override in subclasses.
        Based on parseData in MajorAsset.cs
        """
        pass
    
    def save(self, bw: BinaryIO, context: 'MapDataContext') -> None:
        """
        Save asset to binary stream.
        Based on Save in MajorAsset.cs
        """
        import struct
        
        # Write header
        bw.write(struct.pack('<i', self.id))
        bw.write(struct.pack('<h', self.version))
        
        # Write placeholder for data size
        size_pos = bw.tell()
        bw.write(struct.pack('<i', 0))  # Will be overwritten
        
        # Write asset-specific data
        self.save_data(bw, context)
        
        # Update data size
        end_pos = bw.tell()
        data_size = end_pos - size_pos - 4
        bw.seek(size_pos)
        bw.write(struct.pack('<i', data_size))
        bw.seek(end_pos)
    
    def save_data(self, bw: BinaryIO, context: 'MapDataContext') -> None:
        """
        Save asset-specific data. Override in subclasses.
        Based on saveData in MajorAsset.cs
        """
        pass

