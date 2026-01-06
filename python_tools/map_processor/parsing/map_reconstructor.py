"""
RA3 Map Reconstructor - Save maps back to file format
Based on Ra3Map.cs save methods
"""
import struct
from io import BytesIO
from typing import BinaryIO

from ..core.ra3map_struct import Ra3MapStruct, MapDataContext
from ..utils.constants import UNCOMPRESSED_FLAG
from ..utils.refpack import RefPackDecompressor


class Ra3MapReconstructor:
    """
    Reconstruct RA3 maps from MapDataContext.
    Based on Ra3Map.save and doSaveMap methods.
    """
    
    def save_map(self, context: MapDataContext, output_path: str, compress: bool = True) -> None:
        """
        Save map to file.
        Based on doSaveMap in Ra3Map.cs
        
        Args:
            context: MapDataContext with map data
            output_path: Output file path
            compress: Whether to compress the output (default: True)
        """
        # Write to memory buffer first
        memory_buffer = BytesIO()
        
        # Write uncompressed flag
        memory_buffer.write(struct.pack('<I', UNCOMPRESSED_FLAG))
        
        # Write map structure
        self._save_map_struct(memory_buffer, context)
        
        # Get uncompressed data
        memory_buffer.seek(0)
        uncompressed_data = memory_buffer.read()
        memory_buffer.close()
        
        if compress:
            # For now, save uncompressed (RefPack compression not yet implemented)
            # TODO: Implement RefPack compression if needed
            # Note: Most maps work fine uncompressed
            compressed_data = uncompressed_data
        else:
            compressed_data = uncompressed_data
        
        # Write to file
        with open(output_path, 'wb') as f:
            f.write(compressed_data)
    
    def _save_map_struct(self, bw: BinaryIO, context: MapDataContext) -> None:
        """
        Save map structure to binary stream.
        Based on Ra3MapStruct.save method.
        """
        from ..utils.binary_utils import BinaryUtils
        
        map_struct = context.map_struct
        
        # Write string pool
        # Sort by index (descending) as C# code does
        sorted_strings = sorted(map_struct.string_pool.items(), key=lambda x: x[1], reverse=True)
        bw.write(struct.pack('<i', len(sorted_strings)))
        
        for string_value, index in sorted_strings:
            # C# uses BinaryWriter.Write(string) which uses 7-bit encoded format
            BinaryUtils.write_string_csharp(bw, string_value)
            bw.write(struct.pack('<i', index))
        
        # Write assets
        for asset in map_struct.assets:
            asset.save(bw, context)

