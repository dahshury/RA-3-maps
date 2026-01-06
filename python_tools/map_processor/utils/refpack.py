"""
RefPack Decompression - Python implementation of RefPack algorithm
Based on Ra3Solution/MapCoreLib/Compress/RefpackComrpessor.cs
"""
import struct
from io import BytesIO
from typing import BinaryIO


class RefPackDecompressor:
    """RefPack decompression implementation"""
    
    UNCOMPRESSED_FLAG = 1884121923
    COMPRESSED_FLAG = 5390661
    
    @staticmethod
    def get_uncompressed_size(br: BinaryIO) -> int:
        """
        Get uncompressed size from RefPack header.
        Based on GetUncompressedSize in RefpackComrpessor.cs
        
        Note: This reads from the current position, not from the start.
        The input stream should be positioned at the start of the RefPack header.
        """
        current_pos = br.tell()
        flag = br.read(1)[0]
        
        if (flag & 0x80) != 0:
            # Large file
            br.read(1)
            location = br.tell()
            br.seek(current_pos + 4)  # Seek to position 4 from start of header
            uncompressed_size = struct.unpack('<I', br.read(4))[0]
            br.seek(location)
            br.read(4)  # Skip 4 bytes
        else:
            # Small file
            br.read(1)
            location = br.tell()
            br.seek(current_pos + 4)  # Seek to position 4 from start of header
            uncompressed_size = struct.unpack('<I', br.read(4))[0]
            br.seek(location)
            br.read(3)  # Skip 3 bytes
        
        return uncompressed_size
    
    @staticmethod
    def decompress(input_stream: BinaryIO, output_stream: BinaryIO) -> None:
        """
        Decompress RefPack data.
        Based on Decompress method in RefpackComrpessor.cs
        
        Args:
            input_stream: Binary input stream (positioned at start of RefPack header, at position 8)
            output_stream: Binary output stream for decompressed data
        """
        # CRITICAL: GetUncompressedSize reads the RefPack header and advances the input stream
        # We MUST call it to skip the header, even though we don't use the return value
        RefPackDecompressor.get_uncompressed_size(input_stream)
        
        output_buffer = BytesIO()
        
        while True:
            code_bytes = input_stream.read(1)
            if len(code_bytes) == 0:
                break
            code = code_bytes[0]
            
            if (code & 0x80) == 0:
                # Format: 0xxxxxxx xxxxxxxx
                code2 = input_stream.read(1)[0]
                count = code & 3
                
                # Copy literal bytes
                literal_data = input_stream.read(count)
                output_buffer.write(literal_data)
                old_pos = output_buffer.tell()
                
                # Calculate back reference
                offset = code2 + (code & 0x60) * 8
                new_pos = old_pos - 1 - offset
                
                if new_pos < 0:
                    raise ValueError(f"Invalid back reference: new_pos={new_pos}, old_pos={old_pos}")
                
                repeat_available = old_pos - new_pos
                count = (code & 0x1C) // 4 + 3
                
                # Copy from back reference using SeekOrigin.Current (seek backwards)
                output_buffer.seek(-repeat_available, 1)  # Seek backwards from current position
                temp = output_buffer.read(count)
                output_buffer.seek(0, 2)  # Seek to end
                output_buffer.write(temp)
                
                if count > repeat_available:
                    RefPackDecompressor._copy_repeat(
                        output_buffer, old_pos, new_pos, count, repeat_available
                    )
                    
            elif (code & 0x40) == 0:
                # Format: 10xxxxxx xxxxxxxx xxxxxxxx
                code2 = input_stream.read(1)[0]
                code3 = input_stream.read(1)[0]
                count = code2 >> 6
                
                # Copy literal bytes
                literal_data = input_stream.read(count)
                output_buffer.write(literal_data)
                old_pos = output_buffer.tell()
                
                # Calculate back reference
                offset = ((code2 & 0x3F) << 8) + code3
                new_pos = old_pos - 1 - offset
                
                if new_pos < 0:
                    raise ValueError(f"Invalid back reference: new_pos={new_pos}, old_pos={old_pos}")
                
                repeat_available = old_pos - new_pos
                count = (code & 0x3F) + 4
                
                # Copy from back reference using SeekOrigin.Current (seek backwards)
                output_buffer.seek(-repeat_available, 1)  # Seek backwards from current position
                temp = output_buffer.read(count)
                output_buffer.seek(0, 2)  # Seek to end
                output_buffer.write(temp)
                
                if count > repeat_available:
                    RefPackDecompressor._copy_repeat(
                        output_buffer, old_pos, new_pos, count, repeat_available
                    )
                    
            elif (code & 0x20) == 0:
                # Format: 110xxxxx xxxxxxxx xxxxxxxx xxxxxxxx
                code2 = input_stream.read(1)[0]
                code3 = input_stream.read(1)[0]
                code4 = input_stream.read(1)[0]
                count = code & 3
                
                # Copy literal bytes
                literal_data = input_stream.read(count)
                output_buffer.write(literal_data)
                old_pos = output_buffer.tell()
                
                # Calculate back reference
                offset = (((code & 0x10) >> 4) << 16) + (code2 << 8) + code3
                new_pos = old_pos - 1 - offset
                
                if new_pos < 0:
                    raise ValueError(f"Invalid back reference: new_pos={new_pos}, old_pos={old_pos}")
                
                repeat_available = old_pos - new_pos
                count = (((code & 0xC) >> 2) << 8) + code4 + 5
                
                # Copy from back reference using SeekOrigin.Current (seek backwards)
                output_buffer.seek(-repeat_available, 1)  # Seek backwards from current position
                temp = output_buffer.read(count)
                output_buffer.seek(0, 2)  # Seek to end
                output_buffer.write(temp)
                
                if count > repeat_available:
                    RefPackDecompressor._copy_repeat(
                        output_buffer, old_pos, new_pos, count, repeat_available
                    )
            else:
                # Format: 111xxxxx - literal data
                count = (code & 0x1F) * 4 + 4
                if count > 112:
                    break
                output_buffer.write(input_stream.read(count))
        
        # Final literal bytes
        count = code & 3
        output_buffer.write(input_stream.read(count))
        
        # Write to output stream
        output_buffer.seek(0)
        output_stream.write(output_buffer.read())
        output_buffer.close()
    
    @staticmethod
    def _copy_repeat(output_buffer: BytesIO, old_pos: int, new_pos: int, 
                     count: int, repeat_available: int) -> None:
        """
        Helper method to copy repeating data when count > repeat_available.
        Based on CopyRepeat in RefpackComrpessor.cs
        """
        copy_from_end = count - repeat_available
        i = 0
        while i < copy_from_end:
            output_buffer.seek(old_pos)
            to_copy = min(copy_from_end - i, repeat_available)
            temp = output_buffer.read(to_copy)
            old_pos += len(temp)
            i += len(temp)
            output_buffer.seek(0, 2)  # Seek to end
            output_buffer.write(temp)
