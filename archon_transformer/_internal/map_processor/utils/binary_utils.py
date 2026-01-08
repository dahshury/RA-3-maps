"""
Binary utility functions for reading/writing RA3 map data
Based on StreamExtension.cs and IOUtility.cs
"""
import struct
import numpy as np
from io import BytesIO
from typing import BinaryIO, Tuple, Optional


class BinaryUtils:
    """
    Utility class for binary I/O operations.
    """
    
    @staticmethod
    def from_sage_float16(value: int) -> float:
        """
        Convert SageFloat16 (uint16) to float.
        Based on StreamExtension.FromSageFloat16
        """
        # SageFloat16 format: sign (1 bit) + exponent (5 bits) + mantissa (10 bits)
        sign = 1.0 if (value & 0x8000) == 0 else -1.0
        exponent = (value >> 10) & 0x1F
        mantissa = value & 0x3FF
        
        if exponent == 0:
            if mantissa == 0:
                return 0.0
            else:
                # Denormalized
                return sign * (mantissa / 1024.0) * (2.0 ** (-14))
        elif exponent == 31:
            # Infinity or NaN
            if mantissa == 0:
                return float('inf') if sign > 0 else float('-inf')
            else:
                return float('nan')
        else:
            # Normalized
            return sign * (1.0 + mantissa / 1024.0) * (2.0 ** (exponent - 15))
    
    @staticmethod
    def to_sage_float16(value: float) -> int:
        """
        Convert float to SageFloat16 (uint16).
        Based on StreamExtension.ToSageFloat16
        """
        if value == 0.0:
            return 0
        if np.isnan(value):
            return 0x7FFF  # NaN
        if np.isinf(value):
            return 0x7C00 if value > 0 else 0xFC00
        
        sign = 0 if value >= 0 else 0x8000
        value = abs(value)
        
        # Find exponent
        exponent = 0
        if value >= 1.0:
            while value >= 2.0:
                value /= 2.0
                exponent += 1
        else:
            while value < 1.0:
                value *= 2.0
                exponent -= 1
        
        exponent += 15  # Bias
        if exponent < 0:
            exponent = 0
        elif exponent > 30:
            exponent = 30
        
        mantissa = int((value - 1.0) * 1024.0) & 0x3FF
        
        return sign | (exponent << 10) | mantissa
    
    @staticmethod
    def read_7bit_encoded_int(br: BinaryIO) -> int:
        """
        Read a 7-bit encoded integer (used by C# BinaryReader.ReadString for length).
        Based on C# BinaryReader.ReadString implementation
        """
        count = 0
        shift = 0
        while True:
            byte = br.read(1)[0]
            count |= (byte & 0x7F) << shift
            shift += 7
            if (byte & 0x80) == 0:
                break
        return count
    
    @staticmethod
    def write_7bit_encoded_int(bw: BinaryIO, value: int) -> None:
        """
        Write a 7-bit encoded integer (used by C# BinaryWriter.Write(string) for length).
        Based on C# BinaryWriter.Write(string) implementation
        """
        while value >= 0x80:
            bw.write(struct.pack('B', (value & 0x7F) | 0x80))
            value >>= 7
        bw.write(struct.pack('B', value))
    
    @staticmethod
    def read_string_csharp(br: BinaryIO) -> str:
        """
        Read a string in the format used by C# BinaryReader.ReadString().
        Format: 7-bit encoded length (int) + UTF-8 bytes
        """
        length = BinaryUtils.read_7bit_encoded_int(br)
        if length == 0:
            return ""
        data = br.read(length)
        return data.decode('utf-8')
    
    @staticmethod
    def write_string_csharp(bw: BinaryIO, s: str) -> None:
        """
        Write a string in the format used by C# BinaryWriter.Write(string).
        Format: 7-bit encoded length (int) + UTF-8 bytes
        """
        encoded = s.encode('utf-8')
        BinaryUtils.write_7bit_encoded_int(bw, len(encoded))
        bw.write(encoded)
    
    @staticmethod
    def read_string_default(br: BinaryIO) -> str:
        """
        Read a default string format (used in RA3 maps).
        Format: ushort length (2 bytes, little-endian) + Default encoding bytes (latin-1)
        Based on StreamExtension.readDefaultString (uses Encoding.Default which is latin-1/Windows-1252)
        """
        length = struct.unpack('<H', br.read(2))[0]  # ushort (2 bytes)
        if length == 0:
            return ""
        data = br.read(length)
        return data.decode('latin-1')  # Encoding.Default in C# is typically latin-1/Windows-1252
    
    @staticmethod
    def write_string_default(bw: BinaryIO, s: str) -> None:
        """
        Write a default string format (used in RA3 maps).
        Format: ushort length (2 bytes, little-endian) + Default encoding bytes (latin-1)
        Based on StreamExtension.writeDefaultString
        """
        encoded = s.encode('latin-1')
        bw.write(struct.pack('<H', len(encoded)))  # ushort (2 bytes)
        bw.write(encoded)
    
    @staticmethod
    def read_string_ascii(br: BinaryIO) -> str:
        """
        Read an ASCII string with ushort length prefix.
        Format: ushort length (2 bytes, little-endian) + ASCII bytes
        Based on IOUtility.ReadString from MapCreatorCore
        """
        length = struct.unpack('<H', br.read(2))[0]
        if length == 0:
            return ""
        data = br.read(length)
        # Some real-world maps contain bytes outside strict ASCII. Use latin-1 to preserve bytes 1:1.
        return data.decode('latin-1')
    
    @staticmethod
    def write_string_ascii(bw: BinaryIO, s: str) -> None:
        """
        Write an ASCII string with ushort length prefix.
        Format: ushort length (2 bytes, little-endian) + ASCII bytes
        Based on IOUtility.WriteString from MapCreatorCore
        """
        # Preserve bytes 1:1 for the full 0-255 range (round-trips data from read_string_ascii).
        encoded = s.encode('latin-1', errors='replace')
        bw.write(struct.pack('<H', len(encoded)))
        bw.write(encoded)
    
    @staticmethod
    def read_uint24(br: BinaryIO) -> int:
        """
        Read a 24-bit unsigned integer (3 bytes, little-endian).
        Based on StreamExtension.readUInt24
        """
        bytes_data = br.read(3)
        return struct.unpack('<I', bytes_data + b'\x00')[0]
    
    @staticmethod
    def write_uint24(bw: BinaryIO, value: int) -> None:
        """
        Write a 24-bit unsigned integer (3 bytes, little-endian).
        Based on StreamExtension.writeUInt24
        """
        bw.write(struct.pack('<I', value)[:3])
    
    @staticmethod
    def read_unicode_string(br: BinaryIO) -> str:
        """
        Read a Unicode string (used in some RA3 assets).
        Format: ushort length (2 bytes) + UTF-16LE bytes
        Based on StreamExtension.readUnicodeString
        """
        length = struct.unpack('<H', br.read(2))[0]  # ushort (2 bytes)
        if length == 0:
            return ""
        data = br.read(length * 2)
        # Use 'replace' error handling to handle invalid surrogates (like C# does)
        return data.decode('utf-16-le', errors='replace')
    
    @staticmethod
    def write_unicode_string(bw: BinaryIO, s: str) -> None:
        """
        Write a Unicode string (used in some RA3 assets).
        Format: ushort length (2 bytes) + UTF-16LE bytes
        Based on StreamExtension.writeUnicodeString
        """
        encoded = s.encode('utf-16-le')
        bw.write(struct.pack('<H', len(encoded) // 2))  # ushort (2 bytes)
        bw.write(encoded)
    
    @staticmethod
    def read_vec3d(br: BinaryIO) -> Tuple[float, float, float]:
        """
        Read a Vec3D (3 floats, little-endian).
        Based on StreamExtension.readVec3D
        """
        x = struct.unpack('<f', br.read(4))[0]
        y = struct.unpack('<f', br.read(4))[0]
        z = struct.unpack('<f', br.read(4))[0]
        return (x, y, z)
    
    @staticmethod
    def write_vec3d(bw: BinaryIO, x: float, y: float, z: float) -> None:
        """
        Write a Vec3D (3 floats, little-endian).
        Based on StreamExtension.writeVec3D
        """
        bw.write(struct.pack('<f', x))
        bw.write(struct.pack('<f', y))
        bw.write(struct.pack('<f', z))
    
    @staticmethod
    def read_vec2d(br: BinaryIO) -> Tuple[float, float]:
        """
        Read a Vec2D (2 floats, little-endian).
        Based on Vec2D(BinaryReader br)
        """
        x = struct.unpack('<f', br.read(4))[0]
        y = struct.unpack('<f', br.read(4))[0]
        return (x, y)
    
    @staticmethod
    def write_vec2d(bw: BinaryIO, x: float, y: float) -> None:
        """
        Write a Vec2D (2 floats, little-endian).
        Based on Vec2D.Save(BinaryWriter bw)
        """
        bw.write(struct.pack('<f', x))
        bw.write(struct.pack('<f', y))
    
    @staticmethod
    def read_color_rgbf(br: BinaryIO) -> Tuple[float, float, float]:
        """
        Read a ColorRgbF (3 floats, little-endian).
        Based on StreamExtension.ReadColorRgbF
        """
        r = struct.unpack('<f', br.read(4))[0]
        g = struct.unpack('<f', br.read(4))[0]
        b = struct.unpack('<f', br.read(4))[0]
        return (r, g, b)
    
    @staticmethod
    def write_color_rgbf(bw: BinaryIO, r: float, g: float, b: float) -> None:
        """
        Write a ColorRgbF (3 floats, little-endian).
        Based on StreamExtension.writeColorRgbF
        """
        bw.write(struct.pack('<f', r))
        bw.write(struct.pack('<f', g))
        bw.write(struct.pack('<f', b))
    
    @staticmethod
    def read_map_color_argb(br: BinaryIO) -> Tuple[int, int, int, int]:
        """
        Read a MapColorArgb (ARGB color as uint32, little-endian).
        Based on MapColorArgb.fromStream
        """
        value = struct.unpack('<I', br.read(4))[0]
        a = (value >> 24) & 0xFF
        r = (value >> 16) & 0xFF
        g = (value >> 8) & 0xFF
        b = value & 0xFF
        return (a, r, g, b)
    
    @staticmethod
    def write_map_color_argb(bw: BinaryIO, a: int, r: int, g: int, b: int) -> None:
        """
        Write a MapColorArgb (ARGB color as uint32, little-endian).
        Based on MapColorArgb.saveData
        """
        combined = (a << 24) | (r << 16) | (g << 8) | b
        bw.write(struct.pack('<I', combined))
    
    @staticmethod
    def read_array_2d(br: BinaryIO, width: int, height: int, dtype: type) -> np.ndarray:
        """
        Read a 2D array from binary stream.
        Based on IOUtility.ReadArray
        
        Args:
            br: Binary reader
            width: Array width
            height: Array height
            dtype: numpy dtype (np.uint16, np.uint8, np.bool_, np.int32)
        
        Returns:
            2D numpy array with shape (width, height)
        """
        if width <= 0 or height <= 0:
            raise ValueError(f"Invalid array dimensions: width={width}, height={height}")

        if dtype == np.bool_:
            # Boolean arrays are bit-packed row-major along X (little-endian bit order).
            # Preserve raw bytes for bit-perfect reconstruction.
            bytes_per_row = (width + 7) // 8
            raw_bytes = br.read(bytes_per_row * height)
            if len(raw_bytes) != bytes_per_row * height:
                raise EOFError("Unexpected end of stream while reading boolean array")

            raw = np.frombuffer(raw_bytes, dtype=np.uint8).reshape((height, bytes_per_row))
            # Unpack bits per row, little-endian bit order (x%8 corresponds to bit (x%8))
            bits = np.unpackbits(raw, axis=1, bitorder='little')[:, :width]  # (height, width)
            array = bits.astype(np.bool_).T  # (width, height) with [x, y] indexing
            return (array, raw_bytes)

        if dtype == np.uint16:
            nbytes = width * height * 2
            data = br.read(nbytes)
            if len(data) != nbytes:
                raise EOFError("Unexpected end of stream while reading uint16 array")
            arr_yx = np.frombuffer(data, dtype='<u2').reshape((height, width))
            return arr_yx.T.copy()

        if dtype == np.uint8:
            nbytes = width * height
            data = br.read(nbytes)
            if len(data) != nbytes:
                raise EOFError("Unexpected end of stream while reading uint8 array")
            arr_yx = np.frombuffer(data, dtype=np.uint8).reshape((height, width))
            return arr_yx.T.copy()

        if dtype == np.int32:
            nbytes = width * height * 4
            data = br.read(nbytes)
            if len(data) != nbytes:
                raise EOFError("Unexpected end of stream while reading int32 array")
            arr_yx = np.frombuffer(data, dtype='<i4').reshape((height, width))
            return arr_yx.T.copy()

        raise ValueError(f"Unsupported dtype for ReadArray: {dtype}")
    
    @staticmethod
    def write_array_2d(bw: BinaryIO, array: np.ndarray, dtype: type) -> None:
        """
        Write a 2D array to binary stream.
        Based on WriteArray in IOUtility.cs
        
        Args:
            bw: Binary writer
            array: 2D numpy array (should be [width, height] shape) or tuple (array, raw_bytes) for bool
            dtype: numpy dtype (np.uint16, np.uint8, np.bool_, etc.)
        """
        # Handle tuple for boolean arrays (array, raw_bytes) for bit-perfect reconstruction
        if dtype == np.bool_ and isinstance(array, tuple) and len(array) == 2:
            # array is (numpy_array, raw_bytes) tuple - use raw bytes for bit-perfect reconstruction
            bw.write(array[1])
            return
        
        width, height = array.shape
        
        if dtype == np.bool_:
            # Boolean arrays are bit-packed row-major along X (little-endian bit order).
            # This matches the C# implementation and produces identical padding behavior.
            bits_yx = np.asarray(array, dtype=np.bool_).T  # (height, width)
            packed = np.packbits(bits_yx, axis=1, bitorder='little')  # (height, ceil(width/8))
            bw.write(packed.tobytes())
        elif dtype == np.uint16:
            # ushort arrays: 2 bytes per element (little-endian), stored row-major by Y then X
            arr_yx = np.asarray(array, dtype=np.uint16).T  # (height, width)
            bw.write(arr_yx.astype('<u2', copy=False).tobytes(order='C'))
        elif dtype == np.uint8:
            # byte arrays: 1 byte per element, stored row-major by Y then X
            arr_yx = np.asarray(array, dtype=np.uint8).T  # (height, width)
            bw.write(arr_yx.tobytes(order='C'))
        elif dtype == np.int32:
            # int arrays: 4 bytes per element (little-endian), stored row-major by Y then X
            arr_yx = np.asarray(array, dtype=np.int32).T  # (height, width)
            bw.write(arr_yx.astype('<i4', copy=False).tobytes(order='C'))
        else:
            raise ValueError(f"Unsupported dtype for WriteArray: {dtype}")
