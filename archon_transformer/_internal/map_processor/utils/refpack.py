"""
RefPack Decompression - Python implementation of RefPack algorithm
Based on Ra3Solution/MapCoreLib/Compress/RefpackComrpessor.cs
"""
import struct
from io import BytesIO
from typing import BinaryIO, Optional, List, Tuple


class RefPackDecompressor:
    """RefPack decompression implementation"""
    
    UNCOMPRESSED_FLAG = 1884121923
    COMPRESSED_FLAG = 5390661
    
    @staticmethod
    def get_uncompressed_size(br: BinaryIO) -> int:
        """
        Get uncompressed size from RefPack header.
        This function also advances the stream position past the RefPack header,
        leaving the stream positioned at the first RefPack command byte.

        Header variants observed in real RA3 maps:
        - Small header (5 bytes):  [flag][0xFB][size_be24]
          Example: flag=0x10 in official maps
        - Large header (6 bytes):  [flag|0x80][0xFB][size_be32]

        Note: the outer `.map` header is "EAR\\0" + uint32_le(uncompressed_size),
        and the RefPack header begins at file offset 8.
        """
        flag_b = br.read(1)
        if not flag_b:
            raise EOFError("Unexpected end of stream while reading RefPack header flag")
        flag = flag_b[0]

        # second byte is typically 0xFB; keep it for compatibility but don't enforce
        b2 = br.read(1)
        if not b2:
            raise EOFError("Unexpected end of stream while reading RefPack header byte2")

        if (flag & 0x80) != 0:
            # Large: 4-byte big-endian size
            size_be = br.read(4)
            if len(size_be) != 4:
                raise EOFError("Unexpected end of stream while reading RefPack large header size")
            return int.from_bytes(size_be, "big", signed=False)

        # Small: 3-byte big-endian size (uint24)
        size_be24 = br.read(3)
        if len(size_be24) != 3:
            raise EOFError("Unexpected end of stream while reading RefPack small header size")
        return int.from_bytes(size_be24, "big", signed=False)
    
    @staticmethod
    def decompress(input_stream: BinaryIO, output_stream: BinaryIO) -> None:
        """
        Decompress RefPack data.
        Based on Decompress method in RefpackComrpessor.cs
        
        Args:
            input_stream: Binary input stream (positioned at start of RefPack header, at position 8)
            output_stream: Binary output stream for decompressed data
        """
        # Skip RefPack header (and optionally learn output size)
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


class RefPackCompressor:
    """
    RefPack compressor for RA3 `.map` files.

    Implements the same compression scheme as:
      `Ra3Solution/MapCoreLib/Compress/Compression.cs`

    Notes:
    - This is LZ-style with back-references, producing files comparable in size
      to official maps (hundreds of KB instead of multiple MB).
    - We output the "large header" variant (flag 0x80, 0xFB, size_be32),
      which is what the C# compressor emits. The decompressor supports both.
    """

    @staticmethod
    def compress(data: bytes) -> bytes:
        if data is None:
            raise ValueError("data cannot be None")

        input_bytes = data if isinstance(data, (bytes, bytearray)) else bytes(data)
        n = len(input_bytes)
        if n < 16:
            # Too small to compress effectively; still emit a valid literal-only stream.
            return RefPackCompressor._compress_literal_only(input_bytes)

        level = _CompressionLevel.max()
        compressed = _Compression.compress(input_bytes, level)
        if compressed is None:
            # Fallback (still game-readable), but may be larger.
            return RefPackCompressor._compress_literal_only(input_bytes)
        return compressed

    @staticmethod
    def _compress_literal_only(data: bytes) -> bytes:
        """
        Fallback: valid RefPack stream using only literal blocks.
        This is not size-efficient, but it's always valid.
        """
        size = len(data)
        out = bytearray()
        out += b"EAR\x00"
        out += struct.pack("<I", size)
        # Use large header for consistency (matches C# compressor style)
        out.append(0x80)
        out.append(0xFB)
        out += size.to_bytes(4, "big", signed=False)

        i = 0
        while size - i >= 4:
            block = min(112, (size - i) & ~3)
            ctrl = 0xE0 | ((block >> 2) - 1)
            out.append(ctrl)
            out += data[i : i + block]
            i += block

        rem = size - i
        out.append(0xFC | rem)
        if rem:
            out += data[i:]
        return bytes(out)


class _CompressionLevel:
    """
    Port of C# CompressionLevel (MapCoreLib/Compress/Compression.cs).
    """

    def __init__(self, block_interval: int, search_length: int, same_val_to_track: int, brute_force_length: int):
        self.block_interval = int(block_interval)
        self.search_length = int(search_length)
        self.prequeue_length = self.search_length // self.block_interval
        self.queue_length = 131000 // self.block_interval - self.prequeue_length
        self.same_val_to_track = int(same_val_to_track)
        self.brute_force_length = int(brute_force_length)

    @staticmethod
    def max() -> "_CompressionLevel":
        return _CompressionLevel(1, 1, 10, 64)


class _Compression:
    """
    Port of C# Compress/Compression.cs.
    Returns a fully formed `.map` file bytes (EAR\\0 + size + RefPack stream),
    or None if compression is not beneficial.
    """

    @staticmethod
    def compress(input_bytes: bytes, level: _CompressionLevel) -> Optional[bytes]:
        if len(input_bytes) >= 0xFFFFFFFF:
            raise ValueError("input data is too large")

        end_is_valid = False
        compressed_chunks: List[bytes] = []
        compressed_index = 0
        compressed_length = 0

        # Tracking structures (ported)
        from collections import deque

        block_tracking_queue = deque()  # of (key, pos)
        block_pretracking_queue = deque()  # of (key, pos)
        unused_lists: List[List[int]] = []
        latest_blocks: dict[int, List[int]] = {}
        last_block_stored = 0

        b = input_bytes  # local alias
        n = len(b)

        def _read_i32_le(pos: int) -> int:
            # matches BitConverter.ToInt32 (signed)
            return int.from_bytes(b[pos : pos + 4], "little", signed=True)

        while compressed_index < n:
            while compressed_index > last_block_stored + level.block_interval and (n - compressed_index) > 16:
                if len(block_pretracking_queue) >= level.prequeue_length:
                    key, pos = block_pretracking_queue.popleft()
                    block_tracking_queue.append((key, pos))

                    value_list = latest_blocks.get(key)
                    if value_list is None:
                        value_list = unused_lists.pop() if unused_lists else []
                        value_list.clear()
                        latest_blocks[key] = value_list

                    if len(value_list) >= level.same_val_to_track:
                        # replace earliest (smallest position)
                        earliest_idx = 0
                        earliest_val = value_list[0]
                        for i in range(1, len(value_list)):
                            if value_list[i] < earliest_val:
                                earliest_idx = i
                                earliest_val = value_list[i]
                        value_list[earliest_idx] = pos
                    else:
                        value_list.append(pos)

                    if len(block_tracking_queue) > level.queue_length:
                        key2, pos2 = block_tracking_queue.popleft()
                        vl2 = latest_blocks.get(key2)
                        if vl2 is not None:
                            # remove pos2
                            for i in range(len(vl2)):
                                if vl2[i] == pos2:
                                    vl2.pop(i)
                                    break
                            if not vl2:
                                latest_blocks.pop(key2, None)
                                unused_lists.append(vl2)

                # enqueue new block
                new_key = _read_i32_le(last_block_stored)
                block_pretracking_queue.append((new_key, last_block_stored))
                last_block_stored += level.block_interval

            if (n - compressed_index) < 4:
                # terminal copy: 0xFC | remaining
                rem = n - compressed_index
                chunk = bytes([0xFC | rem]) + b[compressed_index:]
                compressed_chunks.append(chunk)
                compressed_index += rem
                compressed_length += len(chunk)
                end_is_valid = True
                continue

            # Find sequence at or ahead of current index
            seq_start = 0
            seq_length = 0
            seq_index = 0
            is_sequence = False

            found, seq_start, seq_length, seq_index = _Compression._find_sequence(
                b, compressed_index, latest_blocks, level
            )
            if found:
                is_sequence = True
            else:
                # Find next sequence 4-byte aligned ahead
                loop = compressed_index + 4
                while (not is_sequence) and (loop + 3 < n):
                    found2, s2, l2, i2 = _Compression._find_sequence(b, loop, latest_blocks, level)
                    if found2:
                        seq_start, seq_length, seq_index = s2, l2, i2 + (loop - compressed_index)
                        is_sequence = True
                        break
                    loop += 4

                if seq_index == 2**31 - 1:  # int.MaxValue
                    seq_index = n - compressed_index

                # Copy skipped data in 4-byte multiples, up to 112 bytes
                while seq_index >= 4:
                    to_copy = seq_index & ~3
                    if to_copy > 112:
                        to_copy = 112
                    chunk = bytes([0xE0 | ((to_copy >> 2) - 1)]) + b[compressed_index : compressed_index + to_copy]
                    compressed_chunks.append(chunk)
                    compressed_index += to_copy
                    compressed_length += len(chunk)
                    seq_index -= to_copy

            if is_sequence:
                # Sanity check like C# (rarely triggers)
                if _Compression._find_run_length(b, seq_start, compressed_index + seq_index) < seq_length:
                    break

                while seq_length > 0:
                    this_len = seq_length if seq_length <= 1028 else 1028
                    seq_length -= this_len

                    offset = compressed_index - seq_start + seq_index - 1

                    if this_len > 67 or offset > 16383:
                        # 110cccpp oooooooo oooooooo cccccccc
                        first = (
                            0xC0
                            | (seq_index & 0x3)
                            | (((this_len - 5) >> 6) & 0x0C)
                            | ((offset >> 12) & 0x10)
                        )
                        chunk = bytearray(seq_index + 4)
                        chunk[0] = first & 0xFF
                        chunk[1] = (offset >> 8) & 0xFF
                        chunk[2] = offset & 0xFF
                        chunk[3] = (this_len - 5) & 0xFF
                    elif this_len > 10 or offset > 1023:
                        # 10cccccc ppoooooo oooooooo
                        chunk = bytearray(seq_index + 3)
                        chunk[0] = 0x80 | ((this_len - 4) & 0x3F)
                        chunk[1] = (((seq_index << 6) & 0xC0) | ((offset >> 8) & 0x3F)) & 0xFF
                        chunk[2] = offset & 0xFF
                    else:
                        # 0oocccpp oooooooo
                        chunk = bytearray(seq_index + 2)
                        chunk[0] = (
                            (seq_index & 0x3)
                            | (((this_len - 3) << 2) & 0x1C)
                            | ((offset >> 3) & 0x60)
                        ) & 0xFF
                        chunk[1] = offset & 0xFF

                    # Copy literal bytes (read 0-3) at end of chunk
                    if seq_index > 0:
                        start = compressed_index
                        chunk[-seq_index:] = b[start : start + seq_index]

                    compressed_chunks.append(bytes(chunk))
                    compressed_index += this_len + seq_index
                    compressed_length += len(chunk)
                    seq_start += this_len
                    seq_index = 0

        # Only accept if beneficial (matches C# check)
        if compressed_length + 6 >= n:
            return None

        # Build full `.map` output: "EAR\0" + size_le + RefPack header + chunks
        # Use small header (5 bytes) when size fits in 24 bits, otherwise large (6 bytes)
        if n <= 0xFFFFFF:
            # Small header: flag (0x10) + 0xFB + 3-byte size (big-endian)
            header_size = 13  # 4 (magic) + 4 (size) + 5 (refpack header)
            out = bytearray(compressed_length + header_size + (0 if end_is_valid else 1))
            out[0:4] = b"EAR\x00"
            out[4:8] = struct.pack("<I", n)
            out[8] = 0x10
            out[9] = 0xFB
            out[10] = (n >> 16) & 0xFF
            out[11] = (n >> 8) & 0xFF
            out[12] = n & 0xFF
            pos = 13
        else:
            # Large header: flag (0x80) + 0xFB + 4-byte size (big-endian)
            header_size = 14
            out = bytearray(compressed_length + header_size + (0 if end_is_valid else 1))
            out[0:4] = b"EAR\x00"
            out[4:8] = struct.pack("<I", n)
            out[8] = 0x80
            out[9] = 0xFB
            out[10] = (n >> 24) & 0xFF
            out[11] = (n >> 16) & 0xFF
            out[12] = (n >> 8) & 0xFF
            out[13] = n & 0xFF
            pos = 14
        for ch in compressed_chunks:
            out[pos : pos + len(ch)] = ch
            pos += len(ch)

        if not end_is_valid:
            out[-1] = 0xFC
        return bytes(out)

    @staticmethod
    def _find_sequence(
        data: bytes, offset: int, block_tracking: dict[int, List[int]], level: _CompressionLevel
    ) -> Tuple[bool, int, int, int]:
        start = -3 if offset > 4 else offset - 3
        end = -level.brute_force_length if offset >= level.brute_force_length else -offset

        best_start = 0
        best_length = 3
        best_index = 2**31 - 1  # int.MaxValue
        found_run = False

        search_len = 4 if (len(data) - offset) > 4 else (len(data) - offset)
        search = data[offset : offset + search_len]

        while start >= end and best_length < 1028:
            current_byte = data[start + offset]
            for loop in range(search_len):
                if current_byte != search[loop] or start >= loop or (start - loop) < -131072:
                    continue
                src = offset + start
                dst = offset + loop
                l = _Compression._find_run_length(data, src, dst)
                if (
                    (l > best_length or (l == best_length and loop < best_index))
                    and (
                        l >= 5
                        or (l >= 4 and (start - loop) > -16384)
                        or (l >= 3 and (start - loop) > -1024)
                    )
                ):
                    found_run = True
                    best_start = src
                    best_length = l
                    best_index = loop
            start -= 1

        if block_tracking and (len(data) - offset) > 16 and best_length < 1028:
            for loop in range(4):
                this_position = offset + 3 - loop
                adjust = (loop - 3) if loop > 3 else 0
                # BitConverter.ToInt32(data, thisPosition) signed
                val = int.from_bytes(data[this_position : this_position + 4], "little", signed=True)
                positions = block_tracking.get(val)
                if not positions:
                    continue
                for trypos in positions:
                    local_adjust = adjust
                    if trypos + 131072 < offset + 8:
                        continue
                    length = _Compression._find_run_length(data, trypos + local_adjust, this_position + local_adjust)
                    if length >= 5 and length > best_length:
                        found_run = True
                        best_start = trypos + local_adjust
                        best_length = length
                        best_index = 3 - loop if loop < 3 else 0
                    if best_length > 1028:
                        break
                if best_length > 1028:
                    break

        return found_run, best_start, best_length, best_index

    @staticmethod
    def _find_run_length(data: bytes, source: int, destination: int) -> int:
        end_source = source + 1
        end_destination = destination + 1
        n = len(data)
        # assumes first byte already matches; includes it in the count
        while end_destination < n and data[end_source] == data[end_destination] and (end_destination - destination) < 1028:
            end_source += 1
            end_destination += 1
        return end_destination - destination
