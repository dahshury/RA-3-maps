"""
Test for EXACT bit-perfect reconstruction of map_mp_2_rao1.map
This test verifies that we can parse and reconstruct the map with ZERO data loss.
"""
import pytest
import os
from pathlib import Path
import tempfile
from io import BytesIO
import struct

from map_processor.refpack import RefPackDecompressor
from map_processor.ra3map import Ra3Map
from map_processor.constants import UNCOMPRESSED_FLAG, COMPRESSED_FLAG


def get_uncompressed_bytes(map_path: str) -> bytes:
    """Get uncompressed map data for comparison"""
    with open(map_path, 'rb') as f:
        flag = struct.unpack('<I', f.read(4))[0]
        
        if flag == UNCOMPRESSED_FLAG:
            f.seek(0)
            return f.read()
        elif flag == COMPRESSED_FLAG:
            f.seek(8)  # Skip flag + size
            decompressed = BytesIO()
            RefPackDecompressor.decompress(f, decompressed)
            decompressed.seek(0)
            return decompressed.read()
        else:
            raise ValueError(f"Unknown flag: {flag}")


def test_exact_reconstruction_map_mp_2_rao1():
    """
    Test EXACT bit-perfect reconstruction of map_mp_2_rao1.map
    
    This test:
    1. Parses map_mp_2_rao1.map
    2. Reconstructs it to a new file
    3. Compares byte-by-byte to verify EXACT match
    """
    # Find the map file
    maps_dir = Path(__file__).parent.parent.parent / "RA3 Official maps"
    map_file = maps_dir / "2 II" / "map_mp_2_rao1.map"
    
    if not map_file.exists():
        pytest.skip(f"Map file not found: {map_file}")
    
    print(f"\n{'='*70}")
    print(f"TESTING EXACT RECONSTRUCTION: map_mp_2_rao1.map")
    print(f"{'='*70}")
    print(f"Map path: {map_file}")
    
    # Get original uncompressed data
    original_data = get_uncompressed_bytes(str(map_file))
    original_size = len(original_data)
    print(f"\nOriginal uncompressed size: {original_size:,} bytes")
    
    # Parse and reconstruct
    with tempfile.TemporaryDirectory() as tmpdir:
        # Parse original map
        ra3map = Ra3Map(str(map_file))
        ra3map.parse()
        
        # Reconstruct to new file (uncompressed for comparison)
        reconstructed_path = os.path.join(tmpdir, "map_mp_2_rao1_reconstructed.map")
        ra3map.save(reconstructed_path, compress=False)
        
        # Get reconstructed uncompressed data
        reconstructed_data = get_uncompressed_bytes(reconstructed_path)
        reconstructed_size = len(reconstructed_data)
        print(f"Reconstructed uncompressed size: {reconstructed_size:,} bytes")
        
        # Compare sizes
        size_diff = reconstructed_size - original_size
        print(f"Size difference: {size_diff:,} bytes")
        
        if size_diff != 0:
            pytest.fail(f"File size mismatch! Original: {original_size:,}, Reconstructed: {reconstructed_size:,}, Difference: {size_diff:,}")
        
        # Byte-by-byte comparison (skip first 4 bytes which is the flag)
        min_len = min(len(original_data), len(reconstructed_data))
        differences = []
        compare_start = 4  # Skip flag
        compare_end = min_len
        
        print(f"\nComparing bytes {compare_start:,} to {compare_end:,}...")
        
        for offset in range(compare_start, compare_end):
            if original_data[offset] != reconstructed_data[offset]:
                differences.append(offset)
                if len(differences) >= 100:  # Limit to first 100 differences
                    break
        
        if differences:
            print(f"\n{'='*70}")
            print(f"FAILED: Found {len(differences)} byte differences!")
            print(f"{'='*70}")
            
            # Show first 20 differences with context
            print(f"\nFirst {min(20, len(differences))} differences:")
            for idx, offset in enumerate(differences[:20]):
                orig_byte = original_data[offset]
                recon_byte = reconstructed_data[offset]
                
                # Show context (32 bytes)
                ctx_start = max(compare_start, offset - 16)
                ctx_end = min(min_len, offset + 16)
                
                orig_ctx = original_data[ctx_start:ctx_end]
                recon_ctx = reconstructed_data[ctx_start:ctx_end]
                
                orig_hex = ' '.join(f'{b:02x}' for b in orig_ctx)
                recon_hex = ' '.join(f'{b:02x}' for b in recon_ctx)
                marker_pos = offset - ctx_start
                marker = ' ' * (marker_pos * 3) + '^^'
                
                print(f"\n  Difference #{idx + 1} at offset {offset:,} (0x{offset:x}):")
                print(f"    Original byte:     0x{orig_byte:02x} ({orig_byte:3d})")
                print(f"    Reconstructed byte: 0x{recon_byte:02x} ({recon_byte:3d})")
                print(f"    Difference: {orig_byte - recon_byte:+d}")
                print(f"    Context:")
                print(f"      Original:     {orig_hex}")
                print(f"      Reconstructed: {recon_hex}")
                print(f"      Marker:       {marker}")
            
            if len(differences) > 20:
                print(f"\n  ... and {len(differences) - 20} more differences")
            
            # FAIL the test
            pytest.fail(f"NOT BIT-PERFECT! Found {len(differences)} byte differences. Expected 0 differences for exact match.")
        
        else:
            print(f"\n{'='*70}")
            print(f"SUCCESS: EXACT BIT-PERFECT MATCH!")
            print(f"{'='*70}")
            print(f"All {compare_end - compare_start:,} compared bytes match exactly")
            print(f"File sizes match exactly: {original_size:,} bytes")
            print(f"\nmap_mp_2_rao1.map can be parsed and reconstructed EXACTLY!")
            print(f"{'='*70}")


if __name__ == '__main__':
    test_exact_reconstruction_map_mp_2_rao1()

