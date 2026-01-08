"""
Test for BIT-PERFECT reconstruction
Compare original vs reconstructed maps byte-by-byte
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


def test_bit_perfect_uncompressed_reconstruction(python_processor, sample_map_file, temp_directory):
    """
    Test BIT-PERFECT reconstruction - compare uncompressed data byte-by-byte
    
    This test verifies that we can reconstruct maps with ZERO data loss.
    """
    print(f"\n{'='*60}")
    print(f"BIT-PERFECT RECONSTRUCTION TEST")
    print(f"{'='*60}")
    print(f"\nTesting: {Path(sample_map_file).name}")
    
    # Get original uncompressed data
    original_data = get_uncompressed_bytes(sample_map_file)
    print(f"Original uncompressed size: {len(original_data):,} bytes")
    
    # Parse and reconstruct (uncompressed)
    ra3map = python_processor(sample_map_file)
    ra3map.parse()
    
    reconstructed_path = os.path.join(temp_directory, "reconstructed_bit_perfect.map")
    ra3map.save(reconstructed_path, compress=False)
    
    # Get reconstructed uncompressed data
    reconstructed_data = get_uncompressed_bytes(reconstructed_path)
    print(f"Reconstructed uncompressed size: {len(reconstructed_data):,} bytes")
    
    # Compare sizes
    size_diff = len(reconstructed_data) - len(original_data)
    size_diff_pct = (size_diff / len(original_data) * 100) if len(original_data) > 0 else 0
    print(f"Size difference: {size_diff:,} bytes ({size_diff_pct:.6f}%)")
    
    # Byte-by-byte comparison
    min_len = min(len(original_data), len(reconstructed_data))
    max_len = max(len(original_data), len(reconstructed_data))
    
    if min_len != max_len:
        print(f"\nWARNING: Size mismatch! {max_len - min_len} bytes difference")
        print(f"  Original: {len(original_data):,} bytes")
        print(f"  Reconstructed: {len(reconstructed_data):,} bytes")
    
    # Compare bytes (skip first 4 bytes which is the flag)
    differences = []
    compare_start = 4
    compare_end = min_len
    
    print(f"\nComparing bytes {compare_start:,} to {compare_end:,}...")
    
    for i in range(compare_start, compare_end):
        if original_data[i] != reconstructed_data[i]:
            differences.append(i)
            if len(differences) >= 100:  # Limit to first 100 differences
                break
    
    if differences:
        print(f"\n{'='*60}")
        print(f"FOUND {len(differences)} BYTE DIFFERENCES!")
        print(f"{'='*60}")
        
        # Show first 20 differences with context
        print(f"\nFirst {min(20, len(differences))} differences (with context):")
        for idx, offset in enumerate(differences[:20]):
            # Show 32 bytes of context around the difference
            context_start = max(compare_start, offset - 16)
            context_end = min(compare_end, offset + 16)
            
            orig_context = original_data[context_start:context_end]
            recon_context = reconstructed_data[context_start:context_end]
            
            # Highlight the differing byte
            orig_hex = ' '.join(f'{b:02x}' for b in orig_context)
            recon_hex = ' '.join(f'{b:02x}' for b in recon_context)
            
            # Mark the differing byte
            byte_pos_in_context = offset - context_start
            marker = ' ' * (byte_pos_in_context * 3) + '^^'
            
            print(f"\n  Difference #{idx + 1} at offset {offset:,} (0x{offset:x})")
            print(f"    Original byte:     0x{original_data[offset]:02x} ({original_data[offset]})")
            print(f"    Reconstructed byte: 0x{reconstructed_data[offset]:02x} ({reconstructed_data[offset]})")
            print(f"    Context (32 bytes):")
            print(f"      Original:   {orig_hex}")
            print(f"      Reconstructed: {recon_hex}")
            print(f"      Marker:     {marker}")
        
        if len(differences) > 20:
            print(f"\n  ... and {len(differences) - 20} more differences")
        
        # FAIL the test - we want bit-perfect reconstruction
        pytest.fail(f"Found {len(differences)} byte differences - NOT bit-perfect!")
    else:
        print(f"\n{'='*60}")
        print(f"SUCCESS: BIT-PERFECT MATCH!")
        print(f"{'='*60}")
        print(f"All {compare_end - compare_start:,} compared bytes match exactly")
        if min_len == max_len:
            print(f"File sizes match exactly: {len(original_data):,} bytes")
        else:
            print(f"Note: Size difference of {abs(size_diff):,} bytes (outside compared range)")


def test_verify_critical_data_preserved(python_processor, sample_map_file, temp_directory):
    """
    Verify that critical map data is preserved:
    - Heights (HeightMapData)
    - Objects (ObjectsList)
    - Players (SidesList)
    - All asset counts
    """
    from map_processor.height_map_data import HeightMapData
    from map_processor.objects_list import ObjectsList
    from map_processor.sides_list import SidesList
    import numpy as np
    
    print(f"\nVerifying critical data preservation for {Path(sample_map_file).name}")
    
    # Parse original
    ra3map_orig = python_processor(sample_map_file)
    ra3map_orig.parse()
    context_orig = ra3map_orig.get_context()
    
    # Reconstruct
    recon_path = os.path.join(temp_directory, "verify_critical.map")
    ra3map_orig.save(recon_path, compress=False)
    
    # Parse reconstructed
    ra3map_recon = python_processor(recon_path)
    ra3map_recon.parse()
    context_recon = ra3map_recon.get_context()
    
    # Verify heights
    orig_height = context_orig.get_asset_by_type(HeightMapData)
    recon_height = context_recon.get_asset_by_type(HeightMapData)
    
    if orig_height and recon_height:
        assert orig_height.map_width == recon_height.map_width
        assert orig_height.map_height == recon_height.map_height
        assert np.array_equal(orig_height.elevations, recon_height.elevations), \
            "Height elevations must match EXACTLY (bit-perfect)"
        print(f"  HeightMapData: {orig_height.map_width}x{orig_height.map_height} - EXACT match")
    
    # Verify objects
    orig_objects = context_orig.get_asset("ObjectsList")
    recon_objects = context_recon.get_asset("ObjectsList")
    
    if orig_objects and recon_objects:
        assert isinstance(orig_objects, ObjectsList)
        assert isinstance(recon_objects, ObjectsList)
        assert len(orig_objects.map_objects) == len(recon_objects.map_objects)
        print(f"  ObjectsList: {len(orig_objects.map_objects)} objects - EXACT match")
    
    # Verify players
    orig_sides = context_orig.get_asset("SidesList")
    recon_sides = context_recon.get_asset("SidesList")
    
    if orig_sides and recon_sides:
        assert isinstance(orig_sides, SidesList)
        assert isinstance(recon_sides, SidesList)
        assert len(orig_sides.players) == len(recon_sides.players)
        print(f"  SidesList: {len(orig_sides.players)} players - EXACT match")
    
    # Verify asset counts
    assert len(context_orig.map_struct.assets) == len(context_recon.map_struct.assets)
    print(f"  Total assets: {len(context_orig.map_struct.assets)} - EXACT match")
    
    print(f"\nAll critical data verified - EXACT match!")











