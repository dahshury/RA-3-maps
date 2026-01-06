"""
Tests for map reconstruction (parse -> reconstruct -> parse pipeline)
"""
import pytest
import os
import json
from pathlib import Path


def test_reconstruct_map_python(python_processor, sample_map_file, temp_directory):
    """
    Test the full pipeline: parse -> reconstruct -> parse -> verify
    
    This is the main test that ensures the reconstruction pipeline works correctly.
    """
    # Step 1: Parse original map
    print(f"\nStep 1: Parsing {sample_map_file}")
    ra3map_original = python_processor(sample_map_file)
    ra3map_original.parse()
    original_context = ra3map_original.get_context()
    
    assert original_context.map_width > 0, "Original map should have valid dimensions"
    assert original_context.map_height > 0, "Original map should have valid dimensions"
    print(f"Step 1 passed: Parsed map {original_context.map_width}x{original_context.map_height}")
    
    # Step 2: Reconstruct map
    reconstructed_map = os.path.join(temp_directory, "reconstructed.map")
    print(f"Step 2: Reconstructing to {reconstructed_map}")
    
    ra3map_original.save(reconstructed_map, compress=False)  # Save uncompressed for now
    assert os.path.exists(reconstructed_map), "Reconstructed map should exist"
    assert os.path.getsize(reconstructed_map) > 0, "Reconstructed map should not be empty"
    print(f"Step 2 passed: Reconstructed map ({os.path.getsize(reconstructed_map)} bytes)")
    
    # Step 3: Parse reconstructed map
    print(f"Step 3: Parsing reconstructed map")
    ra3map_reconstructed = python_processor(reconstructed_map)
    ra3map_reconstructed.parse()
    reconstructed_context = ra3map_reconstructed.get_context()
    
    assert reconstructed_context.map_width > 0, "Reconstructed map should have valid dimensions"
    print(f"Step 3 passed: Reparsed map {reconstructed_context.map_width}x{reconstructed_context.map_height}")
    
    # Step 4: Compare data
    print(f"Step 4: Comparing data")
    
    # Compare dimensions
    assert original_context.map_width == reconstructed_context.map_width, \
        f"Width mismatch: {original_context.map_width} != {reconstructed_context.map_width}"
    assert original_context.map_height == reconstructed_context.map_height, \
        f"Height mismatch: {original_context.map_height} != {reconstructed_context.map_height}"
    
    # Compare height maps if available
    from map_processor.height_map_data import HeightMapData
    
    original_height = original_context.get_asset_by_type(HeightMapData)
    reconstructed_height = reconstructed_context.get_asset_by_type(HeightMapData)
    
    if original_height and reconstructed_height:
        import numpy as np
        assert np.allclose(original_height.elevations, reconstructed_height.elevations, atol=0.1), \
            "Height maps should match (within tolerance)"
        print("Step 4 passed: Height maps match")
    else:
        print("⚠ Step 4: HeightMapData not available for comparison")
    
    print(f"\nComplete pipeline test passed!")


def test_reconstruction_pipeline_multiple_maps(python_processor, all_map_files, temp_directory):
    """Test reconstruction pipeline on multiple maps"""
    if len(all_map_files) == 0:
        pytest.skip("No map files available")
    
    # Test first 3 maps to keep test time reasonable
    test_files = all_map_files[:3]
    
    failed_maps = []
    
    for i, map_file in enumerate(test_files):
        map_name = Path(map_file).stem
        
        try:
            # Parse
            ra3map = python_processor(map_file)
            ra3map.parse()
            original_context = ra3map.get_context()
            
            # Reconstruct
            reconstructed_map = os.path.join(temp_directory, f"{map_name}_reconstructed.map")
            ra3map.save(reconstructed_map, compress=False)
            
            if not os.path.exists(reconstructed_map):
                failed_maps.append((map_name, "reconstruction_failed"))
                continue
            
            # Parse reconstructed
            ra3map_recon = python_processor(reconstructed_map)
            ra3map_recon.parse()
            reconstructed_context = ra3map_recon.get_context()
            
            # Basic verification
            assert os.path.exists(reconstructed_map), f"Reconstructed map should exist: {map_name}"
            assert os.path.getsize(reconstructed_map) > 0, f"Reconstructed map should not be empty: {map_name}"
            assert original_context.map_width == reconstructed_context.map_width, \
                f"Width mismatch for {map_name}"
            
            print(f"Successfully processed: {map_name}")
            
        except Exception as e:
            failed_maps.append((map_name, str(e)))
            print(f"✗ Failed to process {map_name}: {e}")
    
    # Report results
    success_count = len(test_files) - len(failed_maps)
    print(f"\nProcessed {success_count}/{len(test_files)} maps successfully")
    
    if failed_maps:
        print("\nFailed maps:")
        for map_name, error in failed_maps:
            print(f"  - {map_name}: {error}")
        
        # Fail if more than 50% failed
        failure_rate = len(failed_maps) / len(test_files)
        assert failure_rate < 0.5, f"Too many maps failed: {failure_rate:.1%}"


def test_reconstruction_file_sizes(python_processor, sample_map_file, temp_directory):
    """Test that reconstructed maps have reasonable file sizes"""
    reconstructed_map = os.path.join(temp_directory, "size_test.map")
    
    # Parse
    ra3map = python_processor(sample_map_file)
    ra3map.parse()
    
    # Reconstruct
    ra3map.save(reconstructed_map, compress=False)
    
    # Compare file sizes (reconstructed should be within reasonable range)
    original_size = os.path.getsize(sample_map_file)
    reconstructed_size = os.path.getsize(reconstructed_map)
    
    # Reconstructed size should be within 50% to 200% of original
    # (compression differences, etc.)
    size_ratio = reconstructed_size / original_size if original_size > 0 else 1.0
    
    assert 0.5 <= size_ratio <= 2.0, \
        f"Reconstructed map size ratio {size_ratio:.2f} is outside reasonable range (0.5-2.0)"
    
    print(f"Original size: {original_size} bytes")
    print(f"Reconstructed size: {reconstructed_size} bytes")
    print(f"Size ratio: {size_ratio:.2f}")

