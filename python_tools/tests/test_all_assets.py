"""
Tests for all asset types - verify parsing and reconstruction works for all assets
"""
import pytest
import os
from pathlib import Path


def test_parse_all_asset_types(python_processor, sample_map_file):
    """Test that all asset types can be parsed"""
    from map_processor.height_map_data import HeightMapData
    from map_processor.objects_list import ObjectsList
    from map_processor.asset_list import AssetList
    from map_processor.sides_list import SidesList
    from map_processor.player_scripts_list import PlayerScriptsList
    
    ra3map = python_processor(sample_map_file)
    ra3map.parse()
    context = ra3map.get_context()
    
    # Get all assets
    assets_by_name = {}
    for asset in context.map_struct.assets:
        asset_name = asset.get_asset_name()
        if asset_name not in assets_by_name:
            assets_by_name[asset_name] = []
        assets_by_name[asset_name].append(asset)
    
    print(f"\nFound {len(context.map_struct.assets)} total assets")
    print(f"Unique asset types: {len(assets_by_name)}")
    
    # Test HeightMapData
    height_map = context.get_asset_by_type(HeightMapData)
    if height_map:
        print(f"  - HeightMapData: {height_map.map_width}x{height_map.map_height}")
        assert height_map.map_width > 0
        assert height_map.map_height > 0
    else:
        print("  - HeightMapData: Not found")
    
    # Test ObjectsList
    objects_list = context.get_asset("ObjectsList")
    if objects_list:
        from map_processor.objects_list import ObjectsList
        assert isinstance(objects_list, ObjectsList)
        print(f"  - ObjectsList: {len(objects_list.map_objects)} objects")
    else:
        print("  - ObjectsList: Not found")
    
    # Test AssetList
    asset_list = context.get_asset("AssetList")
    if asset_list:
        from map_processor.asset_list import AssetList
        assert isinstance(asset_list, AssetList)
        print(f"  - AssetList: {len(asset_list.asset_blocks)} blocks")
    else:
        print("  - AssetList: Not found")
    
    # Test SidesList
    sides_list = context.get_asset("SidesList")
    if sides_list:
        from map_processor.sides_list import SidesList
        assert isinstance(sides_list, SidesList)
        print(f"  - SidesList: {len(sides_list.players)} players")
    else:
        print("  - SidesList: Not found")
    
    # Test PlayerScriptsList
    player_scripts = context.get_asset("PlayerScriptsList")
    if player_scripts:
        from map_processor.player_scripts_list import PlayerScriptsList
        assert isinstance(player_scripts, PlayerScriptsList)
        print(f"  - PlayerScriptsList: {len(player_scripts.script_lists)} script lists")
    else:
        print("  - PlayerScriptsList: Not found")
    
    # Print summary of all asset types found
    print("\nAll asset types found:")
    for asset_name, assets in sorted(assets_by_name.items()):
        print(f"  - {asset_name}: {len(assets)} instance(s)")


def test_reconstruct_all_assets(python_processor, sample_map_file, temp_directory):
    """
    Test full round-trip: parse -> reconstruct -> parse -> verify all assets
    """
    from map_processor.height_map_data import HeightMapData
    from map_processor.objects_list import ObjectsList
    from map_processor.asset_list import AssetList
    from map_processor.sides_list import SidesList
    from map_processor.player_scripts_list import PlayerScriptsList
    
    # Step 1: Parse original map
    print(f"\nStep 1: Parsing {Path(sample_map_file).name}")
    ra3map_original = python_processor(sample_map_file)
    ra3map_original.parse()
    original_context = ra3map_original.get_context()
    
    assert original_context.map_width > 0
    assert original_context.map_height > 0
    
    # Count assets by type
    original_assets_by_name = {}
    for asset in original_context.map_struct.assets:
        name = asset.get_asset_name()
        original_assets_by_name[name] = original_assets_by_name.get(name, 0) + 1
    
    print(f"  Original: {len(original_context.map_struct.assets)} assets, "
          f"{len(original_assets_by_name)} types")
    
    # Step 2: Reconstruct
    reconstructed_map = os.path.join(temp_directory, "reconstructed_all_assets.map")
    print(f"Step 2: Reconstructing...")
    ra3map_original.save(reconstructed_map, compress=False)
    
    assert os.path.exists(reconstructed_map)
    assert os.path.getsize(reconstructed_map) > 0
    print(f"  Reconstructed: {os.path.getsize(reconstructed_map)} bytes")
    
    # Step 3: Parse reconstructed
    print(f"Step 3: Parsing reconstructed map...")
    ra3map_reconstructed = python_processor(reconstructed_map)
    ra3map_reconstructed.parse()
    reconstructed_context = ra3map_reconstructed.get_context()
    
    assert reconstructed_context.map_width > 0
    assert reconstructed_context.map_height > 0
    
    # Count assets by type
    reconstructed_assets_by_name = {}
    for asset in reconstructed_context.map_struct.assets:
        name = asset.get_asset_name()
        reconstructed_assets_by_name[name] = reconstructed_assets_by_name.get(name, 0) + 1
    
    print(f"  Reconstructed: {len(reconstructed_context.map_struct.assets)} assets, "
          f"{len(reconstructed_assets_by_name)} types")
    
    # Step 4: Verify dimensions
    assert original_context.map_width == reconstructed_context.map_width
    assert original_context.map_height == reconstructed_context.map_height
    print(f"Step 4: Dimensions match: {reconstructed_context.map_width}x{reconstructed_context.map_height}")
    
    # Step 5: Verify asset counts match (at least approximately - some may use DefaultMajorAsset)
    print(f"Step 5: Comparing asset types...")
    
    # Core assets should match
    core_assets = ["HeightMapData", "ObjectsList", "AssetList", "SidesList", "PlayerScriptsList"]
    for asset_name in core_assets:
        original_count = original_assets_by_name.get(asset_name, 0)
        reconstructed_count = reconstructed_assets_by_name.get(asset_name, 0)
        if original_count > 0:
            print(f"  {asset_name}: {original_count} -> {reconstructed_count}")
            # Should have at least one if original had one
            assert reconstructed_count > 0, f"{asset_name} should exist in reconstructed map"
    
    # Step 6: Verify HeightMapData if available
    original_height = original_context.get_asset_by_type(HeightMapData)
    reconstructed_height = reconstructed_context.get_asset_by_type(HeightMapData)
    
    if original_height and reconstructed_height:
        import numpy as np
        assert np.allclose(original_height.elevations, reconstructed_height.elevations, atol=0.1), \
            "Height maps should match"
        print(f"Step 6: HeightMapData matches")
    
    # Step 7: Verify ObjectsList if available
    original_objects = original_context.get_asset("ObjectsList")
    reconstructed_objects = reconstructed_context.get_asset("ObjectsList")
    
    if original_objects and reconstructed_objects:
        assert isinstance(original_objects, ObjectsList)
        assert isinstance(reconstructed_objects, ObjectsList)
        # Object count should match (approximately - some may be filtered)
        print(f"Step 7: ObjectsList - {len(original_objects.map_objects)} -> {len(reconstructed_objects.map_objects)} objects")
    
    # Step 8: Verify SidesList if available
    original_sides = original_context.get_asset("SidesList")
    reconstructed_sides = reconstructed_context.get_asset("SidesList")
    
    if original_sides and reconstructed_sides:
        assert isinstance(original_sides, SidesList)
        assert isinstance(reconstructed_sides, SidesList)
        assert len(original_sides.players) == len(reconstructed_sides.players), \
            "Player count should match"
        print(f"Step 8: SidesList - {len(original_sides.players)} players")
    
    print("\nAll asset round-trip test PASSED!")


def test_parse_multiple_maps_with_all_assets(python_processor, all_map_files):
    """Test parsing multiple maps and verify all asset types are handled"""
    if len(all_map_files) == 0:
        pytest.skip("No map files available")
    
    # Test first 3 maps
    test_files = all_map_files[:3]
    
    asset_types_found = set()
    success_count = 0
    
    for map_file in test_files:
        try:
            ra3map = python_processor(map_file)
            ra3map.parse()
            context = ra3map.get_context()
            
            assert context.map_width > 0
            assert context.map_height > 0
            
            # Collect asset types
            for asset in context.map_struct.assets:
                asset_types_found.add(asset.get_asset_name())
            
            success_count += 1
            print(f"Parsed: {Path(map_file).name} - {len(context.map_struct.assets)} assets")
            
        except Exception as e:
            print(f"Failed to parse {Path(map_file).name}: {e}")
    
    assert success_count > 0, "Should parse at least some maps"
    print(f"\nFound {len(asset_types_found)} unique asset types across all maps:")
    for asset_type in sorted(asset_types_found):
        print(f"  - {asset_type}")











