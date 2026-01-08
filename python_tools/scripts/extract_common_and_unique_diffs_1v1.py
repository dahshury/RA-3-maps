#!/usr/bin/env python3
"""
Extract common differences and unique differences from the 1v1 diff table.
Creates two separate markdown tables.
"""
import re
from collections import defaultdict

def parse_diff_file(filepath):
    """Parse the markdown diff file and return rows grouped by category/property"""
    rows = []
    with open(filepath, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    # Skip header and separator
    for line in lines[2:]:
        if not line.strip():
            continue
        # Parse markdown table row: | Map | Category | Property | Base Value | GT Value |
        parts = [p.strip() for p in line.split('|')]
        if len(parts) >= 6:
            map_name = parts[1]
            category = parts[2]
            property_name = parts[3]
            base_val = parts[4]
            gt_val = parts[5]
            rows.append((map_name, category, property_name, base_val, gt_val))
    
    return rows

def extract_diffs():
    rows = parse_diff_file('../ALL_1V1_BASES_VS_ARCHON_DIFF.md')
    
    # Group by category + property (ignoring map name)
    # For "String Pool | Only in GT", the string name is in base_val
    grouped = defaultdict(lambda: {'CE': None, 'COC': None, 'CR': None, 'FI': None, 'II': None})
    
    for map_name, category, property_name, base_val, gt_val in rows:
        # Special case: for "String Pool | Only in GT", use the string name (in base_val) as the key
        if category == 'String Pool' and property_name == 'Only in GT':
            key = (category, property_name, base_val)  # base_val contains the string name
        else:
            key = (category, property_name)
        
        if map_name in grouped[key]:
            grouped[key][map_name] = (base_val, gt_val)
    
    # Separate into common and unique
    common_rows = []
    unique_rows = []
    
    # Get all map names
    all_maps = ['CE', 'COC', 'CR', 'FI', 'II']
    
    for key, maps in grouped.items():
        # Handle both 2-tuple and 3-tuple keys
        if len(key) == 3:
            category, property_name, string_name = key
            # For display, combine property_name and string_name
            display_property = f"{property_name}: {string_name}"
        else:
            category, property_name = key
            display_property = property_name
        
        # Count how many maps have this difference
        maps_with_diff = [m for m in all_maps if maps[m] is not None]
        
        if len(maps_with_diff) >= 3:  # Common if appears in at least 3 maps
            # Build a row showing values for all maps that have it
            row_data = [category, display_property]
            for map_name in all_maps:
                if maps[map_name] is not None:
                    base_val, gt_val = maps[map_name]
                    row_data.append(f"{map_name}: Base={base_val}, GT={gt_val}")
                else:
                    row_data.append("")
            # If all maps have the same pattern, show it once
            if len(maps_with_diff) == len(all_maps):
                # Check if all values are similar (for common strings, etc.)
                first_base, first_gt = maps[maps_with_diff[0]]
                all_same = all(maps[m][0] == first_base and maps[m][1] == first_gt for m in maps_with_diff[1:])
                if all_same:
                    common_rows.append((category, display_property, first_base, first_gt, "All maps"))
                else:
                    # Show values for each map
                    for map_name in maps_with_diff:
                        base_val, gt_val = maps[map_name]
                        common_rows.append((category, display_property, base_val, gt_val, map_name))
            else:
                # Show values for maps that have it
                for map_name in maps_with_diff:
                    base_val, gt_val = maps[map_name]
                    common_rows.append((category, display_property, base_val, gt_val, map_name))
        else:
            # Unique: only in 1-2 maps
            for map_name in maps_with_diff:
                base_val, gt_val = maps[map_name]
                unique_rows.append((map_name, category, display_property, base_val, gt_val))
    
    # Output common differences table
    print("# COMMON DIFFERENCES (Appearing in 3+ Maps)")
    print()
    print("| Category | Property | Base Value | GT Value | Maps |")
    print("|----------|----------|------------|----------|------|")
    for category, display_property, base_val, gt_val, maps_info in sorted(common_rows):
        # Escape pipe characters
        category = str(category).replace("|", "\\|")
        display_property = str(display_property).replace("|", "\\|")
        base_val = str(base_val).replace("|", "\\|")
        gt_val = str(gt_val).replace("|", "\\|")
        maps_info = str(maps_info).replace("|", "\\|")
        print(f"| {category} | {display_property} | {base_val} | {gt_val} | {maps_info} |")
    
    print()
    print()
    print("# UNIQUE DIFFERENCES (Map-Specific)")
    print()
    print("| Map | Category | Property | Base Value | GT Value |")
    print("|-----|----------|----------|------------|----------|")
    for map_name, category, display_property, base_val, gt_val in sorted(unique_rows):
        # Escape pipe characters
        map_name = str(map_name).replace("|", "\\|")
        category = str(category).replace("|", "\\|")
        display_property = str(display_property).replace("|", "\\|")
        base_val = str(base_val).replace("|", "\\|")
        gt_val = str(gt_val).replace("|", "\\|")
        print(f"| {map_name} | {category} | {display_property} | {base_val} | {gt_val} |")

if __name__ == "__main__":
    extract_diffs()
