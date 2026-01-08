#!/usr/bin/env python3
"""
Extract common differences and unique differences from the diff table.
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
    rows = parse_diff_file('../ALL_BASES_VS_ARCHON_DIFF.md')
    
    # Group by category + property (ignoring map name)
    # For "String Pool | Only in GT", the string name is in base_val
    grouped = defaultdict(lambda: {'HF': None, 'Caldera': None})
    
    for map_name, category, property_name, base_val, gt_val in rows:
        # Special case: for "String Pool | Only in GT", use the string name (in base_val) as the key
        if category == 'String Pool' and property_name == 'Only in GT':
            key = (category, property_name, base_val)  # base_val contains the string name
        else:
            key = (category, property_name)
        
        if map_name == 'HF':
            grouped[key]['HF'] = (base_val, gt_val)
        elif map_name == 'Caldera':
            grouped[key]['Caldera'] = (base_val, gt_val)
    
    # Separate into common and unique
    common_rows = []
    unique_rows = []
    
    for key, maps in grouped.items():
        # Handle both 2-tuple and 3-tuple keys
        if len(key) == 3:
            category, property_name, string_name = key
            # For display, combine property_name and string_name
            display_property = f"{property_name}: {string_name}"
        else:
            category, property_name = key
            display_property = property_name
        
        hf_data = maps['HF']
        caldera_data = maps['Caldera']
        
        # Common: exists in both maps
        if hf_data is not None and caldera_data is not None:
            # Check if the difference pattern is the same
            hf_base, hf_gt = hf_data
            cal_base, cal_gt = caldera_data
            
            # If both have the same pattern (e.g., both add the same string, both have same property change)
            # Consider it common even if values differ slightly
            common_rows.append((category, display_property, hf_base, hf_gt, cal_base, cal_gt))
        else:
            # Unique: only in one map
            if hf_data is not None:
                unique_rows.append(('HF', category, display_property, hf_data[0], hf_data[1]))
            if caldera_data is not None:
                unique_rows.append(('Caldera', category, display_property, caldera_data[0], caldera_data[1]))
    
    # Output common differences table
    print("# COMMON DIFFERENCES (Both HF and Caldera)")
    print()
    print("| Category | Property | HF Base Value | HF GT Value | Caldera Base Value | Caldera GT Value |")
    print("|----------|----------|---------------|-------------|-------------------|------------------|")
    for category, property_name, hf_base, hf_gt, cal_base, cal_gt in sorted(common_rows):
        # Escape pipe characters
        category = str(category).replace("|", "\\|")
        property_name = str(property_name).replace("|", "\\|")
        hf_base = str(hf_base).replace("|", "\\|")
        hf_gt = str(hf_gt).replace("|", "\\|")
        cal_base = str(cal_base).replace("|", "\\|")
        cal_gt = str(cal_gt).replace("|", "\\|")
        print(f"| {category} | {property_name} | {hf_base} | {hf_gt} | {cal_base} | {cal_gt} |")
    
    print()
    print()
    print("# UNIQUE DIFFERENCES (Map-Specific)")
    print()
    print("| Map | Category | Property | Base Value | GT Value |")
    print("|-----|----------|----------|------------|----------|")
    for map_name, category, property_name, base_val, gt_val in sorted(unique_rows):
        # Escape pipe characters
        map_name = str(map_name).replace("|", "\\|")
        category = str(category).replace("|", "\\|")
        property_name = str(property_name).replace("|", "\\|")
        base_val = str(base_val).replace("|", "\\|")
        gt_val = str(gt_val).replace("|", "\\|")
        print(f"| {map_name} | {category} | {property_name} | {base_val} | {gt_val} |")

if __name__ == "__main__":
    extract_diffs()
