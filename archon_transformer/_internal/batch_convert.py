"""
Batch convert RA3 maps to Archon mode.

Scans input folder for .map files, detects player count,
and converts all 2-player and 3-player maps automatically.

Usage:
  python batch_convert.py [input_folder] [output_folder]
  
If no folders specified:
  - Input: ./maps_to_convert/
  - Output: ./converted_maps/
"""

from __future__ import annotations

import sys
import shutil
from pathlib import Path
from typing import Optional

# Handle PyInstaller bundled mode vs normal Python execution
def get_app_root() -> Path:
    """Get the application root directory, works both for normal Python and PyInstaller."""
    if getattr(sys, 'frozen', False):
        # Running as compiled exe - use exe's directory
        return Path(sys.executable).resolve().parent
    else:
        # Running as script
        return Path(__file__).resolve().parent

_ROOT = get_app_root()
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from map_processor.core.ra3map import Ra3Map
from transform_to_archon import (
    transform_to_archon, 
    find_player_starts,
    are_maps_same_base,
    _is_paired_archon_3p_template,
    _generate_art_tga,
    _write_sidecars,
)


# Note: No wait_for_exit() needed - the batch file handles pausing


def get_player_count(map_path: Path) -> Optional[int]:
    """
    Detect the number of builder players in a map.
    Returns None if map cannot be parsed.
    """
    try:
        ra3map = Ra3Map(str(map_path))
        ra3map.parse()
        context = ra3map.get_context()
        
        # Count Player Start waypoints by unique_id (e.g. "Player_1_Start")
        builder_count = 0
        objs = context.get_asset("ObjectsList")
        if objs:
            for obj in objs.map_objects:
                unique_id = getattr(obj, 'unique_id', None)
                if unique_id and 'Player_' in unique_id and '_Start' in unique_id:
                    # Extract player number
                    try:
                        num = int(unique_id.split('Player_')[1].split('_')[0])
                        if num > builder_count:
                            builder_count = num
                    except (ValueError, IndexError):
                        pass
        
        return builder_count if builder_count > 0 else None
    except Exception as e:
        print(f"  [ERROR] Could not parse {map_path.name}: {e}")
        return None


def find_maps(folder: Path) -> list[Path]:
    """Recursively find all .map files in folder."""
    maps = []
    for path in folder.rglob("*.map"):
        # Skip archon maps (already converted)
        if "[archon]" in path.name.lower():
            continue
        # Skip template maps
        if "template" in str(path).lower():
            continue
        maps.append(path)
    return maps


def get_map_display_name(map_path: Path) -> str:
    """Get a clean display name for the map."""
    name = map_path.stem
    # Remove common prefixes
    for prefix in ["map_mp_", "map_"]:
        if name.lower().startswith(prefix):
            name = name[len(prefix):]
    return name


def convert_map(
    map_path: Path,
    template_2p: Path,
    template_3p: Path,
    output_folder: Path,
    player_count: int
) -> bool:
    """
    Convert a single map to Archon mode.
    Returns True on success, False on failure.
    """
    # Select template based on player count
    template = template_2p if player_count <= 2 else template_3p
    
    # Generate output name
    display_name = get_map_display_name(map_path)
    # Capitalize words and clean up
    clean_name = "_".join(word.capitalize() for word in display_name.replace("-", "_").split("_") if word)
    archon_name = f"[Archon]{clean_name}"
    
    output_dir = output_folder / archon_name
    output_map = output_dir / f"{archon_name}.map"
    
    try:
        print(f"\n  Converting to: {archon_name}")
        print(f"  Using template: {template.name}")
        
        # Load source map
        source_map = Ra3Map(str(map_path))
        source_map.parse()
        source_ctx = source_map.get_context()
        
        # Load template map
        template_map = Ra3Map(str(template))
        template_map.parse()
        template_ctx = template_map.get_context()
        
        # Check for same-base/paired scenarios
        num_builders = len(find_player_starts(source_ctx))
        same_base = are_maps_same_base(source_ctx, template_ctx)
        paired_3p = num_builders == 3 and _is_paired_archon_3p_template(template_ctx)
        
        # Create output directory
        output_dir.mkdir(parents=True, exist_ok=True)
        
        if same_base and paired_3p:
            # Copy template directly for same-base 3p paired maps
            shutil.copy2(template, output_map)
            _generate_art_tga(output_map, source_ctx, source_map_path=map_path)
            _write_sidecars(output_map, source_ctx, template, source_map_path=map_path)
        else:
            # Transform the map
            transform_to_archon(source_ctx, template_ctx, bit_perfect=False)
            
            # Save output
            source_map.save(str(output_map), compress=True)
            
            # Generate minimap and write sidecars
            _generate_art_tga(output_map, source_ctx, source_map_path=map_path)
            _write_sidecars(output_map, source_ctx, template, source_map_path=map_path)
        
        print(f"  [OK] Success! Output: {output_dir}")
        return True
        
    except Exception as e:
        print(f"  [FAIL] Failed: {e}")
        return False


def main():
    print("=" * 60)
    print("  RA3 ARCHON MAP CONVERTER")
    print("=" * 60)
    print()
    
    # Determine paths - user folders are in parent, internal stuff stays in _ROOT
    user_root = _ROOT.parent  # One level up from _internal
    
    if len(sys.argv) >= 2:
        input_folder = Path(sys.argv[1])
    else:
        input_folder = user_root / "maps_to_convert"
    
    if len(sys.argv) >= 3:
        output_folder = Path(sys.argv[2])
    else:
        output_folder = user_root / "converted_maps"
    
    # Template paths (inside _internal)
    template_2p = _ROOT / "templates" / "2p" / "Archon Fire Island [1.4].map"
    template_3p = _ROOT / "templates" / "3p" / "[Archon]Hidden_Fortress_1.2.map"
    
    # Verify templates exist
    if not template_2p.exists():
        print(f"ERROR: 2-player template not found: {template_2p}")
        return 1
    
    if not template_3p.exists():
        print(f"ERROR: 3-player template not found: {template_3p}")
        return 1
    
    # Create folders if needed
    input_folder.mkdir(parents=True, exist_ok=True)
    output_folder.mkdir(parents=True, exist_ok=True)
    
    print(f"Input folder:  {input_folder}")
    print(f"Output folder: {output_folder}")
    print()
    
    # Find maps
    print("Scanning for .map files...")
    maps = find_maps(input_folder)
    
    if not maps:
        print("\nNo .map files found in input folder!")
        print(f"\nPlace your RA3 map files (or folders containing them) in:")
        print(f"  {input_folder}")
        print("\nThe converter will scan all subfolders automatically.")
        return 0
    
    print(f"Found {len(maps)} map file(s)")
    print()
    
    # Analyze and categorize maps
    maps_2p = []
    maps_3p = []
    maps_skipped = []
    
    print("Analyzing player counts...")
    for map_path in maps:
        print(f"  {map_path.name}: ", end="")
        player_count = get_player_count(map_path)
        
        if player_count is None:
            print("Could not detect players - SKIPPED")
            maps_skipped.append((map_path, "Could not parse"))
        elif player_count == 1:
            print("1 player - Will convert as 2-player Archon")
            maps_2p.append(map_path)
        elif player_count == 2:
            print("2 players - OK")
            maps_2p.append(map_path)
        elif player_count == 3:
            print("3 players - OK")
            maps_3p.append(map_path)
        else:
            print(f"{player_count} players - TOO MANY (max 3)")
            maps_skipped.append((map_path, f"{player_count} players - max is 3"))
    
    print()
    print("-" * 60)
    print(f"  2-player maps: {len(maps_2p)}")
    print(f"  3-player maps: {len(maps_3p)}")
    print(f"  Skipped:       {len(maps_skipped)}")
    print("-" * 60)
    
    if maps_skipped:
        print("\nSkipped maps:")
        for path, reason in maps_skipped:
            print(f"  - {path.name}: {reason}")
    
    convertible = len(maps_2p) + len(maps_3p)
    if convertible == 0:
        print("\nNo maps to convert!")
        return 0
    
    print(f"\nReady to convert {convertible} map(s).")
    print()
    
    # Convert maps
    success_count = 0
    fail_count = 0
    
    print("=" * 60)
    print("  CONVERTING 2-PLAYER MAPS")
    print("=" * 60)
    
    for map_path in maps_2p:
        print(f"\n[{success_count + fail_count + 1}/{convertible}] {map_path.name}")
        if convert_map(map_path, template_2p, template_3p, output_folder, 2):
            success_count += 1
        else:
            fail_count += 1
    
    if maps_3p:
        print()
        print("=" * 60)
        print("  CONVERTING 3-PLAYER MAPS")
        print("=" * 60)
        
        for map_path in maps_3p:
            print(f"\n[{success_count + fail_count + 1}/{convertible}] {map_path.name}")
            if convert_map(map_path, template_2p, template_3p, output_folder, 3):
                success_count += 1
            else:
                fail_count += 1
    
    # Summary
    print()
    print("=" * 60)
    print("  CONVERSION COMPLETE")
    print("=" * 60)
    print(f"  Successful: {success_count}")
    print(f"  Failed:     {fail_count}")
    print(f"  Skipped:    {len(maps_skipped)}")
    print()
    print(f"Converted maps saved to: {output_folder}")
    print()
    
    return 0 if fail_count == 0 else 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n\nCancelled by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\n\nFATAL ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

