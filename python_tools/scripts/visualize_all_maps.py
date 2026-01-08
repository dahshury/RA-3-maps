"""
Visualize all maps in a directory structure
Usage: python visualize_all_maps.py [maps_directory] [output_directory]
"""
import sys
import argparse
from pathlib import Path
import os
try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False
    def tqdm(iterable, desc=""):
        return iterable

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from map_processor.core.ra3map import Ra3Map
from map_processor.parsing.parser_config import ParserConfig
from map_processor.utils.map_visualizer import MapVisualizer


def find_map_files(directory: Path) -> list:
    """Find all .map files in directory and subdirectories"""
    map_files = []
    for map_file in directory.rglob("*.map"):
        map_files.append(map_file)
    return sorted(map_files)


def visualize_map_file(
    map_file: Path,
    output_base: Path,
    use_training_config: bool = True,
    generate_heightmap: bool = False,
    use_gpu: bool = False,
) -> dict:
    """
    Visualize a single map file.
    
    Returns:
        dict with status and any error message
    """
    try:
        # Determine output path (preserve directory structure)
        relative_path = map_file.relative_to(map_file.parents[len(map_file.parents) - 2])
        output_dir = output_base / relative_path.parent
        output_dir.mkdir(parents=True, exist_ok=True)
        
        map_name = map_file.stem
        
        # Parse map
        if use_training_config:
            config = ParserConfig.training_config()
            ra3map = Ra3Map(str(map_file), config=config)
        else:
            ra3map = Ra3Map(str(map_file))
        
        ra3map.parse()
        context = ra3map.get_context()
        
        # Generate visualization (only comprehensive by default)
        results = MapVisualizer.visualize_map(
            context, 
            str(output_dir), 
            map_name,
            generate_heightmap=generate_heightmap,
            generate_comprehensive=True,
            use_gpu=use_gpu,
        )
        
        return {
            'status': 'success',
            'map_file': str(map_file),
            'output': results.get('terrain_comprehensive', '')
        }
    except Exception as e:
        return {
            'status': 'error',
            'map_file': str(map_file),
            'error': str(e)
        }


def main():
    parser = argparse.ArgumentParser(description='Visualize all RA3 maps in a directory structure')
    parser.add_argument('maps_directory', type=str, nargs='?', 
                       default='../RA3 Official maps',
                       help='Directory containing map files (default: ../RA3 Official maps)')
    parser.add_argument('output_directory', type=str, nargs='?',
                       default='test_output',
                       help='Output directory (default: test_output)')
    parser.add_argument('--no-training-config', action='store_true',
                       help='Use default config instead of training config')
    parser.add_argument('--heightmap', action='store_true',
                       help='Also generate heightmap images')
    parser.add_argument('--gpu', action='store_true',
                       help='Use GPU acceleration (requires CuPy).')
    parser.add_argument('--workers', type=int, default=max(1, (os.cpu_count() or 4) // 2),
                       help='Number of parallel worker processes (default: half your CPU cores).')
    
    args = parser.parse_args()
    
    maps_dir = Path(args.maps_directory)
    if not maps_dir.exists():
        print(f"Error: Maps directory not found: {maps_dir}")
        return 1
    
    output_base = Path(args.output_directory)
    output_base.mkdir(parents=True, exist_ok=True)
    
    # Find all map files
    print(f"Scanning for map files in: {maps_dir}")
    map_files = find_map_files(maps_dir)
    print(f"Found {len(map_files)} map files")
    
    if len(map_files) == 0:
        print("No map files found!")
        return 1
    
    # Process all maps
    print(f"\nProcessing maps (output: {output_base})...")
    results = []
    errors = []

    if args.workers and args.workers > 1:
        from concurrent.futures import ProcessPoolExecutor, as_completed

        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            futures = [
                executor.submit(
                    visualize_map_file,
                    map_file,
                    output_base,
                    not args.no_training_config,
                    args.heightmap,
                    args.gpu,
                )
                for map_file in map_files
            ]

            iterator = as_completed(futures)
            if HAS_TQDM:
                iterator = tqdm(iterator, desc="Processing maps", total=len(futures))

            for fut in iterator:
                result = fut.result()
                results.append(result)
                if result['status'] == 'error':
                    errors.append(result)
    else:
        for map_file in tqdm(map_files, desc="Processing maps"):
            result = visualize_map_file(
                map_file,
                output_base,
                use_training_config=not args.no_training_config,
                generate_heightmap=args.heightmap,
                use_gpu=args.gpu,
            )
            results.append(result)
            if result['status'] == 'error':
                errors.append(result)
    
    # Summary
    print(f"\n{'='*60}")
    print(f"Processing complete!")
    print(f"  Total maps: {len(map_files)}")
    print(f"  Successful: {len([r for r in results if r['status'] == 'success'])}")
    print(f"  Errors: {len(errors)}")
    
    if errors:
        print(f"\nErrors encountered:")
        for error in errors[:10]:  # Show first 10 errors
            print(f"  {Path(error['map_file']).name}: {error['error']}")
        if len(errors) > 10:
            print(f"  ... and {len(errors) - 10} more errors")
    
    return 0 if len(errors) == 0 else 1


if __name__ == '__main__':
    sys.exit(main())

