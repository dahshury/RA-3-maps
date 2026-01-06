"""
Generate image from RA3 map file
Usage: python generate_map_image.py <map_file> [output_dir]
"""
import sys
import argparse
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from map_processor.ra3map import Ra3Map
from map_processor.map_visualizer import MapVisualizer


def main():
    parser = argparse.ArgumentParser(description='Generate images from RA3 map files')
    parser.add_argument('map_file', type=str, help='Path to .map file')
    parser.add_argument('output_dir', type=str, nargs='?', default='.', help='Output directory (default: current directory)')
    parser.add_argument('--colormap', type=str, choices=['grayscale', 'terrain'], default='terrain',
                       help='Colormap for height visualization (default: terrain)')
    
    args = parser.parse_args()
    
    map_path = Path(args.map_file)
    if not map_path.exists():
        print(f"Error: Map file not found: {map_path}")
        return 1
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Parsing map: {map_path.name}")
    ra3map = Ra3Map(str(map_path))
    ra3map.parse()
    context = ra3map.get_context()
    
    # Generate visualizations
    map_name = map_path.stem
    results = MapVisualizer.visualize_map(context, str(output_dir), map_name)
    
    print(f"\nGenerated images:")
    for key, path in results.items():
        print(f"  {key}: {path}")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())

