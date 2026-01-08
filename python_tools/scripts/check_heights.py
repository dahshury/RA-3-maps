"""Quick check of height map values."""
import sys
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from map_processor.parsing.map_parser import Ra3MapParser
from map_processor.assets.terrain.height_map_data import HeightMapData

map_path = Path(__file__).parent.parent.parent / "RA3 Official maps" / "2 II" / "map_mp_2_rao1.map"

parser = Ra3MapParser()
context = parser.parse(str(map_path))

height_data = context.get_asset_by_type(HeightMapData)

if height_data:
    print(f"HeightMapData found:")
    print(f"  map_width: {height_data.map_width}")
    print(f"  map_height: {height_data.map_height}")
    print(f"  elevations type: {type(height_data.elevations)}")
    
    if height_data.elevations is not None:
        arr = np.array(height_data.elevations)
        print(f"  elevations shape: {arr.shape}")
        print(f"  elevations dtype: {arr.dtype}")
        print(f"  Sample values [0:5, 0:5]:")
        print(arr[0:5, 0:5])
        print(f"  Non-zero count: {np.count_nonzero(arr)}")
        print(f"  Min: {arr.min()}, Max: {arr.max()}")
        
        # Check if all zeros
        if arr.max() == 0 and arr.min() == 0:
            print("\n  WARNING: All heights are 0!")
            print("  This might be a parsing issue or a truly flat map.")









