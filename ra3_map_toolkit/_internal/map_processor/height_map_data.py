"""
Convenience re-export module for HeightMapData.

Some scripts/tests import `map_processor.height_map_data.HeightMapData`; the actual
implementation lives in `map_processor.assets.terrain.height_map_data`.
"""

from .assets.terrain.height_map_data import HeightMapData, HeightMapBorder

__all__ = ["HeightMapData", "HeightMapBorder"]









