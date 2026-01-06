"""
GlobalLightingConfiguration class
Based on GlobalLightingConfiguration.cs
"""
from typing import BinaryIO, TYPE_CHECKING

from .global_light import GlobalLight

if TYPE_CHECKING:
    from ...core.ra3map_struct import MapDataContext


class GlobalLightingConfiguration:
    """
    Global lighting configuration (NOT a MajorAsset, used within GlobalLighting).
    Based on GlobalLightingConfiguration.cs
    """
    
    def __init__(self):
        self.terrain_sun: GlobalLight = None
        self.terrain_accent1: GlobalLight = None
        self.terrain_accent2: GlobalLight = None
    
    def from_stream(self, br: BinaryIO, context: 'MapDataContext') -> 'GlobalLightingConfiguration':
        """
        Parse global lighting configuration from stream.
        Based on fromStream in GlobalLightingConfiguration.cs
        """
        self.terrain_sun = GlobalLight()
        self.terrain_sun.from_stream(br, context)
        self.terrain_accent1 = GlobalLight()
        self.terrain_accent1.from_stream(br, context)
        self.terrain_accent2 = GlobalLight()
        self.terrain_accent2.from_stream(br, context)
        return self
    
    def save_data(self, bw: BinaryIO, context: 'MapDataContext') -> None:
        """
        Save global lighting configuration data.
        Based on saveData in GlobalLightingConfiguration.cs
        """
        self.terrain_sun.save_data(bw, context)
        self.terrain_accent1.save_data(bw, context)
        self.terrain_accent2.save_data(bw, context)

