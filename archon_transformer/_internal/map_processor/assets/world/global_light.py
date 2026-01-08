"""
GlobalLight class
Based on GlobalLight.cs
"""
import struct
from typing import BinaryIO, Tuple, TYPE_CHECKING

from ...utils.binary_utils import BinaryUtils

if TYPE_CHECKING:
    from ...core.ra3map_struct import MapDataContext


class GlobalLight:
    """
    Global light information (not a MajorAsset, used within GlobalLighting).
    Based on GlobalLight.cs
    """
    
    def __init__(self):
        self.direction: Tuple[float, float, float] = (0.0, 0.0, 0.0)
        self.color: Tuple[float, float, float] = (0.0, 0.0, 0.0)
        self.ambient_color: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    
    def from_stream(self, br: BinaryIO, context: 'MapDataContext') -> 'GlobalLight':
        """
        Parse global light from stream.
        Based on fromStream in GlobalLight.cs
        """
        self.direction = BinaryUtils.read_vec3d(br)
        self.color = BinaryUtils.read_vec3d(br)
        self.ambient_color = BinaryUtils.read_vec3d(br)
        return self
    
    def save_data(self, bw: BinaryIO, context: 'MapDataContext') -> None:
        """
        Save global light data.
        Based on saveData in GlobalLight.cs
        """
        BinaryUtils.write_vec3d(bw, self.direction[0], self.direction[1], self.direction[2])
        BinaryUtils.write_vec3d(bw, self.color[0], self.color[1], self.color[2])
        BinaryUtils.write_vec3d(bw, self.ambient_color[0], self.ambient_color[1], self.ambient_color[2])

