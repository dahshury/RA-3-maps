"""
SidesList asset
Based on SidesList.cs
"""
import struct
from typing import BinaryIO, List, TYPE_CHECKING

from ...core.major_asset import MajorAsset
from ...utils.constants import ASSET_SidesList
from ..sides.player import Player

if TYPE_CHECKING:
    from ...core.ra3map_struct import MapDataContext


class SidesList(MajorAsset):
    """
    List of players/sides.
    Based on SidesList.cs
    """
    
    def __init__(self):
        super().__init__()
        self.players: List[Player] = []
    
    def get_asset_name(self) -> str:
        return ASSET_SidesList
    
    def get_version(self) -> int:
        return 6
    
    def from_stream(self, br: BinaryIO, context: 'MapDataContext') -> 'SidesList':
        """
        Parse sides list from stream.
        Based on fromStream in SidesList.cs
        """
        # Call base to read header
        super().from_stream(br, context)
        
        # Read byte (must be 1)
        br.read(1)  # Skip the byte
        
        # Read player count
        player_count = struct.unpack('<i', br.read(4))[0]
        self.players = []
        for i in range(player_count):
            player = Player()
            player.from_stream(br, context)
            self.players.append(player)
        
        return self
    
    def parse_data(self, br: BinaryIO, context: 'MapDataContext') -> None:
        """Not used - parsing handled in from_stream override"""
        pass
    
    def save_data(self, bw: BinaryIO, context: 'MapDataContext') -> None:
        """
        Save sides list data.
        Based on saveData in SidesList.cs
        """
        bw.write(struct.pack('B', 1))  # Write byte (must be 1)
        bw.write(struct.pack('<i', len(self.players)))
        for player in self.players:
            player.save_data(bw, context)

