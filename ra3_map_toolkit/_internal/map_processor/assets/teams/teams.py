"""
Teams asset
Based on Teams.cs
"""
import struct
from typing import BinaryIO, List, TYPE_CHECKING

from ...core.major_asset import MajorAsset
from ...utils.constants import ASSET_Teams
from ..teams.team import Team

if TYPE_CHECKING:
    from ...core.ra3map_struct import MapDataContext


class Teams(MajorAsset):
    """
    Teams list asset.
    Based on Teams.cs
    """
    
    def __init__(self):
        super().__init__()
        self.teams: List[Team] = []
    
    def get_asset_name(self) -> str:
        return ASSET_Teams
    
    def parse_data(self, br: BinaryIO, context: 'MapDataContext') -> None:
        """
        Parse teams data.
        Based on parseData in Teams.cs
        """
        import struct
        count = struct.unpack('<i', br.read(4))[0]
        self.teams = []
        for i in range(count):
            team = Team()
            team.from_stream(br, context)
            self.teams.append(team)
    
    def save_data(self, bw: BinaryIO, context: 'MapDataContext') -> None:
        """
        Save teams data.
        Based on saveData in Teams.cs
        """
        import struct
        bw.write(struct.pack('<i', len(self.teams)))
        for team in self.teams:
            team.save_data(bw, context)

