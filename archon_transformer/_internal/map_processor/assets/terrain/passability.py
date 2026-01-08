"""
Passability enum
Based on Passability.cs
"""
from enum import IntEnum


class Passability(IntEnum):
    """
    Terrain passability types.
    Based on Passability enum in Passability.cs
    """
    Passable = 0
    Impassable = 1
    ImpassableToPlayers = 2
    ImpassableToAirUnits = 3
    ExtraPassable = 4

