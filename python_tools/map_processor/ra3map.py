"""
Convenience re-export module for the public API.

Some scripts/tests import `map_processor.ra3map.Ra3Map`; the actual implementation
lives in `map_processor.core.ra3map`.
"""

from .core.ra3map import Ra3Map

__all__ = ["Ra3Map"]










