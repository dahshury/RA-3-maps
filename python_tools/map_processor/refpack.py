"""
Backward-compatible re-export.

Some tests/scripts import `map_processor.refpack.RefPackDecompressor`, while the
implementation lives in `map_processor.utils.refpack`.
"""

from .utils.refpack import RefPackDecompressor

__all__ = ["RefPackDecompressor"]










