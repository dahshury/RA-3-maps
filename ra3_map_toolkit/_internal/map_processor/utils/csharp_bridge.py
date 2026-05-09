"""
C# Bridge - Interface to C# MapCoreLib for parsing and reconstructing maps
"""
import subprocess
import json
import os
from pathlib import Path
from typing import Optional, Dict, Any


class CSharpMapProcessor:
    """
    Interface to C# MapCoreLib for parsing and reconstructing RA3 maps.
    
    This class provides a Python interface to the C# map processing code.
    It expects a C# CLI tool to be available for map operations.
    """
    
    def __init__(self, csharp_tool_path: Optional[str] = None):
        """
        Initialize the C# bridge.
        
        Args:
            csharp_tool_path: Path to C# CLI tool executable. If None, will look for
                             'MapProcessor.exe' in the project root.
        """
        if csharp_tool_path is None:
            # Look for C# tool in parent directories
            current_dir = Path(__file__).parent.parent.parent.parent
            possible_paths = [
                current_dir / "MapProcessor" / "bin" / "Release" / "MapProcessor.exe",
                current_dir / "Ra3Solution" / "MapProcessor" / "bin" / "Release" / "MapProcessor.exe",
            ]
            
            for path in possible_paths:
                if path.exists():
                    csharp_tool_path = str(path)
                    break
            
            if csharp_tool_path is None:
                raise FileNotFoundError(
                    "C# MapProcessor tool not found. Please build the C# CLI tool first."
                )
        
        self.csharp_tool_path = csharp_tool_path
    
    def parse_map_to_json(self, map_path: str, output_json_path: str) -> Dict[str, Any]:
        """
        Parse a RA3 map file to JSON format.
        
        Args:
            map_path: Path to the .map file
            output_json_path: Path to save the JSON output
            
        Returns:
            Dictionary containing the parsed map data
        """
        cmd = [
            self.csharp_tool_path,
            "parse",
            "--input", map_path,
            "--output", output_json_path
        ]
        
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True
        )
        
        # Load the JSON output
        with open(output_json_path, 'r') as f:
            return json.load(f)
    
    def reconstruct_map_from_json(self, json_path: str, output_map_path: str) -> bool:
        """
        Reconstruct a RA3 map file from JSON format.
        
        Args:
            json_path: Path to the JSON file
            output_map_path: Path to save the reconstructed .map file
            
        Returns:
            True if successful
        """
        cmd = [
            self.csharp_tool_path,
            "reconstruct",
            "--input", json_path,
            "--output", output_map_path
        ]
        
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True
        )
        
        return os.path.exists(output_map_path)
    
    def parse_map_to_json_direct(self, map_path: str) -> Dict[str, Any]:
        """
        Parse a map and return JSON data directly (uses temp file).
        
        Args:
            map_path: Path to the .map file
            
        Returns:
            Dictionary containing the parsed map data
        """
        import tempfile
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as tmp:
            tmp_path = tmp.name
        
        try:
            return self.parse_map_to_json(map_path, tmp_path)
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)


class MapProcessorFallback:
    """
    Fallback processor that uses file-based approach when C# tool is not available.
    This is a placeholder for when we implement direct Python parsing.
    """
    
    def __init__(self):
        self.available = False
    
    def parse_map_to_json(self, map_path: str, output_json_path: str) -> Dict[str, Any]:
        raise NotImplementedError("Direct Python parsing not yet implemented. Use C# tool.")
    
    def reconstruct_map_from_json(self, json_path: str, output_map_path: str) -> bool:
        raise NotImplementedError("Direct Python reconstruction not yet implemented. Use C# tool.")

