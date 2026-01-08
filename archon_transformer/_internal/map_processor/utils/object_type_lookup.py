"""
Exact object type lookup system - no string matching, only exact matches
Based on comprehensive scan of all RA3 maps
"""
import json
from pathlib import Path
from typing import Dict, Set, Optional, Tuple
from dataclasses import dataclass

from .object_categories import ObjectCategory


@dataclass
class ObjectTypeMapping:
    """Mapping from exact object type name to category"""
    type_name: str
    category_key: str
    category_name: str
    color: Tuple[int, int, int]
    size: int


class ObjectTypeLookup:
    """
    Exact-match object type categorization system.
    Uses comprehensive lists extracted from all RA3 maps.
    """
    
    def __init__(self):
        self.type_to_category: Dict[str, ObjectTypeMapping] = {}
        self._load_categorized_types()

    @staticmethod
    def _should_ignore_type(type_name: str) -> bool:
        """
        Filter out known non-gameplay object types that frequently appear in maps.
        These often get mis-bucketed during naive categorization (e.g. ambient audio emitters).
        """
        if not type_name:
            return True
        t = type_name.strip().lower()

        # Waypoints/markers are not gameplay structures.
        if t.startswith("*waypoint") or t.startswith("*waypoints"):
            return True

        # Ambient sound emitters / streaming audio sources.
        if t.startswith("amb_") or t.startswith("ambstream_") or t.startswith("ambstream"):
            return True

        # Common ambient keywords (keep conservative; this is only for the exact-match lookup list).
        ambient_tokens = [
            "birds",
            "wind",
            "stereo",
            "chime",
            "churchbell",
            "creak",
            "stream",
            "music",
            "announcement",
        ]
        if any(tok in t for tok in ambient_tokens):
            return True

        return False
    
    def _load_categorized_types(self):
        """Load categorized object types from JSON file"""
        json_path = Path(__file__).parent.parent.parent / 'categorized_object_types.json'
        
        if not json_path.exists():
            # Fallback: create empty lookup
            return
        
        with open(json_path, 'r') as f:
            categorized = json.load(f)
        
        # Define category colors and sizes
        category_configs = {
            'ore_nodes': {
                'category_key': 'ore_node',
                'category_name': 'OreNode',
                'color': (255, 140, 0),
                'size': 6
            },
            'oil_derricks': {
                'category_key': 'oil_derrick',
                'category_name': 'OilDerrick',
                'color': (255, 255, 0),
                'size': 8
            },
            'garrisons': {
                'category_key': 'garrison',
                'category_name': 'Garrison',
                'color': (200, 150, 100),
                'size': 7
            },
            'buildings': {
                'category_key': 'building',
                'category_name': 'Building',
                'color': (150, 150, 255),
                'size': 6
            }
        }
        
        # Create mappings for each category (with light sanitization).
        for category_type, object_list in categorized.items():
            if category_type in category_configs:
                config = category_configs[category_type]
                for obj_type in object_list:
                    if self._should_ignore_type(obj_type):
                        continue
                    self.type_to_category[obj_type] = ObjectTypeMapping(
                        type_name=obj_type,
                        category_key=config['category_key'],
                        category_name=config['category_name'],
                        color=config['color'],
                        size=config['size']
                    )
    
    def get_category_for_object(self, type_name: str) -> Optional[ObjectTypeMapping]:
        """
        Get category for an object type using exact match.
        
        Args:
            type_name: Exact object type name
            
        Returns:
            ObjectTypeMapping if found, None otherwise
        """
        return self.type_to_category.get(type_name)
    
    def is_known_type(self, type_name: str) -> bool:
        """Check if object type is in our lookup"""
        return type_name in self.type_to_category


