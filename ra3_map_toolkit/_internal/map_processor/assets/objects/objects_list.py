"""
ObjectsList asset
Based on ObjectsList.cs
"""
import struct
from typing import BinaryIO, List, TYPE_CHECKING

from ...core.major_asset import MajorAsset
from ...utils.constants import ASSET_ObjectsList
from ..objects.map_object import MapObject

if TYPE_CHECKING:
    from ...core.ra3map_struct import MapDataContext


class ObjectsList(MajorAsset):
    """
    List of map objects.
    Based on ObjectsList.cs
    """
    
    def __init__(self):
        super().__init__()
        self.map_objects: List[MapObject] = []
    
    def get_asset_name(self) -> str:
        return ASSET_ObjectsList
    
    def get_version(self) -> int:
        return 3
    
    def from_stream(self, br: BinaryIO, context: 'MapDataContext') -> 'ObjectsList':
        """
        Parse objects list from stream.
        Based on fromStream in ObjectsList.cs
        """
        # Call base to read header (sets self.data_start_pos)
        super().from_stream(br, context)
        
        # Check if we should filter objects by category
        from ...parsing.parser_config import ParserConfig
        config = getattr(context, '_parser_config', None)
        
        def _is_category_included(category_key: str, included: set) -> bool:
            """
            Training/visualization filtering supports top-level categories:
            - Including `garrison` implies all `garrison_*` categories
            - Including `building` implies all `building_*` categories
            """
            if not category_key:
                return False
            if category_key in included:
                return True
            if category_key.startswith("garrison_") and "garrison" in included:
                return True
            if category_key.startswith("building_") and "building" in included:
                return True
            return False

        # Read objects until we've consumed all data
        # data_start_pos is set by base.from_stream() - use self.data_start_pos
        while br.tell() - self.data_start_pos < self.data_size:
            obj = MapObject()
            obj.from_stream(br, context)
            
            # Filter by object category if config is set
            if config and config.included_object_categories is not None:
                from ...utils.object_categories import ObjectCategoryConfig
                category_config = ObjectCategoryConfig()
                
                # Check if this is a player start (by unique_id) - include it if player_start is in included categories
                is_player_start = False
                if hasattr(obj, 'unique_id') and obj.unique_id:
                    player_start_ids = {'Player_1_Start', 'Player_2_Start', 'Player_3_Start', 
                                       'Player_4_Start', 'Player_5_Start', 'Player_6_Start'}
                    if obj.unique_id in player_start_ids:
                        is_player_start = True
                
                if is_player_start:
                    # Include player starts if 'player_start' is in included categories
                    if 'player_start' in config.included_object_categories:
                        self.map_objects.append(obj)
                    # Otherwise exclude
                else:
                    category, should_include = category_config.get_category_for_object(obj.type_name)
                    
                    # Find the category key (e.g., 'ore_node', 'oil_derrick')
                    category_key = None
                    if category:
                        # Find the key for this category
                        for key, cat in category_config.get_all_categories().items():
                            if cat.name == category.name:
                                category_key = key
                                break
                    
                    # Check if category key is in included list
                    if category_key and _is_category_included(category_key, config.included_object_categories):
                        self.map_objects.append(obj)
                    elif not category and should_include:
                        # No category matched but should be included - include it
                        self.map_objects.append(obj)
                    # Otherwise exclude
            elif config and config.excluded_object_categories is not None:
                from ...utils.object_categories import ObjectCategoryConfig
                category_config = ObjectCategoryConfig()
                category, should_include = category_config.get_category_for_object(obj.type_name)
                
                # Find the category key
                category_key = None
                if category:
                    for key, cat in category_config.get_all_categories().items():
                        if cat.name == category.name:
                            category_key = key
                            break
                
                # Check if category key is in excluded list
                excluded = config.excluded_object_categories
                # Support top-level excludes:
                # - Excluding `garrison` excludes all `garrison_*`
                # - Excluding `building` excludes all `building_*`
                is_excluded = False
                if category_key and category_key in excluded:
                    is_excluded = True
                elif category_key and category_key.startswith("garrison_") and "garrison" in excluded:
                    is_excluded = True
                elif category_key and category_key.startswith("building_") and "building" in excluded:
                    is_excluded = True

                if is_excluded:
                    # Exclude this object
                    continue
                else:
                    self.map_objects.append(obj)
            else:
                # No filtering - include all
                self.map_objects.append(obj)
        
        return self
    
    def parse_data(self, br: BinaryIO, context: 'MapDataContext') -> None:
        """Not used - parsing handled in from_stream override"""
        pass
    
    def save_data(self, bw: BinaryIO, context: 'MapDataContext') -> None:
        """
        Save objects list data.
        Based on saveData in ObjectsList.cs
        """
        for obj in self.map_objects:
            obj.save(bw, context)

