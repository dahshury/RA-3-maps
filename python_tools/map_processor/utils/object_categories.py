"""
Object Categories for RA3 Map Visualization
Categorizes all building and structure types for visualization with enable/disable capability
"""
from typing import Dict, List, Tuple, Set
from dataclasses import dataclass, field


@dataclass
class ObjectCategory:
    """Category configuration for object types"""
    name: str
    keywords: List[str]  # Keywords to match in object type names
    color: Tuple[int, int, int]  # RGB color for visualization
    size: int  # Size of marker
    enabled: bool = True  # Whether this category is enabled
    description: str = ""  # Description of the category


class ObjectCategoryConfig:
    """
    Comprehensive categorization of RA3 object types.
    Based on building types found in MapCreatorCore and game data.
    """
    
    # Faction prefixes
    ALLIED_PREFIXES = ['allied']
    SOVIET_PREFIXES = ['soviet']
    JAPAN_PREFIXES = ['japan', 'empire']
    CIVILIAN_PREFIXES = ['civilian']
    
    # Decorative/ambient prefixes to exclude
    DECORATIVE_PREFIXES = ['cc_', 'yu_', 'cs_', 'il_', 'hv_', 'amb_', 'yucatan', 'gc_', 'my_', 'sa_', 'mj_', 'th_']
    
    # Decorative object keywords to exclude (even if they match building keywords).
    # Note: 'road', 'street', 'path', 'bridge' have been REMOVED from this list so the
    # dedicated 'road' category below can match them. They are still skipped from being
    # treated as buildings/garrisons because the 'road' category catches them first.
    DECORATIVE_KEYWORDS = ['idol', 'statue', 'rock', 'bush', 'tree', 'palm', 'coral',
                           'grass', 'cliff', 'wall', 'fence', 'bench', 'table', 'cart',
                           'sign', 'fountain', 'umbrella', 'towel', 'board', 'auto',
                           'ship', 'sunken', 'bamboo', 'beach', 'lamp', 'lamppost',
                           'decoration', 'deco',
                           'office', 'drum']  # Office buildings and oil drums are decorative, not gameplay objects
    
    # Waypoint/marker prefixes to exclude
    MARKER_PREFIXES = ['*waypoint', '*waypoints']
    
    def __init__(self):
        self.categories: Dict[str, ObjectCategory] = {}
        self._initialize_categories()
    
    def _initialize_categories(self):
        """Initialize all object categories"""
        
        # Economic/Resource Buildings
        self.categories['ore_node'] = ObjectCategory(
            name='OreNode',
            keywords=['ore', 'node', 'refinery'],  # Include refinery as ore node
            color=(255, 140, 0),  # Dark orange
            size=6,
            enabled=True,
            description='Ore mining nodes and refineries'
        )
        
        self.categories['oil_derrick'] = ObjectCategory(
            name='OilDerrick',
            keywords=['derrick'],  # Only match 'derrick', not 'oil' (to exclude decorative oil drums)
            color=(255, 255, 0),  # Bright yellow
            size=8,
            enabled=True,
            description='Oil derricks (excludes decorative oil drums)'
        )

        # Roads / paths / bridges. Drawn small + light so they trace road shapes
        # without dominating the render. Critical for skin-debug: when objects
        # are stripped, roads visibly disappear.
        self.categories['road'] = ObjectCategory(
            name='Road',
            keywords=['road', 'street', 'path', 'bridge'],
            color=(240, 240, 240),  # Near-white so roads are visible on any biome
            size=3,  # small but readable
            enabled=True,
            description='Road / path / bridge segments (3D meshes placed on tiles)'
        )
        
        # Military Buildings - Construction
        self.categories['construction_yard'] = ObjectCategory(
            name='ConstructionYard',
            keywords=['construction', 'yard'],
            color=(0, 255, 0),  # Green
            size=10,
            enabled=True,
            description='Construction yards'
        )
        
        # Military Buildings - Production
        self.categories['barracks'] = ObjectCategory(
            name='Barracks',
            keywords=['barracks'],
            color=(0, 200, 255),  # Cyan
            size=8,
            enabled=True,
            description='Barracks'
        )
        
        self.categories['war_factory'] = ObjectCategory(
            name='WarFactory',
            keywords=['war', 'factory'],
            color=(255, 100, 100),  # Light red
            size=9,
            enabled=True,
            description='War factories'
        )
        
        self.categories['factory'] = ObjectCategory(
            name='Factory',
            keywords=['factory'],
            color=(255, 0, 255),  # Magenta
            size=9,
            enabled=True,
            description='Factories (general)'
        )
        
        self.categories['airfield'] = ObjectCategory(
            name='Airfield',
            keywords=['airfield'],
            color=(100, 150, 255),  # Light blue
            size=8,
            enabled=True,
            description='Airfields'
        )
        
        self.categories['naval_yard'] = ObjectCategory(
            name='NavalYard',
            keywords=['naval', 'yard'],
            color=(0, 150, 255),  # Blue
            size=8,
            enabled=True,
            description='Naval yards (note: "yard" alone matches construction yards first)'
        )
        
        self.categories['laser_tower'] = ObjectCategory(
            name='LaserTower',
            keywords=['laser', 'tower'],
            color=(255, 150, 0),  # Orange
            size=7,
            enabled=True,
            description='Laser towers'
        )
        
        # Military Buildings - Power
        self.categories['power_plant'] = ObjectCategory(
            name='PowerPlant',
            # IMPORTANT: don't match plain 'plant' (decorative vegetation like ST_PLANT09 would false-positive).
            # Power plants in RA3 type names generally contain a combined token like PowerPlant/Power_Plant.
            keywords=['powerplant', 'power_plant'],
            color=(255, 200, 0),  # Gold
            size=7,
            enabled=True,
            description='Power plants'
        )
        
        # Military Buildings - Defense
        self.categories['base_defense'] = ObjectCategory(
            name='BaseDefense',
            keywords=['defense', 'defence'],
            color=(200, 0, 200),  # Purple
            size=7,
            enabled=True,
            description='Base defense structures'
        )
        
        self.categories['tower'] = ObjectCategory(
            name='Tower',
            keywords=['tower'],
            color=(150, 100, 200),  # Lavender
            size=6,
            enabled=True,
            description='Defense towers'
        )
        
        self.categories['bunker'] = ObjectCategory(
            name='Bunker',
            keywords=['bunker'],
            color=(100, 50, 150),  # Dark purple
            size=7,
            enabled=True,
            description='Bunkers'
        )
        
        # Super Weapons
        self.categories['super_weapon'] = ObjectCategory(
            name='SuperWeapon',
            keywords=['super', 'weapon'],
            color=(255, 0, 100),  # Pink-red
            size=12,
            enabled=True,
            description='Super weapons'
        )
        
        # Specific Garrison Types
        self.categories['garrison_tikihut'] = ObjectCategory(
            name='TikiHut',
            keywords=['tikihut', 'tiki_hut'],
            color=(200, 150, 100),  # Brown/tan
            size=7,
            enabled=True,
            description='Tiki huts (garrisonable)'
        )
        
        self.categories['garrison_house'] = ObjectCategory(
            name='House',
            keywords=['house'],
            color=(180, 140, 90),  # Darker brown/tan
            size=7,
            enabled=True,
            description='Houses (garrisonable)'
        )
        
        self.categories['garrison_warehouse'] = ObjectCategory(
            name='Warehouse',
            keywords=['warehouse'],  # Must match 'warehouse' exactly (not just 'house')
            color=(160, 130, 80),  # Even darker brown
            size=7,
            enabled=True,
            description='Warehouses (garrisonable)'
        )
        
        self.categories['garrison_house'] = ObjectCategory(
            name='House',
            keywords=['house'],  # Match 'house' but not if it's part of 'warehouse'
            color=(180, 140, 90),  # Darker brown/tan
            size=7,
            enabled=True,
            description='Houses (garrisonable)'
        )
        
        self.categories['garrison_other'] = ObjectCategory(
            name='GarrisonOther',
            keywords=['church', 'restaurant', 'shop', 'store', 'villa', 
                     'mansion', 'shack', 'dwelling', 'habitation', 'residential',
                     'civilian',
                     # Common garrisonable building words seen across official maps
                     'hotel', 'apartment', 'townhouse', 'guardhouse', 'lighthouse',
                     # Some maps use small civilian structures with these names
                     'cabin', 'tent',
                     # Often garrisonable / strategic, but would otherwise match the generic `tower` category
                     'watchtower', 'watch_tower', 'guardtower', 'guard_tower',
                     # Castle/clock/stag towers show up as strategic garrison-style structures in official maps
                     'castletower', 'castle_tower', 'clocktower', 'stag', 'stagtower',
                     # Additional tower types that appear in official maps and should be treated as gameplay markers
                     'od_tower', 'gi_tower'],  # Other civilian buildings
            color=(190, 145, 95),  # Medium brown/tan
            size=7,
            enabled=True,
            description='Other garrisonable buildings'
        )
        
        # Specific Building Types
        self.categories['building_observation_post'] = ObjectCategory(
            name='ObservationPost',
            keywords=['observationpost', 'observation_post'],
            color=(150, 150, 255),  # Light blue
            size=6,
            enabled=True,
            description='Observation posts'
        )
        
        self.categories['building_hospital'] = ObjectCategory(
            name='Hospital',
            keywords=['hospital'],
            color=(100, 200, 255),  # Light cyan
            size=6,
            enabled=True,
            description='Hospitals'
        )
        
        self.categories['building_garage'] = ObjectCategory(
            name='Garage',
            keywords=['garage'],
            color=(120, 170, 255),  # Light blue-purple
            size=6,
            enabled=True,
            description='Garages'
        )
        
        self.categories['building_snowy'] = ObjectCategory(
            name='SnowyBuilding',
            keywords=['snowy'],
            color=(170, 200, 255),  # Light blue-white
            size=6,
            enabled=True,
            description='Snowy buildings'
        )
        
        self.categories['building_convention_center'] = ObjectCategory(
            name='ConventionCenter',
            keywords=['convention', 'center'],
            color=(140, 150, 255),  # Medium blue-purple
            size=6,
            enabled=True,
            description='Convention centers'
        )
        
        self.categories['building_port_structure'] = ObjectCategory(
            name='PortStructure',
            keywords=['port', 'structure'],
            color=(110, 180, 255),  # Light blue
            size=6,
            enabled=True,
            description='Port structures'
        )
        
        self.categories['building_airport'] = ObjectCategory(
            name='Airport',
            keywords=['airport'],
            color=(90, 190, 255),  # Cyan-blue
            size=6,
            enabled=True,
            description='Airport structures'
        )
        
        self.categories['building_military'] = ObjectCategory(
            name='MilitaryBuilding',
            keywords=['military'],
            color=(100, 100, 255),  # Blue
            size=6,
            enabled=True,
            description='Military buildings'
        )
        
        self.categories['building_cargo_container'] = ObjectCategory(
            name='CargoContainer',
            keywords=['cargo', 'container'],
            color=(150, 140, 255),  # Purple-blue
            size=6,
            enabled=True,
            description='Cargo containers'
        )
        
        self.categories['building_supply'] = ObjectCategory(
            name='SupplyBuilding',
            keywords=['supply'],
            color=(160, 130, 255),  # Purple
            size=6,
            enabled=True,
            description='Supply buildings'
        )
        
        self.categories['building_veterancy'] = ObjectCategory(
            name='VeterancyStructure',
            keywords=['veterancy'],
            color=(170, 120, 255),  # Purple-pink
            size=6,
            enabled=True,
            description='Veterancy structures'
        )
        
        self.categories['building_shipyard'] = ObjectCategory(
            name='Shipyard',
            keywords=['shipyard', 'ship'],
            color=(80, 200, 255),  # Cyan
            size=6,
            enabled=True,
            description='Shipyard structures'
        )
        
        self.categories['building_tech_structure'] = ObjectCategory(
            name='TechStructure',
            keywords=['tech'],
            color=(120, 160, 255),  # Light blue
            size=6,
            enabled=True,
            description='Tech structures'
        )
        
        self.categories['building_soviet'] = ObjectCategory(
            name='SovietBuilding',
            keywords=['sv_', 'st_'],
            color=(100, 150, 200),  # Blue-gray
            size=6,
            enabled=True,
            description='Soviet buildings'
        )
        
        self.categories['building_other'] = ObjectCategory(
            name='BuildingOther',
            keywords=[
                'building', 'base', 'structure', 'command',
                # Map-specific strategic structures
                'easterislandheaddefense', 'headdefense',
            ],
            color=(130, 160, 255),  # Medium blue
            size=6,
            enabled=True,
            description='Other buildings and structures'
        )
        
        # Player Starts
        self.categories['player_start'] = ObjectCategory(
            name='PlayerStart',
            keywords=['player', 'start'],
            color=(255, 255, 255),  # White
            size=12,
            enabled=True,
            description='Player starting positions'
        )
    
    def get_category_for_object(self, type_name: str) -> Tuple[ObjectCategory, bool]:
        """
        Get the category for an object type name.
        Uses exact-match lookup first, then falls back to string matching.
        
        Returns:
            Tuple of (category, should_draw)
            should_draw is False if object should be skipped (decorative, waypoint, etc.)
        """
        # First, try exact-match lookup (no string matching)
        try:
            from .object_type_lookup import ObjectTypeLookup
            lookup = ObjectTypeLookup()
            mapping = lookup.get_category_for_object(type_name)
            if mapping:
                # Map the category_key to our internal category keys
                # The lookup returns generic keys like 'garrison' or 'building'
                # We need to map them to specific category keys
                if mapping.category_key == 'garrison':
                    # For garrisons, try to find the specific type
                    type_lower = type_name.lower()
                    if 'tikihut' in type_lower or 'tiki_hut' in type_lower:
                        if 'garrison_tikihut' in self.categories:
                            return self.categories['garrison_tikihut'], True
                    elif 'warehouse' in type_lower:
                        if 'garrison_warehouse' in self.categories:
                            return self.categories['garrison_warehouse'], True
                    elif 'house' in type_lower and 'warehouse' not in type_lower:
                        if 'garrison_house' in self.categories:
                            return self.categories['garrison_house'], True
                    # Fall back to garrison_other if specific type not found
                    if 'garrison_other' in self.categories:
                        return self.categories['garrison_other'], True
                elif mapping.category_key == 'building':
                    # For buildings, try to find the specific type
                    type_lower = type_name.lower()
                    # Check in priority order (most specific first)
                    if 'observation' in type_lower and 'post' in type_lower:
                        if 'building_observation_post' in self.categories:
                            return self.categories['building_observation_post'], True
                    elif 'hospital' in type_lower:
                        if 'building_hospital' in self.categories:
                            return self.categories['building_hospital'], True
                    elif 'garage' in type_lower:
                        if 'building_garage' in self.categories:
                            return self.categories['building_garage'], True
                    elif 'snowy' in type_lower:
                        if 'building_snowy' in self.categories:
                            return self.categories['building_snowy'], True
                    elif ('convention' in type_lower and 'center' in type_lower) or 'convention' in type_lower:
                        if 'building_convention_center' in self.categories:
                            return self.categories['building_convention_center'], True
                    elif 'port' in type_lower and 'structure' in type_lower:
                        if 'building_port_structure' in self.categories:
                            return self.categories['building_port_structure'], True
                    elif 'airport' in type_lower:
                        if 'building_airport' in self.categories:
                            return self.categories['building_airport'], True
                    elif 'military' in type_lower:
                        if 'building_military' in self.categories:
                            return self.categories['building_military'], True
                    elif 'cargo' in type_lower or ('container' in type_lower and 'cargo' in type_lower):
                        if 'building_cargo_container' in self.categories:
                            return self.categories['building_cargo_container'], True
                    elif 'supply' in type_lower:
                        if 'building_supply' in self.categories:
                            return self.categories['building_supply'], True
                    elif 'veterancy' in type_lower:
                        if 'building_veterancy' in self.categories:
                            return self.categories['building_veterancy'], True
                    elif 'shipyard' in type_lower or ('ship' in type_lower and 'yard' in type_lower):
                        if 'building_shipyard' in self.categories:
                            return self.categories['building_shipyard'], True
                    elif 'tech' in type_lower:
                        if 'building_tech_structure' in self.categories:
                            return self.categories['building_tech_structure'], True
                    elif type_lower.startswith('sv_') or type_lower.startswith('st_'):
                        if 'building_soviet' in self.categories:
                            return self.categories['building_soviet'], True
                    # Fall back to building_other
                    if 'building_other' in self.categories:
                        return self.categories['building_other'], True
                elif mapping.category_key == 'ore_node':
                    if 'ore_node' in self.categories:
                        return self.categories['ore_node'], True
                elif mapping.category_key == 'oil_derrick':
                    if 'oil_derrick' in self.categories:
                        return self.categories['oil_derrick'], True
        except Exception:
            pass  # Fall back to string matching if lookup fails
        
        type_name_lower = type_name.lower()

        # Explicitly exclude known decorative infrastructure towers that are not gameplay structures.
        # These otherwise match generic keywords like 'tower' and pollute the gameplay object layers.
        if any(tok in type_name_lower for tok in ['cabletower', 'cable_tower', 'powerline', 'power_line']):
            return None, False
        
        # Skip waypoints/markers first (these are never buildings)
        if any(type_name_lower.startswith(prefix) for prefix in self.MARKER_PREFIXES):
            return None, False
        
        # Check decorative prefixes FIRST - objects with decorative prefixes should be excluded
        # even if they match building keywords (e.g., TH_office_building01)
        if any(type_name_lower.startswith(prefix) for prefix in self.DECORATIVE_PREFIXES):
            # Exception: Allow garrisonable buildings with decorative prefixes (e.g., YU_TikiHut01)
            # Check if it's a garrison first (check all garrison category keywords)
            garrison_keywords = []
            for cat_key in ['garrison_tikihut', 'garrison_house', 'garrison_warehouse', 'garrison_other']:
                if cat_key in self.categories:
                    garrison_keywords.extend(self.categories[cat_key].keywords)
            # Exception: Allow road/path/bridge objects even with biome prefixes
            # (e.g., YucatanDirtRoad01, YU_RoadStraight)
            road_keywords = self.categories.get('road').keywords if 'road' in self.categories else []
            if any(keyword in type_name_lower for keyword in garrison_keywords):
                # It's a garrison, don't exclude yet - let it be categorized as garrison
                pass
            elif any(keyword in type_name_lower for keyword in road_keywords):
                # It's a road/path/bridge, don't exclude - let road category catch it
                pass
            else:
                # It's decorative, exclude it
                return None, False
        
        # Check for decorative keywords - these should never be categorized as buildings
        # This check happens early to prevent false positives
        if any(keyword in type_name_lower for keyword in self.DECORATIVE_KEYWORDS):
            # Exception: Allow 'hut' even if it contains decorative keywords (e.g., 'tikihut')
            # But exclude if it's clearly decorative (e.g., 'idol', 'statue', etc.)
            if 'hut' not in type_name_lower:
                return None, False

        # Special-case: some map objects use "Defense" in the name but are not base defenses in the RA3 sense
        # (e.g. EI_EasterIslandHeadDefense). Treat these as general buildings so they don't get filtered out.
        if 'headdefense' in type_name_lower:
            b = self.categories.get('building_other')
            if b and b.enabled:
                return b, True
        
        # Check categories in priority order (most specific first)
        # Order matters - check specific types before general ones
        # IMPORTANT: Check garrison BEFORE decorative filtering, as garrisonable buildings
        # may have decorative prefixes (e.g., YU_TikiHut01)
        priority_order = [
            'road',  # Roads/paths/bridges - check first so they don't get matched as buildings
            'ore_node', 'oil_derrick',  # Removed 'refinery' - now part of ore_node
            # Specific garrison types (check before war_factory to avoid "warehouse" matching "war")
            'garrison_tikihut', 'garrison_house', 'garrison_warehouse', 'garrison_other',
            'construction_yard', 'barracks', 'war_factory', 'factory', 
            'airfield', 'naval_yard', 'power_plant',
            'base_defense', 'tower', 'bunker',
            'super_weapon',
            # Specific building types (check before general, most specific first)
            'building_observation_post', 'building_hospital', 'building_garage', 'building_snowy',
            'building_convention_center', 'building_port_structure', 'building_airport',
            'building_military', 'building_cargo_container', 'building_supply',
            'building_veterancy', 'building_shipyard', 'building_tech_structure',
            'building_soviet', 'building_other',
            'player_start'
        ]
        
        for category_key in priority_order:
            category = self.categories[category_key]
            if not category.enabled:
                continue
            
            # Check if any keyword matches
            for keyword in category.keywords:
                if keyword in type_name_lower:
                    # Special handling for 'warehouse' - must match 'warehouse', not just 'house'
                    if keyword == 'warehouse':
                        if 'warehouse' in type_name_lower:
                            return category, True
                    # Special handling for 'house' - must match 'house' but not if it's part of 'warehouse'
                    elif keyword == 'house':
                        if 'house' in type_name_lower and 'warehouse' not in type_name_lower:
                            return category, True
                    # Special handling for 'tikihut' - must contain both 'tiki' and 'hut'
                    elif keyword == 'tikihut':
                        if 'tiki' in type_name_lower and 'hut' in type_name_lower:
                            return category, True
                    # Special handling for observation posts
                    elif keyword in ['observationpost', 'observation_post']:
                        # Match 'observationpost' or 'observation_post' but exclude decorative posts
                        if 'observation' in type_name_lower and 'post' in type_name_lower:
                            # Exclude if it's a decorative post (lamp, street, etc.)
                            if any(deco in type_name_lower for deco in ['lamp', 'street', 'road', 'path']):
                                continue  # Skip this match, try next category
                            return category, True
                    else:
                        return category, True
        
        # No category matched
        # Note: Decorative prefix check already happened earlier, so we don't need to check again
        return None, False
    
    def enable_category(self, category_name: str, enabled: bool = True):
        """Enable or disable a category by name"""
        if category_name in self.categories:
            self.categories[category_name].enabled = enabled
    
    def enable_all_categories(self, enabled: bool = True):
        """Enable or disable all categories"""
        for category in self.categories.values():
            category.enabled = enabled
    
    def get_all_categories(self) -> Dict[str, ObjectCategory]:
        """Get all categories"""
        return self.categories.copy()
    
    def get_enabled_categories(self) -> Dict[str, ObjectCategory]:
        """Get only enabled categories"""
        return {k: v for k, v in self.categories.items() if v.enabled}
    
    def list_categories(self) -> List[str]:
        """Get list of all category names"""
        return list(self.categories.keys())

