# AI Training Data Catalog - RA3 Map Generation

This document catalogs all available data types and object categories that can be extracted from RA3 maps for AI training purposes. Use this to decide what to include/exclude in your training pipeline.

---

## 📊 Table 1: Object Categories

| Category Name | Category Key | Keywords | Color (RGB) | Size | Description | **Training Status** |
|---------------|--------------|----------|------------|------|-------------|---------------------|
| **OreNode** | `ore_node` | `ore`, `node` | (255, 140, 0) | 6 | Ore mining nodes/refineries | ✅ **INCLUDE** |
| **OilDerrick** | `oil_derrick` | `derrick`, `oil` | (255, 255, 0) | 8 | Oil derricks | ✅ **INCLUDE** |
| **Garrison** | `garrison` | `hut`, `house`, `church`, `restaurant`, `shop`, `store`, `villa`, `mansion`, `shack`, `dwelling`, `habitation`, `residential`, `tikihut`, `civilian` | (200, 150, 100) | 7 | Garrisonable civilian buildings (ALL TYPES) | ✅ **INCLUDE** |
| **Building** | `building` | `building`, `base`, `structure`, `command`, `post`, `tech` | (150, 150, 255) | 6 | Other buildings and structures (ALL TYPES including tech structures) | ✅ **INCLUDE** |
| **PlayerStart** | `player_start` | `player`, `start` | (255, 255, 255) | 12 | Player starting positions | ✅ **INCLUDE** |
| **Refinery** | `refinery` | `refinery` | (255, 0, 0) | 8 | Refineries | ❌ **EXCLUDE** |
| **ConstructionYard** | `construction_yard` | `construction`, `yard` | (0, 255, 0) | 10 | Construction yards | ❌ **EXCLUDE** |
| **Barracks** | `barracks` | `barracks` | (0, 200, 255) | 8 | Barracks | ❌ **EXCLUDE** |
| **WarFactory** | `war_factory` | `war`, `factory` | (255, 100, 100) | 9 | War factories | ❌ **EXCLUDE** |
| **Factory** | `factory` | `factory` | (255, 0, 255) | 9 | Factories (general) | ❌ **EXCLUDE** |
| **Airfield** | `airfield` | `airfield` | (100, 150, 255) | 8 | Airfields | ❌ **EXCLUDE** |
| **NavalYard** | `naval_yard` | `naval`, `yard` | (0, 150, 255) | 8 | Naval yards | ❌ **EXCLUDE** |
| **LaserTower** | `laser_tower` | `laser`, `tower` | (255, 150, 0) | 7 | Laser towers | ❌ **EXCLUDE** |
| **PowerPlant** | `power_plant` | `power`, `plant` | (255, 200, 0) | 7 | Power plants | ❌ **EXCLUDE** |
| **BaseDefense** | `base_defense` | `defense`, `defence` | (200, 0, 200) | 7 | Base defense structures | ❌ **EXCLUDE** |
| **Tower** | `tower` | `tower` | (150, 100, 200) | 6 | Defense towers | ❌ **EXCLUDE** |
| **Bunker** | `bunker` | `bunker` | (100, 50, 150) | 7 | Bunkers | ❌ **EXCLUDE** |
| **SuperWeapon** | `super_weapon` | `super`, `weapon` | (255, 0, 100) | 12 | Super weapons | ❌ **EXCLUDE** |

**Total Object Categories: 18**
**Included for Training: 5 categories** (OreNode, OilDerrick, Garrison, Building, PlayerStart)
**Excluded from Training: 13 categories**

---

## 📊 Table 2: Parsed Data Types (Major Assets)

| Asset Name | Asset Constant | Data Type | Structure | Description | **Training Status** |
|------------|----------------|-----------|-----------|-------------|---------------------|
| **HeightMapData** | `ASSET_HeightMapData` | `float[,]` | 2D array (width × height) | Terrain elevation values | ✅ **INCLUDE** |
| **BlendTileData** | `ASSET_BlendTileData` | Multiple arrays | Complex structure | Texture tiles, blend info, passability | ✅ **INCLUDE** |
| **ObjectsList** | `ASSET_ObjectsList` | `List<MapObject>` | List of objects | All placed objects on map | ✅ **INCLUDE** |
| **WorldInfo** | `ASSET_WorldInfo` | Structure | Metadata | Map metadata (name, description, etc.) | ✅ **INCLUDE** |
| **StandingWaterAreas** | `ASSET_StandingWaterAreas` | `List<StandingWaterArea>` | List of polygons | Standing water bodies | ✅ **INCLUDE** |
| **RiverAreas** | `ASSET_RiverAreas` | `List<RiverArea>` | List of polygons | River paths | ✅ **INCLUDE** |
| **StandingWaveAreas** | `ASSET_StandingWaveAreas` | `List<StandingWaveArea>` | List of polygons | Wave effects | ✅ **INCLUDE** |
| **SidesList** | `ASSET_SidesList` | `List<Player>` | List of players | Player/side definitions | ✅ **INCLUDE** |
| **MPPositionList** | `ASSET_MPPositionList` | `List<MPPositionInfo>` | List of positions | Multiplayer spawn positions | ✅ **INCLUDE** |
| **Teams** | `ASSET_Teams` | `List<Team>` | List of teams | Team definitions | ✅ **INCLUDE** |
| **BuildLists** | `ASSET_BuildLists` | `BuildList[]` | Build lists | Available buildings per player | ✅ **INCLUDE** |
| **GlobalWaterSettings** | `ASSET_GlobalWaterSettings` | Structure | Settings | Global water configuration | ✅ **INCLUDE** |
| **FogSettings** | `ASSET_FogSettings` | Structure | Settings | Fog configuration | ✅ **INCLUDE** |
| **NamedCameras** | `ASSET_NamedCameras` | Structure | Camera data | Camera positions | ✅ **INCLUDE** |
| **GlobalLighting** | `ASSET_GlobalLighting` | Structure | Lighting config | Global lighting settings | ✅ **INCLUDE** |
| **PostEffectsChunk** | `ASSET_PostEffectsChunk` | Structure | Effects data | Post-processing effects | ✅ **INCLUDE** |
| **EnvironmentData** | `ASSET_EnvironmentData` | Structure | Environment data | Environment settings | ✅ **INCLUDE** |
| **AssetList** | `ASSET_AssetList` | Structure | Asset metadata | Asset references | ✅ **INCLUDE** |
| **PlayerScriptsList** | `ASSET_PlayerScriptsList` | `ScriptList[]` | Script trees | Game logic scripts | ❌ **EXCLUDE** |
| **TriggerAreas** | `ASSET_TriggerAreas` | `List<TriggerArea>` | List of areas | Trigger zones | ❌ **EXCLUDE** |
| **MissionHotSpots** | `ASSET_MissionHotSpots` | `List<MissionHotSpot>` | List of spots | Mission objective locations | ❌ **EXCLUDE** |
| **MissionObjectives** | `ASSET_MissionObjectives` | `List<MissionObjective>` | List of objectives | Mission objectives | ❌ **EXCLUDE** |
| **LibraryMaps** | `ASSET_LibraryMaps` | Structure | Library data | Map library references | ❌ **EXCLUDE** |
| **LibraryMapLists** | `ASSET_LibraryMapLists` | Structure | Library lists | Map library lists | ❌ **EXCLUDE** |

**Total Major Assets: 24**
**Included for Training: 18 assets**
**Excluded from Training: 6 assets** (PlayerScriptsList, TriggerAreas, MissionHotSpots, MissionObjectives, LibraryMaps, LibraryMapLists)

---

## 📊 Table 3: BlendTileData Sub-Components

| Component | Data Type | Dimensions | Description | **Training Status** |
|-----------|-----------|------------|-------------|---------------------|
| **tiles** | `ushort[,]` | width × height | Texture tile indices | ✅ **INCLUDE** |
| **blend_infos** | `BlendInfo[,]` | width × height | Blend information | ✅ **INCLUDE** |
| **impassable** | `bool[,]` | width × height | Impassable terrain | ✅ **INCLUDE** |
| **impassable_to_players** | `bool[,]` | width × height | Player-impassable | ✅ **INCLUDE** |
| **extra_passable** | `bool[,]` | width × height | Extra passable areas | ✅ **INCLUDE** |
| **passage_width** | `bool[,]` | width × height | Passage width data | ✅ **INCLUDE** |
| **visibility** | `bool[,]` | width × height | Visibility map | ✅ **INCLUDE** |
| **buildability** | `bool[,]` | width × height | Buildable terrain | ✅ **INCLUDE** |
| **impassable_to_air_units** | `bool[,]` | width × height | Air-impassable | ✅ **INCLUDE** |
| **tiberium_growability** | `bool[,]` | width × height | Resource growth areas | ✅ **INCLUDE** |
| **textures** | `List<Texture>` | Variable | Texture definitions | ✅ **INCLUDE** |

**All BlendTileData components are INCLUDED for training**

---

## 📊 Table 4: MapObject Properties

| Property | Data Type | Description | **Training Status** |
|----------|-----------|-------------|---------------------|
| **typeName** | `string` | Object type identifier | ✅ **INCLUDE** |
| **position** | `Vec3D` (x, y, z) | 3D world coordinates | ✅ **INCLUDE** |
| **angle** | `float` | Rotation in degrees | ✅ **INCLUDE** |
| **roadOption** | `int` | Road connection options | ✅ **INCLUDE** |
| **assetPropertyCollection** | `AssetPropertyCollection` | Additional properties | ✅ **INCLUDE** |

**All MapObject properties are INCLUDED for training**

**Common AssetProperty Types (all included):**
- `originalOwner` (string) - Player who owns the object
- `uniqueID` (string) - Unique identifier
- `objectInitialHealth` (int) - Starting health
- `objectEnabled` (bool) - Whether object is active
- `objectIndestructible` (bool) - Cannot be destroyed
- `objectLayer` (string) - Layer assignment
- `objectName` (string) - Custom name
- And all other properties...

---

## 📊 Table 5: Water Area Structures

| Structure | Properties | Description | **Training Status** |
|-----------|------------|-------------|---------------------|
| **StandingWaterArea** | `points` (polygon), `waterHeight` (float) | Standing water body | ✅ **INCLUDE** |
| **RiverArea** | `points` (polyline), `width` (float) | River path | ✅ **INCLUDE** |
| **StandingWaveArea** | `points` (polygon), wave properties | Wave effects | ✅ **INCLUDE** |

**All Water Area structures are INCLUDED for training**

---

## 📊 Table 6: Map Metadata

| Metadata | Data Type | Description | **Training Status** |
|----------|-----------|-------------|---------------------|
| **map_width** | `int` | Map width in tiles | ✅ **INCLUDE** |
| **map_height** | `int` | Map height in tiles | ✅ **INCLUDE** |
| **border** | `int` | Border size in tiles | ✅ **INCLUDE** |
| **map_name** | `string` | Map name | ✅ **INCLUDE** |

**All Map Metadata is INCLUDED for training**

---

## 🎯 Training Data Selection Summary

### ✅ **INCLUDED FOR TRAINING**

#### Object Categories (5 categories):
1. **OreNode** - Ore mining nodes
2. **OilDerrick** - Oil derricks
3. **Garrison** - ALL garrison types (huts, houses, churches, restaurants, shops, stores, villas, mansions, shacks, dwellings, habitation, residential, tiki huts, civilian buildings)
4. **Building** - ALL building types including tech structures (buildings, bases, structures, command posts, tech structures)
5. **PlayerStart** - Player starting positions

#### Major Assets (18 assets):
1. **HeightMapData** - Terrain elevation
2. **BlendTileData** - All components (see Table 3)
3. **ObjectsList** - All objects (filtered by included categories)
4. **WorldInfo** - Map metadata
5. **StandingWaterAreas** - Water bodies
6. **RiverAreas** - Rivers
7. **StandingWaveAreas** - Wave effects
8. **SidesList** - Player/side definitions
9. **MPPositionList** - Multiplayer spawn positions
10. **Teams** - Team definitions
11. **BuildLists** - Build lists
12. **GlobalWaterSettings** - Water configuration
13. **FogSettings** - Fog configuration
14. **NamedCameras** - Camera positions
15. **GlobalLighting** - Lighting settings
16. **PostEffectsChunk** - Post-processing effects
17. **EnvironmentData** - Environment settings
18. **AssetList** - Asset metadata

#### BlendTileData Components (Table 3 - ALL 11 components):
- tiles, blend_infos, impassable, impassable_to_players, extra_passable, passage_width, visibility, buildability, impassable_to_air_units, tiberium_growability, textures

#### MapObject Properties (Table 4 - ALL properties):
- typeName, position, angle, roadOption, assetPropertyCollection (all properties)

#### Water Area Structures (Table 5 - ALL structures):
- StandingWaterArea, RiverArea, StandingWaveArea

#### Map Metadata (Table 6 - ALL metadata):
- map_width, map_height, border, map_name

### ❌ **EXCLUDED FROM TRAINING**

#### Object Categories (13 categories):
- Refinery, ConstructionYard, Barracks, WarFactory, Factory, Airfield, NavalYard, LaserTower, PowerPlant, BaseDefense, Tower, Bunker, SuperWeapon

#### Major Assets (6 assets):
1. **PlayerScriptsList** - Game logic scripts
2. **TriggerAreas** - Trigger zones
3. **MissionHotSpots** - Mission objective locations
4. **MissionObjectives** - Mission objectives
5. **LibraryMaps** - Map library references
6. **LibraryMapLists** - Map library lists

---

## 📝 Notes

- **Object Categories**: All 18 categories are enabled by default. You can disable specific categories if you don't want to train on them.
- **Decorative Objects**: Objects with decorative prefixes (CC_, YU_, CS_, IL_, HV_, AM_, etc.) are automatically excluded unless they match a category (e.g., YU_TikiHut01 matches Garrison).
- **Data Format**: Most data is stored as 2D arrays (width × height) matching the map dimensions.
- **Coordinate System**: World coordinates use 10 units per tile (divide by 10 to get tile coordinates).
- **Normalization**: Height maps typically range from 0-450, but should be normalized to 0-1 for training.

---

## 🔧 Usage Example

```python
from map_processor.utils.object_categories import ObjectCategoryConfig
from map_processor.core.ra3map import Ra3Map

# Parse map
ra3map = Ra3Map("path/to/map.map")
ra3map.parse()
context = ra3map.get_context()

# Initialize category config and disable excluded categories
config = ObjectCategoryConfig()
excluded_categories = [
    'refinery', 'construction_yard', 'barracks', 'war_factory', 'factory',
    'airfield', 'naval_yard', 'laser_tower', 'power_plant', 'base_defense',
    'tower', 'bunker', 'super_weapon'
]
for cat_key in excluded_categories:
    config.enable_category(cat_key, enabled=False)

# Get all included assets
height_data = context.get_asset('HeightMapData')
blend_data = context.get_asset('BlendTileData')
objects_list = context.get_asset('ObjectsList')
water_areas = context.get_asset('StandingWaterAreas')
river_areas = context.get_asset('RiverAreas')
wave_areas = context.get_asset('StandingWaveAreas')
world_info = context.get_asset('WorldInfo')
sides_list = context.get_asset('SidesList')
mp_positions = context.get_asset('MPPositionList')
teams = context.get_asset('Teams')
build_lists = context.get_asset('BuildLists')
water_settings = context.get_asset('GlobalWaterSettings')
fog_settings = context.get_asset('FogSettings')
cameras = context.get_asset('NamedCameras')
lighting = context.get_asset('GlobalLighting')
post_effects = context.get_asset('PostEffectsChunk')
environment = context.get_asset('EnvironmentData')
asset_list = context.get_asset('AssetList')

# Filter objects by included categories only
included_objects = [
    obj for obj in objects_list.map_objects 
    if config.get_category_for_object(obj.type_name)[1]
]

# Extract training data
training_data = {
    # Terrain data
    'height_map': height_data.elevations,
    
    # BlendTileData - ALL components
    'tiles': blend_data.tiles,
    'blend_infos': blend_data.blend_infos,
    'impassable': blend_data.impassable,
    'impassable_to_players': blend_data.impassable_to_players,
    'extra_passable': blend_data.extra_passable,
    'passage_width': blend_data.passage_width,
    'visibility': blend_data.visibility,
    'buildability': blend_data.buildability,
    'impassable_to_air_units': blend_data.impassable_to_air_units,
    'tiberium_growability': blend_data.tiberium_growability,
    'textures': blend_data.textures,
    
    # Objects - filtered by included categories
    'objects': [
        {
            'typeName': obj.type_name,
            'position': obj.position,
            'angle': obj.angle,
            'roadOption': obj.road_option,
            'properties': obj.asset_property_collection.get_all_properties() if hasattr(obj, 'asset_property_collection') else {}
        }
        for obj in included_objects
    ],
    
    # Water areas - ALL
    'water_areas': water_areas.areas if water_areas else [],
    'river_areas': river_areas.areas if river_areas else [],
    'wave_areas': wave_areas.areas if wave_areas else [],
    
    # Other included assets
    'world_info': world_info,
    'sides_list': sides_list,
    'mp_positions': mp_positions,
    'teams': teams,
    'build_lists': build_lists,
    'water_settings': water_settings,
    'fog_settings': fog_settings,
    'cameras': cameras,
    'lighting': lighting,
    'post_effects': post_effects,
    'environment': environment,
    'asset_list': asset_list,
    
    # Metadata
    'width': context.map_width,
    'height': context.map_height,
    'border': context.border,
    'map_name': context.map_name
}
```

---

**Last Updated**: Based on training requirements
**Total Categories**: 18 object categories (5 included, 13 excluded)
**Total Assets**: 24 major asset types (18 included, 6 excluded)
**Training Focus**: Terrain generation with strategic objects (ores, oil, garrisons, tech structures, player starts)

