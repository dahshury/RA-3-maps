# Object Categories for RA3 Map Visualization

This document describes the object categorization system used for visualizing buildings and structures on RA3 maps.

## Overview

The `ObjectCategoryConfig` class provides a comprehensive categorization system for all object types found in RA3 maps. Categories can be enabled or disabled individually, allowing fine-grained control over what appears in visualizations.

## Categories

### Economic/Resource Buildings
- **OreNode**: Ore mining nodes/refineries (Dark orange, size 6)
- **OilDerrick**: Oil derricks (Bright yellow, size 8)
- **Refinery**: Refineries (Red, size 8)

### Military Buildings - Construction
- **ConstructionYard**: Construction yards (Green, size 10)

### Military Buildings - Production
- **Barracks**: Barracks (Cyan, size 8)
- **WarFactory**: War factories (Light red, size 9)
- **Factory**: Factories (general) (Magenta, size 9)
- **Airfield**: Airfields (Light blue, size 8)
- **NavalYard**: Naval yards (Blue, size 8)

### Military Buildings - Power
- **PowerPlant**: Power plants (Gold, size 7)

### Military Buildings - Defense
- **BaseDefense**: Base defense structures (Purple, size 7)
- **Tower**: Defense towers (Lavender, size 6)
- **Bunker**: Bunkers (Dark purple, size 7)
- **LaserTower**: Laser towers (Orange, size 7)

### Super Weapons
- **SuperWeapon**: Super weapons (Pink-red, size 12)

### Garrisonable Buildings (Civilian)
- **Garrison**: Garrisonable civilian buildings (Brown/tan, size 7)
  - Includes: huts, houses, churches, restaurants, shops, stores, villas, mansions, shacks, dwellings, habitation, residential buildings, tiki huts, and civilian-prefixed objects

### Other Buildings/Structures
- **Building**: Other buildings and structures (Light blue, size 6)
  - Includes: buildings, bases, structures, command posts, tech structures

### Player Starts
- **PlayerStart**: Player starting positions (White, size 12)

## Usage

### Basic Usage

```python
from map_processor.utils.object_categories import ObjectCategoryConfig

# Create configuration
config = ObjectCategoryConfig()

# Get category for an object
category, should_draw = config.get_category_for_object("AlliedConstructionYard")
if should_draw and category:
    print(f"Category: {category.name}, Color: {category.color}, Size: {category.size}")
```

### Enabling/Disabling Categories

```python
# Disable a specific category
config.enable_category('garrison', enabled=False)

# Disable all categories
config.enable_all_categories(enabled=False)

# Re-enable specific categories
config.enable_category('ore_node', enabled=True)
config.enable_category('oil_derrick', enabled=True)
```

### Listing Categories

```python
# Get all categories
all_categories = config.get_all_categories()

# Get only enabled categories
enabled_categories = config.get_enabled_categories()

# Get list of category names
category_names = config.list_categories()
```

## Category Matching Logic

1. **Priority Order**: Categories are checked in a specific order (most specific first)
2. **Keyword Matching**: Object type names are matched against category keywords (case-insensitive)
3. **Decorative Filtering**: Objects with decorative prefixes (CC_, YU_, CS_, IL_, HV_, AM_, etc.) are filtered out UNLESS they match a category first (e.g., YU_TikiHut01 matches "garrison" category)
4. **Waypoint Filtering**: Waypoints and markers are always excluded

## Adding New Categories

To add a new category, edit `object_categories.py` and add to `_initialize_categories()`:

```python
self.categories['new_category'] = ObjectCategory(
    name='NewCategory',
    keywords=['keyword1', 'keyword2'],
    color=(255, 0, 0),  # RGB color
    size=8,
    enabled=True,
    description='Description of the category'
)
```

Then add the category key to the `priority_order` list in `get_category_for_object()` method.

## Building Types Found in Codebase

Based on analysis of MapCreatorCore and game data, the following building types are known:

### Faction-Specific Buildings
- Allied: AlliedConstructionYard, AlliedBarracks, AlliedRefinery, AlliedWarFactory, AlliedNavalYard, AlliedAirfield, AlliedBaseDefense, AlliedSuperWeapon
- Soviet: SovietConstructionYard, SovietBarracks, SovietRefinery, SovietWarFactory, SovietNavalYard, SovietAirfield, SovietBaseDefense, SovietSuperWeapon
- Japan/Empire: JapanConstructionYardEgg, JapanBarracksEgg, JapanRefineryEgg, JapanWarFactoryEgg, JapanNavalYardEgg, JapanAirfieldEgg, JapanBaseDefenseEgg, JapanSuperWeaponEgg

### Civilian/Garrisonable Buildings
- Various civilian buildings with prefixes like YU_, HB_, AM_, etc.
- Examples: YU_TikiHut01, HB_House03, HB_Church, HB_Restaurant01

## Notes

- The system is designed to be comprehensive and catch all building types, even those with unusual naming conventions
- Garrisonable buildings are detected even if they have decorative prefixes (e.g., YU_TikiHut01)
- Categories can be dynamically enabled/disabled without modifying code
- The visualization system automatically uses this categorization when drawing objects on maps











