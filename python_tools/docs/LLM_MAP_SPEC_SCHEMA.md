# RA3 Map Specification Schema for LLM Generation

This document defines the complete JSON schema that an LLM should generate to create a full RA3 map.

## Overview

The schema uses a **hierarchical representation** that balances:
- **Exact positions** for gameplay-critical objects (resources, player starts, garrisons)
- **Zone-based patterns** for decorative objects (trees, bushes, rocks)
- **Path-based descriptions** for linear features (roads, cliffs, rivers)

## Complete Schema

```json
{
  "metadata": {
    "name": "Map Name",
    "size": [580, 580],
    "border": 20,
    "player_count": 4,
    "style": "tropical_mykonos"
  },

  "terrain": {
    "base_height": 280,
    "height_regions": [
      {
        "description": "Main plateau",
        "height": 280,
        "shape": "default",
        "coverage_pct": 55
      },
      {
        "description": "Central lake depression",
        "height": 200,
        "shape": "circle",
        "center": [2900, 2900],
        "radius": 1200
      },
      {
        "description": "Player start valleys",
        "height": 140,
        "shape": "corners",
        "corner_radius": 800
      }
    ],
    "ramps": [
      {
        "from_height": 280,
        "to_height": 200,
        "path": [[2000, 2000], [2200, 2200]],
        "width": 150
      }
    ],
    "cliffs": [
      {
        "along_height_boundary": true,
        "from_height": 280,
        "to_height": 200,
        "steepness": "steep"
      }
    ]
  },

  "textures": {
    "palette": [
      "Dirt_Mykonos02",
      "Grass_Mykonos02", 
      "Sand_Cannes03",
      "Pavement_Mykonos04",
      "Rock_Mykonos01"
    ],
    "zones": [
      {"texture": "Grass_Mykonos02", "where": "high_ground", "height_above": 250},
      {"texture": "Sand_Cannes03", "where": "water_edges", "distance": 50},
      {"texture": "Pavement_Mykonos04", "where": "near_buildings", "radius": 30},
      {"texture": "Dirt_Mykonos02", "where": "default"}
    ]
  },

  "water": {
    "areas": [
      {
        "type": "lake",
        "shape": "polygon",
        "points": [[1213, 2344], [603, 2394], [540, 2702], [1193, 3056]],
        "water_height": 190
      }
    ],
    "rivers": [
      {
        "path": [[500, 0], [600, 500], [700, 1000]],
        "width": 80
      }
    ]
  },

  "passability": {
    "impassable_zones": [
      {"where": "water_deep", "depth_below": 150},
      {"where": "cliff_edges", "steepness_above": 45},
      {"shape": "polygon", "points": [...]}
    ],
    "buildable_zones": [
      {"where": "flat_ground", "slope_below": 10, "height_above": 200}
    ]
  },

  "player_starts": [
    {"player": 1, "position": [742, 799], "facing": "northeast"},
    {"player": 2, "position": [4608, 736], "facing": "northwest"},
    {"player": 3, "position": [4689, 4622], "facing": "southwest"},
    {"player": 4, "position": [766, 4657], "facing": "southeast"}
  ],

  "resources": {
    "ore_nodes": [
      {"position": [896, 2329], "near_player": 1, "distance": "medium"},
      {"position": [3059, 907], "near_player": 2, "distance": "medium"},
      {"position": [2317, 4493], "near_player": 3, "distance": "medium"},
      {"position": [4481, 3071], "near_player": 4, "distance": "medium"},
      {"position": [2730, 2700], "contested": true, "central": true}
    ],
    "oil_derricks": [
      {"position": [1830, 4737], "between_players": [1, 4]},
      {"position": [4711, 3564], "between_players": [3, 4]},
      {"position": [3574, 707], "between_players": [1, 2]},
      {"position": [704, 1829], "between_players": [1, 4]}
    ]
  },

  "tech_structures": [
    {"type": "ObservationPostTechStructure", "position": [2698, 3493], "controls": "center"},
    {"type": "ObservationPostTechStructure", "position": [1896, 2708], "controls": "west"},
    {"type": "ObservationPostTechStructure", "position": [2693, 1894], "controls": "north"},
    {"type": "AirportTechStructure", "position": [2688, 2695], "central": true}
  ],

  "garrison_buildings": [
    {"type": "MY_Restaurant_01", "position": [3309, 123], "near_player": 2},
    {"type": "MY_House_04", "position": [138, 581], "near_player": 1},
    {"type": "MY_Hotel_01", "position": [321, 157], "near_player": 1},
    {"type": "MY_Church_02", "position": [99, 1931], "neutral_zone": true},
    {"type": "MY_Apartment01", "position": [4435, 2409], "contested": true}
  ],

  "bridges": [
    {
      "type": "BridgeWood01",
      "from": [1500, 2000],
      "to": [1600, 2100],
      "over": "water"
    }
  ],

  "decorative_vegetation": {
    "trees": [
      {
        "types": ["CS_Palm01", "CS_Palm02", "CS_Palm03"],
        "density": "medium",
        "zone": "map_edges",
        "avoid": ["water", "buildings", "roads"],
        "total_count": 400
      },
      {
        "types": ["CS_Palm01"],
        "density": "sparse", 
        "zone": {"near_buildings": true, "radius": 50},
        "total_count": 50
      }
    ],
    "bushes": [
      {
        "types": ["MY_BUSH03", "MY_Bush01"],
        "density": "high",
        "zone": "everywhere",
        "avoid": ["water", "roads", "buildings", "cliffs"],
        "total_count": 1200
      }
    ],
    "rocks": [
      {
        "types": ["MY_ROCKS01", "MY_ROCKS03"],
        "density": "medium",
        "zone": "near_cliffs",
        "total_count": 400
      }
    ],
    "coral": [
      {
        "types": ["YU_CORAL02", "YU_CORAL03", "YU_Coral01"],
        "density": "high",
        "zone": "underwater",
        "depth_range": [50, 150],
        "total_count": 150
      }
    ]
  },

  "decorative_structures": {
    "cliff_walls": [
      {
        "types": ["MY_CLIFFWALL01", "MY_CLIFFWALL02", "MY_CLIFFWALL03"],
        "placement": "along_height_transitions",
        "from_height": 280,
        "to_height": 200
      }
    ],
    "fences": [
      {
        "type": "MY_ALLYFENCE01",
        "near": "garrison_buildings",
        "density": "sparse"
      }
    ],
    "lights": [
      {
        "type": "MY_LIGHT01",
        "near": "roads",
        "spacing": 100
      }
    ]
  },

  "roads": [
    {
      "type": "HawaiiDirtRoad01",
      "path": [[100, 100], [500, 300], [900, 500], [1500, 800]],
      "connects": ["player_1_start", "center"]
    },
    {
      "type": "MykonosSidewalk01",
      "around": "garrison_buildings",
      "width": 2
    }
  ],

  "ambient_sounds": [
    {"type": "Amb_Wind1", "zone": "everywhere", "count": 25},
    {"type": "Amb_BirdsDesert", "zone": "vegetation_heavy", "count": 20},
    {"type": "Amb_Water1", "zone": "near_water", "count": 10}
  ],

  "props": [
    {"type": "MY_Barrel01", "near": "buildings", "count": 20},
    {"type": "MY_Crate01", "near": "roads", "count": 15}
  ],

  "vehicles": [
    {"type": "ParkedCar01", "near": "roads", "count": 10},
    {"type": "ParkedBoat01", "near": "water_edge", "count": 5}
  ]
}
```

## Token Estimation

| Section | Estimated Tokens |
|---------|------------------|
| Metadata | 50 |
| Terrain | 500 |
| Textures | 200 |
| Water | 200 |
| Passability | 150 |
| Player starts | 100 |
| Resources | 300 |
| Tech structures | 150 |
| Garrison buildings | 400 |
| Bridges | 50 |
| Decorative vegetation | 600 |
| Decorative structures | 300 |
| Roads | 300 |
| Ambient sounds | 100 |
| Props/Vehicles | 100 |
| **TOTAL** | **~3,500 tokens** |

## Key Principles

1. **Exact positions** for:
   - Player starts
   - Resources (ore nodes, oil derricks)
   - Tech structures
   - Garrison buildings
   - Bridges

2. **Zone-based descriptions** for:
   - Trees, bushes, rocks (distributed with density)
   - Coral (underwater only)
   - Ambient sounds
   - Props and vehicles

3. **Path-based descriptions** for:
   - Roads (connected paths)
   - Rivers (linear paths)
   - Cliff walls (along height boundaries)

4. **Rule-based placement** for:
   - Fences (near buildings)
   - Lights (along roads)
   - Sidewalks (around buildings)

## Renderer Implementation

The renderer must:
1. Generate height map from `terrain.height_regions` and `terrain.ramps`
2. Apply textures based on `textures.zones` rules
3. Place water areas and calculate passability
4. Place all exact-position objects
5. Distribute zone-based decorations using noise/random with density rules
6. Generate road tiles along paths
7. Place cliff walls along height transitions

## Example Prompt for LLM

```
You are an RA3 map generator. Given the example maps in your context, generate a complete map specification following this schema.

USER REQUEST: "Create a 4-player tropical island map with:
- Central contested lake with an airport tech structure
- Each player starts in a corner with 3 ore nodes nearby
- Oil derricks between adjacent players
- Dense vegetation on high ground
- Beaches near water
- Roads connecting player bases to center"

Generate the complete JSON map specification:
```









