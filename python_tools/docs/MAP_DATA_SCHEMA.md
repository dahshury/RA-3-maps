# RA3 Map Data Schema - Complete Feature Reference

<!-- markdownlint-disable MD012 MD022 MD024 MD029 MD031 MD032 MD036 MD040 MD060 -->

This document details **every feature** available in a parsed `.map` file that can be used for learning/generation.

---

## Overview

A parsed RA3 map contains **24 asset types**. Each asset provides specific features:

| Asset | Purpose | Learnable? |
|-------|---------|------------|
| `HeightMapData` | Terrain elevations | ✅ **Critical** |
| `BlendTileData` | Textures, blending, passability | ✅ **Critical** |
| `ObjectsList` | All placed objects | ✅ **Critical** |
| `StandingWaterAreas` | Water bodies | ✅ **Critical** |
| `MPPositionList` | Multiplayer slots | ⚠️ Metadata only |
| `GlobalLighting` | Lighting config | ✅ Style |
| `EnvironmentData` | Sky/cloud textures | ✅ Style |
| `GlobalWaterSettings` | Water reflection settings | ⚠️ Minor |
| `FogSettings` | Fog parameters | ⚠️ Minor |
| Others | Scripts, cameras, etc. | ❌ Not relevant |

---

## 1. HeightMapData (Terrain Elevation)

The **height map** defines terrain elevation across the map grid.

### Data Structure

| Field | Type | Description |
|-------|------|-------------|
| `elevations` | `ndarray (W, H) float32` | Height value at each tile |
| `map_width` | `int` | Total map width in tiles |
| `map_height` | `int` | Total map height in tiles |
| `playable_width` | `int` | Playable area width (excludes border) |
| `playable_height` | `int` | Playable area height (excludes border) |
| `border_width` | `int` | Non-playable border size |
| `borders` | `list[HeightMapBorder]` | Border region definitions |

### Example Values (from 2 II map)

```
Shape: (590, 440) tiles
Height Range: 34.82 - 281.05 units
Mean: 123.53, Std: 71.15
Unique values: 5,720
Border: 30 tiles
Playable: 530 x 380 tiles
```

### Key Insights for Learning

- **Water level is typically 200** - heights below 200 are underwater
- **Height distribution** shows distinct bands (lowlands, mid, highlands)
- **Gradients** indicate cliffs (>20 height diff between neighbors)
- **Format**: Uses `SageFloat16` encoding: `upper * 10 + lower * 9.96 / 256`

---

## 2. BlendTileData (Textures & Passability)

The **blend tile data** defines what textures appear where and movement restrictions.

### Data Structure

| Field | Type | Description |
|-------|------|-------------|
| `tiles` | `ndarray (W, H) uint16` | Base texture tile ID per cell |
| `textures` | `list[Texture]` | Texture palette (names + cell info) |
| `blends` | `ndarray (W, H) uint16` | Blend info index (for smooth transitions) |
| `blend_info` | `list[BlendInfo]` | Blend direction & secondary texture |
| `impassable` | `ndarray (W, H) bool` | True = cannot walk through |
| `buildability` | `ndarray (W, H) bool` | True = can build structures |
| `passability` | `ndarray (W, H) int32` | Passability flags (0=pass, 1=block) |
| `visibility` | `ndarray (W, H) bool` | Vision blocking |
| `cliff_blends` | `ndarray (W, H) uint16` | Cliff-specific blend data |
| `single_edge_blends` | `ndarray (W, H) uint16` | Edge blend for seamless transitions |
| `dynamic_shrubbery` | `ndarray (W, H) uint8` | Auto-generated shrubs |
| `tiberium_growability` | `ndarray (W, H) bool` | Ore can spread here |
| `passage_width` | `ndarray (W, H) bool` | Width-based passage restriction |

### Tile ID Formula

```
texture_index = tile_id // 64  (which texture from palette)
texture_cell = tile_id % 64    (which variant of that texture)
```

### Texture Structure

| Field | Type | Description |
|-------|------|-------------|
| `name` | `str` | Texture name (e.g., "Grass_Yucatan02") |
| `cell_count` | `int` | Number of variants (typically 16) |
| `cell_size` | `int` | Variant size |
| `cell_start` | `int` | Starting cell index |

### BlendInfo Structure

| Field | Type | Description |
|-------|------|-------------|
| `blend_direction` | `int` | Edge direction (bitfield) |
| `secondary_texture_tile` | `int` | Secondary texture for blending |
| `i3`, `i4` | `int` | Additional blend parameters |

### Example Values (from 2 II map)

```
Textures: 40 unique textures in palette
Tile IDs: 0 - 2559 (2,242 unique values)

Top Textures by Coverage:
  Dirt_Yucatan01      54.7%
  Grass_Yucatan02      5.4%
  Dirt_Yucatan04       5.0%
  Dirt_Yucatan02       4.8%
  Grass_Yucatan05      3.4%

Passability:
  Impassable:  7.25% (18,812 cells)
  Passable:   92.75% (240,788 cells)

Buildability:
  Buildable:   0.00% (explicitly set areas only)
```

### Key Insights for Learning

- **Texture names encode style**: `Yucatan`, `Cannes`, `Heidelberg` indicate biome
- **Texture types**: `Dirt_`, `Grass_`, `Rock_`, `Sand_`, `Reef_`, `Transition_`
- **Blend info is deterministic**: given two textures, blend is calculated
- **Impassability correlates with**: cliffs, water edges, placed objects
- **Cell variants (0-63)** provide visual variety for same texture

---

## 3. ObjectsList (Placed Objects)

All objects placed on the map, from player starts to decorative rocks.

### Important: Object Selection for Training

The raw `.map` may contain **thousands** of decorative objects (grass/coral/bushes/audio props). We **do not** want the model to learn/place all of them initially.

For training we will **only supervise a curated subset** (the “trainable objects”). Per your current plan, v1 is:

| Group | Include? | Examples |
|------|----------|----------|
| **Player positions** | ✅ | `*Waypoints/Waypoint` with `Player_{n}_Start` |
| **Resources** | ✅ | `OreNode`, `OilDerrick`, `OilDerrick_OnWater` |
| **Tech buildings** | ✅ | `ObservationPostTechStructure`, `AirportTechStructure`, `HospitalTechStructure`, `VeterancyTechStructure`, `GarageTechStructure`, `ShipYardTechStructure` |
| **Garrison buildings** | ✅ | all garrisonable/civilian buildings (dataset-derived allowlist) |
| **Bridges** | ✅ | any type containing `Bridge` |
| **Decor / ambient / props / vegetation** | ❌ | `Amb_*`, `CC_GRASS*`, `*_BUSH*`, `*_TREE*`, etc. |

**Why**:
- Decorative objects are high-count and style-specific, and they explode label space.
- They can be reintroduced later as **density fields** or via a second “decor placer” model.

### Trainable Object Set v1 (Your Current Scope)

We will still **parse and retain** all objects, but for training we only **supervise / generate** the critical subset:

- **All terrain**: heights, tiles, blends, passability/water
- **All player positions**
- **All bridges**
- **All garrisons and buildings** (k-way classification in the object table)
  - “Buildings” include: oil derricks, hospitals, veterancy academies, garages, observation posts, airports, shipyards/dry-docks.

This aligns with your current goal: keep the map visually cohesive while not exploding supervision with decor spam.

#### Allowlist (gameplay-critical)

- **Player positions (spawns)**:
  - Primary: `*Waypoints/Waypoint` where `uniqueID` is `Player_{n}_Start` (matches `MapCreatorCoreLib` logic)
- **Resources**:
  - `OreNode`
  - `OilDerrick`, `OilDerrick_OnWater` (case variants exist in tooling; normalize in filters)
- **Tech buildings (explicit list from tooling + your additions)**:
  - `ObservationPostTechStructure`
  - `AirportTechStructure`
  - `HospitalTechStructure`
  - `VeterancyTechStructure`
  - `GarageTechStructure`
  - `ShipYardTechStructure` (covers “dry dock” in practice)
- **Bridges**:
  - any object type containing `Bridge` (e.g. `Bridge1`, `RomaniaBridge`, `HB_Bridge_02`, `AS_Bridge`, etc.)
- **Garrisons / civilian buildings (k-way subtype)**:
  - A dataset-driven vocabulary of building types that are considered “garrison buildings”.
  - Example hint from `MapCreatorCore` test utilities: garrison structures were filtered via substring `SV_Building`.
  - In practice, this will be a curated allowlist file derived from `all_object_types.json` + mapset scanning.

#### Denylist (always exclude from supervision)

- **Ambient/audio emitters**: `^Amb_`, contains `Sound`, contains `Audio`, `WaterFountain1AmbientLoop`, etc.
- **Vegetation / ground clutter**: contains `GRASS`, `BUSH`, `TREE`, `CORAL`, `PALM`, `SHRUB`, `PLANT`, etc.
- **Cosmetic props** (non-garrison): benches, fences, umbrellas, crates, barrels, mailboxes, streetlights, signs, etc.
- **Road markings / sidewalk segments**: types containing `RoadMarking`, `Sidewalk`, `StreetSegment`, etc. (not bridges)

These rules should be implemented in the feature builder as configurable filters so we can tweak per-mapset.

### Object Structure

| Field | Type | Description |
|-------|------|-------------|
| `type_name` | `str` | Object type (e.g., "OreNode", "CC_GRASS07") |
| `name` | `str` | Instance name |
| `position` | `tuple (x, y, z)` | World coordinates |
| `angle` | `float` | Rotation in radians |
| `original_owner` | `str` | Owner player/team |
| `road_option` | `int` | Road-related flags |
| `asset_property_collection` | `object` | Additional properties |

### World Coordinates

```
World Position → Tile Position:
  tile_x = world_x / 10.0
  tile_y = world_y / 10.0
```

### Object Categories

| Category | Count | Unique Types | Examples |
|----------|-------|--------------|----------|
| **Gameplay** | 15 | 4 | `OreNode`, `OilDerrick`, `Waypoint`, `TechStructure` |
| **Nature** | 1,707 | 24 | `CC_GRASS07`, `YU_Coral01`, `CS_Palm01`, `HV_Tree01` |
| **Cliffs** | 84 | 14 | `YU_CLIFFWALL06`, `YU_SEACLIFFWALL05` |
| **Roads** | 66 | 1 | `YucatanDirtRoad01` |
| **Audio** | 35 | 5 | `Amb_WaterLakeLight1`, `Amb_CricketsBed1Loop` |
| **Props** | 27 | 12 | `YU_SUNKENSHIP03`, `YU_TikiHut01` |

### Gameplay Object Details

| Object Type | Purpose | Typical Count |
|-------------|---------|---------------|
| `*Waypoints/Waypoint` | Player spawn positions | 2-6 per map |
| `OreNode` | Resource collection point | 8-15 per map |
| `OilDerrick` | Capturable resource building | 2-4 per map |
| `ObservationPostTechStructure` | Tech building | 0-2 per map |
| `Garrison` | Infantry can enter | 2-8 per map |
| `Bridge` | Crossable structure | 0-4 per map |

### Object Naming Conventions

```
CC_*      → Common/Shared objects (grass, rocks)
YU_*      → Yucatan theme
HV_*      → Havana theme
IL_*      → Island theme
CS_*      → Coastal theme
GC_*      → Generic/Common
Amb_*     → Ambient sound emitters
```

### Example Positions (from 2 II map)

```
Player Spawns (Waypoints):
  Player 1: (1390, 1588) → tile (139, 159)
  Player 2: (3981, 2188) → tile (398, 219)

Ore Nodes:
  Near P1: (741, 744), (1101, 2102)
  Near P2: (4517, 3035), (4156, 1635)
  Central: (2587, 2992), (2778, 723)

Oil Derricks:
  (2279, 2770), (2979, 1051)

Tech Structure:
  (2638, 1896) - center map
```

---

## 4. StandingWaterAreas (Water Bodies)

Defines water regions with their visual properties.

### Water Area Structure

| Field | Type | Description |
|-------|------|-------------|
| `water_height` | `float` | Z-level of water surface |
| `points` | `list[tuple]` | Polygon vertices (x, y) |
| `bumpmap_texture` | `str` | Wave texture |
| `sky_texture` | `str` | Reflection texture |
| `depth_colors` | `list` | Color gradient by depth |
| `fx_shader` | `str` | Shader effect name |
| `uv_scroll_speed` | `tuple` | Water animation speed |
| `additive_blending` | `bool` | Blend mode |

### Example (from 2 II map)

```
Water Areas: 1
  Height: 200.0
  Polygon: 4 vertices
    (-234, 4034), (5532, 4029), (5542, -258), (-234, -267)
  Coverage: Full map border (ocean surrounding island)
```

### Key Insights

- **Water height 200** is the standard level
- **Polygons can be complex** - not just rectangles
- **Height < water_height** = submerged terrain

---

## 5. GlobalLighting (Visual Atmosphere)

Controls map lighting and shadows.

### Structure

| Field | Type | Description |
|-------|------|-------------|
| `lighting_configurations` | `list[4]` | Time-of-day configs (dawn/day/dusk/night) |
| `shadow_color` | `tuple (a,r,g,b)` | Shadow tint |
| `time` | `int` | Current time setting (0-3) |
| `no_cloud_factor` | `tuple` | Cloud influence |

### Per-Configuration

| Field | Type | Description |
|-------|------|-------------|
| `terrain_sun` | `GlobalLight` | Main directional light |
| `terrain_accent1` | `GlobalLight` | Fill light 1 |
| `terrain_accent2` | `GlobalLight` | Fill light 2 |

---

## 6. EnvironmentData (Sky & Effects)

Global environmental textures.

| Field | Type | Description |
|-------|------|-------------|
| `cloud_texture` | `str` | Cloud layer texture (e.g., "TSCloudMed") |
| `environment_map` | `str` | Reflection cubemap (e.g., "EVDefault") |
| `macro_texture` | `str` | Large-scale noise overlay (e.g., "TSNoiseUrb") |
| `water_max_alpha` | `float` | Water opacity |
| `water_max_alpha_depth` | `float` | Depth for full opacity |

---

## 7. Other Assets (Reference)

### MPPositionList (Multiplayer Slots)

```
positions: list[MPPositionInfo]
  - is_human: bool
  - is_computer: bool
  - team: int (4294967295 = any team)
```

### GlobalWaterSettings

```
reflection: bool (True)
reflection_plane_z: float (200.0)
```

### FogSettings

```
enabled: bool
start: float
end: float
r, g, b: float (color)
```

---

## Summary: Learnable Features Table

| Feature | Data Type | Size (2 II example) | Importance |
|---------|-----------|---------------------|------------|
| **Height Grid** | float32 grid | 590×440 = 259,600 | 🔴 Critical |
| **Texture Tiles** | uint16 grid | 590×440 = 259,600 | 🔴 Critical |
| **Texture Palette** | string list | 40 textures | 🔴 Critical |
| **Passability** | bool grid | 590×440 = 259,600 | 🟡 Important |
| **Water Polygons** | coordinate list | 4 vertices | 🟡 Important |
| **Gameplay Objects** | typed positions | 15 objects | 🔴 Critical |
| **Decorative Objects** | typed positions | 1,900 objects | 🟢 Optional |
| **Cliff Objects** | typed positions | 84 objects | 🟡 Important |
| **Road Objects** | typed positions | 66 objects | 🟢 Optional |
| **Audio Emitters** | typed positions | 35 objects | 🟢 Optional |
| **Lighting Config** | structured data | 4 configs | 🟢 Style |
| **Environment** | string refs | 3 textures | 🟢 Style |

---

## End-to-End Training Tensors (Spatial Deep Model)

This section answers: **how do we turn the parsed `.map` into tensors for training, end-to-end?**

We are **not using text→map for now**. Instead:
- A **hierarchical CNN + Transformer** generates map tensors.
- A deterministic writer turns tensors into a valid RA3 `.map`.

### Canonical Tensor Sizes (handles up to ~700×700)

Maps have variable size (e.g. 590×440). For training we use a fixed canvas:

| Resolution | Symbol | Size | Notes |
|-----------|--------|------|------|
| Final | \(H_f,W_f\) | **704×704** | divisible by 16; pad/crop |
| Mid | \(H_m,W_m\) | **176×176** | 4× downscale |
| Blueprint | \(H_b,W_b\) | **44×44** | 16× downscale; “intent grid” |

Always provide a mask:

| Tensor | Shape | Meaning |
|--------|-------|---------|
| `valid_mask_final` | `[704, 704, 1]` | 1 where original map exists |

### Training Sample (Inputs → Targets)

Each sample is a single parsed `.map` converted into tensors:

#### Inputs

| Name | Shape | dtype | Purpose |
|------|-------|-------|---------|
| `blueprint` | `[44, 44, Cb]` | float32 | coarse intent (derived deterministically) |
| `valid_mask_final` | `[704, 704, 1]` | float32 | masks padding out of losses |
| `map_meta` | `[M]` | float32/int32 | scalars: original \(W,H\), border, water_height, style IDs |
| `noise_final` (optional) | `[704, 704, Z]` | float32 | stochastic diversity (can be zeros) |

#### Targets

| Name | Shape | dtype | Derived from `.map` |
|------|-------|-------|---------------------|
| `height_final` | `[704, 704, 1]` | float32 | `HeightMapData.elevations` |
| `texture_class_final` | `[704, 704, 1]` | int64 | `BlendTileData.tiles // 64` (K-way class target, **not** K channels) |
| `texture_variant_final` (optional) | `[704, 704, 1]` | int64 | `BlendTileData.tiles % 64` |
| `water_mask_final` | `[704, 704, 1]` | float32 | from `StandingWaterAreas` + water height |
| `impassable_final` | `[704, 704, 1]` | float32 | `BlendTileData.impassable` |
| `road_mask_final` (optional) | `[704, 704, 1]` | float32 | from road tiles/road objects |
| `object_maps_final` (optional) | `[704, 704, Co]` | float32 | **small Co only**: heatmaps for a few key trainable groups |

**Important**: losses are computed with `valid_mask_final` so padded area doesn’t contribute.

### Avoiding `704×704×Co` Explosion

If `Co` is “every object type”, training will be painful and unnecessary.

Recommended strategy:
- Keep `Co` **tiny** (e.g. 4–12): `spawn`, `ore`, `oil`, `tech`, `garrison`, `bridge` (plus maybe `road`).
- Predict textures as **class logits** (shape `[704,704,K]` internally) with **target** `texture_class_final` `[704,704,1]`. Do **not** one-hot encode textures in the dataset.
- For gameplay objects, prefer an **object table head** (`objects_table`) over huge per-type channels.

### Blueprint Channels (44×44)

Blueprint is computed per **16×16 tile block** via simple statistics (no ML).

Suggested `Cb = 12`:

| Channel | Range | How it is computed |
|---------|-------|--------------------|
| `zone_base` | {0,1} | block contains a player start waypoint |
| `zone_expansion` | {0,1} | near ore/oil clusters or intended expansions |
| `zone_neutral` | {0,1} | default |
| `player_id` | -1..N | dominant player within block |
| `importance` | 0..1 | centrality + gameplay-object density |
| `road_strength` | 0..1 | fraction of road tiles/objects in block |
| `height_mean` | normalized | mean height |
| `height_std` | normalized | std height (cliffiness proxy) |
| `water_frac` | 0..1 | fraction underwater |
| `texture_entropy` | 0..1 | entropy of texture distribution |
| `decor_density` | 0..1 | decor objects per tile (clipped) |
| `style_id` | one-hot/scalar | biome/style label (optional) |

### Optional: Gameplay Objects as a Table (instead of only heatmaps)

For gameplay-critical objects, a table head often works better:

| Tensor | Shape | dtype | Notes |
|--------|-------|-------|------|
| `objects_table` | `[Nmax, F]` | float32/int64 | type + owner + normalized coords + angle |
| `objects_mask` | `[Nmax]` | float32 | 1 where row is valid |

Typical values:
- `Nmax = 64`
- `F = 10` (example): `[type_id, owner_id, x_norm, y_norm, angle_sin, angle_cos, radius, flags...]`

### Mapping Back to `.map` (Writer)

| `.map` field | Reconstruction |
|-------------|----------------|
| `HeightMapData.elevations` | crop `height_final` to original \(W,H\) |
| `BlendTileData.tiles` | `texture_class_final*64 + texture_variant_final` (or choose variant deterministically) |
| `BlendTileData.blends` | deterministic blend lookup (still valid) |
| `BlendTileData.impassable` | `impassable_final` + repair rules |
| `StandingWaterAreas` | polygonize `water_mask_final` or keep templates |
| `ObjectsList.map_objects` | from `objects_table` and/or heatmaps + sampler |

## Token Budget Estimation

For LLM-based generation, here's the token cost at different resolutions:

| Data | Raw Size | Compressed | Tokens (est.) |
|------|----------|------------|---------------|
| Height 590×440 | 259,600 floats | 16×16 grid | ~500 |
| Height 590×440 | 259,600 floats | 32×32 grid | ~2,000 |
| Textures | 259,600 uint16 | Palette + zones | ~1,000 |
| Passability | 259,600 bool | Height-derived | ~0 (rule) |
| Gameplay Objects | 15 | Full detail | ~200 |
| Decorative | 1,900 | Pattern rules | ~500 |
| Water | 1 polygon | Full detail | ~50 |
| Style/Lighting | configs | Full | ~100 |
| **Total** | | | **~4,000-5,000** |

---

## Coordinate Systems

### World Coordinates
- Origin: Bottom-left of map
- Units: Game units (10 units = 1 tile)
- X: Left → Right
- Y: Bottom → Top
- Z: Down → Up (height)

### Tile Coordinates
- Origin: Bottom-left
- Units: Tiles (1 tile = 10 world units)
- Range: (0,0) to (width-1, height-1)

### Image Coordinates (for visualization)
- Origin: Top-left
- Y-axis: Flipped from world (top → bottom)
- Formula: `image_y = (height - 1) - tile_y`

---

## File Format Notes

### SageFloat16 (Height Encoding)
```python
# NOT standard IEEE float16!
# Custom RA3 format:
def decode_sage_float16(bytes_2):
    upper = bytes_2[1]  # High byte
    lower = bytes_2[0]  # Low byte
    return upper * 10.0 + lower * 9.96 / 256.0
```

### Tile ID Encoding
```python
texture_index = tile_id // 64
texture_variant = tile_id % 64
tile_id = texture_index * 64 + texture_variant
```

### Object Owner Strings
```
"SkirmishNeutral/teamSkirmishNeutral" - Neutral capturable
"PlyrCreeps/teamPlyrCreeps" - Hostile NPCs
"" - No owner
```

