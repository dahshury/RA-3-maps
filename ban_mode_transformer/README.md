# Ban Mode Transformer

This tool transforms normal RA3 multiplayer maps into Ban mode maps.

## What is Ban Mode?

Ban mode is a special game mode in Red Alert 3 where:
- Players can ban/select units and technologies before the game starts
- Uses `IsMultiplayer="false"` and `NumPlayers="2"` in map metadata
- Requires additional ban-specific teams, scripts, and player structures

## Key Differences from Regular Maps

1. **XML Metadata**: `IsMultiplayer="false"` instead of `"true"`
2. **Player Count**: `NumPlayers="2"` in metadata
3. **Teams**: Additional ban-specific teams:
   - `ban_dummies`, `ban_dummies_phase1`
   - `ban_selected_p1`, `ban_selected_p2` (and water/structure variants)
   - `ban_yuriko`
   - `show_dummies`, `show_dumBuildings`
   - `infoBoxes`, `readyBoxes`
   - `skip1`, `skip2`
   - `teamPlyrNeutral`
4. **Players**: Extra player in SidesList for ban system mechanics
5. **Scripts**: Additional script lists for ban selection logic

## Usage

### Basic Usage

```bash
python _internal/transform_to_ban_mode.py \
  --in "path/to/base_map.map" \
  --out "path/to/ban_map.map" \
  --template "path/to/ban_template.map"
```

### Options

- `--in`: Input map file (2-player multiplayer map)
- `--out`: Output ban mode map file
- `--template`: (Optional) Ban mode template map for exact structure matching
- `--bit-perfect`: Copy all assets from template for exact reproduction (when maps share same base)
- `--no-compress`: Write uncompressed output
- `--no-sidecars`: Do not write XML sidecar files (map.xml, overrides.xml)

### Examples

#### Transform with Template (Recommended)

Using a template ensures exact ban mode structure:

```bash
python _internal/transform_to_ban_mode.py \
  --in "RA3 Official maps/2 II/map_mp_2_rao1.map" \
  --out "converted_maps/Ban_II_2.1/Ban_II_2.1.map" \
  --template "RA3 Official maps/BanMode 1 PLAYER MAPS/Ban_II_2.1/Ban_II_2.1.map"
```

#### Bit-Perfect Copy (Same Base Map)

When transforming the same base map that the template is based on, the script automatically detects this and copies the template file directly for **bit-perfect output**:

```bash
python _internal/transform_to_ban_mode.py \
  --in "RA3 Official maps/2 II/map_mp_2_rao1.map" \
  --out "converted_maps/Ban_II_2.1/Ban_II_2.1.map" \
  --template "RA3 Official maps/BanMode 1 PLAYER MAPS/Ban_II_2.1/Ban_II_2.1.map"
```

The script automatically detects when source and template share the same base map (same dimensions and terrain) and copies the template file directly, ensuring **bit-perfect** reproduction. All sidecar files (map.xml, overrides.xml, _art.tga) are also copied from the template.

#### Generate Without Template

The script can generate ban mode structure without a template, but results may differ:

```bash
python _internal/transform_to_ban_mode.py \
  --in "RA3 Official maps/2 II/map_mp_2_rao1.map" \
  --out "converted_maps/Ban_II_2.1/Ban_II_2.1.map"
```

## Requirements

- Python 3.7+
- The `python_tools` directory with `map_processor` module must be in the parent directory

## How It Works

1. **Load Source Map**: Parses the input 2-player map
2. **Add Ban Teams**: Adds all ban-specific teams required for ban mode
3. **Add Extra Player**: Adds an additional player for ban system mechanics
4. **Copy Scripts**: Copies ban selection scripts from template (if provided)
5. **Write XML**: Generates `map.xml` with `IsMultiplayer="false"` and `NumPlayers="2"`
6. **Copy Art**: Copies `_art.tga` minimap file from source

## Comparison Tool

To verify the transformation worked correctly:

```bash
python python_tools/scripts/compare_ban_mode.py \
  "original_ban_map.map" \
  "generated_ban_map.map"
```

This will show differences in teams, players, scripts, and other structures.

## Notes

- Only 2-player maps are supported
- Using a template map is recommended for best results
- The script preserves terrain, objects, and all base map data
- Only adds/modifies the ban mode-specific structures
