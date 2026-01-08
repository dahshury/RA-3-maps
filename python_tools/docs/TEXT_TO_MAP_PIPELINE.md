# RA3 Tensor-to-Map Generation Pipeline (Spatial Deep Learning)

<!-- markdownlint-disable MD012 MD022 MD024 MD029 MD031 MD032 MD036 MD040 MD060 -->

## Overview

We are **not doing text→map for now**.

This document defines an end-to-end pipeline where a **hierarchical spatial deep model** generates map tensors (grid + objects), and a deterministic writer converts them into a valid RA3 `.map`.

**Core idea**:
- Learn **global structure** at low resolution (blueprint).
- Learn **local detail** at higher resolution (U-Net).
- Use a small rule post-pass only for **hard constraints / repair**.

---

## Why This Approach

| Technique | Verdict | Why |
|----------|---------|-----|
| Reinforcement Learning | ❌ No | No reward signals; hard to specify |
| LLM tile generator | ❌ No | Weak spatial coherence; huge outputs |
| Pure WFC | ❌ No | Local constraints only; weak global layout |
| CNN only | ⚠️ Weak | Limited long-range dependencies |
| Transformer only | ❌ Too big | Memory blowup at 700×700 |
| **Hierarchical CNN + Transformer** | ✅ Best | Global + local, scalable |

---

## Recommended Model (12GB VRAM, minimal code, HuggingFace-friendly)

We want:
- strong **semantic segmentation** performance (textures/water/impassable),
- **pretrained** weights for fast fine-tuning,
- no attention at 704×704 (too expensive),
- easy training loop via `transformers` + `accelerate`.

### Choice: SegFormer (encoder) + lightweight decoder heads

**Why SegFormer**:
- Strong pretrained semantic segmentation models with efficient transformer encoder design.
- Excellent support in HuggingFace (`SegformerForSemanticSegmentation`).
- Outputs dense per-pixel logits cleanly (we don’t hand-write a U-Net from scratch).

**How we keep global coherence**:
- Condition on the **blueprint** (44×44 intent grid). Embed/upsample it and concatenate as extra channels.
- If needed, add a tiny Transformer **only at 44×44** to enforce symmetry/structure (cheap on 12GB).

**What SegFormer does NOT do by itself**:
- Clean set/instance prediction for sparse gameplay objects. For that we add an **object-table head** (see `MAP_DATA_SCHEMA.md`).

---

## Pipeline Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                      TENSOR-TO-MAP TRAINING & GENERATION                     │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Dataset (raw .map folders)                                                  │
│       │                                                                     │
│       v                                                                     │
│  Style Clustering (unsupervised; tile-style)                                 │
│    - parse BlendTileData.tiles + texture palette                             │
│    - compute texture-name histogram features                                 │
│    - KMeans/PCA to group maps by biome/style                                 │
│    - export clustered folders for training/augmentation                       │
│    - HeightMapData.elevations                                                │
│    - BlendTileData.tiles / blends / impassable / etc                          │
│    - ObjectsList.map_objects                                                 │
│    - StandingWaterAreas polygons                                             │
│       │                                                                     │
│       v                                                                     │
│  Feature Builder                                                             │
│    - canonicalize size (pad/crop to 704×704)                                 │
│    - build blueprint tensor (44×44)                                          │
│    - build mid tensor (176×176)                                              │
│    - build final targets (704×704)                                           │
│       │                                                                     │
│       v                                                                     │
│  Hierarchical Spatial Model                                                  │
│    Stage A: blueprint→mid  (Transformer @ 44×44 + CNN blocks)                │
│    Stage B: mid→final      (U-Net @ 176×176/704×704, transformer @ low-res)  │
│    Heads: textures, height, passability, water, roads, objects               │
│       │                                                                     │
│       v                                                                     │
│  Rule Repair Pass (minimal)                                                  │
│    - enforce clearance, slopes, object-tile invariants                       │
│    - fix disconnected roads, illegal placements                              │
│       │                                                                     │
│       v                                                                     │
│  .map Writer                                                                 │
│    - write HeightMapData, BlendTileData, ObjectsList, StandingWaterAreas     │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Style Clustering (New First Step)

Before training, we group maps into **tile-style clusters** using an unsupervised model on parsed texture usage.

Why:
- enables **style-consistent splits** (train/val/test within a biome)
- enables safe augmentation like **palette swapping inside a cluster**
- makes it easier to diagnose model failures (“works on Yucatan, fails on Snow”)

Implementation:
- Script: `RA 3 maps/python_tools/scripts/cluster_map_styles.py`
- Features: texture-name histogram (from `BlendTileData.tiles // 64` → `BlendTileData.textures[idx].name`)
- Model: StandardScaler → PCA → KMeans (auto-select K via silhouette score)
- Output: exported folder `python_tools/style_clusters/.../exported/cluster_XX/<map_folder>/`

## Canonical Map Size (Variable Map Sizes → Fixed Tensors)

RA3 official maps vary (e.g. 590×440 up to ~700×700).
For training we use a fixed canvas:

- **Final canvas**: `Hf×Wf = 704×704` (divisible by 16)
- **Mid**: `Hm×Wm = 176×176` (4× downscale)
- **Blueprint**: `Hb×Wb = 44×44` (16× downscale)

### Padding/Cropping

All grids use a **valid mask**:

| Tensor | Shape | Meaning |
|--------|-------|---------|
| `valid_mask_final` | `[Hf, Wf]` | 1 where original map exists |

During loss computation, multiply by mask so padded regions don’t contribute.

---

## Training Sample: Exact Tensor Shapes (End-to-End)

Each training item is derived from a single `.map`:

### Inputs (conditioning)

| Name | Shape | dtype | Description |
|------|-------|-------|-------------|
| `blueprint` | `[Hb, Wb, Cb]` = `[44, 44, Cb]` | float32 | Coarse semantic intent grid (derived) |
| `noise_mid` | `[Hm, Wm, Zm]` = `[176, 176, Zm]` | float32 | Noise for stochastic variety (optional) |
| `noise_final` | `[Hf, Wf, Zf]` = `[704, 704, Zf]` | float32 | Noise for stochastic variety (optional) |
| `valid_mask_final` | `[704, 704, 1]` | float32 | 1 = valid tile, 0 = padding |
| `map_meta` | `[M]` | float32/int32 | Scalars: original (W,H), border, water_height, style ids |

Notes:
- If you don’t want stochasticity yet, set `noise_* = 0`.
- `map_meta` can include a `style_id` to select palettes/biomes.

### Outputs/Targets (supervised)

| Target | Shape | dtype | Comes from `.map` |
|--------|-------|-------|-------------------|
| `height_final` | `[704, 704, 1]` | float32 | `HeightMapData.elevations` |
| `texture_class_final` | `[704, 704, 1]` | int64 | `BlendTileData.tiles // 64` |
| `texture_variant_final` | `[704, 704, 1]` | int64 | `BlendTileData.tiles % 64` (optional head) |
| `water_mask_final` | `[704, 704, 1]` | float32 | from `StandingWaterAreas` + water height |
| `impassable_final` | `[704, 704, 1]` | float32 | `BlendTileData.impassable` (or derived) |
| `road_mask_final` | `[704, 704, 1]` | float32 | from road objects / road tiles |
| `object_maps_final` | `[704, 704, Co]` | float32 | sparse occupancy masks per object category |

**Object maps** are the key compromise:
- for dense decor (trees/grass/coral), learn **density fields**, not every instance
- for gameplay objects (Ore/Oil/Waypoint/Tech/Garrison), learn **heatmaps** + a small “instance decoder” (below)

### Optional: Instance Table Head (Gameplay Objects)

For “hard” gameplay objects, a table head is often easier than per-pixel classification:

| Target | Shape | dtype | Description |
|--------|-------|-------|-------------|
| `objects_table` | `[Nmax, F]` | float32/int64 | normalized pos + type + owner + angle |
| `objects_mask` | `[Nmax]` | float32 | 1 for real rows |

Typical:
- `Nmax = 256` (safer if we include **all garrison buildings** + bridges + tech buildings)
- `F = 10` (example): `[category_id, subtype_id, owner_id, x_norm, y_norm, angle_sin, angle_cos, radius, flags...]`

This target is derived from `ObjectsList.map_objects`.

#### Object Table Semantics (v1 scope)

We only supervise a **critical subset** of objects (not decor). In v1 the table rows cover:
- **Player positions**: `*Waypoints/Waypoint` with `uniqueID=Player_{n}_Start`
- **Buildings**: OilDerrick / Hospital / Veterancy / Garage / ObservationPost / Airport / ShipYard (dry dock)
- **All garrison buildings**: k-way subtype classification over a curated garrison-building vocabulary
- **Bridges**: bridge types (k-way or grouped)

To keep training stable and preserve cohesion:
- `category_id` is a small enum (e.g. `spawn`, `tech_building`, `garrison_building`, `bridge`, `resource`).
- `subtype_id` is the k-way class within that category (e.g. which specific garrison building or which tech building).
- We train `subtype_id` with cross-entropy, and coords with L1/Huber.
- The backbone predicts terrain/textures and object table jointly so “object↔texture” dependencies can be learned.

Ordering / matching:
- Either sort rows deterministically (by `category_id`, `subtype_id`, `x_norm`, `y_norm`)
- Or use a set-prediction loss (Hungarian matching) if ordering is unstable.

---

## Blueprint Tensor Definition (44×44)

Blueprint is **derived deterministically** from the real map (for training), then later can be generated procedurally.

Suggested `Cb = 12` channels:

| Channel | Range | How to compute (per 16×16 tile block) |
|---------|-------|----------------------------------------|
| `zone_base` | {0,1} | block contains player start waypoint |
| `zone_expansion` | {0,1} | near ore/oil clusters or high buildable potential |
| `zone_neutral` | {0,1} | default |
| `player_id` | 0..N or -1 | which player dominates this block |
| `importance` | 0..1 | weighted sum of object densities + centrality |
| `road_strength` | 0..1 | fraction of road tiles/objects in block |
| `height_mean` | normalized | mean height in block |
| `height_std` | normalized | std height in block (cliffiness proxy) |
| `water_frac` | 0..1 | fraction of tiles underwater |
| `texture_entropy` | 0..1 | entropy of texture_class distribution |
| `decor_density` | 0..1 | decor objects per tile (clipped) |
| `style_id` | one-hot or scalar | biome/style indicator (optional) |

---

## Model Outputs (What the Network Predicts)

We do **multi-task prediction** with separate heads:

| Head | Output | Loss |
|------|--------|------|
| Height | `height_final` (regression) | masked L1 / Huber |
| Texture class | `texture_class_final` (K-way) | masked cross entropy |
| Texture variant (optional) | `texture_variant_final` (64-way) | masked cross entropy |
| Water | `water_mask_final` | masked BCE |
| Impassable | `impassable_final` | masked BCE |
| Roads | `road_mask_final` | masked BCE |
| Objects (heatmaps) | `object_maps_final` | focal loss / BCE |
| Objects (table, optional) | `objects_table` | CE for type/owner + L1 for coords |

“Masked” means multiply the loss by `valid_mask_final`.

---

## Data Augmentation Policy (Allowed)

RA3 maps are grid-structured; we only apply **whole-map right-angle rotations**:

- ✅ Allowed rotations: **0°, 90°, 180°, 270°**
- ❌ Not allowed: arbitrary rotations, warps, elastic transforms, random crops that break borders

When rotating a training sample:
- Rotate **all grids** (height/texture/water/impassable/roads/valid mask) identically.
- Rotate the **object table**:
  - Update `(x_norm, y_norm)` with the same rotation around map center.
  - Add the rotation to object yaw and re-encode as `(sin, cos)`.

Optional Phase-2 augmentation (not v1):
- **Style-consistent texture remapping** (swap a whole texture family to another compatible family),
  which requires a curated mapping (including transition textures + blend behavior).

---

## Post-Pass (Minimal Repair)

Rules are not the generator; they only enforce invariants:

- **Water**: if `height < water_height` → force `water_mask=1`, `impassable=1`
- **Cliffs**: if local slope > threshold → force `impassable=1`
- **Object clearance**: remove/shift objects overlapping water/cliffs
- **Road connectivity**: ensure roads connect base zones (optional)

---

## `.map` Writer Responsibilities

Given final predicted tensors (and optional object table), write:

| `.map` Asset | From tensors |
|-------------|--------------|
| `HeightMapData.elevations` | `height_final` (crop to original size) |
| `BlendTileData.tiles` | `texture_class_final*64 + texture_variant_final` (or choose variants deterministically) |
| `BlendTileData.blends/blend_info` | deterministic blend lookup (still valid here) |
| `BlendTileData.impassable` | `impassable_final` (plus rule repair) |
| `StandingWaterAreas` | polygonize `water_mask_final` or keep template polygons |
| `ObjectsList.map_objects` | from `objects_table` + optional sampled decor density |

---

## Component Status (Updated)

| Component | Status | Approach |
|-----------|--------|----------|
| Tensor dataset spec | ✅ Defined | blueprint/mid/final + masks |
| Height generation | ✅ In-model | regression head |
| Passability | ✅ In-model + repair | predict + enforce invariants |
| Texture blending | ✅ Post writer | deterministic lookup |
| Texture assignment | ✅ In-model | classify texture class (+ optional variant) |
| Gameplay objects | ✅ In-model | heatmaps + optional table head |
| Decorative objects | ✅ In-model | density fields + sampler |

---

## Relationship to `LLM_MAP_SPEC_SCHEMA.md`

That JSON schema is **not the training target anymore**.

What we keep:
- Concepts (zones, paths, object categories) are still useful **as blueprint features**.

What is deprecated for now:
- “LLM outputs full JSON spec”
- token estimation sections

---

## Next Steps

1. Implement a feature builder: `.map` → tensors (blueprint + targets + masks)
2. Choose a minimal channel set (start small: height + texture_class + water + impassable + 4 gameplay heatmaps)
3. Train a baseline hierarchical model
4. Add object table head (optional) after heatmaps work
5. Add decor density fields last

---

*Last Updated: January 2026*

