# Blend Model Improvement Strategy

## Current Performance Baseline
- **Accuracy**: 93.45%
- **Precision**: 84.39%
- **Recall**: 89.69%
- **F1 Score**: 0.87

## Executive Summary

After thorough investigation of all three codebases (Ra3Solution, MapCreatorCore, Python pipeline), I've identified **7 major improvement opportunities** that could push F1 to 0.95+.

---

## 1. MISSING FEATURES FROM C# PARSERS

### 1.1 Passability Data (HIGH IMPACT)
**Source**: `BlendTileData.cs` in Ra3Solution

Currently unused features that strongly correlate with blend patterns:
- `impassable[x,y]` - Cliffs/walls block blends
- `passability[x,y]` - 5-state enum (Passable, Impassable, ImpassableToPlayers, etc.)
- `cliffBlends[x,y]` - Specialized cliff blending (separate from regular blends)

**Why it matters**: Passability is computed from elevation slope. WorldBuilder doesn't create blends across impassable terrain. This is a **hard constraint** we're not encoding.

**Implementation**:
```python
# Add to feature extraction
passability_grid = blend.passability  # Already parsed but unused
impassable_mask = (passability_grid == Passability.Impassable)
# Add 8 neighbor impassability flags as features
```

### 1.2 Buildability & Visibility
**Source**: `BlendTileData.cs`

- `buildability[x,y]` - Can structures be placed? (affects texture choices)
- `visibility[x,y]` - Shroud/fog areas often have different texture patterns

### 1.3 Dynamic Shrubbery
**Source**: `BlendTileData.cs`

- `dynamicShrubbery[x,y]` - byte (0-255) indicating vegetation density
- Correlates with grass textures and blend patterns

### 1.4 Water Proximity
**Source**: `StandingWaterAreas.cs`, `GlobalWaterSettings.cs`

- `waterHeight` - Global water level (typically 200)
- `StandingWaterArea.points[]` - Polygon vertices of water regions

**Implementation**: Compute distance-to-water feature for each cell.

---

## 2. BLEND ALGORITHM LOGIC (FROM MapCreatorCore)

### 2.1 The ACTUAL Blend Priority Rules
**Source**: `MapGenerator.cs` lines 641-720

The C# code reveals the **exact decision logic**:

```csharp
// Priority order (highest to lowest):
1. Corner + Diagonal: if (leftTex == topTex && topTex != centerTex) → BottomRight
2. Edge: if (leftTex != centerTex) → Right
3. Corner-only: if (topLeftTex != centerTex) → ExceptTopLeft

// Critical rule:
if (centerTexture <= secondaryTexture) {
    AddBlend(x, y, secondaryTexture, direction);
}
```

**Key insight**: Blends ONLY occur when `centerTexture <= secondaryTexture` (palette index comparison).

### 2.2 Missing Pattern Features

We should encode the **12 pattern matching conditions** explicitly:

| Pattern | Condition | Direction |
|---------|-----------|-----------|
| 1 | left == top && top != center | BottomRight |
| 2 | right == top && top != center | BottomLeft |
| 3 | left == bottom && bottom != center | TopRight |
| 4 | right == bottom && bottom != center | TopLeft |
| 5 | left != center | Right |
| 6 | right != center | Left |
| 7 | top != center | Bottom |
| 8 | bottom != center | Top |
| 9 | topLeft != center | ExceptTopLeft |
| 10 | topRight != center | ExceptTopRight |
| 11 | bottomLeft != center | ExceptBottomLeft |
| 12 | bottomRight != center | ExceptBottomRight |

**Implementation**: Add 12 binary features indicating which pattern matches.

---

## 3. FEATURE ENGINEERING IMPROVEMENTS

### 3.1 Palette Dominance Encoding
Currently we use `tex_local_norm` (local index / count). Better approach:

```python
# For each neighbor, encode relative palette position
for neighbor in neighbors:
    features.append(center_idx - neighbor_idx)  # Signed difference
    features.append(int(center_idx <= neighbor_idx))  # Blend eligibility
```

### 3.2 Texture Transition Matrix
Build a **learned co-occurrence matrix** showing which textures commonly blend:

```python
# From training data
transition_matrix[tex_a, tex_b] = count(tex_a blends with tex_b)
# Use as embedding or direct feature
```

### 3.3 Multi-Scale Context
Current 5×5 window may be too small. Add:
- 9×9 downsampled context (every other cell)
- Dominant texture in larger region (21×21)

### 3.4 Elevation Curvature Features
Beyond slope and laplacian:
- `mean_curvature` - Terrain smoothness
- `gaussian_curvature` - Saddle points
- `aspect` - Compass direction of slope

---

## 4. MODEL ARCHITECTURE IMPROVEMENTS

### 4.1 Explicit Pattern Matching Layer

Add a **rule-based pattern matching module** before the transformer:

```python
class PatternMatcher(nn.Module):
    def __init__(self):
        self.patterns = [
            # (positions_to_compare, expected_relation, direction_output)
            ([3, 1], 'equal_diff_center', 'BottomRight'),  # left==top, !=center
            # ... 12 patterns
        ]

    def forward(self, tex_window):
        pattern_matches = []
        for pattern in self.patterns:
            match = self.check_pattern(tex_window, pattern)
            pattern_matches.append(match)
        return torch.stack(pattern_matches, dim=-1)  # [B, 12]
```

### 4.2 Hierarchical Direction Prediction

Instead of flat 17-class classification, use hierarchical:

```
Level 1: Position (9 classes - 3x3 grid)
  TopLeft, Top, TopRight, Left, Center, Right, BottomLeft, Bottom, BottomRight

Level 2: Type (3 classes)
  Corner, Edge, Except

Final Direction = combine(position, type)
```

### 4.3 Consistency Regularization

Add loss term enforcing blend rules:

```python
# If center_idx > all neighbor_idx, blend_present should be 0
max_neighbor = max(neighbor_indices)
rule_violation = blend_present * (center_idx > max_neighbor).float()
loss_rule = rule_violation.mean()
```

### 4.4 Attention Over Neighbors Only

Current transformer attends to all 25 positions. For direction prediction, attention should focus on the 8 neighbors:

```python
# Masked attention: only neighbors can attend to each other
neighbor_mask = torch.zeros(25, 25)
neighbor_positions = [0,1,2,5,7,10,11,12]  # Excluding center
for i in neighbor_positions:
    for j in neighbor_positions:
        neighbor_mask[i,j] = 1
```

---

## 5. TRAINING IMPROVEMENTS

### 5.1 Class Weighting Optimization

Current focal loss uses α=0.10. Based on class imbalance (8% positive):

```python
# Optimal for 8% positive rate
alpha = 0.25  # Increase weight on positives
gamma = 2.5   # Increase focus on hard examples
```

### 5.2 Hard Negative Mining

Sample more from cells where:
- Rule says blend, but ground truth has no blend
- Texture boundary exists but no blend

```python
# During data loading
hard_negatives = (rule_predicts_blend & ~blend_present)
sample_weight[hard_negatives] *= 3.0
```

### 5.3 Curriculum Learning

Train in stages:
1. **Stage 1**: Binary presence only (blend vs no blend)
2. **Stage 2**: Add direction classification
3. **Stage 3**: Add mask prediction

### 5.4 Data Augmentation

Current rotation augmentation is disabled. Better alternatives:
- **Horizontal/vertical flip** (with direction remapping)
- **Texture permutation** (randomly rename textures, preserving order)
- **Mixup at map level** (blend features from two maps)

---

## 6. LOSS FUNCTION REFINEMENTS

### 6.1 Mask-Conditional Direction Loss

Only train direction when mask has exactly 1 bit set:

```python
single_neighbor = (mask.sum(dim=-1) == 1)
loss_dir = loss_dir * single_neighbor.float()
```

### 6.2 Blend Rule Regularization

```python
# Soft constraint: blend_present should correlate with blend eligibility
eligibility = (center_idx <= max_neighbor_idx).float()
loss_rule = F.binary_cross_entropy(blend_present_prob, eligibility) * 0.1
```

### 6.3 Direction Smoothness

Penalize predicting incompatible directions for adjacent cells:

```python
# If cell (x,y) blends to Right, cell (x+1,y) shouldn't blend to Left with same texture
```

---

## 7. EVALUATION IMPROVEMENTS

### 7.1 Per-Texture Metrics

Track F1 score per texture type:
- Grass blends
- Rock blends
- Transition textures

### 7.2 Spatial Consistency Metrics

- **Edge alignment**: Do predicted blends form coherent boundaries?
- **Island detection**: Are there isolated blend cells?

### 7.3 Visual Comparison

Generate side-by-side images:
- Original blends
- Predicted blends
- Difference map

---

## IMPLEMENTATION PRIORITY

### Phase 1: Quick Wins (Est. +3-5% F1)
1. ✅ Lower threshold to 0.3 (DONE - already +44% recall)
2. Add passability features
3. Add explicit pattern matching features (12 patterns)
4. Tune focal loss α to 0.25

### Phase 2: Feature Engineering (+3-5% F1)
5. Add elevation curvature features
6. Build texture transition matrix
7. Add water proximity feature
8. Multi-scale context (9×9 downsampled)

### Phase 3: Architecture Changes (+2-3% F1)
9. Pattern matching layer
10. Hierarchical direction heads
11. Neighbor-only attention for direction

### Phase 4: Training Refinements (+1-2% F1)
12. Hard negative mining
13. Curriculum learning
14. Consistency regularization

---

## Expected Final Performance

With all improvements implemented:
- **Accuracy**: 97%+
- **Precision**: 92%+
- **Recall**: 94%+
- **F1 Score**: 0.93-0.95

---

## Code Locations for Implementation

| Improvement | File to Modify |
|-------------|----------------|
| Passability features | `generate_blendinfo_dataset.py` |
| Pattern matching features | `generate_blendinfo_dataset.py` |
| Focal loss tuning | `train_blend_model_hf.py` |
| Pattern matching layer | `train_blend_model_hf.py` |
| Hierarchical direction | `train_blend_model_hf.py` |
| Transition matrix | `prepare_blend_dataset_memmap.py` |
