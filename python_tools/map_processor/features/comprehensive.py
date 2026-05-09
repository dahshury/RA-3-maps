"""Comprehensive feature extractor: parsed Ra3Map -> (C, W, H) feature stack.

Channels are organised in groups. The same module also returns:
  - channel_names: list of human-readable names matching the C axis
  - per-tile target palette index (when target_blend is provided)
  - object tokens: a (N, F) tensor for cross-attention models
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Tuple, Optional

import numpy as np
from scipy.ndimage import distance_transform_edt, gaussian_filter, sobel


# Broad object categories used for density / direction features. These collapse
# the fine-grained ObjectCategoryConfig categories into the buckets that drive
# texture choice in practice.
OBJECT_CATEGORY_NAMES = [
    "ore_node",        # economic
    "oil_derrick",     # economic
    "road",            # roads / paths / bridges
    "structure",       # military buildings (CY, barracks, factory, defense, ...)
    "garrison",        # garrisonable civilian buildings
    "super_weapon",    # super-weapon class structures
    "player_start",    # MP starting markers
]

DENSITY_RADII = (1, 4, 16)   # per-category density ring radii in tiles


# Map fine-grained ObjectCategoryConfig keys -> coarse buckets above.
def _coarse_bucket(category_key: str) -> Optional[str]:
    if category_key in ("ore_node",):
        return "ore_node"
    if category_key in ("oil_derrick",):
        return "oil_derrick"
    if category_key in ("road",):
        return "road"
    if category_key in ("super_weapon",):
        return "super_weapon"
    if category_key in ("player_start",):
        return "player_start"
    if category_key.startswith("garrison_"):
        return "garrison"
    if category_key in (
        "construction_yard", "barracks", "war_factory", "factory",
        "airfield", "naval_yard", "laser_tower", "power_plant",
        "base_defense", "tower", "bunker",
        "building_observation_post", "building_hospital", "building_garage",
        "building_snowy", "building_convention_center", "building_port_structure",
        "building_airport", "building_military", "building_cargo_container",
        "building_supply", "building_veterancy", "building_shipyard",
        "building_tech_structure", "building_soviet", "building_other",
    ):
        return "structure"
    return None


@dataclass
class FeatureStack:
    array: np.ndarray             # (C, W, H) float32
    names: List[str]              # length C
    target_tiles: Optional[np.ndarray] = None      # (W, H) int32, palette index per tile (None unless requested)
    palette: Optional[List[str]] = None            # texture names indexed by palette id
    object_tokens: Optional[np.ndarray] = None     # (N, F) float32 — see TOKEN_FEATURE_NAMES
    token_owners: Optional[List[str]] = None       # list of owners aligned to object_tokens
    width: int = 0
    height: int = 0


TOKEN_FEATURE_NAMES = [
    "x_norm", "y_norm", "z_norm",
    "angle_sin", "angle_cos",
    "category_id",            # int matching OBJECT_CATEGORY_NAMES; -1 for uncategorised
    "scale",                  # placeholder = 1.0
    "is_player_start",
]


# -------------------------- Helpers --------------------------

def _gradient_magnitude(z: np.ndarray) -> np.ndarray:
    gx = sobel(z, axis=0, mode="nearest")
    gy = sobel(z, axis=1, mode="nearest")
    return np.sqrt(gx * gx + gy * gy)


def _gradient_xy(z: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    gx = sobel(z, axis=0, mode="nearest")
    gy = sobel(z, axis=1, mode="nearest")
    return gx, gy


def _local_rank(z: np.ndarray, win: int = 9) -> np.ndarray:
    """Percentile rank of every cell within a win x win window — cheap proxy."""
    # Approximate via mean + std normalisation rather than full window sort.
    blurred = gaussian_filter(z, sigma=win / 4.0)
    diff = z - blurred
    rng = np.std(diff) + 1e-6
    return np.clip(diff / (3 * rng) + 0.5, 0.0, 1.0)


def _local_var(z: np.ndarray, sigma: float = 2.0) -> np.ndarray:
    m = gaussian_filter(z, sigma=sigma)
    var = gaussian_filter((z - m) ** 2, sigma=sigma)
    return np.sqrt(np.maximum(var, 0.0))


def _norm01(x: np.ndarray, lo: Optional[float] = None, hi: Optional[float] = None) -> np.ndarray:
    if lo is None:
        lo = float(np.percentile(x, 1))
    if hi is None:
        hi = float(np.percentile(x, 99))
    if hi - lo < 1e-6:
        return np.zeros_like(x, dtype=np.float32)
    return np.clip((x - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)


# -------------------------- Geometry features --------------------------

def _geometry_features(elev: np.ndarray) -> Tuple[List[np.ndarray], List[str]]:
    chans, names = [], []
    z = elev.astype(np.float32)
    chans.append(_norm01(z));                                names.append("elev_norm")
    gmag = _gradient_magnitude(z)
    chans.append(_norm01(gmag));                             names.append("slope_mag")
    gx, gy = _gradient_xy(z)
    angle = np.arctan2(gy, gx)
    chans.append((np.sin(angle) * 0.5 + 0.5).astype(np.float32)); names.append("slope_sin")
    chans.append((np.cos(angle) * 0.5 + 0.5).astype(np.float32)); names.append("slope_cos")
    # Curvature ~ laplacian of z (concave/convex)
    lap = (np.roll(z, 1, 0) + np.roll(z, -1, 0) + np.roll(z, 1, 1) + np.roll(z, -1, 1) - 4 * z)
    chans.append(_norm01(lap));                              names.append("curvature")
    chans.append(_local_rank(z));                            names.append("elev_local_rank")
    chans.append(_norm01(_local_var(z)));                    names.append("roughness")
    return chans, names


# -------------------------- Water features --------------------------

def _water_features(ctx, W: int, H: int) -> Tuple[List[np.ndarray], List[str]]:
    """Returns inside-water binary + signed distance + water-type one-hots.

    Source: BlendTileData.passability + StandingWaterAreas/RiverAreas/StandingWaveAreas
    polygon AABBs (cheap rasterisation).
    """
    chans, names = [], []
    in_water = np.zeros((W, H), dtype=np.float32)
    type_standing = np.zeros((W, H), dtype=np.float32)
    type_river = np.zeros((W, H), dtype=np.float32)
    type_wave = np.zeros((W, H), dtype=np.float32)

    blend = ctx.get_asset("BlendTileData")
    # Passability=Impassable_or_water heuristic via z<water_level captured by polygons below.

    def _rasterise_areas(areas, mask, attr_points: str = "polygon"):
        for a in areas:
            pts = getattr(a, attr_points, None) or getattr(a, "points", None)
            if not pts:
                continue
            xs = [p[0] / 10.0 for p in pts]
            ys = [p[1] / 10.0 for p in pts]
            x0, x1 = max(0, int(min(xs))), min(W - 1, int(max(xs)))
            y0, y1 = max(0, int(min(ys))), min(H - 1, int(max(ys)))
            if x1 > x0 and y1 > y0:
                mask[x0:x1 + 1, y0:y1 + 1] = 1.0

    standing = ctx.get_asset("StandingWaterAreas")
    rivers = ctx.get_asset("RiverAreas")
    waves = ctx.get_asset("StandingWaveAreas")
    if standing is not None:
        _rasterise_areas(getattr(standing, "water_areas", []), type_standing)
    if rivers is not None:
        _rasterise_areas(getattr(rivers, "areas", []), type_river)
    if waves is not None:
        _rasterise_areas(getattr(waves, "areas", []), type_wave)

    in_water = np.maximum.reduce([type_standing, type_river, type_wave])

    # Signed distance to water (positive on land, 0 at water edge, negative inside).
    if in_water.any():
        dist_out = distance_transform_edt(in_water == 0).astype(np.float32)
        dist_in = distance_transform_edt(in_water > 0).astype(np.float32)
        signed = dist_out - dist_in
    else:
        signed = np.full((W, H), 1e3, dtype=np.float32)

    chans.append(in_water);                                  names.append("in_water")
    # Squash signed distance to ~[0,1]
    chans.append(_norm01(signed, lo=-30.0, hi=30.0));        names.append("dist_to_water_signed")
    chans.append(type_standing);                             names.append("water_type_standing")
    chans.append(type_river);                                names.append("water_type_river")
    chans.append(type_wave);                                 names.append("water_type_wave")
    return chans, names


# -------------------------- Object features --------------------------

def _object_features(ctx, W: int, H: int) -> Tuple[List[np.ndarray], List[str], np.ndarray, List[str]]:
    """Per-tile object density / direction / orientation features.

    Returns (channels, names, object_tokens, token_owners).
    """
    from ..utils.object_categories import ObjectCategoryConfig
    cfg = ObjectCategoryConfig()

    objs = ctx.get_asset("ObjectsList")
    obj_records = []  # (bucket, x_tile, y_tile, z, angle_rad, owner, type_name)

    if objs is not None:
        for obj in objs.map_objects:
            name = obj.type_name or ""
            if not name:
                continue
            cat, draw = cfg.get_category_for_object(name)
            if cat is None or not draw:
                # Player start markers are emitted as "*Waypoints/Player_X_Start" — keep them
                if "player" not in name.lower() or "start" not in name.lower():
                    continue
            bucket = None
            if cat is not None:
                # Find category_key by reverse lookup on cat
                for k, v in cfg.categories.items():
                    if v is cat:
                        bucket = _coarse_bucket(k)
                        break
            if bucket is None and "player" in name.lower() and "start" in name.lower():
                bucket = "player_start"
            if bucket is None:
                continue
            x_t = obj.position[0] / 10.0
            y_t = obj.position[1] / 10.0
            z_t = obj.position[2]
            angle_rad = np.deg2rad(obj.angle or 0.0)
            owner = obj.original_owner or ""
            obj_records.append((bucket, x_t, y_t, z_t, angle_rad, owner, name))

    chans, names = [], []
    # Density at multiple radii per category
    for bucket in OBJECT_CATEGORY_NAMES:
        # Stamp positions into a small base map, then blur with sigma=radius
        base = np.zeros((W, H), dtype=np.float32)
        for rec in obj_records:
            if rec[0] != bucket:
                continue
            xi = int(np.clip(round(rec[1]), 0, W - 1))
            yi = int(np.clip(round(rec[2]), 0, H - 1))
            base[xi, yi] += 1.0
        for r in DENSITY_RADII:
            blurred = gaussian_filter(base, sigma=r)
            chans.append(_norm01(blurred, lo=0.0, hi=blurred.max() + 1e-6))
            names.append(f"density_{bucket}_r{r}")

    # Per-category nearest-distance map (compute via EDT of bucket presence)
    for bucket in OBJECT_CATEGORY_NAMES:
        occ = np.zeros((W, H), dtype=np.float32)
        for rec in obj_records:
            if rec[0] != bucket:
                continue
            xi = int(np.clip(round(rec[1]), 0, W - 1))
            yi = int(np.clip(round(rec[2]), 0, H - 1))
            occ[xi, yi] = 1.0
        if occ.any():
            dist = distance_transform_edt(occ == 0).astype(np.float32)
        else:
            dist = np.full((W, H), float(max(W, H)), dtype=np.float32)
        chans.append(_norm01(dist, lo=0.0, hi=64.0));    names.append(f"dist_nearest_{bucket}")

    # Direction & orientation of nearest object overall
    if obj_records:
        coords = np.array([[r[1], r[2]] for r in obj_records], dtype=np.float32)   # (N, 2)
        angles = np.array([r[4] for r in obj_records], dtype=np.float32)            # (N,)
        # Compute per-tile nearest-object index
        xs, ys = np.meshgrid(np.arange(W), np.arange(H), indexing="ij")
        # Vectorised nearest neighbour (small N, can afford O(N*W*H) for now)
        dx = xs[..., None] - coords[None, None, :, 0]
        dy = ys[..., None] - coords[None, None, :, 1]
        d2 = dx * dx + dy * dy
        nn = np.argmin(d2, axis=-1)
        nn_dx = np.take_along_axis(dx, nn[..., None], axis=-1)[..., 0]
        nn_dy = np.take_along_axis(dy, nn[..., None], axis=-1)[..., 0]
        dir_angle = np.arctan2(nn_dy, nn_dx)
        face_angle = angles[nn]
        chans.append(((np.sin(dir_angle) * 0.5) + 0.5).astype(np.float32));    names.append("dir_to_nearest_sin")
        chans.append(((np.cos(dir_angle) * 0.5) + 0.5).astype(np.float32));    names.append("dir_to_nearest_cos")
        chans.append(((np.sin(face_angle) * 0.5) + 0.5).astype(np.float32));   names.append("face_of_nearest_sin")
        chans.append(((np.cos(face_angle) * 0.5) + 0.5).astype(np.float32));   names.append("face_of_nearest_cos")
    else:
        z = np.zeros((W, H), dtype=np.float32)
        chans += [z, z, z, z]
        names += ["dir_to_nearest_sin", "dir_to_nearest_cos",
                  "face_of_nearest_sin", "face_of_nearest_cos"]

    # Object tokens for cross-attention
    if obj_records:
        cat_to_id = {n: i for i, n in enumerate(OBJECT_CATEGORY_NAMES)}
        tokens = np.zeros((len(obj_records), len(TOKEN_FEATURE_NAMES)), dtype=np.float32)
        for i, rec in enumerate(obj_records):
            bucket, x, y, z, ang, owner, name = rec
            tokens[i, 0] = x / max(W, 1)
            tokens[i, 1] = y / max(H, 1)
            tokens[i, 2] = z / 100.0
            tokens[i, 3] = (np.sin(ang) + 1) * 0.5
            tokens[i, 4] = (np.cos(ang) + 1) * 0.5
            tokens[i, 5] = float(cat_to_id.get(bucket, -1))
            tokens[i, 6] = 1.0
            tokens[i, 7] = 1.0 if bucket == "player_start" else 0.0
        token_owners = [r[5] for r in obj_records]
    else:
        tokens = np.zeros((0, len(TOKEN_FEATURE_NAMES)), dtype=np.float32)
        token_owners = []

    return chans, names, tokens, token_owners


# -------------------------- Mask features --------------------------

ALL_BLEND_MASK_NAMES = (
    "pass_passable", "pass_impassable", "pass_imp_players", "pass_imp_air", "pass_extra",
    "buildability", "visibility", "tib_growability", "dynamic_shrubbery",
)


def _mask_features(ctx, W: int, H: int,
                   subset: Optional[List[str]] = None) -> Tuple[List[np.ndarray], List[str]]:
    """Passability / buildability / visibility / tib / shrubbery channels.

    Parameters
    ----------
    subset : if provided, only return mask channels whose name is in this list.
             Order is preserved against ALL_BLEND_MASK_NAMES regardless of input order.
    """
    chans, names = [], []
    keep = set(subset) if subset is not None else set(ALL_BLEND_MASK_NAMES)
    blend = ctx.get_asset("BlendTileData")

    def _bool_chan(arr):
        if arr is None:
            return np.zeros((W, H), dtype=np.float32)
        a = np.asarray(arr).astype(np.float32)
        out = np.zeros((W, H), dtype=np.float32)
        cw, ch = min(W, a.shape[0]), min(H, a.shape[1])
        out[:cw, :ch] = a[:cw, :ch]
        return out

    if blend is None:
        z = np.zeros((W, H), dtype=np.float32)
        for n in ALL_BLEND_MASK_NAMES:
            if n in keep:
                chans.append(z); names.append(n)
        return chans, names

    # Passability one-hot
    pas = np.asarray(blend.passability, dtype=np.int32)
    pw, ph = pas.shape
    target = np.zeros((W, H), dtype=np.int32)
    cw, ch = min(W, pw), min(H, ph)
    target[:cw, :ch] = pas[:cw, :ch]
    pas = target
    for code, label in enumerate([
        "pass_passable", "pass_impassable", "pass_imp_players", "pass_imp_air", "pass_extra",
    ]):
        if label in keep:
            chans.append((pas == code).astype(np.float32));    names.append(label)

    if "buildability" in keep:
        chans.append(_bool_chan(blend.buildability));          names.append("buildability")
    if "visibility" in keep:
        chans.append(_bool_chan(blend.visibility));            names.append("visibility")
    if "tib_growability" in keep:
        chans.append(_bool_chan(blend.tiberium_growability));  names.append("tib_growability")
    if "dynamic_shrubbery" in keep:
        if blend.dynamic_shrubbery is not None:
            ds = np.asarray(blend.dynamic_shrubbery, dtype=np.float32) / 255.0
            out = np.zeros((W, H), dtype=np.float32)
            cw, ch = min(W, ds.shape[0]), min(H, ds.shape[1])
            out[:cw, :ch] = ds[:cw, :ch]
            chans.append(out)
        else:
            chans.append(np.zeros((W, H), dtype=np.float32))
        names.append("dynamic_shrubbery")
    return chans, names


# -------------------------- Strategic / position features --------------------------

def _strategic_features(ctx, W: int, H: int, *,
                        slope_mag: Optional[np.ndarray] = None,
                        use_blend_masks: bool = True) -> Tuple[List[np.ndarray], List[str]]:
    chans, names = [], []
    xs, ys = np.meshgrid(np.arange(W), np.arange(H), indexing="ij")
    xs = xs.astype(np.float32); ys = ys.astype(np.float32)

    cx, cy = (W - 1) / 2.0, (H - 1) / 2.0
    dist_center = np.sqrt((xs - cx) ** 2 + (ys - cy) ** 2)
    chans.append(_norm01(dist_center));                      names.append("dist_to_center")

    dist_edge = np.minimum(np.minimum(xs, W - 1 - xs), np.minimum(ys, H - 1 - ys))
    chans.append(_norm01(dist_edge));                        names.append("dist_to_edge")

    # Distance to nearest MP spawn (use MPPositionList + objects-flagged-as-player-start as fallback)
    mp = ctx.get_asset("MPPositionList")
    objs = ctx.get_asset("ObjectsList")
    spawns: List[Tuple[float, float]] = []
    if objs is not None:
        for obj in objs.map_objects:
            n = (obj.type_name or "").lower()
            if "player" in n and "start" in n:
                spawns.append((obj.position[0] / 10.0, obj.position[1] / 10.0))
    occ = np.zeros((W, H), dtype=np.float32)
    for sx, sy in spawns:
        ix = int(np.clip(round(sx), 0, W - 1))
        iy = int(np.clip(round(sy), 0, H - 1))
        occ[ix, iy] = 1.0
    if occ.any():
        dist_spawn = distance_transform_edt(occ == 0).astype(np.float32)
    else:
        dist_spawn = np.full((W, H), float(max(W, H)), dtype=np.float32)
    chans.append(_norm01(dist_spawn, lo=0.0, hi=64.0));      names.append("dist_to_nearest_spawn")

    # Quadrant / side relative to spawn(s): for the closest spawn, normalised dx/dy
    if spawns:
        coords = np.array(spawns, dtype=np.float32)
        dx = xs[..., None] - coords[None, None, :, 0]
        dy = ys[..., None] - coords[None, None, :, 1]
        d2 = dx * dx + dy * dy
        nn = np.argmin(d2, axis=-1)
        nn_dx = np.take_along_axis(dx, nn[..., None], axis=-1)[..., 0]
        nn_dy = np.take_along_axis(dy, nn[..., None], axis=-1)[..., 0]
        chans.append(_norm01(nn_dx, lo=-W / 2.0, hi=W / 2.0));    names.append("offset_dx_to_spawn")
        chans.append(_norm01(nn_dy, lo=-H / 2.0, hi=H / 2.0));    names.append("offset_dy_to_spawn")
    else:
        z = np.zeros((W, H), dtype=np.float32)
        chans += [z, z]; names += ["offset_dx_to_spawn", "offset_dy_to_spawn"]

    # "Defensibility" — fraction of cliff-like tiles in radius.
    # Source: passability mask if available (paint-time), else slope-magnitude
    # threshold (geometry only). The slope-based path is what an unpainted
    # textureless map gives us.
    blend = ctx.get_asset("BlendTileData")
    cliff_mask = None
    if use_blend_masks and blend is not None and blend.passability is not None:
        pas = np.asarray(blend.passability, dtype=np.int32)
        pw, ph = pas.shape
        impass = (pas != 0).astype(np.float32)
        out = np.zeros((W, H), dtype=np.float32)
        cw, ch = min(W, pw), min(H, ph)
        out[:cw, :ch] = impass[:cw, :ch]
        cliff_mask = out
    elif slope_mag is not None:
        # Top quartile of slope magnitude approximates "cliff-like" terrain.
        thr = float(np.percentile(slope_mag, 88))
        cliff_mask = (slope_mag > thr).astype(np.float32)

    if cliff_mask is None:
        chans.append(np.zeros((W, H), dtype=np.float32));    names.append("defensibility")
        chans.append(np.zeros((W, H), dtype=np.float32));    names.append("dist_to_cliff")
    else:
        defensibility = gaussian_filter(cliff_mask, sigma=8.0)
        chans.append(_norm01(defensibility));                names.append("defensibility")
        if cliff_mask.any():
            cliff_dist = distance_transform_edt(cliff_mask == 0).astype(np.float32)
        else:
            cliff_dist = np.full((W, H), 64.0, dtype=np.float32)
        chans.append(_norm01(cliff_dist, lo=0.0, hi=32.0));  names.append("dist_to_cliff")

    return chans, names


# -------------------------- Top-level extractor --------------------------

def extract_features(
    ra3_map,
    *,
    extract_target: bool = False,
    style_id: Optional[int] = None,
    n_styles: int = 8,
    include_blend_features: bool = True,
    blend_mask_subset: Optional[List[str]] = None,
) -> FeatureStack:
    """Extract a (C, W, H) feature stack from a parsed Ra3Map.

    Parameters
    ----------
    ra3_map : Ra3Map (already parsed)
    extract_target : if True, also returns target_tiles + palette derived from BlendTileData.
    style_id : optional style cluster id; broadcast as one-hot across the map.
    n_styles : total number of style clusters (for one-hot length).
    include_blend_features : if False, omit features that come from BlendTileData
        masks (passability/buildability/visibility/tib/shrubbery) and substitute
        slope-derived equivalents for defensibility/cliff-distance. Use False
        when simulating inference on a textureless user-built map where those
        masks haven't been painted yet.
    blend_mask_subset : optional explicit list of mask names to keep. When set,
        overrides include_blend_features and includes only the named masks.
        Names: pass_passable, pass_impassable, pass_imp_players, pass_imp_air,
        pass_extra, buildability, visibility, tib_growability, dynamic_shrubbery.
    """
    ctx = ra3_map.get_context()
    blend = ctx.get_asset("BlendTileData")
    h_asset = ctx.get_asset("HeightMapData")
    if blend is None or h_asset is None:
        raise ValueError("Map missing BlendTileData or HeightMapData")

    W = int(blend.map_width)
    H = int(blend.map_height)

    # Crop heightmap to BlendTileData footprint
    elev = np.asarray(h_asset.elevations, dtype=np.float32)
    eW, eH = elev.shape
    cw, ch = min(W, eW), min(H, eH)
    elev_t = np.zeros((W, H), dtype=np.float32)
    elev_t[:cw, :ch] = elev[:cw, :ch]
    elev = elev_t

    chans: List[np.ndarray] = []
    names: List[str] = []

    g_chans, g_names = _geometry_features(elev)
    chans.extend(g_chans); names.extend(g_names)
    # Reuse the slope_mag channel produced by _geometry_features for the
    # slope-based cliff fallback when blend masks are unavailable.
    slope_mag_idx = g_names.index("slope_mag") if "slope_mag" in g_names else None
    slope_mag = g_chans[slope_mag_idx] if slope_mag_idx is not None else None

    w_chans, w_names = _water_features(ctx, W, H)
    chans.extend(w_chans); names.extend(w_names)

    o_chans, o_names, tokens, token_owners = _object_features(ctx, W, H)
    chans.extend(o_chans); names.extend(o_names)

    # Resolve which mask channels to include.
    if blend_mask_subset is not None:
        effective_subset = list(blend_mask_subset)
    elif include_blend_features:
        effective_subset = list(ALL_BLEND_MASK_NAMES)
    else:
        effective_subset = []

    if effective_subset:
        m_chans, m_names = _mask_features(ctx, W, H, subset=effective_subset)
        chans.extend(m_chans); names.extend(m_names)

    # Strategic features can use the passability mask only when at least one
    # passability channel is present in the kept subset; otherwise fall back to
    # slope-derived defensibility/cliff distance.
    have_passability = any(n.startswith("pass_") for n in effective_subset)
    s_chans, s_names = _strategic_features(
        ctx, W, H, slope_mag=slope_mag, use_blend_masks=have_passability,
    )
    chans.extend(s_chans); names.extend(s_names)

    if style_id is not None:
        for k in range(n_styles):
            v = 1.0 if k == style_id else 0.0
            chans.append(np.full((W, H), v, dtype=np.float32))
            names.append(f"style_{k}")

    array = np.stack(chans, axis=0).astype(np.float32)

    target_tiles = None
    palette = None
    if extract_target:
        # Derive per-tile palette index using BlendTileData.get_texture
        target_tiles = np.zeros((W, H), dtype=np.int32)
        for x in range(W):
            for y in range(H):
                target_tiles[x, y] = blend.get_texture(x, y)
        palette = [t.name for t in blend.textures]

    return FeatureStack(
        array=array,
        names=names,
        target_tiles=target_tiles,
        palette=palette,
        object_tokens=tokens,
        token_owners=token_owners,
        width=W,
        height=H,
    )
