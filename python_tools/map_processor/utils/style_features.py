"""Shared feature extraction for the style-conditioned texture transfer model.

Used by both the training dataset prep script and the inference swap script
to ensure identical channel layout.
"""
from __future__ import annotations

import re
from typing import List, Tuple

import numpy as np


CATEGORY_RULES = [
    ("cliff",      [r"CLIFFWALL\d", r"SEACLIFFWALL\d"]),
    ("road",       [r"\bROAD", r"\bSTREET", r"\bPATH", r"_PATH_", r"BRIDGE"]),
    ("resource",   [r"OilDerrick", r"OreNode", r"\bRefiner", r"_TIBERIUM", r"\bORE_", r"\bOIL_"]),
    ("building",   [r"^Allied", r"^Soviet", r"^Japan", r"^Empire", r"^Civilian",
                    r"Hospital", r"Garage", r"Garrison", r"ObservationPost", r"Veterancy",
                    r"Shipyard", r"PortStructure", r"BuildingOther", r"TikiHut",
                    r"^GLA", r"_HOUSE", r"_TOWER", r"_BUNKER", r"_FACTORY"]),
    ("decoration", [r"TREE", r"PALM", r"BUSH", r"GRASS", r"ROCK", r"STATUE",
                    r"STONE", r"FLOWER", r"BAMBOO", r"SHRUB", r"LOG", r"FENCE",
                    r"SIGN", r"LAMPPOST", r"BENCH", r"FOUNTAIN", r"TABLE",
                    r"UMBRELLA", r"DECO", r"AMB_", r"CC_", r"YU_(?!CLIFF)", r"IL_",
                    r"HV_", r"CS_", r"GC_", r"MY_", r"SA_", r"MJ_", r"TH_",
                    r"DRUM", r"BARREL", r"CRATE"]),
]
CATEGORY_NAMES: List[str] = ["resource", "building", "decoration", "road", "cliff"]
_CAT_RES = [(name, [re.compile(p, re.I) for p in pats]) for name, pats in CATEGORY_RULES]


def categorize(type_name: str) -> str | None:
    for name, regs in _CAT_RES:
        for r in regs:
            if r.search(type_name):
                return name
    return None


def gaussian_blur(arr: np.ndarray, sigma: float) -> np.ndarray:
    from scipy.ndimage import gaussian_filter
    return gaussian_filter(arr, sigma=sigma, mode="nearest")


def compute_slope(height: np.ndarray) -> np.ndarray:
    gx = np.gradient(height, axis=0)
    gy = np.gradient(height, axis=1)
    s = np.sqrt(gx * gx + gy * gy).astype(np.float32)
    if s.max() > 0:
        s = s / s.max()
    return s


def normalize_height(h: np.ndarray) -> np.ndarray:
    lo, hi = float(h.min()), float(h.max())
    if hi - lo < 1e-6:
        return np.zeros_like(h, dtype=np.float32)
    return ((h - lo) / (hi - lo)).astype(np.float32)


def extract_input_channels(
    blend, h_asset, objs, world_to_tile: float = 10.0, sigma: float = 2.0,
) -> Tuple[np.ndarray, int, int]:
    """Returns (X (10, W, H) float32, W, H). Same channel order as training."""
    tW, tH = blend.tiles.shape
    elev_full = h_asset.elevations.astype(np.float32)
    eW, eH = elev_full.shape
    W = min(tW, eW); H = min(tH, eH)
    elev = elev_full[:W, :H]
    height_n = normalize_height(elev)
    slope = compute_slope(elev)
    buildability = blend.buildability[:W, :H].astype(np.float32)
    impassable_b = blend.impassable[:W, :H]
    passability = (1.0 - impassable_b.astype(np.float32))
    low = float(np.quantile(elev, 0.05))
    water_mask = ((elev <= low + 1e-3) & impassable_b).astype(np.float32)

    chans = {c: np.zeros((W, H), dtype=np.float32) for c in CATEGORY_NAMES}
    for obj in objs.map_objects:
        cat = categorize(obj.type_name)
        if cat is None:
            continue
        tx = int(obj.position[0] / world_to_tile)
        ty = int(obj.position[1] / world_to_tile)
        if 0 <= tx < W and 0 <= ty < H:
            chans[cat][tx, ty] += 1.0
    for c in CATEGORY_NAMES:
        chans[c] = gaussian_blur(chans[c], sigma)
        m = chans[c].max()
        if m > 0:
            chans[c] /= m

    X = np.stack([
        height_n, slope, water_mask, buildability, passability,
        chans["resource"], chans["building"], chans["decoration"],
        chans["road"], chans["cliff"],
    ], axis=0)
    return X, W, H
