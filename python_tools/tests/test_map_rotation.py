import math

import numpy as np


def _angle_diff_deg(a: float, b: float) -> float:
    """Smallest absolute difference between two angles in degrees."""
    d = (a - b) % 360.0
    if d > 180.0:
        d = 360.0 - d
    return abs(d)


def test_rotate_map_roundtrip_in_memory(sample_map_file, python_processor):
    Ra3Map = python_processor

    m = Ra3Map(sample_map_file)
    m.parse()
    ctx = m.get_context()

    from map_processor.utils.map_rotation import rotate_context_right_angles
    from map_processor.assets.terrain.height_map_data import HeightMapData
    from map_processor.assets.terrain.blend_tile_data import BlendTileData
    from map_processor.assets.objects.objects_list import ObjectsList

    height = ctx.get_asset_by_type(HeightMapData)
    blend = ctx.get_asset_by_type(BlendTileData)
    objs = ctx.get_asset_by_type(ObjectsList)

    assert height is not None
    assert blend is not None
    assert objs is not None

    orig_w, orig_h = ctx.map_width, ctx.map_height

    # Snapshot a few key structures
    h_raw0 = None if height._elevations_raw is None else np.asarray(height._elevations_raw).copy()
    tiles0 = None if blend.tiles is None else np.asarray(blend.tiles).copy()
    pass0 = None if blend.passability is None else np.asarray(blend.passability).copy()
    objs0 = [(o.position, o.angle, o.type_name) for o in objs.map_objects[:100]]

    # Rotate 90 CW then 270 CW (net 360)
    rotate_context_right_angles(ctx, degrees=90, clockwise=True)
    assert (ctx.map_width, ctx.map_height) == (orig_h, orig_w)

    rotate_context_right_angles(ctx, degrees=270, clockwise=True)
    assert (ctx.map_width, ctx.map_height) == (orig_w, orig_h)

    if h_raw0 is not None:
        np.testing.assert_array_equal(height._elevations_raw, h_raw0)
    if tiles0 is not None:
        np.testing.assert_array_equal(blend.tiles, tiles0)
    if pass0 is not None:
        np.testing.assert_array_equal(blend.passability, pass0)

    for (pos0, ang0, type0), obj in zip(objs0, objs.map_objects[:100]):
        assert obj.type_name == type0
        x0, y0, z0 = pos0
        x1, y1, z1 = obj.position
        assert abs(x1 - x0) < 1e-4
        assert abs(y1 - y0) < 1e-4
        assert abs(z1 - z0) < 1e-4
        assert _angle_diff_deg(obj.angle, ang0) < 1e-4


def test_rotate_map_90_formula_holds(sample_map_file, python_processor):
    """
    Spot-check that our CW90 world-coordinate transform matches the intended rectangle mapping:
      (x, y) -> (y, max_x - x)  where max_x=(playable_W)*10
    """
    Ra3Map = python_processor

    m = Ra3Map(sample_map_file)
    m.parse()
    ctx = m.get_context()

    from map_processor.utils.map_rotation import rotate_context_right_angles, WORLD_UNITS_PER_TILE
    from map_processor.assets.objects.objects_list import ObjectsList

    objs = ctx.get_asset_by_type(ObjectsList)
    assert objs is not None
    assert len(objs.map_objects) > 0

    # Objects are in playable-area space in this repo (see map_visualizer.py)
    old_w = ctx.map_width - 2 * ctx.border
    old_h = ctx.map_height - 2 * ctx.border
    max_x = float(old_w) * WORLD_UNITS_PER_TILE

    # pick a reasonably "normal" object (not necessarily bounded, but transform should still apply)
    o0 = objs.map_objects[0]
    x0, y0, z0 = o0.position
    ang0 = o0.angle

    rotate_context_right_angles(ctx, degrees=90, clockwise=True)
    o1 = objs.map_objects[0]
    x1, y1, z1 = o1.position

    assert abs(x1 - y0) < 1e-4
    assert abs(y1 - (max_x - x0)) < 1e-4
    assert abs(z1 - z0) < 1e-4
    # yaw should rotate with the map (CW => subtract 90 degrees)
    assert _angle_diff_deg(o1.angle, (ang0 - 90.0) % 360.0) < 1e-4


def test_rotate_map_180_formula_holds(sample_map_file, python_processor):
    """
    Spot-check CW180:
      (x, y) -> (max_x - x, max_y - y)
    """
    Ra3Map = python_processor

    m = Ra3Map(sample_map_file)
    m.parse()
    ctx = m.get_context()

    from map_processor.utils.map_rotation import rotate_context_right_angles, WORLD_UNITS_PER_TILE
    from map_processor.assets.objects.objects_list import ObjectsList

    objs = ctx.get_asset_by_type(ObjectsList)
    assert objs is not None
    assert len(objs.map_objects) > 0

    old_w = ctx.map_width - 2 * ctx.border
    old_h = ctx.map_height - 2 * ctx.border
    max_x = float(old_w) * WORLD_UNITS_PER_TILE
    max_y = float(old_h) * WORLD_UNITS_PER_TILE

    o0 = objs.map_objects[0]
    x0, y0, z0 = o0.position
    ang0 = o0.angle

    rotate_context_right_angles(ctx, degrees=180, clockwise=True)
    o1 = objs.map_objects[0]
    x1, y1, z1 = o1.position

    assert abs(x1 - (max_x - x0)) < 1e-4
    assert abs(y1 - (max_y - y0)) < 1e-4
    assert abs(z1 - z0) < 1e-4
    assert _angle_diff_deg(o1.angle, (ang0 - 180.0) % 360.0) < 1e-4


