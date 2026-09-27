"""Metric arrival checks from a VLM image point and current RGB-D only."""
import array
import base64
import math
import statistics
import sys
import zlib


def ground_visual_anchor(check, views, robot_position, max_distance=3.0):
    view = next((v for v in views if v['label'] == check.get('anchor_view')), None)
    pixel = check.get('anchor_pixel_xy')
    if view is None or not isinstance(pixel, list) or len(pixel) != 2:
        return {'valid': False, 'reason': 'missing visual anchor'}
    width, height = view['image_size']
    if any(type(p) not in (int, float) or not math.isfinite(p) for p in pixel):
        return {'valid': False, 'reason': 'invalid anchor pixel'}
    u, v = map(round, pixel)
    if not (1 <= u < width-1 and 1 <= v < height-1):
        return {'valid': False, 'reason': 'anchor outside image interior'}
    try:
        depth = array.array('H', zlib.decompress(base64.b64decode(view['depth_mm'])))
        if sys.byteorder != 'little': depth.byteswap()
    except (KeyError, ValueError, zlib.error):
        return {'valid': False, 'reason': 'invalid depth frame'}
    if len(depth) != width*height:
        return {'valid': False, 'reason': 'depth dimensions differ'}
    samples = [depth[(v+dy)*width+u+dx]/1000 for dy in [-1,0,1] for dx in [-1,0,1]
               if 0 < depth[(v+dy)*width+u+dx] <= 40000]
    if len(samples) < 5 or max(samples)-min(samples) > .4:
        return {'valid': False, 'reason': 'missing or discontinuous anchor depth'}
    d = statistics.median(samples)
    fx, fy, cx, cy = view['intrinsics']
    local = [d, (cx-u)*d/fx, (cy-v)*d/fy]
    point = [view['position'][i] + sum(view['rotation'][i][j]*local[j] for j in range(3)) for i in range(3)]
    distance = math.dist(point[:2], robot_position[:2])
    on_furniture = .2 <= point[2] <= 1.2
    valid = .2 <= distance <= max_distance and on_furniture
    reason = 'anchor is on floor or above table height' if not on_furniture else ('near observed furniture' if valid else 'observed furniture too far')
    return {'valid': valid, 'reason': reason,
            'horizontal_distance_m': distance, 'anchor_world_xyz': point,
            'anchor_view': view['label'], 'anchor_pixel_xy': [u,v], 'depth_m': d}
