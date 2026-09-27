"""Metric arrival checks from a VLM image point and current RGB-D only."""
import array
import base64
import math
import statistics
import sys
import zlib


def _surface_evidence(depth, width, height, u, v, intrinsics, rotation):
    """Estimate normals at several pixel scales; require an observed level surface.

    This rejects walls/legs, but is not an object classifier. RGB task evidence
    and repeated observations are still required by the caller.
    """
    fx, fy, cx, cy = intrinsics

    def point(x, y):
        if not (0 <= x < width and 0 <= y < height):
            return None
        d = depth[y*width+x]/1000
        if not 0 < d <= 40:
            return None
        local = [d, (cx-x)*d/fx, (cy-y)*d/fy]
        return [sum(rotation[i][j]*local[j] for j in range(3)) for i in range(3)]

    normals = []
    centre = point(u, v)
    residuals = []
    for radius in (3, 5, 7, 9):
        points = [point(u-radius, v), point(u+radius, v), point(u, v-radius), point(u, v+radius)]
        if centre is None or any(p is None for p in points):
            continue
        a = [points[1][i]-points[0][i] for i in range(3)]
        b = [points[3][i]-points[2][i] for i in range(3)]
        n = [a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0]]
        norm = math.sqrt(sum(x*x for x in n))
        if norm < 1e-8:
            continue
        n = [x/norm for x in n]
        normals.append(abs(n[2]))
        residuals.append(max(abs(sum(n[i]*(p[i]-centre[i]) for i in range(3))) for p in points))
    consistent = sum(z >= .85 and r <= .015 for z, r in zip(normals, residuals))
    return {"valid": consistent >= 3, "scales": len(normals), "consistent_scales": consistent,
            "normal_up_cosines": normals, "plane_residuals_m": residuals}


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
    surface = _surface_evidence(depth, width, height, u, v, view['intrinsics'], view['rotation'])
    valid = .2 <= distance <= max_distance and on_furniture and surface['valid']
    if not on_furniture:
        reason = 'anchor is on floor or above table height'
    elif not surface['valid']:
        reason = 'anchor lacks a consistent horizontal tabletop surface'
    else:
        reason = 'near observed tabletop surface' if valid else 'observed furniture too far'
    return {'valid': valid, 'reason': reason,
            'horizontal_distance_m': distance, 'anchor_world_xyz': point,
            'anchor_view': view['label'], 'anchor_pixel_xy': [u,v], 'depth_m': d, 'surface': surface}
