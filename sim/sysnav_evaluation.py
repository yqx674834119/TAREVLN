"""Evaluator-only object acceptance. Never mounted in the policy sandbox."""
import math


def box_distance(point, target, dimensions=3):
    return math.sqrt(sum(max(target['min'][i]-point[i], 0., point[i]-target['max'][i])**2
                         for i in range(dimensions)))


def evaluate_object_stop(position, predicted_world_xyz, visible_prims, definition):
    floor = definition['base_z_range_m']
    matches = []
    for target in definition['targets']:
        distance = box_distance(position, target, 2)
        grounding = box_distance(predicted_world_xyz, target) if predicted_world_xyz is not None else None
        visible = target['prim'] in visible_prims
        passed = (floor[0] <= position[2] <= floor[1]
                  and distance <= definition['maximum_footprint_distance_m']
                  and grounding is not None and grounding <= definition['maximum_grounding_error_m']
                  and (visible or not definition['requires_visible_target']))
        matches.append(dict(prim=target['prim'], footprint_distance_m=distance,
                            grounding_error_m=grounding, visible=visible, passed=passed))
    return dict(passed=any(m['passed'] for m in matches), targets=matches,
                predicted_world_xyz=predicted_world_xyz)


def visible_targets(query, camera_origin, definition):
    """First-hit mesh visibility; an AABB alone cannot see through a wall."""
    visible = []
    for target in definition['targets']:
        lo, hi = target['min'], target['max']
        for fx, fy, fz in [(0.5,0.5,0.5),(0.2,0.5,0.6),(0.8,0.5,0.6),
                           (0.5,0.2,0.6),(0.5,0.8,0.6)]:
            point = [lo[i]+(hi[i]-lo[i])*f for i,f in enumerate((fx,fy,fz))]
            delta = [point[i]-camera_origin[i] for i in range(3)]
            length = math.sqrt(sum(x*x for x in delta))
            if length <= 1e-6:
                continue
            hit = query.raycast_closest(tuple(map(float,camera_origin)),
                                       tuple(x/length for x in delta), length+0.25)
            if hit['hit'] and any(str(hit.get(key,'' )).startswith(target['prim']+'/')
                                 or str(hit.get(key,'')) == target['prim']
                                 for key in ('collision','rigidBody')):
                visible.append(target['prim'])
                break
    return visible
