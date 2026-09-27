"""Validate semantic decisions without granting the VLM motion authority."""
import math


def validate_decision(result, candidates, view_candidates=None):
    if not isinstance(result, dict) or result.get("action") not in {"explore", "inspect", "confirm"}:
        raise ValueError("Invalid VLM action")
    for field in ("target_visible", "inside_target_area"):
        if type(result.get(field)) is not bool:
            raise ValueError(f"{field} must be a boolean")
    confidence = result.get("confidence")
    if type(confidence) not in (int, float) or not math.isfinite(confidence) or not 0 <= confidence <= 1:
        raise ValueError("Invalid VLM confidence")
    selected = None
    if result["action"] == "inspect":
        choice = result.get("candidate_id")
        selected = next((c for c in candidates if type(choice) is int and c["id"] == choice), None)
        if selected is None:
            raise ValueError("VLM selected an unavailable candidate")
        if view_candidates is not None:
            view = result.get("selected_view")
            if view not in view_candidates or choice not in view_candidates[view]:
                raise ValueError(f"Candidate {choice} is not projected in selected_view {view}; allowed IDs by view: {view_candidates}")
        if len(selected["xyz"]) != 3 or not all(math.isfinite(v) for v in selected["xyz"]):
            raise ValueError("Invalid candidate geometry")
    return selected


def confirmed_from_two_positions(previous, current):
    """Distinct recent camera poses, not duplicate frames or camera rotation."""
    dt = current["time"] - previous["time"]
    distance = math.dist(current["position"][:2], previous["position"][:2])
    if previous.get('surface_voxels_xy_10cm') is not None or current.get('surface_voxels_xy_10cm') is not None:
        a={tuple(p) for p in previous.get('surface_voxels_xy_10cm',[])}
        b={tuple(p) for p in current.get('surface_voxels_xy_10cm',[])}
        if min(len(a),len(b))<4 or len(a&b)<max(4,.1*min(len(a),len(b))):
            return False
        if abs(previous['surface_z_m']-current['surface_z_m'])>.06:
            return False
    if previous.get("anchor_world_xyz") is not None and current.get("anchor_world_xyz") is not None:
        if math.dist(previous["anchor_world_xyz"], current["anchor_world_xyz"]) > 1.5:
            return False
    return 0 < dt <= 60 and .6 <= distance <= 4.


def unvisited_candidates(candidates, visited, minimum_distance=.6):
    """Search memory uses traveled poses only, never scene labels or goal regions."""
    return [c for c in candidates if not visited or
            min(math.dist(c["xyz"][:2], p[:2]) for p in visited) >= minimum_distance]
