"""Validate semantic decisions without granting the VLM motion authority."""
import math


def validate_decision(result, candidates):
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
        if len(selected["xyz"]) != 3 or not all(math.isfinite(v) for v in selected["xyz"]):
            raise ValueError("Invalid candidate geometry")
    return selected


def confirmed_from_two_positions(previous, current):
    """Distinct recent camera poses, not duplicate frames or camera rotation."""
    dt = current["time"] - previous["time"]
    distance = math.dist(current["position"][:2], previous["position"][:2])
    return 0 < dt <= 60 and .6 <= distance <= 4.


def unvisited_candidates(candidates, visited, minimum_distance=1.2):
    """Search memory uses traveled poses only, never scene labels or goal regions."""
    return [c for c in candidates if not visited or
            min(math.dist(c["xyz"][:2], p[:2]) for p in visited) >= minimum_distance]
