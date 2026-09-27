"""Separate PhysX proximity reports from touching/load-bearing contacts."""
import math

DEFINITION = {"impulse_threshold_ns": 1e-5, "touch_separation_m": .0005,
              "rule": "physical if impulse norm exceeds threshold OR separation is at most touch tolerance"}


def accumulate_contact(pairs, robot, environment, time, points):
    key = robot + " | " + environment
    item = pairs.setdefault(key, {"headers": 0, "first_time": time, "max_impulse": 0.,
                                 "point_count": 0, "min_separation": None, "physical_steps": 0})
    item["headers"] += 1
    item["point_count"] += len(points)
    physical = False
    for p in points:
        impulse = math.sqrt(sum(float(x)**2 for x in p["impulse"]))
        separation = float(p["separation"])
        if not math.isfinite(impulse) or not math.isfinite(separation):
            raise ValueError("Nonfinite contact evidence")
        item["max_impulse"] = max(item["max_impulse"], impulse)
        item["min_separation"] = min(item["min_separation"], separation) if item["min_separation"] is not None else separation
        physical |= impulse > DEFINITION["impulse_threshold_ns"] or separation <= DEFINITION["touch_separation_m"]
    item["physical_steps"] += physical


def contact_quality(pairs):
    def foot(pair):
        return any(part.endswith("_foot") for part in pair.split(" | ")[0].split("/"))
    raw = [p for p in pairs if not foot(p)]
    actual = [p for p in raw if pairs[p]["physical_steps"]]
    support = any(foot(p) and v["max_impulse"] > DEFINITION["impulse_threshold_ns"] for p,v in pairs.items())
    return {"nonfoot_contact_pairs": raw, "nonfoot_physical_contact_pairs": actual,
            "contact_definition": DEFINITION, "contact_quality_passed": support and not actual}
