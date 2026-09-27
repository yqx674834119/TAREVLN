"""Simulator-owned elevator skill, with proximity, occupancy, and door interlocks."""
from __future__ import annotations

import math


class Elevator:
    def __init__(self, spec):
        self.spec = spec
        self.floor = spec["initial_floor"]
        self.z = (self.floor - 1) * 3.0
        self.target = self.floor
        self.state = "closed"
        self.elapsed = 0.0
        self.door_fraction = 0.0
        self.trip_start = self.z
        self.duration = 0.0
        self.closing_start = 0.0

    def inside(self, xyz):
        return (abs(xyz[0] - self.spec["x"]) < 1.0 and -10.65 < xyz[1] < -8.55
                and self.z < xyz[2] < self.z + 1.0)

    def call(self, floor, xyz):
        if floor not in self.spec["floors"]:
            return {"ok": False, "reason": "UNSERVED_FLOOR"}
        if math.hypot(xyz[0] - self.spec["x"], xyz[1] + 8) > 2.7 or abs(xyz[2] - ((floor - 1) * 3 + 0.4)) > 0.8:
            return {"ok": False, "reason": "NOT_AT_LANDING"}
        if self.state in {"moving", "closing"}:
            return {"ok": False, "reason": "BUSY"}
        if floor == self.floor and self.state in {"open", "opening"}:
            return {"ok": True, "assistance": "ELEVATOR_API_ASSISTED"}
        self.target = floor
        self.elapsed = 0.0
        self.closing_start = self.door_fraction
        self.state = "opening" if floor == self.floor else "closing"
        return {"ok": True, "assistance": "ELEVATOR_API_ASSISTED"}

    def select(self, floor, xyz):
        if floor not in self.spec["floors"]:
            return {"ok": False, "reason": "UNSERVED_FLOOR"}
        if not self.inside(xyz) or self.state != "open":
            return {"ok": False, "reason": "NOT_SAFELY_BOARDED"}
        self.target = floor
        self.elapsed = 0.0
        self.closing_start = self.door_fraction
        self.state = "closing"
        return {"ok": True, "assistance": "ELEVATOR_API_ASSISTED"}

    def tick(self, dt):
        self.elapsed += dt
        if self.state == "opening":
            self.door_fraction = min(1.0, self.elapsed)
            if self.door_fraction == 1:
                self.state = "open"
        elif self.state == "closing":
            self.door_fraction = self.closing_start * max(0.0, 1.0 - self.elapsed)
            if self.elapsed >= 1:
                self.state = "moving"
                self.trip_start = self.z
                self.duration = max(1.0, abs((self.target - 1) * 3 - self.z) / 0.6)
                self.elapsed = 0
        elif self.state == "moving":
            u = min(1.0, self.elapsed / self.duration)
            self.z = self.trip_start + ((self.target - 1) * 3 - self.trip_start) * (3 * u**2 - 2 * u**3)
            if u == 1:
                self.floor = self.target
                self.state = "opening"
                self.elapsed = 0

    def feedback(self):
        # No oracle endpoint identity is returned. Read the landing sign to identify arrival.
        return {"state": self.state, "door_open": self.state == "open"}


def evaluate_stop(spec, xyz, speed, policy_identity, dwell_seconds):
    goal = spec["goal"]
    in_cabin = any(abs(xyz[0] - lift["x"]) < 1.6 and -11.2 < xyz[1] < -8 for lift in spec["lifts"])
    conditions = {"correct_xy": math.hypot(xyz[0] - goal["center_xy"][0], xyz[1] - goal["center_xy"][1]) <= goal["radius"],
                  "correct_height": goal["surface_z"] + 0.15 < xyz[2] < goal["surface_z"] + 0.9,
                  "outside_elevator": not in_cabin, "stopped": speed < 0.06 and dwell_seconds >= 1.0,
                  "identity_verified": policy_identity == {"building": goal["building"], "floor": goal["floor"]}}
    # Simulator coordinates are NumPy scalars; keep the public report JSON-native.
    conditions = {key: bool(value) for key, value in conditions.items()}
    return {"success": all(conditions.values()), "conditions": conditions}
