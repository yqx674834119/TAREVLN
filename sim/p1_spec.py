"""Evaluator-owned P1 geometry. Nothing in this module is mounted in the policy sandbox."""
from __future__ import annotations

import json
from pathlib import Path


def make_spec(bridge: bool = False) -> dict:
    boxes, signs, lifts, doors = [], [], [], []

    def box(name, pos, size, color=(0.65, 0.68, 0.72), **extra):
        boxes.append(dict(name=name, pos=list(pos), size=list(size), color=list(color), **extra))

    def sign(name, pos, yaw, payload, label, size=1.5):
        signs.append(dict(name=name, pos=list(pos), yaw=yaw, payload=payload, label=label, size=size))

    box("Ground", (0, 0, -0.22), (90, 50, 0.4), (0.24, 0.32, 0.24))
    box("Walkway", (0, 0, -0.10), (28, 3, 0.2), (0.48, 0.48, 0.48))
    for building, cx, floors in (("A", -22, 5), ("B", 22, 3)):
        lx = cx - 5 if building == "A" else cx + 5
        lift_id = "E" + building
        lifts.append(dict(id=lift_id, building=building, x=lx, y=-9.6,
                          floors=list(range(1, floors + 1)), initial_floor=5 if building == "A" else 1))
        box(f"{lift_id}_ShaftBack", (lx, -11.2, floors * 1.5), (3.2, 0.2, floors * 3))
        for dx in (-1.6, 1.6):
            box(f"{lift_id}_ShaftSide_{'L' if dx < 0 else 'R'}", (lx + dx, -9.6, floors * 1.5),
                (0.2, 3.2, floors * 3))
        for floor in range(1, floors + 1):
            z = (floor - 1) * 3.0
            prefix = f"{building}_F{floor}"
            box(prefix + "_Floor", (cx, 0, z - 0.1), (16, 16, 0.2),
                (0.65, 0.55, 0.4) if building == "A" else (0.35, 0.48, 0.6))
            box(prefix + "_North", (cx, 8, z + 1.45), (16, 0.2, 2.9))
            # The elevator shaft is outside the slab: no floor crosses the cabin's travel volume.
            for side, low, high in (("L", cx - 8, lx - 1.2), ("R", lx + 1.2, cx + 8)):
                box(prefix + "_South" + side, ((low + high) / 2, -8, z + 1.45), (high - low, 0.2, 2.9))
            box(prefix + "_LiftHeader", (lx, -8, z + 2.65), (2.4, 0.2, 0.5))
            outer = cx - 8 if building == "A" else cx + 8
            portal = cx + 8 if building == "A" else cx - 8
            box(prefix + "_Outer", (outer, 0, z + 1.45), (0.2, 16, 2.9))
            gap = 0 if floor == 1 else (6 if bridge and floor == 2 else None)
            if gap is None:
                box(prefix + "_Inner", (portal, 0, z + 1.45), (0.2, 16, 2.9))
            else:
                for side, lo, hi in (("S", -8, gap - 1.25), ("N", gap + 1.25, 8)):
                    box(prefix + "_Inner" + side, (portal, (lo + hi) / 2, z + 1.45), (0.2, hi - lo, 2.9))
                # A real movable collision door, initially open in the baseline.
                doors.append(dict(id=prefix + "_PortalDoor", pos=[portal, gap, z + 1.2],
                                  size=[0.16, 2.3, 2.4], open_offset=[0, 2.5, 0], initial_open=True))
            doors.append(dict(id=prefix + "_LiftDoor", lift=lift_id, floor=floor,
                              pos=[lx, -8, z + 1.2], size=[2.4, 0.12, 2.4],
                              open_offset=[2.5, 0, 0], initial_open=False))
            # World-facing +Y panels at the lift lobby; text is actually rendered.
            lift_payload = {"kind": "lift", "id": lift_id, "floors": list(range(1, floors + 1))}
            if bridge and building == "A":
                lift_payload["link_B"] = 2
            sign(prefix + "_LiftSign", (lx, -7.82, z + 1.0), 90, lift_payload,
                 f"ELEVATOR {lift_id}  FLOORS 1-{floors}")
            signs[-1]["attach_door"] = prefix + "_LiftDoor"
            # Facing the cabin, identity panel can be read only after reaching this landing.
            sign(prefix + "_FloorSign", (lx, -4.0, z + 1.0), -90,
                 {"kind": "floor", "building": building, "floor": floor}, f"{building}  {floor}F")
            sign(prefix + "_LobbyID", (cx, -3, z + 1.0), 90,
                 {"kind": "floor", "building": building, "floor": floor}, f"{building}  {floor}F")
            if floor == 1:
                sign(prefix + "_ExitSign", (portal - (0.2 if building == "A" else -0.2), 2, z + 1.1),
                     180 if building == "A" else 0, {"kind": "exit", "building": building}, "EXIT  ->")
            if bridge and floor == 2:
                sign(prefix + "_BridgeSign", (portal - (0.2 if building == "A" else -0.2), 6, z + 1),
                     180 if building == "A" else 0,
                     {"kind": "bridge", "to": "B" if building == "A" else "A", "floor": 2}, "2F LINK  ->")
        # Name board faces outdoor space; it is evidence, not prior indoor topology.
        sign(building + "_Name", (cx + (8.2 if building == "A" else -8.2), -2, 1.1),
             0 if building == "A" else 180, {"kind": "building", "building": building}, "BUILDING " + building)
    if bridge:
        box("Skybridge", (0, 6, 2.9), (28, 2.5, 0.2), (0.3, 0.4, 0.5))
        for y in (4.65, 7.35):
            box("BridgeRail" + str(y).replace(".", "_"), (0, y, 3.55), (28, 0.15, 1.1))
    # Evaluator-owned directed topology. These endpoints are never policy input.
    connections = []
    for lift in lifts:
        for source in lift["floors"]:
            for target in lift["floors"]:
                if source != target:
                    connections.append({"id": f'{lift["id"]}_{source}_{target}', "kind": "elevator",
                                        "from": f'{lift["building"]}/F{source}', "to": f'{lift["building"]}/F{target}',
                                        "facility": lift["id"], "directed": True,
                                        "capabilities": ["walk", "elevator_api_assisted"],
                                        "conditions": ["at_landing", "door_open_to_board", "safely_boarded", "doors_closed_to_move"],
                                        "availability": "available"})
    for building in ("A", "B"):
        for source, target in ((f"{building}/F1", "outdoor/path"), ("outdoor/path", f"{building}/F1")):
            connections.append({"id": source + "->" + target, "kind": "door", "from": source, "to": target,
                                "directed": True, "capabilities": ["walk"], "conditions": ["door_open"],
                                "door": building + "_F1_PortalDoor", "initial_state": "open"})
    if bridge:
        for source, target in (("A/F2", "B/F2"), ("B/F2", "A/F2")):
            connections.append({"id": source + "->" + target, "kind": "skybridge", "from": source, "to": target,
                                "directed": True, "capabilities": ["walk"], "conditions": ["both_portal_doors_open"],
                                "initial_state": "open"})
    return {"version": "p1-v1", "boxes": boxes, "doors": doors, "signs": signs, "lifts": lifts,
            "connections": connections,
            "bridge": bridge, "start_xyz": [-22, 0, 12.4],
            "goal": {"building": "B", "floor": 3, "center_xy": [22, 0], "radius": 2.0,
                     "surface_z": 6.0, "outside_elevator": True}}


def public_prior() -> dict:
    return {"provider": "synthetic_OSM_style", "coordinate_system": "local_ENU_m",
            "buildings": [{"id": b, "center_xy": [x, 0], "outline_size_xy": [16, 16]}
                          for b, x in (("A", -22), ("B", 22))],
            "outdoor_footways": [{"from_xy": [-10, 0], "to_xy": [10, 0]}],
            "known_start": {"building": "A", "floor": 5},
            "goal": {"building": "B", "floor": 3, "region": "public_area"},
            "capabilities": ["walk", "elevator_api_assisted"]}


def write_spec(root: Path, bridge: bool = False) -> dict:
    (root / "public").mkdir(parents=True, exist_ok=True)
    (root / "private").mkdir(parents=True, exist_ok=True)
    spec = make_spec(bridge)
    (root / "private" / "oracle.json").write_text(json.dumps(spec, indent=2))
    (root / "public" / "prior.json").write_text(json.dumps(public_prior(), indent=2))
    return spec
