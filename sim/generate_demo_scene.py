"""Generate a small editable USD campus plus a separated evaluation oracle.

This is an environment asset, not an autonomy or Go2 closed-loop result.
The elevator transitions are declared for a future skill API; geometry alone
does not make a robot capable of calling or riding an elevator.
"""

from __future__ import annotations

import json
from pathlib import Path


FLOOR_HEIGHT_M = 3.0
BUILDINGS = {"A": {"x": -22.0, "floors": 5}, "B": {"x": 22.0, "floors": 3}}


def _cube(name: str, position: tuple[float, float, float], size: tuple[float, float, float],
          color: tuple[float, float, float], collision: bool = True) -> str:
    x, y, z = position
    sx, sy, sz = size
    r, g, b = color
    schema = ' (\n        prepend apiSchemas = ["PhysicsCollisionAPI"]\n    )' if collision else ""
    physics = "        bool physics:collisionEnabled = true\n" if collision else ""
    return (f'    def Cube "{name}"{schema}\n    {{\n'
            f'        double size = 1\n'
            f'        double3 xformOp:translate = ({x}, {y}, {z})\n'
            f'        double3 xformOp:scale = ({sx}, {sy}, {sz})\n'
            f'        uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:scale"]\n'
            f'        color3f[] primvars:displayColor = [({r}, {g}, {b})]\n'
            f'{physics}    }}\n')


def generate(output: Path) -> dict:
    output.mkdir(parents=True, exist_ok=True)
    parts = ['#usda 1.0\n(\n    defaultPrim = "World"\n    metersPerUnit = 1\n    upAxis = "Z"\n)\n',
             'def Xform "World"\n{\n',
             '    def PhysicsScene "PhysicsScene"\n    {\n'
             '        vector3f physics:gravityDirection = (0, 0, -1)\n'
             '        float physics:gravityMagnitude = 9.81\n    }\n']
    grey = (0.58, 0.62, 0.67)
    blue = (0.24, 0.40, 0.75)
    tan = (0.72, 0.63, 0.51)
    parts.append(_cube("Ground", (0, 0, -0.2), (100, 45, 0.4), (0.38, 0.52, 0.38)))
    parts.append(_cube("OutdoorWalkway", (0, 0, 0.03), (28, 3, 0.12), grey))
    parts.append(_cube("SkybridgeF2", (0, 6, FLOOR_HEIGHT_M), (28, 2, 0.22), blue))
    for building, spec in BUILDINGS.items():
        cx = spec["x"]
        for floor_index in range(spec["floors"]):
            floor = floor_index + 1
            z = floor_index * FLOOR_HEIGHT_M
            prefix = f"{building}_F{floor}"
            parts.append(_cube(f"{prefix}_Slab", (cx, 0, z), (16, 16, 0.22), tan if building == "A" else blue))
            wall_z = z + 1.5
            for side, y in (("North", 8), ("South", -8)):
                parts.append(_cube(f"{prefix}_Wall{side}", (cx, y, wall_z), (16, 0.22, 3), grey))
            outer_x = cx - 8 if building == "A" else cx + 8
            portal_x = cx + 8 if building == "A" else cx - 8
            parts.append(_cube(f"{prefix}_WallOuter", (outer_x, 0, wall_z), (0.22, 16, 3), grey))
            if floor == 1:
                # Ground entrance: 2.4 m gap centred on the outdoor footway.
                parts.append(_cube(f"{prefix}_PortalWallS", (portal_x, -4.6, wall_z), (0.22, 6.8, 3), grey))
                parts.append(_cube(f"{prefix}_PortalWallN", (portal_x, 4.6, wall_z), (0.22, 6.8, 3), grey))
            elif floor == 2:
                # Skybridge entrance at y=6: a distinct, initially hidden connector.
                parts.append(_cube(f"{prefix}_BridgeWallS", (portal_x, -1.5, wall_z), (0.22, 13, 3), grey))
                parts.append(_cube(f"{prefix}_BridgeWallN", (portal_x, 7.5, wall_z), (0.22, 1, 3), grey))
            else:
                parts.append(_cube(f"{prefix}_WallPortalSide", (portal_x, 0, wall_z), (0.22, 16, 3), grey))
            # Colored panels are placeholders; rendered text/OCR has not been validated.
            parts.append(_cube(f"{prefix}_ElevatorPanel", (cx - 5 if building == "A" else cx + 5, -3, z + 1.4),
                               (0.08, 0.8, 0.8), (0.95, 0.78, 0.18), collision=False))
    parts.append("}\n")
    stage_file = output / "demo_campus.usda"
    stage_file.write_text("\n".join(parts))
    public_prior = {
        "provider": "synthetic_OSM_style",
        "coordinate_system": "local_ENU_m",
        "buildings": [{"id": key, "center_xy": [value["x"], 0], "outline_size_xy": [16, 16]}
                      for key, value in BUILDINGS.items()],
        "outdoor_footways": [{"from_xy": [-14, 0], "to_xy": [14, 0]}],
        "known_start": {"building": "A", "floor": 5},
        "goal": {"building": "B", "floor": 3, "region": "public_area"},
        "not_provided": ["entrances", "elevators", "stairs", "skybridge", "indoor_floorplans"],
    }
    oracle = {
        "scene": "demo_campus.usda",
        "floor_height_m": FLOOR_HEIGHT_M,
        "connectors": [
            {"id": "A_elevator", "building": "A", "floors": [1, 2, 3, 4, 5], "position_xy": [-27, -3], "skill": "ELEVATOR_API_ASSISTED"},
            {"id": "B_elevator", "building": "B", "floors": [1, 2, 3], "position_xy": [27, -3], "skill": "ELEVATOR_API_ASSISTED"},
            {"id": "A_exit", "from": "A/F1", "to": "outdoor", "position_xy": [-14, 0], "skill": "WALK"},
            {"id": "B_entrance", "from": "outdoor", "to": "B/F1", "position_xy": [14, 0], "skill": "WALK"},
            {"id": "skybridge", "from": "A/F2", "to": "B/F2", "position_xy": [0, 6], "skill": "WALK"},
        ],
        "goal_region": {"building": "B", "floor": 3, "center_xyz": [22, 0, 6.5], "radius_m": 2.0, "outside_elevator": True},
        "evaluation_only": True,
    }
    (output / "prior.json").write_text(json.dumps(public_prior, indent=2) + "\n")
    (output / "oracle.json").write_text(json.dumps(oracle, indent=2) + "\n")
    return {"stage": str(stage_file), "slabs": 8, "connectors": 5,
            "warning": "Geometry only; MuJoCo Go2 smoke is separate, Isaac/elevator not validated"}


if __name__ == "__main__":
    print(json.dumps(generate(Path(__file__).resolve().parents[1] / "data" / "sim" / "demo"), indent=2))
