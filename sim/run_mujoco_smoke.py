"""Check that the generated campus and a Go2 model compile and make contact.

This is not a locomotion policy, connector skill, or navigation acceptance.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import mujoco
import mujoco_menagerie as mm
import numpy as np


PROJECT = Path(__file__).resolve().parents[1]
OUTPUT = PROJECT / "data" / "sim" / "demo"
TRIPLE = re.compile(r"\(([^)]+)\)")


def _triple(block: str, key: str) -> list[float]:
    line = next(line for line in block.splitlines() if key in line)
    return [float(value.strip()) for value in TRIPLE.search(line).group(1).split(",")]


def read_cubes(path: Path) -> list[dict]:
    text = path.read_text()
    boxes = []
    for match in re.finditer(r'def Cube "([^"]+)"([^\{]*)\{([^\}]+)\}', text, re.S):
        name, metadata, block = match.groups()
        boxes.append({
            "name": name,
            "pos": _triple(block, "xformOp:translate"),
            "size": _triple(block, "xformOp:scale"),
            "color": _triple(block, "primvars:displayColor"),
            "collision": "PhysicsCollisionAPI" in metadata,
        })
    return boxes


def main() -> None:
    boxes = read_cubes(OUTPUT / "demo_campus.usda")
    if len(boxes) < 50:
        raise RuntimeError(f"expected a multi-floor campus, found only {len(boxes)} boxes")
    robot = mm.get("unitree_go2")
    spec = robot.spec("scene")
    for box in boxes:
        if box["name"] == "Ground":
            continue  # The Go2 reference scene already contains a ground plane.
        geom = spec.worldbody.add_geom(
            name=f"campus_{box['name']}",
            type=mujoco.mjtGeom.mjGEOM_BOX,
            pos=box["pos"],
            size=[dimension / 2 for dimension in box["size"]],
            rgba=[*box["color"], 1.0],
        )
        if not box["collision"]:
            geom.contype = 0
            geom.conaffinity = 0
    model = spec.compile()
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, 0)
    data.qpos[0:3] = [-22.0, 0.0, 12.38]  # A/F5; F1 is at z=0.
    data.ctrl[:] = model.key_ctrl[0]
    mujoco.mj_forward(model, data)
    contacts_at_start = data.ncon
    for _ in range(400):
        mujoco.mj_step(model, data)
    final_xyz = data.qpos[:3].tolist()
    if not np.isfinite(final_xyz).all() or not (11.9 < final_xyz[2] < 13.0):
        raise RuntimeError(f"Go2 did not remain on A/F5: {final_xyz}")
    report = {
        "simulator": "MuJoCo 3.7.0",
        "robot": "Menagerie Unitree Go2",
        "robot_oid": robot.oid,
        "model_nq": model.nq,
        "campus_boxes": len(boxes),
        "spawn": "A/F5",
        "final_base_xyz": final_xyz,
        "contacts_at_start": contacts_at_start,
        "contacts_at_end": data.ncon,
        "steps": 400,
        "scope": "physics load, static support, and optional rendered frame only; no route or elevator execution",
    }
    try:
        from PIL import Image
        renderer = mujoco.Renderer(model, height=480, width=640)
        renderer.update_scene(data)
        Image.fromarray(renderer.render()).save(OUTPUT / "mujoco_smoke.png")
        camera = mujoco.MjvCamera()
        camera.type = mujoco.mjtCamera.mjCAMERA_FREE
        camera.lookat[:] = [-22.0, 0.0, 12.2]
        camera.distance = 2.5
        camera.azimuth = 125
        camera.elevation = -20
        renderer.update_scene(data, camera=camera)
        Image.fromarray(renderer.render()).save(OUTPUT / "mujoco_go2_a5f.png")
        renderer.close()
        report["rendered_frame"] = "mujoco_smoke.png"
        report["robot_closeup"] = "mujoco_go2_a5f.png"
    except Exception as exc:
        report["render_error"] = f"{type(exc).__name__}: {exc}"
    (OUTPUT / "mujoco_smoke.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
