"""Actual articulated Go2 stand/walk check using the published flat-terrain policy."""
import argparse
import json
import math
import os
from pathlib import Path
import traceback

from isaacsim import SimulationApp

parser = argparse.ArgumentParser()
parser.add_argument("--turn", action="store_true")
args = parser.parse_args()

ROOT = Path(__file__).resolve().parents[1]
VLN = ROOT.parent
app = SimulationApp({"headless": True, "open_usd": str(ROOT / "data/sim/p1/private/scene.usda"),
                     "create_new_stage": False, "multi_gpu": False, "active_gpu": 0, "physics_gpu": 0,
                     "anti_aliasing": 0, "width": 640, "height": 480, "fast_shutdown": True})
try:
    import numpy as np
    import omni.usd
    from isaacsim.core.api import World
    from isaacsim.core.utils.rotations import quat_to_rot_matrix
    from pxr import Gf, UsdGeom
    from go2_policy import Go2Policy

    world = World(physics_dt=.005, rendering_dt=.1, stage_units_in_meters=1.0, device="cpu")
    controller = Go2Policy(omni.usd.get_context().get_stage(), VLN / "assets/go2/go2_lab.usd",
                           VLN / "assets/go2/checkpoint.pt", [-22, 0, 12.4])
    world.scene.add(controller.robot)
    world.reset()
    controller.initialize()
    points = []
    clearance, upright, yaws = [], [], []
    stage = omni.usd.get_context().get_stage()
    # Use the Cube's physical size and transform. Authored extent metadata in
    # this asset is already scaled and would incorrectly apply scale twice.
    shape = stage.GetPrimAtPath("/World/Go2/base/collisions/mesh_0")
    if not shape.IsA(UsdGeom.Cube):
        raise ValueError("expected a primitive base collision box")
    half = float(UsdGeom.Cube(shape).GetSizeAttr().Get()) / 2
    xf = UsdGeom.XformCache()
    relative = xf.GetLocalToWorldTransform(shape) * xf.GetLocalToWorldTransform(stage.GetPrimAtPath("/World/Go2/base")).GetInverse()
    corners = np.array([list(relative.Transform(Gf.Vec3d(x, y, z))) for x in (-half, half)
                        for y in (-half, half) for z in (-half, half)])
    if not np.isfinite(corners).all() or not np.all((np.ptp(corners, axis=0) > 0) & (np.ptp(corners, axis=0) < 1)):
        raise ValueError("invalid physical collision corners")
    for step in range(3200 if args.turn else 2000):
        command = np.array([0.0 if step < 400 else .4, 0, 0])
        if args.turn and step >= 400:
            command = np.array([0 if step < 1600 else .15, 0, .7])
        controller.step(command)
        world.step(render=False)
        if step % 20 == 0:
            world.render()
        if step % 200 == 0:
            p, q = controller.robot.get_world_pose()
            points.append(p.tolist())
            rotation = quat_to_rot_matrix(q)
            yaws.append(math.atan2(rotation[1, 0], rotation[0, 0]))
            if step >= 400:
                clearance.append(float((corners @ rotation.T + p)[:, 2].min() - 12))
                upright.append(float(rotation[2, 2]))
            print("GO2", step, p.tolist(), q.tolist(), flush=True)
    p, q = controller.robot.get_world_pose()
    report = {"joint_names": controller.robot.dof_names, "positions": points, "final": p.tolist(),
              "forward_displacement_m": float(p[0] + 22), "standing_height_m": float(p[2] - 12),
              "minimum_base_collision_clearance_m": min(clearance), "minimum_upright_cosine": min(upright),
              "sampled_yaws": yaws, "mode": "turn_diagnostic" if args.turn else "forward_check",
              "passed": bool(not args.turn and 2 < p[0] + 22 < 4.5 and min(clearance) > .025 and min(upright) > .8)}
    (ROOT / ("data/sim/p1/go2_turn_check.json" if args.turn else "data/sim/p1/go2_check_v3.json")).write_text(json.dumps(report, indent=2))
    print("GO2_REPORT", json.dumps(report), flush=True)
except Exception:
    traceback.print_exc()
finally:
    app.close()
