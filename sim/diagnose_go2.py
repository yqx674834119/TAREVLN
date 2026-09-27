"""Independent, contact-audited Go2 locomotion diagnostic (simulation only)."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import shutil
import traceback

parser = argparse.ArgumentParser()
parser.add_argument("--out", required=True)
parser.add_argument("--physics", choices=("baseline", "isaaclab"), default="baseline")
parser.add_argument("--yaw", type=float, default=0.)
parser.add_argument("--threshold", type=float, default=0.)
args = parser.parse_args()
out = Path(args.out)
out.mkdir(parents=True, exist_ok=False)
ROOT = Path(__file__).resolve().parents[1]
VLN = ROOT.parent
for file in (Path(__file__), ROOT / "sim/go2_policy.py"):
    shutil.copyfile(file, out / file.name)

from isaacsim import SimulationApp
app = SimulationApp({"headless": True, "multi_gpu": False, "active_gpu": 0, "physics_gpu": 0,
                     "width": 640, "height": 480, "fast_shutdown": True})
try:
    import numpy as np
    import omni.usd
    from omni.physx import get_physx_simulation_interface
    from isaacsim.core.api import World
    from isaacsim.core.api.objects import FixedCuboid
    from isaacsim.core.api.materials import PhysicsMaterial
    from isaacsim.core.utils.rotations import quat_to_rot_matrix
    from pxr import UsdPhysics, PhysxSchema, PhysicsSchemaTools, UsdShade
    from go2_policy import Go2Policy

    dt = .005
    world = World(physics_dt=dt, rendering_dt=.1, stage_units_in_meters=1., device="cpu")
    stage = omni.usd.get_context().get_stage()
    material = PhysicsMaterial("/World/GroundMaterial", static_friction=.8, dynamic_friction=.6, restitution=0.)
    world.scene.add(FixedCuboid("/World/Ground", position=np.array([0., 0., -.05]),
                               scale=np.array([100., 100., .1]), physics_material=material))
    if args.threshold:
        world.scene.add(FixedCuboid("/World/Threshold", position=np.array([1., 0., args.threshold / 2]),
                                   scale=np.array([.1, 20., args.threshold]), physics_material=material))
    policy = Go2Policy(stage, VLN / "assets/go2/go2_lab.usd", VLN / "assets/go2/checkpoint.pt", [0., 0., .4])
    world.scene.add(policy.robot)
    for prim in stage.Traverse():
        if not str(prim.GetPath()).startswith("/World/Go2/"):
            continue
        if prim.HasAPI(UsdPhysics.RigidBodyAPI):
            PhysxSchema.PhysxContactReportAPI.Apply(prim).CreateThresholdAttr(0.)
            if args.physics == "isaaclab":
                rigid = PhysxSchema.PhysxRigidBodyAPI.Apply(prim)
                rigid.CreateLinearDampingAttr(0.)
                rigid.CreateAngularDampingAttr(0.)
                rigid.CreateMaxLinearVelocityAttr(1000.)
                rigid.CreateMaxAngularVelocityAttr(math.degrees(1000.))
                rigid.CreateMaxDepenetrationVelocityAttr(1.)
        if args.physics == "isaaclab":
            if prim.HasAPI(UsdPhysics.ArticulationRootAPI):
                PhysxSchema.PhysxArticulationAPI.Apply(prim).CreateEnabledSelfCollisionsAttr(False)
            if prim.IsA(UsdPhysics.RevoluteJoint):
                PhysxSchema.PhysxJointAPI.Apply(prim).CreateMaxJointVelocityAttr(math.degrees(30.))
            if prim.HasAPI(UsdPhysics.CollisionAPI):
                UsdShade.MaterialBindingAPI.Apply(prim).Bind(material.material, materialPurpose="physics")
    world.reset()
    policy.initialize()
    if args.physics == "isaaclab":
        policy.robot.set_solver_position_iteration_count(4)
        policy.robot.set_solver_velocity_iteration_count(0)
    policy.robot.set_world_pose(np.array([0., 0., .4]), np.array([math.cos(args.yaw/2), 0., 0., math.sin(args.yaw/2)]))
    contact_stream = (out / "contacts.jsonl").open("w")
    trace = (out / "states.jsonl").open("w")
    current_phase = "settle"
    pairs = {}

    def on_contact(headers, data):
        for h in headers:
            paths = [str(PhysicsSchemaTools.intToSdfPath(p)) for p in (h.collider0, h.collider1)]
            if not any(p.startswith("/World/Go2/") for p in paths):
                continue
            details = [{"position": list(d.position), "normal": list(d.normal), "impulse": list(d.impulse),
                        "separation": float(d.separation)}
                       for d in data[h.contact_data_offset:h.contact_data_offset+h.num_contact_data]]
            record = {"t": float(world.current_time), "phase": current_phase, "colliders": paths, "points": details}
            contact_stream.write(json.dumps(record) + "\n")
            for path in paths:
                if path.startswith("/World/Go2/") and details:
                    pair = " | ".join(paths)
                    item = pairs.setdefault(pair, {"steps": 0, "post_settle_steps": 0, "max_impulse": 0.,
                                                   "support_steps": 0, "touch_steps": 0, "min_separation": None,
                                                   "nonfoot": "_foot/" not in path, "first_time": record["t"]})
                    item["steps"] += 1
                    item["post_settle_steps"] += current_phase != "settle"
                    item["max_impulse"] = max(item["max_impulse"], *(float(np.linalg.norm(d["impulse"])) for d in details))
                    separation = min(d["separation"] for d in details)
                    item["min_separation"] = min(item["min_separation"], separation) if item["min_separation"] is not None else separation
                    if current_phase != "settle":
                        item["support_steps"] += any(np.linalg.norm(d["impulse"]) > 1e-5 for d in details)
                        item["touch_steps"] += separation <= .0005
                    break

    subscription = get_physx_simulation_interface().subscribe_contact_report_events(on_contact)
    schedule = [("settle", 2., [0, 0, 0]), ("stand", 3., [0, 0, 0]),
                ("forward", 4., [.3, 0, 0]), ("stop_1", 2., [0, 0, 0]),
                ("backward", 4., [-.2, 0, 0]), ("left", 4., [0, .2, 0]),
                ("right", 4., [0, -.2, 0]), ("yaw", 4., [0, 0, .5]), ("stop_2", 3., [0, 0, 0])]
    metrics = {}
    for name, duration, command in schedule:
        current_phase = name
        samples = []
        for i in range(round(duration/dt)):
            policy.step(np.array(command))
            world.step(render=False)
            if i % 20 == 0:
                world.render()
            if i % 4 == 0:
                p, q = policy.robot.get_world_pose()
                rot = quat_to_rot_matrix(q)
                v = rot.T @ policy.robot.get_linear_velocity()
                w = rot.T @ policy.robot.get_angular_velocity()
                state = {"t": float(world.current_time), "phase": name, "command": command,
                         "position": p.tolist(), "quaternion": q.tolist(), "body_velocity": v.tolist(),
                         "body_angular_velocity": w.tolist(), "upright": float(rot[2, 2]),
                         "joint_position": policy.robot.get_joint_positions()[policy.indices].tolist(),
                         "joint_velocity": policy.robot.get_joint_velocities()[policy.indices].tolist(),
                         "action": policy.previous.tolist(), "target": policy.target.tolist(),
                         "effort": policy.robot.get_applied_action().joint_efforts.tolist()}
                trace.write(json.dumps(state) + "\n")
                samples.append(state)
        steady = samples[50:]  # First second is transition; raw data retained.
        error = np.array([[s["body_velocity"][0], s["body_velocity"][1], s["body_angular_velocity"][2]] for s in steady]) - command
        metrics[name] = {"minimum_height": min(s["position"][2] for s in samples),
                         "minimum_upright": min(s["upright"] for s in samples),
                         "mean_velocity_error": error.mean(axis=0).tolist(),
                         "rms_velocity_error": np.sqrt((error**2).mean(axis=0)).tolist(),
                         "final_position": samples[-1]["position"]}
        print("PHASE", name, json.dumps(metrics[name]), flush=True)
    nonfoot = [p for p, item in pairs.items() if item["nonfoot"] and (item["support_steps"] or item["touch_steps"])]
    report = {"physics": args.physics, "yaw": args.yaw, "threshold": args.threshold,
              "joint_names": policy.robot.dof_names, "phases": metrics, "contact_pairs": pairs,
              "nonfoot_pairs_after_settling": nonfoot,
              "contact_definition": "impulse > 1e-5 N s or separation <= 0.0005 m after settling; all proximity events retained",
              "contact_passed": bool(pairs) and not nonfoot,
              "source_hashes": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in (Path(__file__), ROOT / "sim/go2_policy.py")}}
    (out / "report.json").write_text(json.dumps(report, indent=2))
    contact_stream.close()
    trace.close()
    print("DIAGNOSTIC_COMPLETE", out, flush=True)
except Exception:
    (out / "error.txt").write_text(traceback.format_exc())
    traceback.print_exc()
    raise
finally:
    app.close()
