"""Closed-loop Isaac P1 runner: real Go2 dynamics, RGB-D signs, lidar, API elevators.

The policy is a deliberately simple sign/geometry baseline in a restricted
process. This is integration evidence, not TARE/VLM or real-robot validation.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import time
import traceback

parser = argparse.ArgumentParser()
parser.add_argument("--scene", type=Path, default=Path(__file__).resolve().parents[1] / "data/sim/p1")
parser.add_argument("--name", default="baseline")
parser.add_argument("--max-seconds", type=float, default=500)
parser.add_argument("--sensor-only", action="store_true")
parser.add_argument("--start-yaw", type=float, default=0.0)
parser.add_argument("--gpu", type=int, default=0)
args = parser.parse_args()
ROOT = Path(__file__).resolve().parents[1]
VLN = ROOT.parent
scene = args.scene.resolve()
output = scene / "runs" / args.name
output.mkdir(parents=True, exist_ok=True)
report = {"status": "starting", "success": False, "simulation": "Isaac Sim 5.1",
          "policy": "RGB-D QR-sign + lidar A* diagnostic baseline", "assistance": "ELEVATOR_API_ASSISTED",
          "robot": "Go2 articulated dynamics + NVIDIA published flat-terrain policy", "odometry": "ideal simulator odometry",
          "platform_compensation": "simulator cabin vertical velocity supplied to locomotion adapter"}
(output / "report.json").write_text(json.dumps(report, indent=2))

from isaacsim import SimulationApp

app = SimulationApp({"headless": True, "open_usd": str(scene / "private/scene.usda"), "create_new_stage": False,
                     "multi_gpu": False, "active_gpu": args.gpu, "physics_gpu": args.gpu, "anti_aliasing": 0,
                     "width": 640, "height": 480, "fast_shutdown": True})
policy = None
try:
    import numpy as np
    from PIL import Image
    import omni.usd
    from omni.physx import get_physx_scene_query_interface, get_physx_simulation_interface
    from pxr import Gf, UsdGeom, UsdPhysics, PhysxSchema, PhysicsSchemaTools
    from isaacsim.core.api import World
    from isaacsim.core.utils.rotations import quat_to_rot_matrix
    from isaacsim.sensors.camera import Camera
    from isaacsim.sensors.physics import IMUSensor
    from go2_policy import Go2Policy
    from p1_interactions import Elevator, evaluate_stop
    from p1_perception import SignFrontend
    from p1_sandbox import PolicyProcess

    spec = json.loads((scene / "private/oracle.json").read_text())
    stage = omni.usd.get_context().get_stage()
    world = World(physics_dt=.005, rendering_dt=.1, stage_units_in_meters=1.0, device="cpu")
    controller = Go2Policy(stage, VLN / "assets/go2/go2_lab.usd", VLN / "assets/go2/checkpoint.pt", spec["start_xyz"])
    world.scene.add(controller.robot)
    for prim in stage.Traverse():
        if str(prim.GetPath()).startswith("/World/Go2/") and prim.HasAPI(UsdPhysics.RigidBodyAPI):
            PhysxSchema.PhysxContactReportAPI.Apply(prim).CreateThresholdAttr(0.0)
    # The runner renders at 10 Hz. Acquire every rendered frame: Camera's own
    # 10-Hz floating-point timer otherwise skips timestamp updates at 0.1 s.
    camera = Camera("/World/P1Camera", frequency=-1, resolution=(1280, 960))
    camera.set_focal_length(2.4)
    camera.set_horizontal_aperture(4.8)
    camera.set_vertical_aperture(3.6)
    camera.set_clipping_range(.05, 100)
    imu = IMUSensor("/World/Go2/base/p1_imu", frequency=200, translation=np.array([0., 0., .05]))
    world.reset()
    controller.initialize()
    if args.start_yaw:
        controller.robot.set_world_pose(np.asarray(spec["start_xyz"]), np.array([math.cos(args.start_yaw/2), 0, 0, math.sin(args.start_yaw/2)]))
    camera.initialize()
    camera.add_distance_to_image_plane_to_frame()
    imu.initialize()
    calibration = {"rgbd": {"width": 1280, "height": 960, "frequency_hz": 10,
                            "intrinsics": camera.get_intrinsics_matrix().tolist(),
                            "base_translation_m": [.28, 0, .18], "axes": "X-forward Z-up", "noise": "none",
                            "clock": "every explicitly rendered 10-Hz frame"},
                   "lidar": {"model": "ideal PhysX planar raycast", "rays": 180, "range_m": 12, "frequency_hz": 10,
                             "base_translation_m": [0, 0, .3]},
                   "imu": {"frequency_hz": 200, "base_translation_m": [0, 0, .05]},
                   "note": "P1 virtual sensor profile, not a measured real Go2/MID360 calibration"}
    (output / "calibration.json").write_text(json.dumps(calibration, indent=2))
    lifts = {s["id"]: Elevator(s) for s in spec["lifts"]}
    frontend = SignFrontend()
    query = get_physx_scene_query_interface()
    command = np.zeros(3)
    feedback, active_lift = {}, None
    last_result = {}
    events = []
    contacts = set()
    contact_event_count = 0
    contact_details = {}
    def contact_callback(headers, contact_data):
        global contact_event_count
        for header in headers:
            p0 = str(PhysicsSchemaTools.intToSdfPath(header.actor0))
            p1 = str(PhysicsSchemaTools.intToSdfPath(header.actor1))
            robot, environment = (p0, p1) if p0.startswith("/World/Go2") else (p1, p0)
            if not robot.startswith("/World/Go2") or environment.startswith("/World/Go2"):
                continue
            contact_event_count += 1
            c0 = str(PhysicsSchemaTools.intToSdfPath(header.collider0))
            c1 = str(PhysicsSchemaTools.intToSdfPath(header.collider1))
            robot_shape, environment_shape = (c0, c1) if p0.startswith("/World/Go2") else (c1, c0)
            surface = environment_shape.rsplit("/", 1)[-1]
            support = surface in {"Floor", "Ground", "Walkway", "Skybridge"} or surface.endswith("_Floor")
            # An actor can combine a cabin's floor and walls, or a leg and foot.
            # Classify actual colliders; only foot-on-support contact is allowed.
            if "_foot/" not in robot_shape or not support:
                pair = (robot_shape, environment_shape)
                contacts.add(pair)
                item = contact_details.setdefault(pair, {"first_time": float(world.current_time), "headers": 0, "max_impulse_norm": 0.0})
                item["headers"] += 1
                for i in range(header.contact_data_offset, header.contact_data_offset + header.num_contact_data):
                    item["max_impulse_norm"] = max(item["max_impulse_norm"], float(np.linalg.norm(contact_data[i].impulse)))

    contact_subscription = get_physx_simulation_interface().subscribe_contact_report_events(contact_callback)
    stderr = open(output / "policy.stderr", "w")
    policy = PolicyProcess(ROOT, scene / "public/prior.json", stderr)
    audit = policy.ask({"op": "audit", "paths": [str(scene / "private/oracle.json"), str(scene / "private/scene.usda"),
                                                  "/input/../private/oracle.json", "/proc/1/root/home/yqx/VLN/cross-building-floor-nav/data/sim/p1/private/oracle.json"]})
    (output / "isolation.json").write_text(json.dumps(audit, indent=2))
    if any(v != "DENIED" for v in audit["audit"].values()):
        raise RuntimeError("policy can read a privileged file")
    telemetry = open(output / "trajectory.jsonl", "w")
    readings = open(output / "observations.jsonl", "w")
    previous_state = None
    action = {"state": "warmup", "identity": {}, "stop": False, "verified_edges": 0}
    stopped_duration = 0.0
    image_count = 0
    observed_kinds = set()
    last_selection_time = 0.0
    goal_identity_time = -1.0
    imu_valid = False
    started = time.monotonic()
    for step in range(int(args.max_seconds / .005)):
        position, quat = controller.robot.get_world_pose()
        platform_velocity = 0.0
        for lift in lifts.values():
            before = lift.z
            lift.tick(.005)
            stage.GetPrimAtPath("/World/Cabins/" + lift.spec["id"]).GetAttribute("xformOp:translate").Set(Gf.Vec3d(lift.spec["x"], lift.spec["y"], lift.z))
            if lift.inside(position):
                platform_velocity = (lift.z - before) / .005
        for door in spec["doors"]:
            fraction = 1.0 if door["initial_open"] else 0.0
            if "lift" in door:
                lift = lifts[door["lift"]]
                fraction = lift.door_fraction if lift.floor == door["floor"] and lift.state != "moving" else 0
            pos = [p + fraction * offset for p, offset in zip(door["pos"], door["open_offset"])]
            stage.GetPrimAtPath("/World/Doors/" + door["id"]).GetAttribute("xformOp:translate").Set(Gf.Vec3d(*pos))
        for sign in spec["signs"]:
            if "attach_door" in sign:
                door = next(d for d in spec["doors"] if d["id"] == sign["attach_door"])
                dpos = stage.GetPrimAtPath("/World/Doors/" + door["id"]).GetAttribute("xformOp:translate").Get()
                pos = [sign["pos"][k] + dpos[k] - door["pos"][k] for k in range(3)]
            elif "attach_lift" in sign:
                lift = lifts[sign["attach_lift"]]
                pos = [sign["pos"][0], sign["pos"][1], lift.z + 1.0]
            else:
                continue
            stage.GetPrimAtPath("/World/Signs/" + sign["name"]).GetAttribute("xformOp:translate").Set(Gf.Vec3d(*pos))
        controller.step(command, platform_velocity=platform_velocity)
        world.step(render=False)
        if step % 20:
            continue
        t = float(world.current_time)
        position, quat = controller.robot.get_world_pose()
        rotation = quat_to_rot_matrix(quat)
        yaw = math.atan2(rotation[1, 0], rotation[0, 0])
        camera_position = position + rotation @ np.array([.28, 0, .18])
        camera.set_world_pose(camera_position, quat, camera_axes="world")
        world.render()
        rgb, depth = camera.get_rgba(), camera.get_depth()
        camera_time = camera.get_current_frame().get("rendering_time")
        if step >= 400 and (camera_time is None or abs(camera_time - t) > .01):
            raise RuntimeError(f"unsynchronized camera timestamp: camera={camera_time}, physics={t}")
        signs = frontend.decode(rgb, depth, camera.get_intrinsics_matrix(), camera_position, rotation)
        for sign in signs:
            observed_kinds.add(sign["text"]["kind"])
            if sign["text"].get("kind") == "floor" and all(sign["text"].get(k) == spec["goal"][k] for k in ("building", "floor")):
                goal_identity_time = t
        scan = []
        for index in range(180):
            angle = yaw + index * 2 * math.pi / 180
            hit = query.raycast_closest((float(position[0]), float(position[1]), float(position[2] + .3)),
                                       (math.cos(angle), math.sin(angle), 0), 12.0)
            scan.append([angle, float(hit["distance"]) if hit["hit"] else 12.0])
        imuframe = imu.get_current_frame()
        imu_valid = imu_valid or bool(imuframe.get("physics_step", 0))
        feedback = lifts[active_lift].feedback() if active_lift else {}
        feedback["last_result"] = last_result
        observation = {"time": t, "odom": [*position.tolist(), yaw], "signs": signs, "scan": scan,
                       "skill_feedback": feedback,
                       "imu": {k: v.tolist() if hasattr(v, "tolist") else v for k, v in imuframe.items()},
                       "camera_time": camera_time}
        if step >= 400:
            action = policy.ask(observation)
            command = np.asarray(action["cmd_vel"], dtype=float)
            if command.shape != (3,) or not np.isfinite(command).all():
                raise ValueError("invalid policy velocity command")
            command = np.clip(command, [-.4, -.3, -.7], [.6, .3, .7])
            request = action.get("request")
            if request:
                lift = lifts.get(request.get("lift"))
                if lift is None or request.get("op") not in {"call", "select"} or not isinstance(request.get("floor"), int):
                    last_result = {"ok": False, "reason": "INVALID_REQUEST"}
                else:
                    active_lift = request["lift"]
                    last_result = getattr(lift, request["op"])(request["floor"], position)
                events.append({"time": t, "request": request, "result": last_result})
                if request.get("op") == "select" and last_result.get("ok"):
                    last_selection_time = t
        transition = action["state"] != previous_state
        if transition or step % 1000 == 0:
            print("P1", round(t, 2), action["state"], [round(float(p), 2) for p in position],
                  [s["text"] for s in signs], flush=True)
        if rgb is not None and rgb.size and depth is not None and (transition or step % 200 == 0):
            Image.fromarray(rgb[:, :, :3]).save(output / f"rgb_{step:07d}.png")
            np.savez_compressed(output / f"depth_{step:07d}.npz", depth=depth, time=t, camera_position=camera_position, camera_orientation=quat)
            image_count += 1
        previous_state = action["state"]
        readings.write(json.dumps(observation, allow_nan=False) + "\n")
        speed = float(np.linalg.norm(controller.robot.get_linear_velocity()))
        row = {"time": t, "position": position.tolist(), "orientation": quat.tolist(), "speed": speed,
               "action": action, "lift_states": {k: {"z": e.z, "state": e.state} for k, e in lifts.items()}}
        telemetry.write(json.dumps(row) + "\n")
        telemetry.flush()
        if action.get("stop"):
            command[:] = 0
            stopped_duration = stopped_duration + .1 if speed < .06 else 0
            if stopped_duration >= 1.0:
                report.update(evaluate_stop(spec, position, speed, action["identity"], stopped_duration))
                report["conditions"]["arrival_visual_evidence"] = goal_identity_time > last_selection_time
                report["success"] = all(report["conditions"].values())
                report["status"] = "completed" if report["success"] else "failed_stop"
                break
        if args.sensor_only and t > 3.0:
            report["status"] = "sensor_check"
            break
        supporting_z = min((0., 3., 6., 9., 12.), key=lambda z: abs(position[2] - z - .35))
        if any(lift.inside(position) for lift in lifts.values()):
            supporting_z = next(lift.z for lift in lifts.values() if lift.inside(position))
        if step > 400 and (position[2] < supporting_z + .14 or rotation[2, 2] < .55):
            report["status"] = "fallen"
            break
    else:
        report["status"] = "timeout"
    report.update({"simulation_seconds": t, "wall_seconds": time.monotonic() - started,
                   "final_policy_state": action["state"], "final_position": position.tolist(),
                   "verified_edges": action.get("verified_edges"), "distance_m": action.get("distance_m"),
                   "contact_pairs": sorted(contacts), "contact_event_count": contact_event_count,
                   "sensor_frames_saved": image_count,
                   "observed_sign_kinds": sorted(observed_kinds), "imu_valid": imu_valid,
                   "api_events": events, "isolation_passed": True})
    report["contact_quality_passed"] = contact_event_count > 0 and not contacts
    (output / "contact_details.json").write_text(json.dumps(
        [{"robot_shape": pair[0], "environment_shape": pair[1], **value} for pair, value in contact_details.items()], indent=2))
    telemetry.close()
    readings.close()
except Exception:
    report["status"] = "error"
    report["error"] = traceback.format_exc()
    traceback.print_exc()
finally:
    try:
        # Persist evidence before cleanup; cleanup must not discard the result.
        temporary = output / "report.json.tmp"
        temporary.write_text(json.dumps(report, indent=2))
        temporary.replace(output / "report.json")
        print("P1_REPORT", json.dumps(report), flush=True)
    finally:
        try:
            if policy:
                policy.close()
        finally:
            app.close()
