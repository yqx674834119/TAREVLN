"""Single-scene native TARE / DeepSeek vision / articulated Go2 experiment."""
from __future__ import annotations
import argparse
import base64
import io
import json
import math
import hashlib
import shutil
from pathlib import Path
import time
import traceback

ap = argparse.ArgumentParser()
ap.add_argument("--name", required=True)
ap.add_argument("--task", choices=["meeting", "lounge"], default="meeting")
ap.add_argument("--max-seconds", type=float, default=600)
ap.add_argument("--geometry-only", action="store_true")
ap.add_argument("--preview", action="store_true")
ap.add_argument("--spawn", type=float, nargs=3)
ap.add_argument("--yaw", type=float, default=0.)
args = ap.parse_args()
ROOT = Path(__file__).resolve().parents[1]
VLN = ROOT.parent
private = ROOT / "data/sim/office/private"
out = ROOT / "data/sim/office/runs" / args.name
out.mkdir(parents=True, exist_ok=False)
report = {"status": "starting", "success": False, "policy": "SysNav native TARE + DeepSeek Flash vision task adapter",
          "robot": "articulated Go2 with NVIDIA flat-terrain checkpoint", "odometry": "ideal simulator pose",
          "mode": "preview" if args.preview else "geometry_only" if args.geometry_only else "vlm_task",
          "source_revision": "SysNav 478727e5a559be22822d5d093b57431449336287 + recorded patch",
          "task": args.task, "requested_spawn": args.spawn, "requested_yaw": args.yaw, "max_seconds": args.max_seconds}
(out / "report.json").write_text(json.dumps(report, indent=2))
archive = out / "source"
archive.mkdir()
hashes = {}
for relative in ["sim/run_office.py", "sim/office_ros_worker.py", "sim/office_sandbox.py", "sim/go2_policy.py",
                 "src/vln_core/vision_api.py", "src/vln_policy/office_contract.py", "src/vln_policy/office_decision.py", "src/vln_policy/office_tasks.py", "config/office-ros.yaml",
                 "config/sysnav-provenance.json", "patches/sysnav-isaac-mission.patch"]:
    file = ROOT / relative
    if file.exists():
        target = archive / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(file, target)
        hashes[relative] = hashlib.sha256(file.read_bytes()).hexdigest()
(archive / "sha256.json").write_text(json.dumps(hashes, indent=2))
native = VLN / "deps/sysnav/build/tare_planner"
if native.exists():
    evidence = {"binary_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                for p in native.glob("*.so")},
                "cmake_flags": [line for line in (native / "CMakeCache.txt").read_text().splitlines()
                if line.startswith(("CMAKE_CXX_FLAGS:", "CMAKE_CXX_FLAGS_RELEASE:", "CMAKE_CXX_FLAGS_RELWITHDEBINFO:", "CMAKE_BUILD_TYPE:"))]}
    (archive / "native-build.json").write_text(json.dumps(evidence, indent=2))
from isaacsim import SimulationApp
app = SimulationApp({"headless": True, "open_usd": str(private / "scene.usda"), "create_new_stage": False,
                     "multi_gpu": False, "active_gpu": 0, "physics_gpu": 0, "anti_aliasing": 0,
                     "width": 640, "height": 480, "fast_shutdown": True})
worker = None
try:
    import numpy as np
    from PIL import Image
    import omni.usd
    from omni.physx import get_physx_scene_query_interface, get_physx_simulation_interface
    from pxr import Gf, UsdPhysics, PhysxSchema, PhysicsSchemaTools
    from isaacsim.core.api import World
    from isaacsim.core.utils.rotations import quat_to_rot_matrix
    from isaacsim.sensors.camera import Camera
    from go2_policy import Go2Policy
    from office_sandbox import OfficeProcess

    oracle_file = private / ("oracle.json" if args.task == "meeting" else "oracle_lounge.json")
    oracle = json.loads(oracle_file.read_text()) if oracle_file.exists() else {}
    build = json.loads((private / "build.json").read_text())
    manifest = json.loads((VLN / "assets/usd-cache/office-download.json").read_text())
    if not args.preview and (not oracle or (not manifest["complete"] and not args.geometry_only)):
        raise RuntimeError("Task run requires complete assets and private evaluation definition")
    spawn = args.spawn or oracle["start_xyz"]
    report["start_position"] = spawn
    for filename in (oracle_file.name, "scene.usda", "build.json"):
        if (private / filename).exists():
            shutil.copyfile(private / filename, archive / filename)
    stage = omni.usd.get_context().get_stage()
    world = World(physics_dt=.005, rendering_dt=.1, stage_units_in_meters=1., device="cpu")
    controller = Go2Policy(stage, VLN / "assets/go2/go2_lab.usd", VLN / "assets/go2/checkpoint.pt", spawn)
    world.scene.add(controller.robot)
    for prim in stage.Traverse():
        if str(prim.GetPath()).startswith("/World/Go2/") and prim.HasAPI(UsdPhysics.RigidBodyAPI):
            PhysxSchema.PhysxContactReportAPI.Apply(prim).CreateThresholdAttr(0.)
    camera = Camera("/World/TaskCamera", frequency=-1, resolution=(640, 480))
    camera.set_focal_length(2.4)
    camera.set_horizontal_aperture(4.8)
    camera.set_vertical_aperture(3.6)
    camera.set_clipping_range(.05, 40)
    chase = Camera("/World/ReviewCamera", frequency=-1, resolution=(640,480))
    chase.set_focal_length(2.4)
    chase.set_horizontal_aperture(4.8)
    chase.set_vertical_aperture(3.6)
    world.reset()
    controller.initialize()
    controller.robot.set_world_pose(np.asarray(spawn), np.array([math.cos(args.yaw/2), 0., 0., math.sin(args.yaw/2)]))
    camera.initialize()
    chase.initialize()
    camera.add_distance_to_image_plane_to_frame()
    query = get_physx_scene_query_interface()
    k = camera.get_intrinsics_matrix()
    contacts = {}
    def contact_callback(headers, data):
        for header in headers:
            a, b = str(PhysicsSchemaTools.intToSdfPath(header.collider0)), str(PhysicsSchemaTools.intToSdfPath(header.collider1))
            robot, env = (a, b) if a.startswith("/World/Go2/") else (b, a)
            if not robot.startswith("/World/Go2/") or env.startswith("/World/Go2/"):
                continue
            # Retain raw collider pairs, including calf/floor contacts; never
            # infer collision-free locomotion from task completion alone.
            pair = robot + " | " + env
            item = contacts.setdefault(pair, {"headers": 0, "first_time": float(world.current_time), "max_impulse": 0.})
            item["headers"] += 1
            for i in range(header.contact_data_offset, header.contact_data_offset + header.num_contact_data):
                item["max_impulse"] = max(item["max_impulse"], float(np.linalg.norm(data[i].impulse)))
    subscription = get_physx_simulation_interface().subscribe_contact_report_events(contact_callback)
    stderr = open(out / "worker.log", "w")
    if not args.preview:
        worker = OfficeProcess(ROOT, out / "agent", stderr, geometry_only=args.geometry_only, task=args.task)
    trace = open(out / "trajectory.jsonl", "w")
    command = np.zeros(3)
    action = {}
    samples, distance, dwell = 0, 0., 0.
    last_pos = np.asarray(spawn)
    start = time.monotonic()
    # A virtual 360 x 11-channel scanning lidar, not a MID360 noise simulation.
    angles = [(a, e) for a in np.arange(360)*math.pi/180 for e in np.deg2rad([-40,-25,-15,-7,0,7,15,25,40,55,70])]
    for step in range(int(args.max_seconds/.005)):
        if (out / "STOP").exists():
            report["status"] = "operator_stopped"
            break
        controller.step(command)
        world.step(render=False)
        if step % 20:
            continue
        t = float(world.current_time)
        pos, quat = controller.robot.get_world_pose()
        rot = quat_to_rot_matrix(quat)
        yaw = math.atan2(rot[1,0], rot[0,0])
        vel = controller.robot.get_linear_velocity()
        speed = float(np.linalg.norm(vel))
        distance += float(np.linalg.norm(pos[:2]-last_pos[:2]))
        last_pos = pos.copy()
        obs = {"time": t, "position": pos.tolist(), "orientation": quat.tolist(), "velocity": vel.tolist(), "yaw": yaw}
        if step % 40 == 0:
            points = []
            origin = pos + np.array([0.,0.,.45])
            for a, e in angles:
                direction = np.array([math.cos(a)*math.cos(e),math.sin(a)*math.cos(e),math.sin(e)])
                hit = query.raycast_closest(tuple(map(float, origin)), tuple(map(float, direction)), 10.)
                if hit["hit"] and hit["distance"] > .4:
                    xyz = origin + direction*hit["distance"]
                    points.append([*xyz.tolist(), 0.])
            obs["points"] = points
        if step % 400 == 0:
            views = []
            for index, offset in enumerate([0,math.pi/2,math.pi,-math.pi/2]):
                cyaw = yaw + offset
                cq = np.array([math.cos(cyaw/2),0.,0.,math.sin(cyaw/2)])
                cp = pos + np.array([0.,0.,.45])
                camera.set_world_pose(cp, cq, camera_axes="world")
                # Each virtual camera orientation needs fresh render history.
                for _ in range(8):
                    world.render()
                rgb = camera.get_rgba()
                if rgb is None or not rgb.size:
                    if t > 2: raise RuntimeError("Camera produced no frame")
                    continue
                image = Image.fromarray(rgb[:,:,:3])
                data = io.BytesIO()
                image.save(data, format="JPEG", quality=85)
                image.save(out / f"rgb_{step:07d}_{index}.jpg")
                depth = camera.get_depth()
                camera_time = camera.get_current_frame().get("rendering_time")
                np.savez_compressed(out / f"depth_{step:07d}_{index}.npz", depth=depth, time=t, camera_time=camera_time, position=cp, quaternion=cq)
                views.append({"label": ["front","left","rear","right"][index], "jpeg": base64.b64encode(data.getvalue()).decode(),
                              "rotation": quat_to_rot_matrix(cq).tolist(), "position": cp.tolist(), "intrinsics": [float(k[0,0]),float(k[1,1]),float(k[0,2]),float(k[1,2])]})
                samples += 1
            obs["views"] = views
            # Review only: never supplied to the navigation or VLM process.
            eye = pos + np.array([-2.0*math.cos(yaw),-2.0*math.sin(yaw),1.6])
            look = Gf.Matrix4d().SetLookAt(Gf.Vec3d(*map(float,eye)), Gf.Vec3d(*map(float,pos+np.array([0.,0.,.15]))), Gf.Vec3d(0,0,1)).GetInverse().ExtractRotationQuat()
            chase.set_world_pose(eye, np.array([look.GetReal(),*look.GetImaginary()]), camera_axes="usd")
            for _ in range(8): world.render()
            review_rgb = chase.get_rgba()
            if review_rgb is not None and review_rgb.size:
                Image.fromarray(review_rgb[:,:,:3]).save(out / f"chase_{step:07d}.jpg")
        if "points" in obs:
            np.savez_compressed(out / f"scan_{step:07d}.npz", points=np.asarray(obs["points"],dtype=np.float32), time=t, position=pos, quaternion=quat)
        if worker and t > 2:
            action = worker.ask(obs)
            command = np.asarray(action["cmd_vel"], dtype=float)
            if command.shape != (3,) or not np.isfinite(command).all():
                raise ValueError("Invalid pathFollower command")
            command = np.clip(command, [-.35,-.25,-.7], [.45,.25,.7])
        row = {"time": t, "position": pos.tolist(), "orientation": quat.tolist(), "speed": speed, "command": command.tolist(), "action": action,
               "joint_positions": controller.robot.get_joint_positions().tolist()}
        trace.write(json.dumps(row) + "\n")
        trace.flush()
        if step % 2000 == 0:
            print("OFFICE", round(t,1), [round(float(x),2) for x in pos], action.get("reason"), action.get("vlm_calls"), action.get("counts"), flush=True)
        if action.get("stop"):
            command[:] = 0
            dwell = dwell + .1 if speed < .06 else 0.
            if dwell >= 1.:
                region = oracle["goal_rectangle"]
                regions = oracle.get("goal_rectangles", [region])
                floor_range = oracle.get("goal_z_range", [.05,.7])
                inside = (floor_range[0] <= pos[2] <= floor_range[1] and
                          any(r[0] <= pos[0] <= r[2] and r[1] <= pos[1] <= r[3] for r in regions))
                report.update(status="completed" if inside else "false_positive_stop", success=bool(inside and not args.geometry_only), stop_dwell=dwell, independent_goal_region=bool(inside))
                break
        if t > 2 and (pos[2] < -.05 or rot[2,2] < .55):
            report["status"] = "fallen"
            break
    else:
        report["status"] = "preview_complete" if args.preview else "timeout"
    report.update(simulation_seconds=t, wall_seconds=time.monotonic()-start, distance_m=distance, final_position=pos.tolist(),
                  sensor_frames=samples, final_action=action, assets_complete=manifest["complete"], collision_meshes=build["collision_meshes"])
    (out / "contacts.json").write_text(json.dumps(contacts, indent=2))
    report["nonfoot_contact_pairs"] = [pair for pair in contacts if "_foot/" not in pair.split(" | ")[0]]
    report["contact_quality_passed"] = bool(contacts) and not report["nonfoot_contact_pairs"]
    trace.close()
except (Exception, KeyboardInterrupt) as error:
    report.update(status="interrupted" if isinstance(error, KeyboardInterrupt) else "error", error=traceback.format_exc())
    if "t" in locals():
        report.update(simulation_seconds=t, distance_m=distance, final_position=pos.tolist(), final_action=action)
    traceback.print_exc()
finally:
    if worker:
        worker.close()
    if "contacts" in locals():
        (out / "contacts.json").write_text(json.dumps(contacts, indent=2))
        report["nonfoot_contact_pairs"] = [pair for pair in contacts if "_foot/" not in pair.split(" | ")[0]]
        report["contact_quality_passed"] = bool(contacts) and not report["nonfoot_contact_pairs"]
    (out / "report.json").write_text(json.dumps(report, indent=2))
    print("OFFICE_REPORT", json.dumps(report), flush=True)
    app.close()
