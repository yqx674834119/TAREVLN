"""Single-scene native TARE / DeepSeek vision / articulated Go2 experiment."""
from __future__ import annotations
import argparse
import base64
import io
import json
import math
import hashlib
import shutil
import sys
from pathlib import Path
import time
import traceback
import zlib
from contact_audit import accumulate_contact, contact_quality

ap = argparse.ArgumentParser()
ap.add_argument("--name", required=True)
ap.add_argument("--task", choices=["meeting", "lounge"], default="meeting")
ap.add_argument("--max-seconds", type=float, default=600)
ap.add_argument("--geometry-only", action="store_true")
ap.add_argument("--preview", action="store_true")
ap.add_argument("--spawn", type=float, nargs=3)
ap.add_argument("--yaw", type=float, default=0.)
ap.add_argument("--checkpoint", type=Path)
ap.add_argument("--odometry", choices=["ideal", "gicp"], default="ideal")
ap.add_argument("--lidar-profile", choices=["legacy11", "uniform64"], default="legacy11")
ap.add_argument("--api-proxy", help="Optional project-owned loopback CONNECT relay, e.g. http://127.0.0.1:18119")
ap.add_argument('--sam',action='store_true',help='Require SAM2-refined observed surfaces for arrival verification')
ap.add_argument('--sam-gpu',type=int,default=2)
args = ap.parse_args()
ROOT = Path(__file__).resolve().parents[1]
VLN = ROOT.parent
private = ROOT / "data/sim/office/private"
out = ROOT / "data/sim/office/runs" / args.name
out.mkdir(parents=True, exist_ok=False)
report = {"status": "starting", "success": False, "policy": "SysNav native TARE + DeepSeek Flash vision task adapter",
          "robot": "articulated Go2 with recorded simulation-only checkpoint", "odometry": args.odometry,
          "localization_control_ready": False, "physical_hardware_ready": False,
          "lidar_profile": args.lidar_profile,
          "api_proxy": args.api_proxy,
          "sam_enabled":args.sam,"sam_gpu":args.sam_gpu if args.sam else None,
          "mode": "preview" if args.preview else "geometry_only" if args.geometry_only else "vlm_task",
          "source_revision": "SysNav 478727e5a559be22822d5d093b57431449336287 + recorded patch",
          "task": args.task, "requested_spawn": args.spawn, "requested_yaw": args.yaw, "max_seconds": args.max_seconds}
(out / "report.json").write_text(json.dumps(report, indent=2))
archive = out / "source"
archive.mkdir()
hashes = {}
for relative in ["sim/run_office.py", "sim/office_ros_worker.py", "sim/office_sandbox.py", "sim/go2_policy.py",
                 "sim/contact_audit.py", "src/vln_core/vision_api.py", "src/vln_policy/office_contract.py", "src/vln_policy/office_decision.py", "src/vln_policy/office_tasks.py", "src/vln_policy/office_grounding.py", "src/vln_policy/office_surface.py", "src/vln_policy/office_candidates.py", "config/office-ros.yaml",
                 "config/sysnav-provenance.json", "patches/sysnav-isaac-mission.patch", "scripts/odometry_worker.py",
                 "scripts/sam_worker.py","config/sam2-provenance.json",*[str(p.relative_to(ROOT)) for p in (ROOT/'src/vln_perception').glob('*.py')],
                 "src/vln_policy/inspection_handoff.py",
                 *[str(p.relative_to(ROOT)) for p in (ROOT/'src/vln_localization').glob('*.py')]]:
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
odometry = None
sam = None
try:
    import numpy as np
    from PIL import Image
    import omni.usd
    from omni.physx import get_physx_scene_query_interface, get_physx_simulation_interface
    from pxr import Gf, UsdPhysics, PhysxSchema, PhysicsSchemaTools
    from isaacsim.core.api import World
    from isaacsim.core.utils.rotations import quat_to_rot_matrix, rot_matrix_to_quat
    from isaacsim.sensors.camera import Camera
    from go2_policy import Go2Policy
    from office_sandbox import OfficeProcess
    sys.path.insert(0, str(ROOT/'src'))
    from vln_localization.process import OdometryProcess
    from vln_perception.process import SamProcess

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
    checkpoint = args.checkpoint or VLN / "assets/go2/checkpoint.pt"
    report.update(robot_checkpoint=str(checkpoint), robot_checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest())
    controller = Go2Policy(stage, VLN / "assets/go2/go2_lab.usd", checkpoint, spawn)
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
    contact_stream = open(out / "contacts.jsonl", "w")
    def contact_callback(headers, data):
        for header in headers:
            a, b = str(PhysicsSchemaTools.intToSdfPath(header.collider0)), str(PhysicsSchemaTools.intToSdfPath(header.collider1))
            robot, env = (a, b) if a.startswith("/World/Go2/") else (b, a)
            if not robot.startswith("/World/Go2/") or env.startswith("/World/Go2/"):
                continue
            points = [{"position": list(d.position), "normal": list(d.normal), "impulse": list(d.impulse),
                       "separation": float(d.separation)}
                      for d in data[header.contact_data_offset:header.contact_data_offset+header.num_contact_data]]
            t = float(world.current_time)
            contact_stream.write(json.dumps({"time":t,"robot":robot,"environment":env,"points":points,
                "event_type":str(header.type),"data_offset":header.contact_data_offset,
                "data_count":header.num_contact_data,"callback_data_length":len(data)})+"\n")
            accumulate_contact(contacts, robot, env, t, points)
    subscription = get_physx_simulation_interface().subscribe_contact_report_events(contact_callback)
    stderr = open(out / "worker.log", "w")
    if args.sam and not args.preview and not args.geometry_only:
        sam=SamProcess(ROOT,out/'sam',args.sam_gpu)
        report['sam_isolation']=sam.isolation
    if not args.preview:
        worker = OfficeProcess(ROOT, out / "agent", stderr, geometry_only=args.geometry_only, task=args.task, api_proxy=args.api_proxy,sam_ipc=sam.ipc if sam else None)
    if args.odometry == 'gicp':
        odometry = OdometryProcess(ROOT, out/'odometry', out)
        odom_trace = (out/'odometry/poses.jsonl').open('w')
        # Public initialization: robot starts level, at nominal standing height.
        # Neither the initial world XY nor world yaw is supplied to navigation.
        estimated_pose = np.eye(4)
        estimated_pose[2, 3] = .4
        previous_estimate, pose_time, estimated_velocity = None, None, np.zeros(3)
        odometry_errors, odometry_rot_errors, odometry_times = [], [], []
        report['sensor_model'] = 'rigid base-mounted instantaneous virtual lidar and RGB-D; no noise or IMU fusion'
        report['odometry_initialization'] = 'relative origin, level orientation, nominal base height 0.4 m'
    else:
        report['sensor_model'] = 'legacy world-stabilized virtual lidar and RGB-D'
    trace = open(out / "trajectory.jsonl", "w")
    command = np.zeros(3)
    action = {}
    samples, distance, dwell = 0, 0., 0.
    last_pos = np.asarray(spawn)
    start = time.monotonic()
    # Neither ray pattern is a calibrated model of a physical Go2/MID360 sensor.
    elevations = [-40,-25,-15,-7,0,7,15,25,40,55,70] if args.lidar_profile == 'legacy11' else np.linspace(-40,70,64)
    angles = [(a, e) for a in np.arange(360)*math.pi/180 for e in np.deg2rad(elevations)]
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
            raw_points = []
            mount = np.array([0.,0.,.45])
            origin = pos + (rot @ mount if odometry else mount)
            for a, e in angles:
                local_direction = np.array([math.cos(a)*math.cos(e),math.sin(a)*math.cos(e),math.sin(e)])
                direction = rot @ local_direction if odometry else local_direction
                hit = query.raycast_closest(tuple(map(float, origin)), tuple(map(float, direction)), 10.)
                if hit["hit"] and hit["distance"] > .4:
                    if odometry and any(str(hit.get(k, '')).startswith('/World/Go2') for k in ('rigidBody', 'collision')):
                        continue
                    xyz = origin + direction*hit["distance"]
                    points.append([*xyz.tolist(), 0.])
                    if odometry:
                        raw_points.append((mount + local_direction*hit['distance']).tolist())
            obs["points"] = points
            if odometry:
                # Save failed frames too; these files stay evaluator-private.
                np.savez_compressed(out/f'raw_scan_{step:07d}.npz', points_base_xyz=np.asarray(raw_points,dtype=np.float32), time=t)
                np.savez_compressed(out/f'scan_{step:07d}.npz', points=np.asarray(points,dtype=np.float32), time=t, position=pos, quaternion=quat)
                result = odometry.ask(t, raw_points)
                if not result.get('valid'):
                    command[:] = 0
                    (out/'odometry/failure.json').write_text(json.dumps({'time':t, **result},indent=2))
                    raise RuntimeError('LiDAR odometry failed: '+str(result.get('reason')))
                estimated_pose = np.asarray(result['odom_from_base'])
                estimated_pose[2, 3] += .4
                if previous_estimate is not None:
                    estimated_velocity = estimated_pose[:3,:3].T @ ((estimated_pose[:3,3]-previous_estimate[:3,3])/(t-pose_time))
                previous_estimate, pose_time = estimated_pose.copy(), t
                registered = np.asarray(raw_points) @ estimated_pose[:3,:3].T + estimated_pose[:3,3]
                obs['points'] = np.column_stack((registered, np.zeros(len(registered)))).tolist()
                # All truth transforms below are evaluator-only and never sent to either worker.
                truth_pose = np.eye(4); truth_pose[:3,:3] = rot; truth_pose[:3,3] = pos
                if 'evaluation_world_from_odometry' not in report:
                    report['evaluation_world_from_odometry'] = (truth_pose @ np.linalg.inv(estimated_pose)).tolist()
                aligned = np.asarray(report['evaluation_world_from_odometry']) @ estimated_pose
                delta = np.linalg.inv(truth_pose) @ aligned
                error = float(np.linalg.norm(delta[:3,3]))
                angle_error = math.acos(float(np.clip((np.trace(delta[:3,:3])-1)/2,-1,1)))
                odometry_errors.append(error); odometry_rot_errors.append(angle_error)
                odometry_times.append(result['processing_seconds'])
                odom_trace.write(json.dumps({'time':t, 'estimate':result, 'truth_pose':truth_pose.tolist(),
                    'aligned_estimate':aligned.tolist(), 'translation_error_m':error, 'rotation_error_rad':angle_error})+'\n')
                odom_trace.flush()
        if odometry:
            estimated_rot, estimated_pos = estimated_pose[:3,:3], estimated_pose[:3,3]
            obs.update(position=estimated_pos.tolist(), orientation=rot_matrix_to_quat(estimated_rot).tolist(),
                       velocity=estimated_velocity.tolist(), yaw=math.atan2(estimated_rot[1,0],estimated_rot[0,0]), pose_time=pose_time)
        if step % 400 == 0:
            views = []
            for index, offset in enumerate([0,math.pi/2,math.pi,-math.pi/2]):
                cyaw = yaw + offset
                cq = np.array([math.cos(cyaw/2),0.,0.,math.sin(cyaw/2)])
                cp = pos + np.array([0.,0.,.45])
                if odometry:
                    view_rot = np.array([[math.cos(offset),-math.sin(offset),0.],
                                         [math.sin(offset),math.cos(offset),0.], [0.,0.,1.]])
                    cp = pos + rot @ mount
                    cq = rot_matrix_to_quat(rot @ view_rot)
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
                depth_mm = (np.clip(np.nan_to_num(depth, nan=0., posinf=0., neginf=0.), 0, 65.535)*1000).astype('<u2')
                policy_rotation = estimated_rot @ view_rot if odometry else quat_to_rot_matrix(cq)
                policy_position = estimated_pos + estimated_rot @ mount if odometry else cp
                views.append({"label": ["front","left","rear","right"][index], "jpeg": base64.b64encode(data.getvalue()).decode(),
                              "depth_mm": base64.b64encode(zlib.compress(depth_mm.tobytes())).decode(), "image_size": [640,480],
                              "rotation": policy_rotation.tolist(), "position": policy_position.tolist(), "intrinsics": [float(k[0,0]),float(k[1,1]),float(k[0,2]),float(k[1,2])]})
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
            # Preserve legacy evaluator scan format: true world-frame scan, never read by policy.
            np.savez_compressed(out / f"scan_{step:07d}.npz", points=np.asarray(points,dtype=np.float32), time=t, position=pos, quaternion=quat)
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
    report.update(contact_quality(contacts))
    trace.close()
except (Exception, KeyboardInterrupt) as error:
    report.update(status="interrupted" if isinstance(error, KeyboardInterrupt) else "error", error=traceback.format_exc())
    if "t" in locals():
        report.update(simulation_seconds=t, distance_m=distance, final_position=pos.tolist(), final_action=action)
    traceback.print_exc()
finally:
    if odometry:
        odometry.close()
        if 'odom_trace' in locals():
            odom_trace.close()
        if 'odometry_errors' in locals() and odometry_errors:
            report['localization_evaluation'] = {'alignment':'first pose only; evaluator private',
                'frames':len(odometry_errors), 'translation_rmse_m':float(np.sqrt(np.mean(np.square(odometry_errors)))),
                'maximum_translation_error_m':max(odometry_errors), 'final_translation_error_m':odometry_errors[-1],
                'maximum_rotation_error_deg':math.degrees(max(odometry_rot_errors)),
                'processing_p95_ms':float(np.quantile(odometry_times,.95)*1000)}
    if worker:
        worker.close()
    if sam:
        sam.close()
    if "contacts" in locals():
        (out / "contacts.json").write_text(json.dumps(contacts, indent=2))
        report.update(contact_quality(contacts))
        contact_stream.close()
    (out / "report.json").write_text(json.dumps(report, indent=2))
    print("OFFICE_REPORT", json.dumps(report), flush=True)
    app.close()
