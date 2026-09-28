"""Single-scene native TARE / DeepSeek vision / articulated Go2 experiment."""
from __future__ import annotations
import argparse
import base64
import io
import json
import math
import hashlib
import itertools
import os
import shutil
import sys
from pathlib import Path
import time
import traceback
import zlib
from contact_audit import accumulate_contact, contact_quality
from sysnav_evaluation import evaluate_object_stop, visible_targets
from review_camera import unobstructed_eye
from camera_readiness import render_rgba
import subprocess

ap = argparse.ArgumentParser()
ap.add_argument("--name", required=True)
ap.add_argument("--task", choices=["meeting", "lounge", "meeting_crossroom"], default="meeting")
ap.add_argument('--api-budget',type=int,default=150)
ap.add_argument('--enable-room-early-stop',action='store_true',help='Explicitly enable the branch hard-disabled in the pinned Go2 release')
ap.add_argument('--mp3d-scene',help='Converted MP3D scene; evaluator loads the original R2R-CE instruction')
ap.add_argument('--mp3d-episode',type=int)
ap.add_argument("--max-seconds", type=float, default=600)
ap.add_argument("--geometry-only", action="store_true")
ap.add_argument("--preview", action="store_true")
ap.add_argument("--spawn", type=float, nargs=3)
ap.add_argument("--yaw", type=float, default=0.)
ap.add_argument("--checkpoint", type=Path)
ap.add_argument("--odometry", choices=["ideal", "gicp"], default="ideal")
ap.add_argument("--gicp-correspondence",type=float,default=.5)
ap.add_argument("--lidar-profile", choices=["legacy11", "uniform64"], default="legacy11")
ap.add_argument("--api-proxy", help="Optional project-owned loopback CONNECT relay, e.g. http://127.0.0.1:18119")
ap.add_argument('--sam',action='store_true',help='Require SAM2-refined observed surfaces for arrival verification')
ap.add_argument('--sam-gpu',type=int,default=2)
ap.add_argument('--lidar-backend',choices=['auto','physx','cuda'],default='auto',help='CUDA is validated for the static office; auto uses PhysX for other scenes')
ap.add_argument('--lidar-gpu',type=int,default=3)
ap.add_argument('--panorama-backend',choices=['reference','atlas'],default='atlas')
ap.add_argument('--fabric',action='store_true',help='Experimental PhysX Fabric transform updates')
ap.add_argument('--profile',action='store_true',help='Record cProfile of the main loop, excluding setup')
ap.add_argument('--gui',action='store_true',help='Show Isaac Sim GUI and refresh the viewport during the run')
ap.add_argument('--isaac-gpu',type=int,default=0)
ap.add_argument('--vision-gpu',type=int,default=1)
ap.add_argument('--vlm-gpu',type=int,default=3)
ap.add_argument('--surface-geometry',choices=['pixel_cross','metric_plane'],default='pixel_cross')
args = ap.parse_args()
if args.geometry_only or args.sam or args.preview:
    ap.error("Full SysNav requires all semantic modules; no geometry-only/preview/external-SAM mode")
if args.api_budget!=-1 and not 1<=args.api_budget<=1500:ap.error('API budget must be -1 or 1..1500')
if (args.max_seconds <= 0 and args.max_seconds != -1) or not math.isfinite(args.max_seconds):ap.error('--max-seconds must be a positive value or -1 for unlimited runtime')
if args.gui and not (os.environ.get('DISPLAY') or os.environ.get('WAYLAND_DISPLAY')):
    ap.error('GUI requires an accessible desktop display')
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from vln_policy.sysnav_tasks import TASKS
VLN = ROOT.parent
private = ROOT / "data/sim/office/private"
mp3d=None
if args.mp3d_scene:
    if args.mp3d_episode is None or args.spawn is not None or args.yaw!=0.:
        ap.error('MP3D requires an episode id and its original start pose')
    from mp3d_episode import load_episode
    mp3d=load_episode(VLN,args.mp3d_scene,args.mp3d_episode)
    private=VLN/'assets/mp3d-isaac'/args.mp3d_scene
    args.yaw=mp3d['yaw']
elif args.mp3d_episode is not None:
    ap.error('--mp3d-episode requires --mp3d-scene')
instruction=mp3d['instruction'] if mp3d else TASKS[args.task][1]
out = ROOT / "data/sim/office/runs" / args.name
out.mkdir(parents=True, exist_ok=False)
report = {"status": "starting", "success": False, "policy": "Full native SysNav + DeepSeek Flash",
          "robot": "articulated Go2 with recorded simulation-only checkpoint", "odometry": args.odometry,
          "gicp_correspondence_m":args.gicp_correspondence,"localization_control_ready": False, "physical_hardware_ready": False,
          "lidar_profile": args.lidar_profile,
          "api_proxy": args.api_proxy,
          "api_budget": args.api_budget,
          "room_early_stop_enabled":args.enable_room_early_stop,
          "surface_geometry":"native_sam2_lidar_fusion","sam_enabled":True,"sam_gpu":args.sam_gpu,
          "mode": "sysnav_full",
          "terminal_control": "one-second torque-limited PD blend to standing joints after native SysNav STOP",
          "instruction": instruction,
          "source_revision": "SysNav 478727e5a559be22822d5d093b57431449336287 + recorded patch",
          "task": args.task, "requested_spawn": args.spawn, "requested_yaw": args.yaw, "max_seconds": args.max_seconds,
          "gui": args.gui, "gpu_allocation": {"isaac": args.isaac_gpu, "physics": "cpu", "vision": args.vision_gpu, "sam": args.sam_gpu, "vlm": args.vlm_gpu}}
if mp3d:
    report.update(mode='sysnav_full_mp3d',task='mp3d:'+str(args.mp3d_episode),scene=args.mp3d_scene,
                  episode_id=args.mp3d_episode,protocol='Isaac transfer of R2R-CE val_unseen; not official Habitat scores',
                  evaluation_pending=True)
    preflight_file=private/'preflight'/f'episode-{args.mp3d_episode}'/'report.json'
    preflight=json.loads(preflight_file.read_text()) if preflight_file.exists() else {}
    checkpoint=args.checkpoint or VLN/'assets/go2/checkpoint.pt'
    verified=bool(preflight.get('version')=='go2-mp3d-spawn-v2' and preflight.get('physical_smoke_pass')
        and preflight.get('scene_sha256')==hashlib.sha256((private/'scene.usda').read_bytes()).hexdigest()
        and preflight.get('checkpoint_sha256')==hashlib.sha256(checkpoint.read_bytes()).hexdigest()
        and preflight.get('episode_id')==args.mp3d_episode)
    report['spawn_preflight_passed']=verified
    if not verified:
        report.update(status='invalid_spawn',evaluation_pending=False,error='Go2 spawn preflight missing, stale or failed')
        (out/'report.json').write_text(json.dumps(report,indent=2))
        raise SystemExit(2)
    shutil.copyfile(preflight_file,out/'spawn_preflight.json')
(out / "report.json").write_text(json.dumps(report, indent=2))
archive = out / "source"
archive.mkdir()
hashes = {}
for relative in ["sim/run_sysnav_full.py", "sim/sysnav_full_worker.py", "sim/sysnav_full_sandbox.py", "sim/go2_policy.py",
                 "sim/sysnav_sensor_bridge.py", "sim/sysnav_runtime_audit.py", "config/sysnav-isaac-overrides.json",
                 "config/sysnav-parity-manifest.json", "config/sysnav-navigation-boundary.ply", "config/sysnav-source-review.json",
                 "src/vln_core/sysnav_sensor_contract.py",
                 "src/vln_policy/sysnav_tasks.py", "sim/sysnav_evaluation.py", "sim/fixtures/sysnav-object-evaluation.json",
                 "sim/sysnav_go2_policy.py", "sim/camera_readiness.py",
                 "sim/planning_recorder.py", "sim/runtime_metrics.py", "sim/gpu_lidar.py", "src/vln_core/planning_geometry.py",
                 "sim/mp3d_episode.py", "sim/build_mp3d_isaac.py", "sim/mp3d_textures.py", "sim/review_camera.py", "sim/check_mp3d_isaac.py", "scripts/benchmarks/run_isaac_mp3d.py", "scripts/benchmarks/evaluate_isaac_mp3d.py",
                 "sim/contact_audit.py", "src/vln_core/vision_api.py", "src/vln_policy/office_contract.py", "src/vln_policy/office_decision.py", "src/vln_policy/office_tasks.py", "src/vln_policy/office_grounding.py", "src/vln_policy/office_surface.py", "src/vln_policy/office_candidates.py", "config/sysnav-full-ros.yaml", "sim/sysnav_panorama.py",
                 "config/sysnav-full-provenance.json", "patches/sysnav-full-isaac.patch", "scripts/odometry_worker.py",
                 "scripts/sam_worker.py","config/sam2-provenance.json",*[str(p.relative_to(ROOT)) for p in (ROOT/'src/vln_perception').glob('*.py')],
                 "src/vln_policy/inspection_handoff.py", "src/vln_policy/office_planes.py",
                 *[str(p.relative_to(ROOT)) for p in (ROOT/'src/vln_localization').glob('*.py')]]:
    file = ROOT / relative
    if file.exists():
        target = archive / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(file, target)
    hashes[relative] = hashlib.sha256(file.read_bytes()).hexdigest()
if mp3d:(archive/'episode.json').write_text(json.dumps(mp3d,indent=2))
# Snapshot exact upstream implementation and model manifest for this run.
source_root=VLN/'deps/sysnav-full'
for subtree in ('src/semantic_mapping/semantic_mapping','src/vlm_node/vlm_node'):
    for file in (source_root/subtree).glob('*.py'):
        relative=Path('sysnav-source')/file.relative_to(source_root)
        target=archive/relative;target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(file,target);hashes[str(relative)]=hashlib.sha256(file.read_bytes()).hexdigest()
for file,label in [(VLN/'assets/sysnav/provenance.json','models.json'),(VLN/'logs/sysnav-full-python.lock','python.lock')]:
    if file.exists():shutil.copyfile(file,archive/label)
(archive/'upstream.diff').write_text(subprocess.check_output(['git','-C',str(source_root),'diff','HEAD'],text=True))
(archive/'git-revisions.json').write_text(json.dumps({
    'platform_head':subprocess.check_output(['git','-C',str(ROOT),'rev-parse','HEAD'],text=True).strip(),
    'sysnav_head':subprocess.check_output(['git','-C',str(source_root),'rev-parse','HEAD'],text=True).strip(),
    'sysnav_status':subprocess.check_output(['git','-C',str(source_root),'status','--porcelain'],text=True).splitlines()
},indent=2))
(archive / "sha256.json").write_text(json.dumps(hashes, indent=2))
native = VLN / "deps/sysnav-full/build/tare_planner"
if native.exists():
    evidence = {"binary_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                for p in native.glob("*.so")},
                "cmake_flags": [line for line in (native / "CMakeCache.txt").read_text().splitlines()
                if line.startswith(("CMAKE_CXX_FLAGS:", "CMAKE_CXX_FLAGS_RELEASE:", "CMAKE_CXX_FLAGS_RELWITHDEBINFO:", "CMAKE_BUILD_TYPE:"))]}
    (archive / "native-build.json").write_text(json.dumps(evidence, indent=2))
from isaacsim import SimulationApp
app = SimulationApp({"headless": not args.gui, "open_usd": str(private / "scene.usda"), "create_new_stage": False,
                     "multi_gpu": False, "active_gpu": args.isaac_gpu, "physics_gpu": args.isaac_gpu, "anti_aliasing": 0,
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
    from sysnav_go2_policy import SysNavGo2Policy as Go2Policy
    from sysnav_full_sandbox import FullSysNavProcess
    from sysnav_panorama import Panorama, cube_rotations
    sys.path.insert(0, str(ROOT/'src'))
    from vln_localization.process import OdometryProcess
    from vln_perception.process import SamProcess

    oracle_file = private / ("oracle_lounge.json" if args.task == "lounge" else "oracle.json")
    oracle = json.loads(oracle_file.read_text()) if oracle_file.exists() else {}
    object_evaluation = json.loads((ROOT/'sim/fixtures/sysnav-object-evaluation.json').read_text())
    object_definition = None if mp3d else object_evaluation['tasks'].get(args.task)
    report['evaluation_definition'] = object_evaluation['version'] if object_definition else 'legacy-public-area-only'
    build = json.loads((private / "build.json").read_text())
    manifest = {'complete':True} if mp3d else json.loads((VLN / "assets/usd-cache/office-download.json").read_text())
    if not mp3d and not args.preview and (not oracle or (not manifest["complete"] and not args.geometry_only)):
        raise RuntimeError("Task run requires complete assets and private evaluation definition")
    spawn = mp3d['spawn'] if mp3d else (args.spawn or (object_definition or {}).get('start_xyz') or oracle["start_xyz"])
    if mp3d:report['evaluation_definition']='R2R-CE goal geodesic <3 m and native STOP; separate Habitat navmesh evaluator'
    report["start_position"] = spawn
    for filename in (oracle_file.name, "scene.usda", "build.json"):
        if (private / filename).exists():
            shutil.copyfile(private / filename, archive / filename)
    stage = omni.usd.get_context().get_stage()
    world = World(physics_dt=.005, rendering_dt=.1, stage_units_in_meters=1., device="cpu")
    if args.fabric:world.get_physics_context().enable_fabric(True)
    report['fabric_enabled']=args.fabric
    checkpoint = args.checkpoint or VLN / "assets/go2/checkpoint.pt"
    report.update(robot_checkpoint=str(checkpoint), robot_checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest())
    controller = Go2Policy(stage, VLN / "assets/go2/go2_lab.usd", checkpoint, spawn)
    world.scene.add(controller.robot)
    for prim in stage.Traverse():
        if str(prim.GetPath()).startswith("/World/Go2/") and prim.HasAPI(UsdPhysics.RigidBodyAPI):
            PhysxSchema.PhysxContactReportAPI.Apply(prim).CreateThresholdAttr(0.)
    camera = Camera("/World/TaskCamera", frequency=-1, resolution=(640, 640))
    camera.set_focal_length(2.4)
    camera.set_horizontal_aperture(4.8)
    camera.set_vertical_aperture(4.8)
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
    panorama = Panorama([k[0,0],k[1,1],k[0,2],k[1,2]],backend=args.panorama_backend)
    report['panorama_backend']=args.panorama_backend
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
        worker = FullSysNavProcess(ROOT, out / "agent", stderr, task=args.task, instruction=instruction,
                                   api_proxy=args.api_proxy,api_budget=args.api_budget,
                                   enable_room_early_stop=args.enable_room_early_stop,
                                   vision_gpu=args.vision_gpu,sam_gpu=args.sam_gpu,vlm_gpu=args.vlm_gpu)
    if args.odometry == 'gicp':
        odometry = OdometryProcess(ROOT, out/'odometry', out,gicp_correspondence=args.gicp_correspondence)
        odom_trace = (out/'odometry/poses.jsonl').open('w')
        # Public initialization: robot starts level, at nominal standing height.
        # Neither the initial world XY nor world yaw is supplied to navigation.
        estimated_pose = np.eye(4)
        estimated_pose[2, 3] = .4
        previous_estimate, pose_time, estimated_velocity = None, None, np.zeros(3)
        odometry_errors, odometry_rot_errors, odometry_times = [], [], []
        report['sensor_model'] = 'rigid base-mounted instantaneous virtual lidar and RGB panorama; no noise or IMU fusion'
        report['odometry_initialization'] = 'relative origin, level orientation, nominal base height 0.4 m'
    else:
        report['sensor_model'] = 'rigid base-mounted virtual lidar and RGB panorama; ideal ground-truth base pose'
    trace = open(out / "trajectory.jsonl", "w")
    command = np.zeros(3)
    action = {}
    samples, distance, dwell = 0, 0., 0.
    last_pos = np.asarray(spawn)
    start = time.monotonic()
    from runtime_metrics import RuntimeMetrics
    metrics=RuntimeMetrics(out,float(world.current_time),args.gui)
    worker_latency_ms=None
    if args.gui:
        from omni.kit.viewport.utility import get_active_viewport
        viewport=get_active_viewport()
        if viewport: viewport.camera_path='/World/ReviewCamera'
    # Neither ray pattern is a calibrated model of a physical Go2/MID360 sensor.
    elevations = [-40,-25,-15,-7,0,7,15,25,40,55,70] if args.lidar_profile == 'legacy11' else np.linspace(-40,70,64)
    angles = [(a, e) for a in np.arange(360)*math.pi/180 for e in np.deg2rad(elevations)]
    # The beam pattern is fixed. Batch pose transforms; keep identical PhysX hit rules.
    local_directions=np.asarray([[math.cos(a)*math.cos(e),math.sin(a)*math.cos(e),math.sin(e)] for a,e in angles])
    cuda_lidar=None
    if args.lidar_backend=='cuda' or (args.lidar_backend=='auto' and not mp3d):
        try:
            from gpu_lidar import VerifiedCudaLidar
            cuda_lidar=VerifiedCudaLidar(stage,controller.robot,len(local_directions),device=f'cuda:{args.lidar_gpu}')
            warm_pos,warm_quat=controller.robot.get_world_pose();warm_rot=quat_to_rot_matrix(warm_quat)
            warm_start=time.perf_counter()
            cuda_lidar.scan(query,warm_pos+warm_rot@np.array([0.,0.,.45]),local_directions@warm_rot.T,bool(odometry))
            report['lidar_warmup_seconds']=time.perf_counter()-warm_start
            report['lidar_backend']=dict(cuda_lidar.info)
        except Exception as error:
            if args.lidar_backend=='cuda':raise
            report['lidar_backend']={'backend':'physx','fallback_reason':str(error)}
    else:report['lidar_backend']={'backend':'physx'}
    # Backend construction, JIT and same-state startup verification are setup.
    start=time.monotonic();metrics.start=time.perf_counter()
    metrics.previous=(metrics.start,float(world.current_time),0,0)
    step_limit=None if args.max_seconds<0 else int(args.max_seconds/.005)
    last_metrics=None
    t=float(world.current_time)
    pos=np.asarray(spawn)
    if args.profile:
        import cProfile
        profiler=cProfile.Profile();profiler.enable()
    for step in itertools.count():
        if step_limit is not None and step>=step_limit:
            report["status"] = "timeout"
            break
        if not app.is_running() or (out / "STOP").exists():
            report["status"] = "operator_stopped"
            break
        tick=time.perf_counter()
        controller.step(command)
        metrics.add('control',tick)
        tick=time.perf_counter()
        render_requested=args.gui and step % 20 == 0
        world.step(render=False)
        metrics.add('physics_step',tick)
        if render_requested:
            tick=time.perf_counter();world.render();metrics.add('gui_render',tick)
        if render_requested:
            metrics.renders += 1
            metrics.viewport_frames += 1
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
        obs = {"time": t, "position": pos.tolist(), "orientation": quat.tolist(), "velocity": (rot.T @ vel).tolist(), "yaw": yaw}
        if step % 40 == 0:
            tick=time.perf_counter()
            mount = np.array([0.,0.,.45])
            origin = pos + rot @ mount
            world_directions=local_directions @ rot.T
            ray_origin=tuple(origin.tolist())
            if cuda_lidar:
                all_ranges=cuda_lidar.scan(query,origin,world_directions,bool(odometry))
                indices=np.flatnonzero(all_ranges>0);ranges=all_ranges[indices,None]
                report['lidar_backend']=dict(cuda_lidar.info)
            else:
                hit_indices=[]; distances=[]
                for beam,direction in enumerate(world_directions.tolist()):
                    hit=query.raycast_closest(ray_origin,tuple(direction),10.)
                    if hit['hit'] and hit['distance']>.4:
                        if odometry and any(str(hit.get(k,'')).startswith('/World/Go2') for k in ('rigidBody','collision')):
                            continue
                        hit_indices.append(beam); distances.append(hit['distance'])
                indices=np.asarray(hit_indices,dtype=np.int64)
                ranges=np.asarray(distances)[:,None]
            xyz=origin+world_directions[indices]*ranges
            points=np.column_stack((xyz,np.zeros(len(xyz)))).tolist()
            raw_points=(mount+local_directions[indices]*ranges).tolist() if odometry else []
            metrics.add('lidar_raycast',tick)
            obs["points"] = points
            if odometry:
                # Save failed frames too; these files stay evaluator-private.
                tick=time.perf_counter()
                np.savez_compressed(out/f'raw_scan_{step:07d}.npz', points_base_xyz=np.asarray(raw_points,dtype=np.float32), time=t)
                metrics.add('raw_scan_write',tick)
                tick=time.perf_counter()
                result = odometry.ask(t, raw_points)
                metrics.add('gicp',tick)
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
        if step % 200 == 0:
            tick=time.perf_counter()
            faces = []
            for index, view_rot in enumerate(cube_rotations()):
                cp = pos + rot @ np.array([0.,0.,.45])
                cq = rot_matrix_to_quat(rot @ view_rot)
                camera.set_world_pose(cp, cq, camera_axes="world")
                capture_started=time.perf_counter()
                rgb,render_count=render_rgba(world,camera)
                metrics.add('cube_render_readback',capture_started)
                metrics.renders += render_count
                if render_count>4:
                    report.setdefault('camera_readiness_retries',[]).append(dict(time=t,face=index,renders=render_count))
                faces.append(rgb[:,:,:3].copy())
                if index == 0:
                    Image.fromarray(rgb[:,:,:3]).save(out / f"rgb_{step:07d}_0.jpg")
                samples += 1
            tick_stitch=time.perf_counter()
            pano = Image.fromarray(panorama.stitch(faces))
            metrics.add('panorama_stitch',tick_stitch)
            tick_encode=time.perf_counter()
            pano.save(out / f"panorama_{step:07d}.jpg")
            data = io.BytesIO(); pano.save(data, format='JPEG', quality=90)
            obs['panorama'] = base64.b64encode(data.getvalue()).decode()
            metrics.add('panorama_encode_write',tick_encode)
            tick_review=time.perf_counter()
            # Review only: never supplied to the navigation or VLM process.
            eye = pos + np.array([-2.0*math.cos(yaw),-2.0*math.sin(yaw),1.6])
            look_at=pos+np.array([0.,0.,.15])
            eye=unobstructed_eye(query,look_at,eye)
            look = Gf.Matrix4d().SetLookAt(Gf.Vec3d(*map(float,eye)), Gf.Vec3d(*map(float,look_at)), Gf.Vec3d(0,0,1)).GetInverse().ExtractRotationQuat()
            chase.set_world_pose(eye, np.array([look.GetReal(),*look.GetImaginary()]), camera_axes="usd")
            for _ in range(8):
                world.render()
                metrics.renders += 1
            review_rgb = chase.get_rgba()
            if review_rgb is not None and review_rgb.size:
                Image.fromarray(review_rgb[:,:,:3]).save(out / f"chase_{step:07d}.jpg")
            metrics.add('review_render_readback_encode',tick_review)
        if "points" in obs:
            # Preserve legacy evaluator scan format: true world-frame scan, never read by policy.
            tick=time.perf_counter()
            np.savez_compressed(out / f"scan_{step:07d}.npz", points=np.asarray(points,dtype=np.float32), time=t, position=pos, quaternion=quat)
            metrics.add('world_scan_write',tick)
        if worker and t > 2:
            ask_started=time.perf_counter()
            action = worker.ask(obs)
            worker_latency_ms=(time.perf_counter()-ask_started)*1000
            metrics.add('ros_bridge',ask_started)
            action["worker_latency_ms"]=worker_latency_ms
            command = np.asarray(action["cmd_vel"], dtype=float)
            if command.shape != (3,) or not np.isfinite(command).all():
                raise ValueError("Invalid pathFollower command")
            command = np.clip(command, [-.35,-.25,-.7], [.45,.25,.7])
        row = {"time": t, "position": pos.tolist(), "orientation": quat.tolist(), "speed": speed, "command": command.tolist(), "action": action,
               "control_mode": "terminal_stand" if controller.stop_start is not None else "velocity_policy",
               "joint_positions": controller.robot.get_joint_positions().tolist()}
        trace.write(json.dumps(row) + "\n")
        trace.flush()
        last_metrics=metrics.sample(t,step+1,worker_latency_ms)
        if step % 2000 == 0:
            print("OFFICE", round(t,1), [round(float(x),2) for x in pos], action.get("reason"), action.get("vlm_calls"), action.get("counts"), flush=True)
        if action.get("stop"):
            command[:] = 0
            controller.request_terminal_stand()
            max_stop_speed = object_definition['maximum_stop_speed_m_s'] if object_definition else .06
            minimum_dwell = object_definition['minimum_stop_seconds'] if object_definition else 1.
            dwell = dwell + .1 if speed < max_stop_speed else 0.
            if dwell >= minimum_dwell:
                if mp3d:
                    report.update(status='stopped_pending_evaluation',native_stop=True,stop_dwell=dwell)
                    break
                region = oracle["goal_rectangle"]
                regions = oracle.get("goal_rectangles", [region])
                floor_range = oracle.get("goal_z_range", [.05,.7])
                inside = (floor_range[0] <= pos[2] <= floor_range[1] and
                          any(r[0] <= pos[0] <= r[2] and r[1] <= pos[1] <= r[3] for r in regions))
                accepted = inside
                if object_definition:
                    predicted = action.get('sysnav',{}).get('object_xyz')
                    if predicted is not None and odometry:
                        predicted = (np.asarray(report['evaluation_world_from_odometry']) @ np.array([*predicted,1.]))[:3].tolist()
                    visible = visible_targets(query, pos + rot @ np.array([0.,0.,.45]), object_definition)
                    evaluation = evaluate_object_stop(pos, predicted, visible, object_definition)
                    report['independent_object_evaluation'] = evaluation
                    accepted = evaluation['passed']
                report.update(status="completed" if accepted else "false_positive_stop", success=bool(accepted),
                              stop_dwell=dwell, independent_goal_region=bool(inside))
                break
        else:
            dwell = 0.
        minimum_z=build.get('bounds_min',[0,0,0])[2]-.5 if mp3d else -.05
        if t > 2 and (pos[2] < minimum_z or rot[2,2] < .55):
            report["status"] = "fallen"
            break
    report.update(simulation_seconds=t, wall_seconds=time.monotonic()-start, distance_m=distance, final_position=pos.tolist(),
                  sensor_frames=samples, final_action=action, assets_complete=manifest["complete"], collision_meshes=build.get("collision_meshes",build.get('meshes')))
    if last_metrics is not None:
        report.update(realtime_factor=last_metrics['realtime_factor'],render_fps=last_metrics['render_fps'],
                     physics_hz=last_metrics['physics_hz'],vlm_api_latency_ms=last_metrics.get('vlm_api_last_ms'),
                     gpu_allocation=last_metrics.get('gpu_allocation',report.get('gpu_allocation')))
        report['runtime_metrics']=last_metrics
        report['measured_stage_wall_seconds']=last_metrics['stage_wall_seconds']
    (out / "contacts.json").write_text(json.dumps(contacts, indent=2))
    report.update(contact_quality(contacts))
    trace.close()
except (Exception, KeyboardInterrupt) as error:
    report.update(status="interrupted" if isinstance(error, KeyboardInterrupt) else "error", error=traceback.format_exc())
    if "t" in locals():
        report.update(simulation_seconds=t, distance_m=distance, final_position=pos.tolist(), final_action=action)
    traceback.print_exc()
finally:
    if 'profiler' in locals():
        profiler.disable();profiler.dump_stats(str(out/'loop.pstats'))
        import pstats
        with (out/'profile.txt').open('w') as stream:pstats.Stats(profiler,stream=stream).sort_stats('cumulative').print_stats(60)
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
    if 'metrics' in locals():
        if metrics.latest: report['runtime_metrics']=metrics.latest
        metrics.close()
    if worker:
        worker.close()
    if sam:
        sam.close()
    if "contacts" in locals():
        (out / "contacts.json").write_text(json.dumps(contacts, indent=2))
        report.update(contact_quality(contacts))
        contact_stream.close()
    report['acceptance_passed'] = bool(report['success'] and report.get('contact_quality_passed'))
    (out / "report.json").write_text(json.dumps(report, indent=2))
    print("OFFICE_REPORT", json.dumps(report), flush=True)
    app.close()
