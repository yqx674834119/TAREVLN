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
parser.add_argument("--physics", choices=("baseline", "isaaclab", "exact_lab"), default="baseline")
parser.add_argument("--snapshot-only", action="store_true")
parser.add_argument("--disable-sleep", action="store_true")
parser.add_argument("--controller", choices=('legacy','lab'), default='legacy')
parser.add_argument("--ground", choices=('box','plane','plane_clean'), default='box')
parser.add_argument("--ground-torsion-radius", type=float, help='Controlled override of ground/threshold torsional friction patch radii')
parser.add_argument('--ground-contact-offset',type=float,help='Controlled ground/threshold contact generation distance; rest geometry unchanged')
parser.add_argument("--yaw", type=float, default=0.)
parser.add_argument("--threshold", type=float, default=0.)
parser.add_argument("--checkpoint", type=Path)
parser.add_argument("--video", action="store_true")
args = parser.parse_args()
out = Path(args.out)
out.mkdir(parents=True, exist_ok=False)
ROOT = Path(__file__).resolve().parents[1]
VLN = ROOT.parent
source_files=[Path(__file__),ROOT/'config/readiness/gait-gates.json',*[ROOT/'sim'/name for name in ('go2_policy.py','go2_lab_policy.py','go2_runtime_audit.py')]]
source_hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in source_files}
gates=json.loads((ROOT/'config/readiness/gait-gates.json').read_text())
for file in source_files:
    shutil.copyfile(file, out / file.name)

from isaacsim import SimulationApp
app = SimulationApp({"headless": True, "multi_gpu": False, "active_gpu": 0, "physics_gpu": 0,
                     "width": 640, "height": 480, "fast_shutdown": True})
try:
    import numpy as np
    import omni.usd
    from omni.physx import get_physx_simulation_interface
    from isaacsim.core.api import World
    from isaacsim.core.api.objects import FixedCuboid, GroundPlane
    from isaacsim.core.api.materials import PhysicsMaterial
    from isaacsim.core.utils.rotations import quat_to_rot_matrix
    from pxr import Gf, UsdLux, UsdPhysics, PhysxSchema, PhysicsSchemaTools, UsdShade
    from go2_policy import Go2Policy

    dt = .005
    world = World(physics_dt=dt, rendering_dt=.1, stage_units_in_meters=1., device="cpu",
                  backend='torch' if args.controller=='lab' else 'numpy')
    stage = omni.usd.get_context().get_stage()
    material = PhysicsMaterial("/World/GroundMaterial", static_friction=1. if args.physics=='exact_lab' else .8,
                               dynamic_friction=1. if args.physics=='exact_lab' else .6, restitution=0.)
    if args.physics=='exact_lab':
        PhysxSchema.PhysxMaterialAPI.Apply(stage.GetPrimAtPath('/World/GroundMaterial')).CreateFrictionCombineModeAttr('multiply')
    if args.ground in ('plane', 'plane_clean'):
        world.scene.add(GroundPlane('/World/Ground',z_position=0.,physics_material=material))
        if args.ground == 'plane_clean':
            # Isaac 5.1 GroundPlane additionally enables collision on /geom.
            # Keep only the analytic plane, explicitly binding its material.
            for prim in stage.Traverse():
                if str(prim.GetPath()).startswith('/World/Ground/') and prim.HasAPI(UsdPhysics.CollisionAPI):
                    UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Set(prim.GetTypeName() == 'Plane')
                    if prim.GetTypeName() == 'Plane':
                        UsdShade.MaterialBindingAPI.Apply(prim).Bind(material.material, materialPurpose='physics')
    else:
        world.scene.add(FixedCuboid("/World/Ground", name='ground', position=np.array([0., 0., -.05]),
                                   scale=np.array([100., 100., .1]), physics_material=material))
    if args.threshold:
        world.scene.add(FixedCuboid("/World/Threshold", name='threshold', position=np.array([math.cos(args.yaw), math.sin(args.yaw), args.threshold / 2]),
                                   orientation=np.array([math.cos(args.yaw/2),0,0,math.sin(args.yaw/2)]),
                                   scale=np.array([.1, 20., args.threshold]), physics_material=material))
    if args.ground_torsion_radius is not None:
        if args.ground_torsion_radius < 0: raise ValueError('Negative torsional radius')
        for prim in stage.Traverse():
            if str(prim.GetPath()).startswith(('/World/Ground', '/World/Threshold')) and prim.HasAPI(UsdPhysics.CollisionAPI):
                collision = PhysxSchema.PhysxCollisionAPI.Apply(prim)
                collision.CreateTorsionalPatchRadiusAttr(args.ground_torsion_radius)
                collision.CreateMinTorsionalPatchRadiusAttr(args.ground_torsion_radius)
    if args.ground_contact_offset is not None:
        if not 0<args.ground_contact_offset<=.1:raise ValueError('Ground contact offset outside diagnostic range')
        for prim in stage.Traverse():
            if str(prim.GetPath()).startswith(('/World/Ground','/World/Threshold')) and prim.HasAPI(UsdPhysics.CollisionAPI):
                PhysxSchema.PhysxCollisionAPI.Apply(prim).CreateContactOffsetAttr(args.ground_contact_offset)
    camera = None
    if args.video:
        from isaacsim.sensors.camera import Camera
        from PIL import Image
        UsdLux.DomeLight.Define(stage,"/World/ReviewLight").CreateIntensityAttr(1500.)
        camera = Camera("/World/ReviewCamera",frequency=-1,resolution=(640,480))
        camera.set_focal_length(2.4)
        camera.set_horizontal_aperture(4.8)
        (out/'frames').mkdir()
    checkpoint = args.checkpoint or VLN / "assets/go2/checkpoint.pt"
    if args.controller=='lab':
        from go2_lab_policy import Go2LabPolicy
        policy=Go2LabPolicy(stage, VLN / 'assets/go2/go2_lab.usd', checkpoint,[0.,0.,.4])
    else:
        policy = Go2Policy(stage, VLN / "assets/go2/go2_lab.usd", checkpoint, [0., 0., .4])
        world.scene.add(policy.robot)
    for prim in stage.Traverse():
        if not str(prim.GetPath()).startswith("/World/Go2/"):
            continue
        if args.disable_sleep and prim.HasAPI(UsdPhysics.ArticulationRootAPI):
            PhysxSchema.PhysxArticulationAPI.Apply(prim).CreateSleepThresholdAttr(0.)
        if prim.HasAPI(UsdPhysics.RigidBodyAPI):
            PhysxSchema.PhysxContactReportAPI.Apply(prim).CreateThresholdAttr(0.)
            if args.physics in ("isaaclab", "exact_lab"):
                rigid = PhysxSchema.PhysxRigidBodyAPI.Apply(prim)
                rigid.CreateLinearDampingAttr(0.)
                rigid.CreateAngularDampingAttr(0.)
                rigid.CreateMaxLinearVelocityAttr(1000.)
                rigid.CreateMaxAngularVelocityAttr(math.degrees(1000.))
                rigid.CreateMaxDepenetrationVelocityAttr(1.)
        if args.physics in ("isaaclab", "exact_lab"):
            if prim.HasAPI(UsdPhysics.ArticulationRootAPI):
                PhysxSchema.PhysxArticulationAPI.Apply(prim).CreateEnabledSelfCollisionsAttr(False)
            if prim.IsA(UsdPhysics.RevoluteJoint):
                PhysxSchema.PhysxJointAPI.Apply(prim).CreateMaxJointVelocityAttr(math.degrees(30.))
            if prim.HasAPI(UsdPhysics.CollisionAPI):
                UsdShade.MaterialBindingAPI.Apply(prim).Bind(material.material, materialPurpose="physics")
    world.reset()
    ground_audit = []
    for prim in stage.Traverse():
        if str(prim.GetPath()).startswith('/World/Ground') and prim.HasAPI(UsdPhysics.CollisionAPI):
            bound, _ = UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial(materialPurpose='physics')
            ground_audit.append({'path':str(prim.GetPath()), 'type':prim.GetTypeName(),
                'enabled':UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Get(),
                'physics_material':str(bound.GetPath()) if bound else None,
                'contact_offset':PhysxSchema.PhysxCollisionAPI(prim).GetContactOffsetAttr().Get(),
                'rest_offset':PhysxSchema.PhysxCollisionAPI(prim).GetRestOffsetAttr().Get(),
                'torsional_patch_radius':PhysxSchema.PhysxCollisionAPI(prim).GetTorsionalPatchRadiusAttr().Get(),
                'minimum_torsional_patch_radius':PhysxSchema.PhysxCollisionAPI(prim).GetMinTorsionalPatchRadiusAttr().Get()})
    (out/'ground-colliders.json').write_text(json.dumps(ground_audit, indent=2))
    policy.initialize()
    if args.physics in ("isaaclab", "exact_lab"):
        policy.robot.set_solver_position_iteration_count(4)
        policy.robot.set_solver_velocity_iteration_count(0)
    if args.physics=='exact_lab':
        # USD Traverse omits instance proxies. Set the physical shape materials directly.
        pv=policy.robot._articulation_view._physics_view
        mats=pv.get_material_properties().copy()
        mats[:]=np.array([.8,.6,0.],dtype=np.float32)
        pv.set_material_properties(mats,np.array([0],dtype=np.uint32))
    from go2_runtime_audit import physics_snapshot
    (out/'physics.json').write_text(json.dumps(physics_snapshot(policy.robot._articulation_view._physics_view),indent=2))
    if args.snapshot_only:
        raise SystemExit(0)
    policy.robot.set_world_pose(np.array([0., 0., .4]), np.array([math.cos(args.yaw/2), 0., 0., math.sin(args.yaw/2)]))
    if camera:
        camera.initialize()
    frame_index = 0
    foot_indices = [i for i,n in enumerate(policy.robot._articulation_view.body_names) if n.endswith('_foot')]
    if len(foot_indices) != 4:
        raise RuntimeError('Expected four foot links for crossing witness')
    maximum_rear_foot_progress = -math.inf
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
            record = {"t": float(world.current_time), "phase": current_phase, "colliders": paths, "points": details,
                      "event_type":str(h.type), "data_offset":h.contact_data_offset,
                      "data_count":h.num_contact_data, "callback_data_length":len(data)}
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
                ("forward", 8. if args.threshold else 4., [.3, 0, 0]), ("stop_1", 2., [0, 0, 0]),
                ("backward", 4., [-.2, 0, 0]), ("left", 4., [0, .2, 0]),
                ("right", 4., [0, -.2, 0]), ("yaw", 4., [0, 0, .5]),
                ("yaw_reverse", 4., [0, 0, -.5]), ("stop_2", 3., [0, 0, 0])]
    metrics = {}
    for name, duration, command in schedule:
        current_phase = name
        samples = []
        for i in range(round(duration/dt)):
            policy.step(np.array(command))
            world.step(render=False)
            if i % 20 == 0:
                if camera:
                    p,_ = policy.robot.get_world_pose()
                    eye = p + np.array([-2*math.cos(args.yaw)-math.sin(args.yaw),-2*math.sin(args.yaw)+math.cos(args.yaw),1.0])
                    q = Gf.Matrix4d().SetLookAt(Gf.Vec3d(*map(float,eye)),Gf.Vec3d(*map(float,p+np.array([0,0,.05]))),Gf.Vec3d(0,0,1)).GetInverse().ExtractRotationQuat()
                    camera.set_world_pose(eye,np.array([q.GetReal(),*q.GetImaginary()]),camera_axes='usd')
                world.render()
                if camera:
                    rgb=camera.get_rgba()
                    if rgb is not None and rgb.size:
                        Image.fromarray(rgb[:,:,:3]).save(out/'frames'/f'{frame_index:06d}.jpg')
                        frame_index += 1
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
                feet = policy.robot._articulation_view._physics_view.get_link_transforms()[0,foot_indices,:3]
                if hasattr(feet,'numpy'):feet=feet.numpy()
                state['feet_xyz'] = feet.tolist()
                if name == 'forward':
                    maximum_rear_foot_progress = max(maximum_rear_foot_progress,float(np.min(feet[:,:2]@np.array([math.cos(args.yaw),math.sin(args.yaw)]))))
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
    loaded_feet={p.split('/collisions/')[0].split('/')[-1] for pair,item in pairs.items()
                 if not item['nonfoot'] and item['support_steps']>0
                 for p in pair.split(' | ') if '_foot/' in p}
    report = {"physics": args.physics, "ground":args.ground,"ground_torsion_radius":args.ground_torsion_radius,"ground_contact_offset":args.ground_contact_offset,"controller":args.controller, "disable_sleep":args.disable_sleep, "yaw": args.yaw, "threshold": args.threshold,
              "checkpoint": str(checkpoint), "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
              "joint_names": policy.robot.dof_names, "phases": metrics, "contact_pairs": pairs,
              "nonfoot_pairs_after_settling": nonfoot,
              "contact_definition": "impulse > 1e-5 N s or separation <= 0.0005 m after settling; all proximity events retained",
              "loaded_feet":sorted(loaded_feet),
              "contact_passed": len(loaded_feet)==4 and not nonfoot,
              "source_hashes":source_hashes}
    report['gates']=gates
    report['tracking_passed'] = all(max(v['rms_velocity_error'][:2]) <= gates['maximum_steady_translation_rms_error_mps']
                                   and v['rms_velocity_error'][2] <= gates['maximum_steady_yaw_rms_error_radps']
                                   for k,v in metrics.items() if k != 'settle')
    report['upright_passed'] = all(v['minimum_upright'] >= gates['minimum_upright_cosine'] for k,v in metrics.items() if k != 'settle')
    report['maximum_rear_foot_progress_m'] = maximum_rear_foot_progress
    report['threshold_crossing_witnessed'] = bool(args.threshold and maximum_rear_foot_progress > 1.075)
    report['passed'] = report['tracking_passed'] and report['upright_passed'] and report['contact_passed'] and (not args.threshold or report['threshold_crossing_witnessed'])
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
