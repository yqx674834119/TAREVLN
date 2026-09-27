"""Privileged asset smoke test; not a navigation experiment or a policy input."""
import argparse
import hashlib
import json
from pathlib import Path
import math
import numpy as np
from mp3d_episode import load_episode
from review_camera import unobstructed_eye
from contact_audit import accumulate_contact, contact_quality
ROOT=Path(__file__).resolve().parents[1];VLN=ROOT.parent
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--scene',default='x8F5xyUWy9e')
parser.add_argument('--episode-id',type=int,default=7)
args=parser.parse_args()
task=load_episode(VLN,args.scene,args.episode_id)
asset=VLN/'assets/mp3d-isaac'/args.scene
out=asset/'preflight'/f'episode-{args.episode_id}';out.mkdir(parents=True,exist_ok=True)
from isaacsim import SimulationApp
app=SimulationApp(dict(headless=True,open_usd=str(asset/'scene.usda'),create_new_stage=False,
    multi_gpu=False,active_gpu=0,physics_gpu=0,fast_shutdown=True))
try:
    from PIL import Image
    import omni.usd
    from omni.physx import get_physx_scene_query_interface, get_physx_simulation_interface
    from pxr import Gf, UsdPhysics, PhysxSchema, PhysicsSchemaTools
    from isaacsim.core.api import World
    from isaacsim.sensors.camera import Camera
    from sysnav_go2_policy import SysNavGo2Policy
    stage=omni.usd.get_context().get_stage()
    world=World(physics_dt=.005,rendering_dt=.1,stage_units_in_meters=1.,device='cpu')
    checkpoint=ROOT/'data/readiness/training/go2-control-v5-rails/model_3595.pt'
    robot=SysNavGo2Policy(stage,VLN/'assets/go2/go2_lab.usd',checkpoint,task['spawn'])
    world.scene.add(robot.robot)
    for prim in stage.Traverse():
        if str(prim.GetPath()).startswith('/World/Go2/') and prim.HasAPI(UsdPhysics.RigidBodyAPI):
            PhysxSchema.PhysxContactReportAPI.Apply(prim).CreateThresholdAttr(0.)
    camera=Camera('/World/PreflightCamera',frequency=-1,resolution=(960,640))
    camera.set_focal_length(2.4);camera.set_horizontal_aperture(4.8);camera.set_vertical_aperture(3.2)
    camera.set_clipping_range(.05,40.)
    world.reset();robot.initialize();camera.initialize()
    yaw=task['yaw'];robot.robot.set_world_pose(np.array(task['spawn']),np.array([math.cos(yaw/2),0,0,math.sin(yaw/2)]))
    contacts={}
    def on_contact(headers,data):
        for header in headers:
            a,b=(str(PhysicsSchemaTools.intToSdfPath(p)) for p in (header.collider0,header.collider1))
            body,env=(a,b) if a.startswith('/World/Go2/') else (b,a)
            if not body.startswith('/World/Go2/') or env.startswith('/World/Go2/'):continue
            points=[dict(impulse=list(d.impulse),separation=float(d.separation))
                    for d in data[header.contact_data_offset:header.contact_data_offset+header.num_contact_data]]
            accumulate_contact(contacts,body,env,float(world.current_time),points)
    subscription=get_physx_simulation_interface().subscribe_contact_report_events(on_contact)
    query=get_physx_scene_query_interface();support=[];positions=[]
    # Downward rays start below the base; excludes robot hits explicitly.
    for dx,dy in [(0,0),(.2,.15),(.2,-.15),(-.2,.15),(-.2,-.15)]:
        origin=(task['spawn'][0]+dx,task['spawn'][1]+dy,task['spawn'][2]-.1)
        hits=[]
        def on_hit(hit):
            if not str(hit.collision).startswith('/World/Go2'):
                hits.append(dict(distance=float(hit.distance),position=list(hit.position),collision=str(hit.collision)))
            return True
        query.raycast_all(origin,(0.,0.,-1.),1.,on_hit)
        support.append(min(hits,key=lambda h:h['distance']) if hits else None)
    for step in range(1000):
        robot.step(np.zeros(3));world.step(render=False)
        if step%20==0:positions.append(robot.robot.get_world_pose()[0].tolist())
    pos,quat=robot.robot.get_world_pose()
    for name,eye,at in [('sensor',pos+np.array([0,0,.45]),pos+np.array([3*math.cos(yaw),3*math.sin(yaw),.45])),
                        ('go2',pos+np.array([-2.0,-2.0,1.5]),pos)]:
        if name=='go2':eye=unobstructed_eye(query,at,eye)
        q=Gf.Matrix4d().SetLookAt(Gf.Vec3d(*map(float,eye)),Gf.Vec3d(*map(float,at)),Gf.Vec3d(0,0,1)).GetInverse().ExtractRotationQuat()
        camera.set_world_pose(eye,np.array([q.GetReal(),*q.GetImaginary()]),camera_axes='usd')
        for _ in range(15):world.render()
        rgb=camera.get_rgba()
        if rgb is None or not rgb.size:raise RuntimeError('No rendered frame')
        Image.fromarray(rgb[:,:,:3]).save(out/(name+'.jpg'))
    a=np.asarray(positions);height=pos[2]-support[0]['position'][2] if support[0] else float('nan')
    quality=contact_quality(contacts)
    up_dot=float(1-2*(quat[1]**2+quat[2]**2))
    drift=float(np.linalg.norm(pos[:2]-a[0,:2]))
    passed=bool(all(support) and .2<height<.6 and up_dot>math.cos(math.radians(10))
                and drift<.2 and quality['contact_quality_passed'])
    result=dict(version='go2-mp3d-spawn-v2',episode_id=args.episode_id,scene=args.scene,
        scene_sha256=hashlib.sha256((asset/'scene.usda').read_bytes()).hexdigest(),
        checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        initial_spawn=task['spawn'],initial_yaw=yaw,final_position=pos.tolist(),floor_rays=support,
        settled_base_height_m=float(height),horizontal_drift_m=float(np.linalg.norm(pos[:2]-a[0,:2])),
        minimum_base_z=float(a[:,2].min()),maximum_base_z=float(a[:,2].max()),
        final_up_dot=up_dot,maximum_tilt_degrees=10.,maximum_drift_m=.2,
        physics_seconds=5.,synthetic_floor=False,visual_review_required=True,
        physical_smoke_pass=passed,**quality)
    (out/'contacts.json').write_text(json.dumps(contacts,indent=2))
    (out/'report.json').write_text(json.dumps(result,indent=2));print('MP3D_PREFLIGHT',json.dumps(result),flush=True)
finally:app.close()
if not passed:raise SystemExit(2)
