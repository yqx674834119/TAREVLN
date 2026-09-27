"""Privileged reachability control, NEVER an autonomous navigation result.

Uses the independent full-mesh audit and goal fixture to construct a route, then
executes it with the same articulated Go2/checkpoint/physics as the failed run.
No SysNav process is started and no oracle information enters a policy sandbox.
"""
import argparse
import hashlib
import heapq
import json
import math
from pathlib import Path
import time
import traceback

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
VLN = ROOT.parent
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--out', type=Path, required=True)
parser.add_argument('--radius', type=float, default=.40)
parser.add_argument('--max-seconds', type=float, default=160.)
args = parser.parse_args()
args.out.mkdir(parents=True, exist_ok=False)
definition = json.loads((ROOT/'sim/fixtures/sysnav-object-evaluation.json').read_text())['tasks']['meeting_crossroom']
grid_file = ROOT/'data/readiness/office-geometry-r2/private-occupancy.npz'
grid = np.load(grid_file)
clearance = grid['clearance']
resolution = .05
origin = np.array([-25., 10.])
free = clearance >= args.radius + resolution / math.sqrt(2)


def ij(p):
    return tuple(np.rint((np.asarray(p[:2])-origin)[::-1]/resolution).astype(int))


def xy(index):
    return np.array(index[::-1])*resolution+origin


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


scene = ROOT/'data/sim/office/private/scene.usda'
checkpoint = ROOT/'data/readiness/training/go2-control-v5-rails/model_3595.pt'
report = dict(scope='evaluator-only oracle route control; NOT SysNav or autonomous success',
              status='starting', physics_route_passed=False, oracle_grounding=True,
              radius_m=args.radius, grid_resolution_m=resolution,
              scene_sha256=digest(scene), checkpoint_sha256=digest(checkpoint),
              grid_sha256=digest(grid_file), source_sha256=digest(Path(__file__)),
              task='meeting_crossroom', start=definition['start_xyz'])
from isaacsim import SimulationApp
app = SimulationApp(dict(headless=True, open_usd=str(scene), create_new_stage=False,
                         multi_gpu=False, active_gpu=0, physics_gpu=0, anti_aliasing=0,
                         width=640, height=480, fast_shutdown=True))
try:
    import omni.usd
    from omni.physx import get_physx_scene_query_interface, get_physx_simulation_interface
    from pxr import Gf, UsdPhysics, PhysxSchema, PhysicsSchemaTools
    from PIL import Image
    from isaacsim.core.api import World
    from isaacsim.core.utils.rotations import quat_to_rot_matrix
    from isaacsim.sensors.camera import Camera
    from sysnav_go2_policy import SysNavGo2Policy
    from sysnav_evaluation import visible_targets, evaluate_object_stop, box_distance
    from contact_audit import accumulate_contact, contact_quality
    from camera_readiness import render_rgba
    from review_camera import unobstructed_eye

    stage = omni.usd.get_context().get_stage()
    world = World(physics_dt=.005, rendering_dt=.1, stage_units_in_meters=1., device='cpu')
    robot = SysNavGo2Policy(stage, VLN/'assets/go2/go2_lab.usd', checkpoint, definition['start_xyz'])
    world.scene.add(robot.robot)
    for prim in stage.Traverse():
        if str(prim.GetPath()).startswith('/World/Go2/') and prim.HasAPI(UsdPhysics.RigidBodyAPI):
            PhysxSchema.PhysxContactReportAPI.Apply(prim).CreateThresholdAttr(0.)
    camera = Camera('/World/DiagnosticCamera', frequency=-1, resolution=(960,640))
    camera.set_focal_length(2.4)
    camera.set_horizontal_aperture(4.8)
    camera.set_vertical_aperture(3.2)
    camera.set_clipping_range(.05,40.)
    world.reset()
    robot.initialize()
    robot.robot.set_world_pose(np.asarray(definition['start_xyz']),np.array([1.,0.,0.,0.]))
    camera.initialize()
    query = get_physx_scene_query_interface()
    contacts = {}

    def on_contact(headers, data):
        for header in headers:
            a,b = (str(PhysicsSchemaTools.intToSdfPath(p)) for p in (header.collider0, header.collider1))
            body,env = (a,b) if a.startswith('/World/Go2/') else (b,a)
            if not body.startswith('/World/Go2/') or env.startswith('/World/Go2/'):
                continue
            points = [dict(impulse=list(d.impulse), separation=float(d.separation))
                      for d in data[header.contact_data_offset:header.contact_data_offset+header.num_contact_data]]
            accumulate_contact(contacts, body, env, float(world.current_time), points)

    subscription = get_physx_simulation_interface().subscribe_contact_report_events(on_contact)
    # The goal set enforces room, distance AND first-hit visibility to the TV.
    goal_masks = []
    candidates_report = []
    yy,xx = np.indices(free.shape)
    xs,ys = xx*resolution+origin[0], yy*resolution+origin[1]
    for target in definition['targets']:
        r = target['room_rectangle']
        d = np.hypot(np.maximum.reduce([target['min'][0]-xs, xs-target['max'][0], np.zeros_like(xs)]),
                     np.maximum.reduce([target['min'][1]-ys, ys-target['max'][1], np.zeros_like(ys)]))
        mask = free & (xs>=r[0]) & (xs<=r[2]) & (ys>=r[1]) & (ys<=r[3]) & (d>=1.2) & (d<=1.75)
        visible_mask = np.zeros_like(mask)
        for j,i in np.argwhere(mask):
            p = [xs[j,i], ys[j,i], .78]
            if visible_targets(query, p, dict(targets=[target])):
                visible_mask[j,i] = True
        goal_masks.append(visible_mask)
        candidates_report.append(dict(prim=target['prim'], geometric_stop_cells=int(mask.sum()),
                                      visible_stop_cells=int(visible_mask.sum())))
    start = ij(definition['start_xyz'])
    queue = [(0.,start)] if free[start] else []
    costs = {start:0.}
    parents = {}
    paths = {}
    while queue:
        distance,current = heapq.heappop(queue)
        if distance != costs[current]:
            continue
        for k,mask in enumerate(goal_masks):
            if k not in paths and mask[current]:
                indices = [current]
                while indices[-1] in parents:
                    indices.append(parents[indices[-1]])
                indices.reverse()
                paths[k] = dict(length_m=distance, route=[xy(v).tolist() for v in indices],
                                minimum_center_clearance_m=float(min(clearance[v] for v in indices)))
        if len(paths) == len(goal_masks):
            break
        for dy,dx in ((0,1),(0,-1),(1,0),(-1,0),(1,1),(1,-1),(-1,1),(-1,-1)):
            n = current[0]+dy,current[1]+dx
            if not (0<=n[0]<free.shape[0] and 0<=n[1]<free.shape[1]) or not free[n]:
                continue
            if dx and dy and (not free[current[0]+dy,current[1]] or not free[current[0],current[1]+dx]):
                continue
            nd = distance+math.hypot(dx,dy)*resolution
            if nd < costs.get(n,math.inf):
                costs[n] = nd
                parents[n] = current
                heapq.heappush(queue,(nd,n))
    report['goals'] = [dict(**r, reachable=k in paths,
                            **{a:b for a,b in paths.get(k,{}).items() if a!='route'})
                       for k,r in enumerate(candidates_report)]
    (args.out/'private-routes.json').write_text(json.dumps(paths,indent=2))
    if not paths:
        raise RuntimeError('No visible stop pose reachable at the requested clearance')
    chosen = min(paths, key=lambda k:paths[k]['length_m'])
    target = definition['targets'][chosen]
    route = np.array(paths[chosen]['route'])
    center = (np.array(target['min'])+target['max'])/2
    report.update(chosen_target=target['prim'], planned_endpoint=route[-1].tolist())
    (args.out/'report.json').write_text(json.dumps(report,indent=2))
    print('ROUTE_READY',json.dumps(report),flush=True)
    # Verify support along the center route using actual environment colliders.
    supports = []
    for point in route[::4]:
        hits = []
        def on_hit(hit):
            if not str(hit.collision).startswith('/World/Go2/'):
                hits.append(float(hit.position[2]))
            return True
        query.raycast_all((float(point[0]),float(point[1]),.7),(0.,0.,-1.),1.,on_hit)
        supports.append(max(hits) if hits else None)
    report['route_floor_support'] = dict(samples=len(supports), missing=sum(x is None for x in supports),
                                        minimum=min(x for x in supports if x is not None),
                                        maximum=max(x for x in supports if x is not None))
    if any(x is None or abs(x)>.08 for x in supports):
        raise RuntimeError('Route has a missing/raised floor support; flat-route control invalid')

    def capture(name, eye, at):
        q = Gf.Matrix4d().SetLookAt(Gf.Vec3d(*map(float,eye)),Gf.Vec3d(*map(float,at)),Gf.Vec3d(0,0,1)).GetInverse().ExtractRotationQuat()
        camera.set_world_pose(np.asarray(eye),np.array([q.GetReal(),*q.GetImaginary()]),camera_axes='usd')
        rgb,_ = render_rgba(world,camera,minimum_renders=12)
        Image.fromarray(rgb[:,:,:3]).save(args.out/(name+'.jpg'))

    progress = 0
    command = np.zeros(3)
    trace = []
    dwell = 0.
    started = time.monotonic()
    for step in range(int(args.max_seconds/.005)):
        robot.step(command)
        world.step(render=False)
        if step%20:
            continue
        t = float(world.current_time)
        pos,quat = robot.robot.get_world_pose()
        rotation = quat_to_rot_matrix(quat)
        speed = float(np.linalg.norm(robot.robot.get_linear_velocity()))
        yaw = math.atan2(rotation[1,0],rotation[0,0])
        progress += int(np.argmin(np.linalg.norm(route[progress:progress+30]-pos[:2],axis=1)))
        remaining = float(np.linalg.norm(route[-1]-pos[:2]))
        if t < 2.:
            command[:] = 0.
        elif robot.stop_start is not None:
            command[:] = 0.
            dwell = dwell+.1 if speed<definition['maximum_stop_speed_m_s'] else 0.
        elif remaining < .12:
            desired = math.atan2(center[1]-pos[1],center[0]-pos[0])
            err = math.atan2(math.sin(desired-yaw),math.cos(desired-yaw))
            command[:] = [0.,0.,np.clip(1.2*err,-.4,.4)]
            if abs(err)<.10:
                robot.request_terminal_stand()
        else:
            aim = route[min(progress+7,len(route)-1)]-pos[:2]
            desired = math.atan2(aim[1],aim[0])
            err = math.atan2(math.sin(desired-yaw),math.cos(desired-yaw))
            v = min(.30, .8*remaining)*max(0.,math.cos(err))
            command[:] = [v,0.,np.clip(1.4*err,-.6,.6)]
        visible = visible_targets(query, pos+rotation@np.array([0.,0.,.45]),dict(targets=[target]))
        evaluation = evaluate_object_stop(pos,center.tolist(),visible,dict(definition,targets=[target]))
        trace.append(dict(time=t,position=pos.tolist(),yaw=yaw,speed=speed,command=command.tolist(),
                          progress=progress,endpoint_distance_m=remaining,up_dot=float(rotation[2,2]),
                          visible=bool(visible),goal_geometry_passed=evaluation['passed']))
        if step%2000==0:
            print('ORACLE_CONTROL',round(t,1),pos.tolist(),progress,len(route),flush=True)
        if step%4000==0:
            eye = unobstructed_eye(query,pos+np.array([0,0,.15]),pos+np.array([-2,-2,1.7]))
            capture('review-'+str(step),eye,pos)
        if dwell>=definition['minimum_stop_seconds']:
            report.update(status='completed',stop_dwell=dwell,goal_evaluation=evaluation,
                          physics_route_passed=bool(evaluation['passed'] and contact_quality(contacts)['contact_quality_passed']))
            break
        if t>2 and (pos[2]<.1 or rotation[2,2]<.8):
            report['status']='fallen_or_excessive_tilt'
            break
    else:
        report['status']='timeout'
    report.update(simulation_seconds=t,wall_seconds=time.monotonic()-started,final_position=pos.tolist(),
                  distance_m=float(sum(np.linalg.norm(np.array(a['position'])[:2]-np.array(b['position'])[:2]) for a,b in zip(trace,trace[1:]))),
                  minimum_up_dot=min(x['up_dot'] for x in trace),**contact_quality(contacts))
    (args.out/'trajectory.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in trace))
    capture('goal-view',pos+rotation@np.array([0.,0.,.45]),center)
    eye=unobstructed_eye(query,pos+np.array([0,0,.15]),pos+np.array([-2,-2,1.7]))
    capture('goal-go2',eye,pos+np.array([0,0,.15]))
except Exception:
    report.update(status='error',error=traceback.format_exc())
    traceback.print_exc()
finally:
    if 'contacts' in locals():
        report.update(contact_quality(contacts))
        (args.out/'contacts.json').write_text(json.dumps(contacts,indent=2))
    try:
        serialized=json.dumps(report,indent=2,default=lambda value:value.item() if isinstance(value,np.generic) else value)
        (args.out/'report.json').write_text(serialized)
        print('REACHABILITY_REPORT',serialized,flush=True)
    finally:
        app.close()
