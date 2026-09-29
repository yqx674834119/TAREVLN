"""Compare CUDA and PhysX on exactly the same settled/replayed sensor states."""
import argparse,json,time,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'sim'))
ap=argparse.ArgumentParser();ap.add_argument('--name',required=True);ap.add_argument('--states',type=int,default=64);ap.add_argument('--fabric',action='store_true');args=ap.parse_args()
out=ROOT/'data/readiness/gpu-lidar'/args.name;out.mkdir(parents=True,exist_ok=False)
from isaacsim import SimulationApp
app=SimulationApp({'headless':True,'open_usd':str(ROOT/'data/sim/office/private/scene.usda'),'create_new_stage':False,'multi_gpu':False,'active_gpu':0,'fast_shutdown':False})
try:
 import omni.usd
 from isaacsim.core.api import World
 from isaacsim.core.utils.rotations import quat_to_rot_matrix
 from omni.physx import get_physx_scene_query_interface
 from sysnav_go2_policy import SysNavGo2Policy
 from gpu_lidar import CudaLidar,cpu_ranges,compare_ranges
 world=World(physics_dt=.005,rendering_dt=.1,device='cpu')
 if args.fabric:world.get_physics_context().enable_fabric(True)
 robot=SysNavGo2Policy(omni.usd.get_context().get_stage(),ROOT.parent/'assets/go2/go2_lab.usd',ROOT/'data/readiness/training/go2-control-v5-rails/model_3595.pt',[-11,29.5,.4]);world.scene.add(robot.robot)
 world.reset();robot.initialize();query=get_physx_scene_query_interface()
 rays=np.array([[np.cos(a)*np.cos(e),np.sin(a)*np.cos(e),np.sin(e)] for a in np.deg2rad(np.arange(360)) for e in np.deg2rad(np.linspace(-40,70,64))])
 lidar=CudaLidar(omni.usd.get_context().get_stage(),robot.robot,len(rays))
 rows=[json.loads(s) for s in (ROOT/'data/sim/office/runs/ui-20260928-160914-12420/trajectory.jsonl').read_text().splitlines()]
 result={'build':lidar.info,'states':[],'scope':'same-state sensor equivalence; no navigation evaluation'}
 for ix in np.linspace(0,len(rows)-1,args.states,dtype=int):
  row=rows[ix];robot.robot.set_world_pose(np.array(row['position']),np.array(row['orientation']));robot.robot.set_joint_positions(np.array(row['joint_positions']));world.step(render=False)
  pos,q=robot.robot.get_world_pose();rot=quat_to_rot_matrix(q);origin=pos+rot@np.array([0,0,.45]);directions=rays@rot.T
  st=time.perf_counter();cpu=cpu_ranges(query,origin,directions);cpu_ms=1000*(time.perf_counter()-st)
  gpu=lidar.scan(query,origin,directions)
  bad=np.flatnonzero(((cpu>0)!=(gpu>0))|((cpu>0)&(gpu>0)&(np.abs(cpu-gpu)>.002)))
  diagnostics=lidar.diagnostic.numpy();faces=lidar.faces.numpy()
  for i in bad:
   hit=query.raycast_closest(tuple(map(float,origin)),tuple(map(float,directions[i])),10.)
   fi=faces[i];mi=np.searchsorted(lidar.face_ends,fi,side='right')
   print('MISMATCH',json.dumps({'ray':int(i),'cpu':float(cpu[i]),'gpu':float(gpu[i]),'diagnostic':diagnostics[i].tolist(),'face':int(fi),'gpu_mesh':lidar.mesh_paths[mi] if fi>=0 else None,'cpu_hit':{k:str(v) for k,v in hit.items()}}),flush=True)
  record={'source_time':row['time'],'cpu_ms':cpu_ms,**lidar.last,**compare_ranges(cpu,gpu)}
  result['states'].append(record);np.savez_compressed(out/f'state-{ix}.npz',reference=cpu,actual=gpu,origin=origin,directions=directions)
  print('VALIDATE',json.dumps(record),flush=True)
  (out/'report.json').write_text(json.dumps(result,indent=2))
 result['passed']=all(x['passed'] for x in result['states']);(out/'report.json').write_text(json.dumps(result,indent=2))
except Exception:
 import traceback;traceback.print_exc()
 result={'passed':False,'error':traceback.format_exc()};(out/'report.json').write_text(json.dumps(result,indent=2))
finally:app.close()

raise SystemExit(0 if result.get('passed') else 1)
