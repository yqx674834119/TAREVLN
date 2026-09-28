"""Compare sequential and simultaneous RGB capture at identical camera poses."""
import argparse,json,time,sys,traceback
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'sim'))
ap=argparse.ArgumentParser();ap.add_argument('--name',required=True);args=ap.parse_args()
out=ROOT/'data/readiness/camera-rig'/args.name;out.mkdir(parents=True,exist_ok=False)
from isaacsim import SimulationApp
app=SimulationApp({'headless':True,'open_usd':str(ROOT/'data/sim/office/private/scene.usda'),'create_new_stage':False,'multi_gpu':False,'active_gpu':0,'anti_aliasing':0,'fast_shutdown':True})
try:
 from isaacsim.core.api import World
 from isaacsim.sensors.camera import Camera
 from isaacsim.core.utils.rotations import quat_to_rot_matrix,rot_matrix_to_quat
 from camera_rig import CubeCameraRig,set_render_enabled
 from camera_readiness import render_rgba
 from sysnav_panorama import cube_rotations,Panorama
 from PIL import Image
 world=World(physics_dt=.005,rendering_dt=.1,device='cpu')
 camera=Camera('/World/TaskCamera',frequency=-1,resolution=(640,640));camera.set_focal_length(2.4);camera.set_horizontal_aperture(4.8);camera.set_vertical_aperture(4.8);camera.set_clipping_range(.05,40)
 review=Camera('/World/ReviewCamera',frequency=-1,resolution=(640,480));review.set_focal_length(2.4);review.set_horizontal_aperture(4.8);review.set_vertical_aperture(3.6)
 rig=CubeCameraRig(camera,review);world.reset();camera.initialize();review.initialize();rig.initialize()
 k=camera.get_intrinsics_matrix();pano=Panorama([k[0,0],k[1,1],k[0,2],k[1,2]])
 result={'scope':'same-pose RGB capture comparison; not navigation','states':[]}
 for ix,(pos,yaw) in enumerate([([-11,29.5,.4],0),([-11,29.5,.4],1.1),([-17,30.8,.35],-1.2)]):
  pos=np.array(pos);c,s=np.cos(yaw),np.sin(yaw);rot=np.array([[c,-s,0],[s,c,0],[0,0,1.]])
  review.set_world_pose(pos+np.array([0,0,1.5]),rot_matrix_to_quat(rot),camera_axes='world')
  # Warm the seven actual render products once; exclude shader startup.
  rig.capture(world,pos,rot)
  set_render_enabled(camera,True);set_render_enabled(review,True)
  st=time.perf_counter();reference=[]
  for r in cube_rotations():
   camera.set_world_pose(pos+rot@np.array([0,0,.45]),rot_matrix_to_quat(rot@r),camera_axes='world')
   rgb,count=render_rgba(world,camera);reference.append(rgb[:,:,:3].copy())
  for _ in range(8):world.render()
  old_ms=1000*(time.perf_counter()-st);rig.set_enabled(False)
  st=time.perf_counter();actual,rev,count=rig.capture(world,pos,rot);new_ms=1000*(time.perf_counter()-st)
  face_errors=[]
  for j,(a,b) in enumerate(zip(reference,actual)):
   diff=np.abs(a.astype(float)-b.astype(float));mae=float(diff.mean());mse=float((diff**2).mean())
   record={'face':j,'mae_255':mae,'psnr_db':float(10*np.log10(255**2/max(mse,1e-8))),'reference_std':float(a.std()),'actual_std':float(b.std()),'passed':bool(mae<8 and a.std()>1 and b.std()>1)}
   face_errors.append(record)
  Image.fromarray(pano.stitch(reference)).save(out/f'{ix}-reference.jpg');Image.fromarray(pano.stitch(actual)).save(out/f'{ix}-batch.jpg')
  rec={'position':pos.tolist(),'yaw':yaw,'sequential_ms':old_ms,'batch_ms':new_ms,'batch_renders':count,'faces':face_errors,'passed':all(e['passed'] for e in face_errors)}
  result['states'].append(rec);print('CAMERA',json.dumps(rec),flush=True)
  (out/'report.json').write_text(json.dumps(result,indent=2))
 result['passed']=all(r['passed'] for r in result['states']);(out/'report.json').write_text(json.dumps(result,indent=2))
except Exception:traceback.print_exc()
finally:app.close()
