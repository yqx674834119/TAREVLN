"""Isolated odometry replay. Ground truth is used only by exporter/evaluator.

Legacy recordings store ideal world-registered virtual scans. The exporter
undoes that registration to construct base-frame measurements. This is an
ideal-sensor replay, not a physical sensor recording or a navigation run.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import shutil
import sys
import numpy as np

ap=argparse.ArgumentParser()
ap.add_argument('--run',type=Path,required=True)
ap.add_argument('--out',type=Path,required=True)
ap.add_argument('--voxel-size',type=float,default=.1)
ap.add_argument('--initial-threshold',type=float,default=.5)
ap.add_argument('--fixed-threshold',type=float)
ap.add_argument('--method',choices=('kiss','gicp'),default='kiss')
ap.add_argument('--prediction',choices=('constant_velocity','static'),default='constant_velocity')
ap.add_argument('--raw-scans',action='store_true',help='Use recorded rigid-mounted base-frame rays, without undoing truth registration')
args=ap.parse_args()
root=Path(__file__).resolve().parents[1]
vln=root.parent
args.run=args.run.resolve();args.out=args.out.resolve()
args.out.mkdir(parents=True,exist_ok=False)
archive=args.out/'source';archive.mkdir()
sources=[Path(__file__),root/'scripts/odometry_worker.py',*sorted((root/'src/vln_localization').glob('*.py'))]
for p in sources:shutil.copyfile(p,archive/p.name)
(archive/'sha256.json').write_text(json.dumps({p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},indent=2))
sys.path.insert(0,str(root/'src'))
from vln_localization.process import OdometryProcess
child=OdometryProcess(root,args.out,args.run,args.method,args.voxel_size,args.initial_threshold,args.fixed_threshold,args.prediction)
def rotation(q):
    w,x,y,z=q
    return np.array([[1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w)],
                     [2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w)],
                     [2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)]])
rows=[];first=None
try:
    with (args.out/'poses.jsonl').open('w') as stream:
        for file in sorted(args.run.glob('scan_*.npz')):
            frame=np.load(file)
            gt=np.eye(4);gt[:3,:3]=rotation(frame['quaternion']);gt[:3,3]=frame['position']
            if args.raw_scans:
                points=np.load(file.with_name('raw_'+file.name))['points_base_xyz']
            else:
                points=(frame['points'][:,:3]-gt[:3,3])@gt[:3,:3]
            # No pose, world point, frame filename, or labels cross this boundary.
            result=child.ask(float(frame['time']),points.tolist())
            row={'source_scan':file.name,'source_sha256':hashlib.sha256(file.read_bytes()).hexdigest(),
                 'truth_pose':gt.tolist(),'estimate':result}
            if first is None:first=gt.copy()
            if result.get('valid'):
                estimate=first@np.asarray(result['odom_from_base'])
                error=np.linalg.inv(gt)@estimate
                row.update(aligned_estimate=estimate.tolist(),translation_error_m=float(np.linalg.norm(error[:3,3])),
                    rotation_error_rad=math.acos(float(np.clip((np.trace(error[:3,:3])-1)/2,-1,1))))
            stream.write(json.dumps(row)+'\n');stream.flush();rows.append(row)
            if not result.get('valid'):break
    good=[r for r in rows if r['estimate'].get('valid')]
    errors=np.array([r['translation_error_m'] for r in good])
    times=np.array([r['estimate']['processing_seconds'] for r in good])
    report={'method':args.method,'prediction':args.prediction,'raw_scans':args.raw_scans,
            'scope':'isolated LiDAR-only odometry replay of virtual scans; not physical SLAM or navigation',
            'alignment':'first pose only, no trajectory fit','input_contract':['time','points_base_xyz'],
            'frames':len(rows),'valid_frames':len(good),'control_ready':False,'complete':len(rows)==len(list(args.run.glob('scan_*.npz'))) and len(good)==len(rows),
            'translation_rmse_m':float(np.sqrt(np.mean(errors**2))) if len(good) else None,
            'maximum_translation_error_m':float(errors.max()) if len(good) else None,
            'final_translation_error_m':float(errors[-1]) if len(good) else None,
            'maximum_rotation_error_deg':max((math.degrees(r['rotation_error_rad']) for r in good),default=None),
            'processing_median_ms':float(np.median(times)*1000) if len(good) else None,
            'processing_p95_ms':float(np.quantile(times,.95)*1000) if len(good) else None,
            'last_result':rows[-1]['estimate'] if rows else None}
    gates=json.loads((root/'config/readiness/odometry-replay-gates.json').read_text())
    report['screening_passed']=(report['complete'] and report['maximum_translation_error_m']<=gates['maximum_translation_error_m']
        and report['final_translation_error_m']<=gates['maximum_final_translation_error_m']
        and report['maximum_rotation_error_deg']<=gates['maximum_rotation_error_deg']
        and report['processing_p95_ms']<=gates['maximum_processing_p95_ms'])
    report['gates']=gates
    (args.out/'report.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report),flush=True)
finally:
    child.close()
