"""Measure marker occlusion using recorded sensor depth and estimated poses only."""
import argparse
import json
from pathlib import Path
import numpy as np

ap=argparse.ArgumentParser();ap.add_argument('run',type=Path);ap.add_argument('--out',type=Path,required=True)
args=ap.parse_args();args.out.mkdir(parents=True,exist_ok=False)
poses=[json.loads(line) for line in (args.run/'odometry/poses.jsonl').read_text().splitlines()]
records=[]
for request in sorted((args.run/'agent/vlm').glob('call-*/request.json')):
    context=json.loads(request.read_text())['context']
    if context['purpose']!='navigation_decision':continue
    response=request.with_name('response.json')
    if not response.exists():continue
    decision=json.loads(response.read_text()).get('result',{})
    t=context['time'];step=round(t/2)*400
    row=min(poses,key=lambda r:abs(r['time']-t))
    if abs(row['time']-t)>.03:raise ValueError('RGB-D and estimated pose are not synchronized')
    pose=np.asarray(row['estimate']['odom_from_base']);pose[2,3]+=.4
    rotation=pose[:3,:3];position=pose[:3,3]
    image_results=[]
    for index,(view,offset) in enumerate(zip(('front','left','rear','right'),(0,np.pi/2,np.pi,-np.pi/2))):
        depth=np.load(args.run/f'depth_{step:07d}_{index}.npz')['depth']
        height,width=depth.shape
        turn=np.array([[np.cos(offset),-np.sin(offset),0],[np.sin(offset),np.cos(offset),0],[0,0,1.]])
        cam_rotation=rotation@turn;cam_position=position+rotation@np.array([0,0,.45])
        for c in context['candidates']:
            if c['id'] not in context['candidate_view_ids'].get(view,[]):continue
            point=np.asarray(c['xyz']).copy();point[2]=context['position'][2]-.2
            local=cam_rotation.T@(point-cam_position)
            u,v=round(320-320*local[1]/local[0]),round(240-320*local[2]/local[0])
            patch=depth[max(0,v-2):min(height,v+3),max(0,u-2):min(width,u+3)]
            valid=patch[np.isfinite(patch)&(patch>0)&(patch<40)]
            measured=float(np.median(valid)) if len(valid)>=5 else None
            image_results.append({'view':view,'id':c['id'],'pixel':[u,v],'marker_depth_m':float(local[0]),
                'observed_depth_m':measured,'occluded':bool(measured is not None and measured+.15<local[0]),
                'selected':decision.get('selected_view')==view and decision.get('candidate_id')==c['id']})
    records.append({'call':request.parent.name,'time':t,'markers':image_results})
selected=[m for r in records for m in r['markers'] if m['selected']]
summary={'scope':'development visibility audit, no scene files or truth poses used',
         'queries':len(records),'selected_markers':len(selected),
         'selected_occluded':sum(m['occluded'] for m in selected),
         'selected_unknown_depth':sum(m['observed_depth_m'] is None for m in selected),
         'all_markers':sum(len(r['markers']) for r in records),
         'occluded_markers':sum(m['occluded'] for r in records for m in r['markers'])}
(args.out/'report.json').write_text(json.dumps({'summary':summary,'records':records},indent=2))
print(summary)
