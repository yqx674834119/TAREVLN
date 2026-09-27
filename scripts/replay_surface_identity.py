"""Recheck recorded RGB-D anchors without simulator metadata or goal labels."""
import argparse
import base64
import hashlib
import json
from pathlib import Path
import zlib
import numpy as np
from vln_core.vision_api import VisionAPI
from vln_policy.office_grounding import ground_visual_anchor
from vln_policy.office_surface import verify_table_surface
from vln_policy.office_contract import confirmed_from_two_positions

parser=argparse.ArgumentParser()
parser.add_argument('--run',type=Path,required=True)
parser.add_argument('--out',type=Path,required=True)
parser.add_argument('--samples',type=Path,help='Optional fixed assistant-reviewed pixel cases, not benchmark ground truth')
args=parser.parse_args()
args.out.mkdir(parents=True,exist_ok=False)
api=VisionAPI(args.out/'vlm',max_calls=20)
rows=[]
specs=(json.loads(args.samples.read_text()) if args.samples else
       [{'response':str(p.relative_to(args.run))} for p in sorted((args.run/'agent/vlm').glob('*/response.json'))])
for spec in specs:
    response=args.run/spec['response']
    record=json.loads(response.read_text())
    context=record['context']
    if context['purpose'] not in ('arrival_verification','depth_anchor_repair'):
        continue
    check=record.get('result',{})
    if 'pixel' in spec:
        check=dict(check,anchor_pixel_xy=spec['pixel'])
    if not check.get('anchor_pixel_xy') or check.get('anchor_view') not in ('front','left','rear','right'):
        continue
    index=['front','left','rear','right'].index(check['anchor_view'])
    step=round(context['time']/.005/400)*400
    file=args.run/f'depth_{step:07d}_{index}.npz'
    frame=np.load(file)
    if abs(float(frame['time'])-context['time'])>.03:
        raise ValueError('Recorded RGB-D time does not match observation')
    w,x,y,z=frame['quaternion'].tolist()
    rotation=[[1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w)],
              [2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w)],
              [2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)]]
    mm=(np.clip(np.nan_to_num(frame['depth'],nan=0.,posinf=0.,neginf=0.),0,65.535)*1000).astype('<u2')
    rgb=args.run/f'rgb_{step:07d}_{index}.jpg'
    view={'label':check['anchor_view'],'jpeg':base64.b64encode(rgb.read_bytes()).decode(),
          'depth_mm':base64.b64encode(zlib.compress(mm.tobytes())).decode(),
          'image_size':[640,480],'intrinsics':[320.,320.,320.,240.],
          'rotation':rotation,'position':frame['position'].tolist()}
    grounding=ground_visual_anchor(check,[view],context['position'])
    row={'source':str(response),'time':context['time'],'position':context['position'],
         'review':spec,
         'depth_sha256':hashlib.sha256(file.read_bytes()).hexdigest(),
         'rgb_sha256':hashlib.sha256(rgb.read_bytes()).hexdigest(),'grounding':grounding}
    if grounding['valid']:
        row['surface']=verify_table_surface(api,check,grounding,[view],{'source':str(response)})
        print(response.parent.name,row['surface']['valid'],row['surface'].get('identity'),flush=True)
    row['accepted']=bool(row.get('surface',{}).get('valid'))
    if 'expected_valid' in spec:
        row['matches_review']=row['accepted']==spec['expected_valid']
    rows.append(row)
    (args.out/'report.json').write_text(json.dumps({'scope':'recorded observation replay; not live task success','samples':rows},indent=2))
valid=[dict(time=r['time'],position=r['position'],anchor_world_xyz=r['grounding']['anchor_world_xyz'],
            **{k:r['surface'][k] for k in ('surface_z_m','surface_voxels_xy_10cm')})
       for r in rows if r.get('surface',{}).get('valid')]
pairs=[{'times':[a['time'],b['time']],'confirmed':confirmed_from_two_positions(a,b)}
       for i,a in enumerate(valid) for b in valid[i+1:]]
(args.out/'association.json').write_text(json.dumps(pairs,indent=2))
print(json.dumps(pairs),flush=True)
