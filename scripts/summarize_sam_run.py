"""Review actual mask-supported VLM selections, without rerunning perception."""
import argparse
import base64
import io
import json
from pathlib import Path
import statistics
from PIL import Image,ImageDraw
from vln_perception.sam2_masks import decode_mask
from vln_policy.office_tasks import TASKS

ap=argparse.ArgumentParser();ap.add_argument('run',type=Path);args=ap.parse_args()
out=args.run/'review';out.mkdir(exist_ok=True)
task=TASKS[json.loads((args.run/'report.json').read_text())['task']]
requests=[json.loads(p.read_text()) for p in sorted((args.run/'sam').glob('request-*.json'))]
images={r['result']['image_sha256']:r['request']['jpeg'] for r in requests}
panels=[];selections=[]
for file in sorted((args.run/'agent/vlm').glob('call-*/request.json')):
    context=json.loads(file.read_text()).get('context',{})
    response=file.with_name('response.json')
    if context.get('purpose')!='surface_proposal_verification' or not response.exists():continue
    result=json.loads(response.read_text()).get('result',{})
    candidate=next((c for c in context.get('surface_candidates',[]) if c['id']==result.get('surface_id')),None)
    if not candidate or not candidate.get('sam'):continue
    support=candidate['sam']
    original=Image.open(io.BytesIO(base64.b64decode(images[support['image_sha256']]))).convert('RGB')
    mask=decode_mask(support['mask_bits'],support['image_size'])
    shown=original.copy()
    shown.paste(Image.blend(original,Image.new('RGB',original.size,'#22c55e'),.4),mask=Image.fromarray(mask.astype('uint8')*255))
    draw=ImageDraw.Draw(shown)
    u,v=candidate['pixel_xy'];draw.ellipse((u-5,v-5,u+5,v+5),fill='red')
    panel=Image.new('RGB',(1280,520),'#0f172a');panel.paste(original,(0,40));panel.paste(shown,(640,40))
    features_pass=all(result.get(k) is True for k in task['features']) and result.get('inside_area') is True and result.get('confidence',0)>=.8
    label='Task features pass; identity checked separately' if features_pass else 'REJECTED: task features / confidence incomplete'
    text=f"{file.parent.name} | t={context.get('simulation_time',context.get('time')):.1f}s | {candidate['view']} {candidate['id']} | {label}"
    ImageDraw.Draw(panel).text((12,12),text,fill='#86efac' if features_pass else '#fca5a5')
    panels.append(panel)
    selections.append({'call':file.parent.name,'simulation_time':context.get('simulation_time',context.get('time')),
                       'id':candidate['id'],'view':candidate['view'],'anchor_xyz':candidate['anchor_xyz'],
                       'mask_pixels':support['mask_pixels'],'measured_support_pixels':support['support_pixels'],
                       'vlm_verification':result})
if panels:
    montage=Image.new('RGB',(1280,520*len(panels)))
    for i,p in enumerate(panels):montage.paste(p,(0,520*i))
    montage.save(out/'sam-selections.jpg',quality=92)
times=[r['result']['processing_seconds'] for r in requests]
summary={'sam_requests':len(requests),'mask_count':sum(len(r['result']['masks']) for r in requests),
         'processing_seconds':{'median':statistics.median(times),'max':max(times)} if times else {},
         'scope':'actual online SAM/geometry support and VLM choices, not an independently labeled segmentation benchmark',
         'selections':selections}
(out/'sam-summary.json').write_text(json.dumps(summary,indent=2))
print({k:v for k,v in summary.items() if k!='selections'})
