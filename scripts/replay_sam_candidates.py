"""Inspect SAM2 masks for recorded, geometry-generated candidate boxes."""
import argparse
import base64
import hashlib
import json
import inspect
from pathlib import Path
import shutil
import numpy as np
from PIL import Image,ImageDraw,ImageFilter
from vln_perception.sam2_masks import Sam2Masks,decode_mask

ap=argparse.ArgumentParser()
ap.add_argument('--run',type=Path,required=True)
ap.add_argument('--proposals',type=Path,required=True)
ap.add_argument('--checkpoint',type=Path,required=True)
ap.add_argument('--out',type=Path,required=True)
ap.add_argument('--step',type=int,default=12000)
args=ap.parse_args()
args.out.mkdir(parents=True,exist_ok=False)
source=args.out/'source';source.mkdir()
for p in [Path(__file__),Path(inspect.getfile(Sam2Masks))]:shutil.copyfile(p,source/p.name)
frame=min(json.loads(args.proposals.read_text())['frames'],key=lambda f:abs(f['context']['time']-args.step*.005))
if abs(frame['context']['time']-args.step*.005)>.03:raise ValueError('Wrong proposal frame')
model=Sam2Masks(args.checkpoint)
report={'checkpoint':str(args.checkpoint),'checkpoint_sha256':hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
        'scope':'development mask replay, not online navigation or segmentation accuracy benchmark','step':args.step,'views':[]}
for index,label in enumerate(['front','left','rear','right']):
    candidates=[c for c in frame['candidates'] if c['view']==label]
    if not candidates:continue
    image_file=args.run/f'rgb_{args.step:07d}_{index}.jpg'
    image=Image.open(image_file).convert('RGB');width,height=image.size
    boxes=[[max(0,c['bbox_xyxy'][0]-8),max(0,c['bbox_xyxy'][1]-8),min(width,c['bbox_xyxy'][2]+8),min(height,c['bbox_xyxy'][3]+8)] for c in candidates]
    result=model.segment({'jpeg':base64.b64encode(image_file.read_bytes()).decode(),'boxes_xyxy':boxes})
    output=args.out/label;output.mkdir()
    panels=[]
    for i,(c,encoded) in enumerate(zip(candidates,result['masks'])):
        mask=decode_mask(encoded,result['image_size'])
        Image.fromarray(mask.astype('uint8')*255).save(output/f"{c['id']}-mask.png")
        shown=image.copy();overlay=Image.new('RGB',image.size,'#22c55e')
        shown.paste(Image.blend(image,overlay,.35),mask=Image.fromarray(mask.astype('uint8')*255))
        draw=ImageDraw.Draw(shown);draw.rectangle(boxes[i],outline='cyan',width=2)
        u,v=c['pixel_xy'];draw.ellipse((u-4,v-4,u+4,v+4),fill='red')
        draw.text((10,10),f"{c['id']} | SAM predicted IoU {result['scores'][i]:.3f}",fill='white',stroke_width=2,stroke_fill='black')
        shown.save(output/f"{c['id']}-overlay.jpg",quality=92);panels.append(shown)
    montage=Image.new('RGB',(width*2,height*((len(panels)+1)//2)))
    for i,p in enumerate(panels):montage.paste(p,(i%2*width,i//2*height))
    montage.save(output/'montage.jpg',quality=92)
    report['views'].append({'view':label,'candidate_ids':[c['id'] for c in candidates],'boxes_xyxy':boxes,**result})
    print(label,len(candidates),'masks',result['mask_pixels'],'scores',result['scores'],'seconds',result['processing_seconds'],flush=True)
(args.out/'report.json').write_text(json.dumps(report,indent=2))
