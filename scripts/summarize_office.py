"""Produce review artifacts from a completed Office run; never used by policy."""
import argparse
import json
import math
import subprocess
import sys
import shutil
import textwrap
from collections import Counter
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ap=argparse.ArgumentParser()
ap.add_argument('run',type=Path)
ap.add_argument('--oracle',type=Path)
ap.add_argument('--video',action='store_true')
args=ap.parse_args()
run=args.run
out=run/'review';out.mkdir(exist_ok=True)
report=json.loads((run/'report.json').read_text())
rows=[json.loads(s) for s in (run/'trajectory.jsonl').read_text().splitlines()]
positions=np.asarray([r['position'] for r in rows])
calls=[]
for request in sorted((run/'agent/vlm').glob('call-*/request.json')):
 r=json.loads(request.read_text());response=request.with_name('response.json')
 if response.exists():r.update(json.loads(response.read_text()))
 calls.append(r)
fig,ax=plt.subplots(figsize=(9,9))
for path in sorted(run.glob('scan_*.npz'))[::10]:
 points=np.load(path)['points']
 if len(points):
  points=points[(points[:,2]>.15)&(points[:,2]<1.7)]
  ax.scatter(points[:,0],points[:,1],s=.15,c='#94a3b8',alpha=.25)
if args.oracle:
 for r in json.loads(args.oracle.read_text()).get('goal_rectangles',[]):
  ax.add_patch(plt.Rectangle((r[0],r[1]),r[2]-r[0],r[3]-r[1],facecolor='#14b8a6',alpha=.13,edgecolor='#0f766e'))
ax.plot(positions[:,0],positions[:,1],color='#2563eb',lw=2,label='Measured robot trajectory')
ax.scatter(*positions[0,:2],s=80,marker='o',color='#16a34a',label='Start')
ax.scatter(*positions[-1,:2],s=90,marker='X',color='#dc2626',label='Final position')
for i,call in enumerate(calls,1):
 p=call.get('context',{}).get('position')
 if p:
  if report.get('evaluation_world_from_odometry'):
   p=(np.asarray(report['evaluation_world_from_odometry'])@np.array([*p,1.]))[:3]
  ax.annotate(str(i),p[:2],fontsize=8,color='#7c3aed')
ax.set_aspect('equal');ax.set_xlabel('World x (m)');ax.set_ylabel('World y (m)');ax.grid(alpha=.2)
ax.set_title(f"{run.name}: {report['status']}\nVLM call numbers; LiDAR points; evaluator goal areas")
ax.legend(loc='best');fig.tight_layout();fig.savefig(out/'trajectory.png',dpi=150);plt.close(fig)
summary={k:report.get(k) for k in ['status','success','task','start_position','simulation_seconds','wall_seconds','distance_m','final_position','assets_complete','contact_quality_passed']}
summary['vlm_calls']=len(calls)
summary['vlm_call_purposes']=dict(Counter(c.get('context',{}).get('purpose','unspecified') for c in calls))
summary['network_attempts']=sum(len(c.get('attempts',[])) for c in calls)
summary['vlm_latency_seconds']=[round(c.get('latency_seconds',0),3) for c in calls]
summary['vlm_decisions']=[{'call':i,'simulation_time':c.get('context',{}).get('simulation_time'),'position':c.get('context',{}).get('position'),'result':c.get('result'),'status':c.get('status')} for i,c in enumerate(calls,1)]
summary['native_counts']=report.get('final_action',{}).get('counts')
(out/'summary.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False))
frames=sorted(run.glob('rgb_*_0.jpg'))
if frames:
 chosen=[frames[0],frames[len(frames)//2],frames[-1]]
 collage=Image.new('RGB',(1280,3*510),'#0f172a');draw=ImageDraw.Draw(collage)
 for row,frame in enumerate(chosen):
  number=int(frame.stem.split('_')[1])
  draw.text((12,row*510+8),f'{run.name}  |  simulated time ~{number*.005:.1f} s',fill='white')
  for col,view in enumerate([0,1]):
   img=Image.open(frame.with_name(frame.name.replace('_0.jpg',f'_{view}.jpg'))).convert('RGB')
   collage.paste(img,(640*col,row*510+30))
 collage.save(out/'keyframes.jpg',quality=90)
 if args.video:
  ffmpeg=shutil.which('ffmpeg') or str(Path(sys.executable).parent/'ffmpeg')
  encoder=subprocess.Popen([ffmpeg,'-y','-loglevel','error','-f','rawvideo','-pixel_format','rgb24',
      '-video_size','1280x560','-framerate','2','-i','-','-an','-c:v','libx264','-crf','23',
      '-pix_fmt','yuv420p',str(out/'autonomous-review.mp4')],stdin=subprocess.PIPE)
  try:
   for frame in frames:
    step=int(frame.stem.split('_')[1]);t=step*.005
    canvas=Image.new('RGB',(1280,560),'#0f172a');draw=ImageDraw.Draw(canvas)
    canvas.paste(Image.open(frame).convert('RGB'),(0,40))
    chase=run/f'chase_{step:07d}.jpg'
    second=chase if chase.exists() else frame.with_name(frame.name.replace('_0.jpg','_1.jpg'))
    canvas.paste(Image.open(second).convert('RGB'),(640,40))
    draw.text((12,12),f'{run.name} | simulation {t:.1f} s | 4x simulation playback',fill='white')
    nearest=min(rows,key=lambda r:abs(r['time']-t))
    action=nearest.get('action',{});decision=action.get('last_decision') or {}
    caption=f"VLM calls: {action.get('vlm_calls',0)} | {action.get('reason','initializing')} | {decision.get('action','')} {decision.get('reason','')}"
    draw.text((12,524),'\n'.join(textwrap.wrap(caption,width=170)[:2]),fill='white')
    encoder.stdin.write(canvas.tobytes())
  finally:
   encoder.stdin.close()
   if encoder.wait()!=0:raise RuntimeError('Video encoding failed')
print(json.dumps(summary,ensure_ascii=False))
