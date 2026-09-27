"""Export recorded native TARE layers without consulting an evaluator map."""
import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('run',type=Path)
parser.add_argument('--time',type=float,default=float('inf'))
args=parser.parse_args()
folder=args.run/'agent/planning'
entries=[json.loads(line) for line in (folder/'index.jsonl').open()]
entries=[entry for entry in entries if entry['time']<=args.time]
if not entries:raise SystemExit('No recorded planning frame before requested time')
frame=json.loads((folder/entries[-1]['file']).read_text())
robot=np.asarray(frame['robot']['position']);layers=frame['layers']
fig,ax=plt.subplots(figsize=(8,7),constrained_layout=True)
counts={}
for key,color,size,label in [('free','#80d6b8',11,'Free-space samples'),
                            ('candidates','#188663',5,'Candidate viewpoints'),
                            ('obstacles','#df737b',5,'Collision cloud')]:
    points=np.asarray(layers.get(key,{}).get('points',[])).reshape(-1,3)
    counts[key]=len(points) if key in layers else 'not recorded'
    points=points[np.abs(points[:,2]-robot[2])<=.8]
    if len(points):ax.scatter(points[:,0],points[:,1],s=size,c=color,label=label,rasterized=True)
for key,color,label in [('coverage_path','#8f9dac','Coverage path'),('path','#7948cb','Local path')]:
    points=np.asarray(layers.get(key,{}).get('points',[])).reshape(-1,3)
    if len(points):ax.plot(points[:,0],points[:,1],c=color,lw=2,label=label)
trail=np.array([entry['robot']['position'] for entry in entries])
ax.plot(trail[:,0],trail[:,1],c='#3d76c8',lw=1.2,label='Estimated trajectory')
waypoint=layers.get('waypoint',{}).get('points',[])
if waypoint:ax.scatter(*waypoint[0][:2],marker='*',s=190,c='#efac30',edgecolor='#775300',label='Waypoint',zorder=5)
yaw=frame['robot']['yaw']
ax.arrow(*robot[:2],.35*np.cos(yaw),.35*np.sin(yaw),width=.06,color='#153d68',zorder=6)
ax.set(xlim=(robot[0]-6,robot[0]+6),ylim=(robot[1]-6,robot[1]+6),
       xlabel='Odometry x (m)',ylabel='Odometry y (m)',
       title=f"Native TARE snapshot at {frame['time']:.1f} s\n"
             f"free={counts['free']} · candidates={counts['candidates']} · obstacles={counts['obstacles']}")
ax.set_aspect('equal');ax.grid(alpha=.15);ax.legend(loc='upper left',fontsize=8)
fig.supxlabel('Unknown space is unfilled. Current-height slice; no oracle geometry.',fontsize=10)
output=args.run/'review';output.mkdir(exist_ok=True)
fig.savefig(output/'native-planning.png',dpi=180);plt.close(fig)
print(output/'native-planning.png')
