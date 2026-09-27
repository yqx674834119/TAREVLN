"""Export measured motion versus commands from an independent diagnostic."""
import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ap=argparse.ArgumentParser()
ap.add_argument('run',type=Path)
args=ap.parse_args()
rows=[json.loads(line) for line in (args.run/'states.jsonl').read_text().splitlines()]
t=np.array([r['t'] for r in rows])
commands=np.array([r['command'] for r in rows])
actual=np.array([[*r['body_velocity'][:2],r['body_angular_velocity'][2]] for r in rows])
fig,axes=plt.subplots(4,1,figsize=(12,9),sharex=True)
for i,label in enumerate(('Forward velocity (m/s)','Lateral velocity (m/s)','Yaw rate (rad/s)')):
    axes[i].plot(t,actual[:,i],color='#2563eb',lw=.8,label='Measured')
    axes[i].plot(t,commands[:,i],color='#f97316',lw=1.7,label='Command')
    axes[i].set_ylabel(label);axes[i].grid(alpha=.2)
axes[0].legend(loc='upper right')
axes[3].plot(t,[r['position'][2] for r in rows],color='#2563eb',label='Base height')
if 'feet_xyz' in rows[0]:
    for leg in range(4):axes[3].plot(t,[r['feet_xyz'][leg][2] for r in rows],lw=.6,label=f'Foot {leg} height')
axes[3].set_ylabel('World height (m)');axes[3].set_xlabel('Simulation time (s)');axes[3].grid(alpha=.2)
axes[3].legend(loc='upper right',ncol=3)
previous=None
for row in rows:
    if row['phase']!=previous:
        axes[0].text(row['t'],1.02,row['phase'],transform=axes[0].get_xaxis_transform(),fontsize=7,rotation=35)
        for ax in axes:ax.axvline(row['t'],color='#94a3b8',alpha=.2,lw=.7)
        previous=row['phase']
report=json.loads((args.run/'report.json').read_text())
fig.suptitle(f"{args.run.name}: gate {'PASS' if report['passed'] else 'FAIL'} — independent simulation, no physical robot",fontsize=12,y=.99)
fig.tight_layout(rect=(0,0,1,.97))
fig.savefig(args.run/'tracking.png',dpi=150)
print(args.run/'tracking.png')
