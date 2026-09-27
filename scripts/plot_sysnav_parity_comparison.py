"""Evaluator-only comparison; never supply these routes or target regions to policy."""
import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]
ap=argparse.ArgumentParser(description=__doc__)
ap.add_argument('run',type=Path)
args=ap.parse_args()
report=json.loads((args.run/'report.json').read_text())
if report['status']=='starting':raise SystemExit('Wait for the frozen experiment to finish')
original=ROOT/'data/sim/office/runs/ui-20260928-022005-2b2f3'
control=ROOT/'data/readiness/crossroom-reachability-r3'

def trajectory(folder):
    return np.array([json.loads(s)['position'] for s in (folder/'trajectory.jsonl').read_text().splitlines()])

fig,axes=plt.subplots(1,2,figsize=(12,9),constrained_layout=True)
targets=json.loads((ROOT/'sim/fixtures/sysnav-object-evaluation.json').read_text())['tasks']['meeting_crossroom']['targets']
for ax,folder,title,color in zip(axes,[original,args.run],['Before system alignment','After system alignment'],['#64748b','#2563eb']):
    for scan in sorted(folder.glob('scan_*.npz'))[::50]:
        points=np.load(scan)['points']
        points=points[(points[:,2]>.2)&(points[:,2]<1.4)]
        ax.scatter(points[:,0],points[:,1],s=.15,color='#94a3b8',alpha=.2,rasterized=True)
    for i,target in enumerate(targets):
        r=target['room_rectangle']
        ax.add_patch(plt.Rectangle(r[:2],r[2]-r[0],r[3]-r[1],fill=False,color='#16a34a',lw=1.2,
                                  label='Eligible goal rooms (truth)' if i==0 else None))
        center=(np.array(target['min'])+target['max'])/2
        ax.scatter(*center[:2],marker='s',s=22,color='#16a34a')
    route=trajectory(control)
    ax.plot(route[:,0],route[:,1],color='#16a34a',ls='--',lw=1.8,label='Physical oracle control (46.9 s)')
    p=trajectory(folder)
    ax.plot(p[:,0],p[:,1],color=color,lw=1.7,label='Autonomous SysNav trajectory')
    ax.scatter(*p[0,:2],s=50,c='#0f172a',label='Start',zorder=4)
    ax.scatter(*p[-1,:2],s=70,c='#dc2626',marker='X',label='Final',zorder=4)
    r=json.loads((folder/'report.json').read_text())
    ax.set(xlim=(-23,7),ylim=(10,62),xlabel='World x (m)',ylabel='World y (m)',
           title=f"{title}\n{r['status']} | {r['simulation_seconds']:.1f} s | {r['distance_m']:.1f} m")
    ax.set_aspect('equal');ax.grid(alpha=.15)
axes[1].legend(loc='upper left',fontsize=8)
fig.suptitle('Same Office, start, task, Go2 checkpoint and 300 s budget',fontsize=14)
fig.supxlabel('Evaluator view only. Green route proves physical reachability; it is not autonomous success.',fontsize=10)
out=args.run/'review';out.mkdir(exist_ok=True)
fig.savefig(out/'system-parity-comparison.png',dpi=160)
plt.close(fig)

# Separate policy-visible evidence: the actual native room mask, doors and
# observed trajectory in odometry coordinates, without evaluator goal truth.
folder=args.run/'agent/planning'
events=[json.loads(s) for s in (args.run/'agent/events.jsonl').read_text().splitlines()]
index=[json.loads(s) for s in (folder/'index.jsonl').read_text().splitlines()]
fig,axes=plt.subplots(1,2,figsize=(12,8),constrained_layout=True)
for ax,t in zip(axes,[75.5,index[-1]['time']]):
    entries=[r for r in index if r['time']<=t]
    frame=json.loads((folder/entries[-1]['file']).read_text())
    masks=sorted(folder.glob('rooms-*.npz'))
    masks=[p for p in masks if int(p.stem.split('-')[1])/1000<=t]
    if masks:
        data=np.load(masks[-1]);mask=data['mask']
        rr,cc=np.nonzero(mask)
        if len(rr):
            r0,r1=max(0,rr.min()-8),min(mask.shape[0],rr.max()+9)
            c0,c1=max(0,cc.min()-8),min(mask.shape[1],cc.max()+9)
            view=mask[r0:r1,c0:c1].T.astype(float);view[view==0]=np.nan
            ax.imshow(view,origin='lower',extent=((r0-1500)*.1,(r1-1500)*.1,(c0-1500)*.1,(c1-1500)*.1),
                      cmap='Pastel1',vmin=1,vmax=9,interpolation='nearest',alpha=.9)
    for key,color,size in [('doors','#c2410c',30),('candidates','#16a34a',2)]:
        p=np.array(frame['layers'].get(key,{}).get('points',[])).reshape(-1,3)
        if len(p):ax.scatter(p[:,0],p[:,1],c=color,s=size,label=key)
    snapshots=[r for r in events if r['topic']=='/room_nodes_list' and r['time']<=t]
    labels={r['id']:r['label'] for r in events if r['topic']=='/room_type_answer' and r['time']<=t}
    if snapshots:
        for room in snapshots[-1]['rooms']:
            ax.text(*room['centroid'][:2],f"R{room['id']} {labels.get(room['id'],'unlabeled')}\n{room['area']:.1f} m²",fontsize=8,
                    bbox=dict(facecolor='white',alpha=.8,edgecolor='none'))
    trail=np.array([r['robot']['position'] for r in entries]);ax.plot(trail[:,0],trail[:,1],c='#2563eb',lw=1,label='Sensor odometry')
    ax.scatter(*trail[-1,:2],marker='^',c='#0f172a',s=70)
    wp=frame['layers'].get('waypoint',{}).get('points',[])
    if wp:ax.scatter(*wp[0][:2],marker='*',c='#f59e0b',s=180,label='Waypoint')
    ax.set(xlabel='Odometry x (m)',ylabel='Odometry y (m)',title=f"Native room state at {frame['time']:.1f} s")
    ax.set_aspect('equal');ax.grid(alpha=.1);ax.legend(fontsize=8)
fig.supxlabel('Recorded native masks and messages only. White means unlabeled; no goal truth is shown.',fontsize=10)
fig.savefig(out/'native-room-state.png',dpi=160)
