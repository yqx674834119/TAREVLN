"""Review a recorded full SysNav run, without feeding truth back to navigation."""
import argparse
from collections import Counter
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ap=argparse.ArgumentParser()
ap.add_argument('run',type=Path)
args=ap.parse_args()
run=args.run
out=run/'review';out.mkdir(exist_ok=True)
report=json.loads((run/'report.json').read_text())
rows=[json.loads(s) for s in (run/'trajectory.jsonl').read_text().splitlines()]
calls=[json.loads(s) for s in (run/'agent/api.jsonl').read_text().splitlines()]
summary={k:report.get(k) for k in ('status','success','acceptance_passed','instruction',
    'simulation_seconds','wall_seconds','distance_m','final_position','stop_dwell',
    'independent_goal_region','independent_object_evaluation','evaluation_definition',
    'contact_quality_passed','localization_evaluation','physical_hardware_ready','terminal_control')}
summary.update(vlm_calls=len(calls), api_status=dict(Counter(c['status'] for c in calls)),
    vlm_purposes=dict(Counter(c['purpose'] for c in calls)),
    calls_with_images=sum(c['images']>0 for c in calls),
    native_counts=report.get('final_action',{}).get('counts',{}),
    rooms=report.get('final_action',{}).get('rooms'),objects=report.get('final_action',{}).get('objects'),
    isolation=json.loads((run/'agent/isolation.json').read_text()))
if calls:
    summary['api_latency_s']={k:float(np.quantile([c['seconds'] for c in calls],q))
                              for k,q in [('median',.5),('p95',.95)]}
    summary['tokens']={k:sum((c.get('usage') or {}).get(k,0) for c in calls)
                      for k in ('prompt_tokens','completion_tokens','total_tokens')}
(out/'summary.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False))
pos=np.asarray([r['position'] for r in rows]);times=np.array([r['time'] for r in rows])
fig,axs=plt.subplots(1,2,figsize=(12,5),constrained_layout=True)
ax=axs[0]
for scan in sorted(run.glob('scan_*.npz'))[::20]:
    points=np.load(scan)['points']
    points=points[(points[:,2]>.15)&(points[:,2]<1.5)]
    ax.scatter(points[:,0],points[:,1],s=.1,c='#94a3b8',alpha=.3)
fixture=run/'source/sim/fixtures/sysnav-object-evaluation.json'
if fixture.exists():
    targets=json.loads(fixture.read_text())['tasks'].get(report['task'],{}).get('targets',[])
    for i,t in enumerate(targets):
        lo,hi=t['min'],t['max']
        ax.add_patch(plt.Rectangle(lo[:2],hi[0]-lo[0],hi[1]-lo[1],color='#16a34a',alpha=.4,
                                  label='Target object truth' if i==0 else None))
        if t.get('room_rectangle'):
            r=t['room_rectangle']
            ax.add_patch(plt.Rectangle(r[:2],r[2]-r[0],r[3]-r[1],fill=False,color='#16a34a',lw=.8))
ax.plot(pos[:,0],pos[:,1],color='#2563eb',lw=2,label='Actual trajectory')
ax.scatter(*pos[0,:2],c='#111827',s=45,label='Start')
ax.scatter(*pos[-1,:2],c='#dc2626',marker='X',s=65,label='Final position')
ax.set(xlabel='World x (m)',ylabel='World y (m)',title='Evaluator view; no truth supplied to policy')
ax.set_aspect('equal');ax.legend(fontsize=8);ax.grid(alpha=.2)
ax=axs[1]
ax.plot(times,[r['speed'] for r in rows],color='#2563eb',lw=1,label='3D base speed')
ax.axhline(.06,color='#dc2626',ls='--',label='Stop threshold (0.06 m/s)')
stops=[r['time'] for r in rows if r.get('action',{}).get('stop')]
if stops:ax.axvline(stops[0],color='#16a34a',ls=':',label='Native SysNav STOP')
ax.set(xlabel='Simulation time (s)',ylabel='Speed (m/s)',title='Physical stopping, including vertical motion')
ax.grid(alpha=.2);ax.legend(fontsize=8)
fig.suptitle(f"Full SysNav + DeepSeek + Isaac Go2 | {report['status']}")
fig.savefig(out/'trajectory-and-stop.png',dpi=170)
plt.close(fig)
print(json.dumps(summary,ensure_ascii=False))
