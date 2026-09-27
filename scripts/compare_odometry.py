"""Evaluator-only comparison of completed replay reports and error curves."""
import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ap=argparse.ArgumentParser()
ap.add_argument('directories',type=Path,nargs='+')
ap.add_argument('--out',type=Path,required=True)
args=ap.parse_args()
reports=[json.loads((d/'report.json').read_text()) for d in args.directories]
if not all(r['complete'] for r in reports):raise ValueError('Only complete replays may be compared')
args.out.mkdir(parents=True,exist_ok=False)
fig,axes=plt.subplots(1,2,figsize=(13,4.5))
summary=[]
for d,r in zip(args.directories,reports):
    rows=[json.loads(line) for line in (d/'poses.jsonl').read_text().splitlines()]
    times=[row['estimate']['time'] for row in rows]
    axes[0].plot(times,[row['translation_error_m'] for row in rows],label=d.name)
    axes[1].plot(times,[row['estimate']['processing_seconds']*1000 for row in rows],alpha=.65)
    summary.append({'name':d.name,'directory':str(d.resolve()),**{k:r[k] for k in (
        'frames','translation_rmse_m','maximum_translation_error_m','final_translation_error_m',
        'processing_p95_ms','screening_passed')}})
axes[0].axhline(reports[0]['gates']['maximum_translation_error_m'],c='red',ls='--',label='Fixed max-error gate')
axes[0].set_ylabel('Position error (m)');axes[0].legend(fontsize=6)
axes[1].set_ylabel('Processing time (ms)')
for ax in axes:ax.set_xlabel('Simulation time (s)');ax.grid(alpha=.2)
fig.suptitle('Virtual LiDAR replay | first-pose alignment | no physical localization claim')
fig.tight_layout();fig.savefig(args.out/'comparison.png',dpi=160)
(args.out/'summary.json').write_text(json.dumps(summary,indent=2))
