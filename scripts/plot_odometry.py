"""Evaluator-only trajectory and localization error review artifact."""
import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ap=argparse.ArgumentParser()
ap.add_argument('directory',type=Path)
args=ap.parse_args()
rows=[json.loads(line) for line in (args.directory/'poses.jsonl').read_text().splitlines()]
rows=[r for r in rows if 'aligned_estimate' in r]
if not rows:raise RuntimeError('No aligned estimates to plot')
truth=np.asarray([r['truth_pose'] for r in rows])[:,:3,3]
estimate=np.asarray([r['aligned_estimate'] for r in rows])[:,:3,3]
times=np.asarray([r['estimate']['time'] for r in rows])
fig,axes=plt.subplots(1,3,figsize=(15,4.5))
ax=axes[0]
ax.plot(truth[:,0],truth[:,1],label='Simulator truth')
ax.plot(estimate[:,0],estimate[:,1],label='Odometry, first pose alignment')
ax.set_aspect('equal');ax.set_xlabel('World x (m)');ax.set_ylabel('World y (m)');ax.legend(fontsize=8)
axes[1].plot(times,[r['translation_error_m'] for r in rows],label='Position error (m)')
axes[1].axhline(.3,color='red',ls='--',label='0.3 m screening limit')
axes[1].set_xlabel('Simulation time (s)');axes[1].legend(fontsize=8)
axes[2].plot(times,[1000*r['estimate']['processing_seconds'] for r in rows])
axes[2].set_xlabel('Simulation time (s)');axes[2].set_ylabel('Estimator processing (ms)')
for ax in axes:ax.grid(alpha=.2)
fig.suptitle(args.directory.name+' | simulated data; not physical localization validation')
fig.tight_layout();fig.savefig(args.directory/'localization.png',dpi=160)
