"""Replay recorded Isaac geometry inputs to diagnose native planner failures.

This is a sensor replay, not an autonomous closed-loop task result.
"""
import argparse
import json
from pathlib import Path
import sys
import time
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'sim'))
from office_sandbox import OfficeProcess
ap = argparse.ArgumentParser()
ap.add_argument('run', type=Path)
ap.add_argument('output', type=Path)
args = ap.parse_args()
args.output.mkdir(parents=True, exist_ok=False)
worker = OfficeProcess(ROOT, args.output / 'agent', open(args.output / 'worker.log', 'w'), geometry_only=True)
previous = None
start_time, scans = None, 0
try:
    for line in (args.run / 'trajectory.jsonl').open():
        row = json.loads(line)
        t, pos, q = row['time'], row['position'], row['orientation']
        if start_time is None: start_time = t
        if t < 2: continue
        velocity = [0., 0., 0.] if previous is None else [(v-p)/(t-previous['time']) for v,p in zip(pos,previous['position'])]
        obs = {'time':t,'position':pos,'orientation':q,'velocity':velocity,'yaw':0.}
        step = round((t-start_time)/.005)
        scan = args.run / f'scan_{step:07d}.npz'
        if scan.exists():
            obs['points'] = np.load(scan)['points'].tolist()
            scans += 1
        response = worker.ask(obs)
        previous = row
        time.sleep(.075)
    if scans < 10: raise RuntimeError('Insufficient recorded point clouds')
    print(json.dumps({'mode':'recorded_sensor_replay','finished':True,'scans':scans,'last_time':t,'response':response}))
finally:
    worker.close()
