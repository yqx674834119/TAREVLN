"""Evaluator-only, time-aligned diagnostics after a system-parity run.

Never imported by navigation. Reads archived messages instead of filling in
unobserved room/door state or treating API replies as successful execution.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))


def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def analyze(run):
    report=json.loads((run/'report.json').read_text())
    events=rows(run/'agent/events.jsonl')
    trace=rows(run/'trajectory.jsonl')
    index=rows(run/'agent/planning/index.jsonl')
    odometry=rows(run/'odometry/poses.jsonl')
    # Use the evaluator's saved aligned estimate, not the true pose, so this
    # checks the adapter contract independently of GICP estimation accuracy.
    transform=np.asarray(report['evaluation_world_from_odometry'])
    odom_times=np.array([r['time'] for r in odometry])
    errors=[]
    for row in index:
        t=row['time'];i=np.searchsorted(odom_times,t+1e-7,side='right')-1
        if i<0:continue
        base=np.linalg.inv(transform)@np.asarray(odometry[i]['aligned_estimate'])
        expected=base[:3,3]+base[:3,:3]@np.array([0.,0.,.45])
        errors.append(np.linalg.norm(np.array(row['robot']['position'])-expected))
    queries=[r for r in events if r['topic']=='/room_navigation_query']
    nonempty=[r for r in queries if r.get('query',{}).get('rooms')]
    room_snapshots=[r for r in events if r['topic']=='/room_nodes_list']
    answers=[r for r in events if r['topic']=='/room_navigation_answer']
    first_followup=None
    if answers:
        first=answers[0]
        until=next((r['time'] for r in answers[1:] if r['room_id']!=first['room_id']),float('inf'))
        follow=[r for r in index if first['time']<=r['time']<until]
        if follow:
            distances=[float(np.linalg.norm(np.array(r['robot']['position'][:2])-first['anchor'][:2])) for r in follow]
            first_followup=dict(room_id=first['room_id'],answer_type=first['answer_type'],answer_time=first['time'],
                last_time=follow[-1]['time'],initial_anchor_distance_m=distances[0],
                minimum_anchor_distance_m=min(distances),final_anchor_distance_m=distances[-1],
                note='Distance to the returned observation anchor, not an oracle goal or proof that native transit was activated.')
    positions=np.array([r['position'] for r in trace])
    definition=json.loads((run/'source/sim/fixtures/sysnav-object-evaluation.json').read_text())['tasks'][report['task']]
    entries=[]
    for target in definition['targets']:
        rect=target.get('room_rectangle')
        if rect is None:continue
        inside=(positions[:,0]>=rect[0])&(positions[:,1]>=rect[1])&(positions[:,0]<=rect[2])&(positions[:,1]<=rect[3])
        first=np.flatnonzero(inside)
        entries.append(dict(target=target['prim'],samples=int(inside.sum()),first_time=trace[first[0]]['time'] if len(first) else None))
    root_sha={path:hashlib.sha256((run/'source'/path).read_bytes()).hexdigest()
              for path in ('sim/sysnav_sensor_bridge.py','sim/sysnav_full_worker.py','src/vln_core/sysnav_sensor_contract.py','config/sysnav-full-ros.yaml')}
    result={k:report.get(k) for k in ('status','success','acceptance_passed','simulation_seconds','distance_m','final_position','contact_quality_passed','localization_evaluation')}
    result.update(sensor_origin_check=dict(samples=len(errors),maximum_error_m=float(max(errors)) if errors else None,
                    passed=bool(errors and max(errors)<1e-6)),
                  room_queries=dict(total=len(queries),empty=len(queries)-len(nonempty),
                    first_nonempty_time=nonempty[0]['time'] if nonempty else None,
                    candidate_ids=[dict(time=q['time'],ids=[r['room id'] for r in q['query']['rooms']]) for q in nonempty]),
                  room_answers=answers,first_selection_followup=first_followup,
                  room_snapshots=dict(count=len(room_snapshots),last=room_snapshots[-1] if room_snapshots else None),
                  target_room_entries=entries,observed_topics=dict(Counter(r['topic'] for r in events)),
                  checked_archive_sha256=root_sha,
                  interpretation='Single-run diagnostics; system repairs were bundled, so no per-fix causal attribution.')
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run',type=Path)
    args=parser.parse_args()
    result=analyze(args.run)
    out=args.run/'review';out.mkdir(exist_ok=True)
    (out/'system-diagnostics.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))
