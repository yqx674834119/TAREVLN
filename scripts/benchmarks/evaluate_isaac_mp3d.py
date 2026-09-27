"""Privileged, post-run R2R-CE scoring. Never imported by the navigation policy.

Uses the supplied Habitat navmesh solely for geodesic evaluation. Isaac's
Go2 embodiment and sensors differ from the official VLN-CE protocol.
"""
import argparse
import gzip
import hashlib
import json
import math
from pathlib import Path
import sys
import habitat_sim
import numpy as np
from metrics import evaluate

PROJECT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(PROJECT/'sim'))
from mp3d_episode import isaac_to_habitat


def score(run):
    report=json.loads((run/'report.json').read_text())
    if report.get('mode')!='sysnav_full_mp3d':raise ValueError('Not an Isaac MP3D run')
    episode=json.loads((run/'source/episode.json').read_text())['episode']
    assets=PROJECT.parent/'assets/benchmarks'
    mesh=assets/'scene_datasets/mp3d'/report['scene']/(report['scene']+'.navmesh')
    finder=habitat_sim.PathFinder()
    if not finder.load_nav_mesh(str(mesh)):raise RuntimeError('Cannot load official navmesh')
    def distance(a,b):
        path=habitat_sim.ShortestPath();path.requested_start=a;path.requested_end=b
        finder.find_path(path)
        return float(path.geodesic_distance)
    goal=episode['goals'][0]['position']
    initial=distance(episode['start_position'],goal)
    if not math.isfinite(initial):raise RuntimeError('Original episode is unreachable on supplied navmesh')
    rows=[json.loads(s) for s in (run/'trajectory.jsonl').read_text().splitlines()]
    if not rows:raise ValueError('No trajectory')
    positions=[episode['start_position']];distances=[initial];audit=[]
    for row in rows:
        base=np.asarray(isaac_to_habitat(row['position']))
        foot=np.asarray(finder.snap_point(base))
        horizontal=float(np.linalg.norm((foot-base)[[0,2]]))
        vertical=float(abs(foot[1]-base[1]))
        # Refuse to silently project across a wall, scan hole, or another floor.
        valid=bool(np.isfinite(foot).all() and horizontal<=.35 and vertical<=.65)
        audit.append(dict(time=row['time'],base=base.tolist(),foot=foot.tolist() if np.isfinite(foot).all() else None,
                          horizontal_projection_m=horizontal if math.isfinite(horizontal) else None,
                          vertical_projection_m=vertical if math.isfinite(vertical) else None,valid=valid))
        positions.append(foot.tolist());distances.append(distance(foot,goal) if valid else float('inf'))
    projection_valid=all(r['valid'] for r in audit) and all(math.isfinite(d) for d in distances)
    completed=report['status'] in ('timeout','completed','false_positive_stop','stopped_pending_evaluation','fallen')
    spawn_valid=report.get('spawn_preflight_passed') is True
    valid=projection_valid and completed and spawn_valid
    result=dict(protocol=report['protocol'],episode_id=report['episode_id'],scene=report['scene'],
        original_instruction=episode['instruction']['instruction_text'],valid=valid,
        projection_valid=projection_valid,protocol_completed=completed,spawn_preflight_passed=spawn_valid,
        native_stop=bool(report.get('native_stop')),termination=report['status'],
        dataset_geodesic_m=episode['info']['geodesic_distance'],initial_geodesic_m=initial,
        navmesh_sha256=hashlib.sha256(mesh.read_bytes()).hexdigest(),
        projection_failures=sum(not r['valid'] for r in audit),
        maximum_horizontal_projection_m=max((r['horizontal_projection_m'] for r in audit if r['horizontal_projection_m'] is not None),default=None),
        maximum_vertical_projection_m=max((r['vertical_projection_m'] for r in audit if r['vertical_projection_m'] is not None),default=None),
        metric_path='all 10 Hz base samples projected onto supplied navmesh; no policy access',
        official_protocol_comparable=False,final_geodesic_m=distances[-1] if math.isfinite(distances[-1]) else None)
    if projection_valid:
        gtfile=assets/'tasks/R2R_VLNCE_v1-3_preprocessed/val_unseen/val_unseen_gt.json.gz'
        with gzip.open(gtfile,'rt') as f:gt=json.load(f)[str(report['episode_id'])]['locations']
        metrics=evaluate(positions,distances,result['native_stop'],initial,gt)
        if valid:result.update(metrics)
        else:result['diagnostic_metrics_only']=metrics
    (run/'benchmark_projection.json').write_text(json.dumps(audit,indent=2))
    (run/'benchmark_result.json').write_text(json.dumps(result,indent=2))
    report.update(evaluation_pending=False,benchmark_result=result,success=bool(valid and result.get('sr')))
    if report['status']=='stopped_pending_evaluation':
        report['status']='completed' if report['success'] else 'false_positive_stop'
    report['acceptance_passed']=bool(report['success'] and report.get('contact_quality_passed'))
    (run/'report.json').write_text(json.dumps(report,indent=2))
    return result


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('run',type=Path)
    print(json.dumps(score(ap.parse_args().run),indent=2))
