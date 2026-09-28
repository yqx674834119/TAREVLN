"""Owned simulation process followed by an isolated, privileged scoring process."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

ap=argparse.ArgumentParser()
ap.add_argument('--name',required=True)
ap.add_argument('--episode-id',required=True,type=int)
ap.add_argument('--max-seconds',type=int,default=300)
ap.add_argument('--gui',action='store_true')
args=ap.parse_args()
project=Path(__file__).resolve().parents[2];root=project.parent
preflight=[str(root/'envs/isaac51/bin/python'),'-u','sim/check_mp3d_isaac.py','--episode-id',str(args.episode_id)]
# A floor ray does not establish that a quadruped fits an original Habitat start.
# Always run the five-second contact/standing check before starting the policy.
preflight_code=subprocess.run(preflight,cwd=project).returncode
if preflight_code:
    run=project/'data/sim/office/runs'/args.name
    run.mkdir(parents=True,exist_ok=False)
    report=dict(mode='sysnav_full_mp3d',episode_id=args.episode_id,scene='x8F5xyUWy9e',
        task='mp3d:'+str(args.episode_id),status='invalid_spawn',success=False,
        spawn_preflight_passed=False,evaluation_pending=False,acceptance_passed=False,
        error='Original episode start failed the Go2 contact/standing preflight')
    (run/'report.json').write_text(json.dumps(report,indent=2))
    sys.exit(preflight_code)
command=[str(root/'envs/isaac51/bin/python'),'-u','sim/run_sysnav_full.py',
    '--name',args.name,'--mp3d-scene','x8F5xyUWy9e','--mp3d-episode',str(args.episode_id),
    '--max-seconds',str(args.max_seconds),'--odometry','gicp','--lidar-profile','uniform64',
    '--api-proxy','http://127.0.0.1:18119','--api-budget','-1' if args.max_seconds==-1 else '600',
    '--checkpoint',str(project/'data/readiness/training/go2-control-v5-rails/model_3595.pt')]
if args.gui: command+=['--gui']
code=subprocess.run(command,cwd=project).returncode
run=project/'data/sim/office/runs'/args.name
if (run/'trajectory.jsonl').exists() and (run/'trajectory.jsonl').stat().st_size:
    code2=subprocess.run([str(root/'envs/habitat-bench/bin/python'),
        'scripts/benchmarks/evaluate_isaac_mp3d.py',str(run)],cwd=project).returncode
    code=code or code2
elif (run/'report.json').exists():
    report=json.loads((run/'report.json').read_text())
    report.update(evaluation_pending=False,evaluation_error='No recorded trajectory; navigation was not evaluated')
    (run/'report.json').write_text(json.dumps(report,indent=2))
    code=code or 2
sys.exit(code)
