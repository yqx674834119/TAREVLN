"""Sequential frozen-source simulation gate; retains every attempted episode."""
import argparse
import hashlib
import itertools
import json
import os
from pathlib import Path
import subprocess

ap=argparse.ArgumentParser()
ap.add_argument('--kind',choices=('gait','office'),required=True)
ap.add_argument('--checkpoint',type=Path,required=True)
ap.add_argument('--out',type=Path,required=True)
ap.add_argument('--ground-torsion-radius',type=float)
ap.add_argument('--ground-contact-offset',type=float)
ap.add_argument('--gait-gpu',type=int,default=0)
ap.add_argument('--sam',action='store_true')
ap.add_argument('--api-proxy')
ap.add_argument('--odometry',choices=('ideal','gicp'),default='ideal')
ap.add_argument('--lidar-profile',choices=('legacy11','uniform64'),default='legacy11')
args=ap.parse_args()
root=Path(__file__).resolve().parents[1]
args.checkpoint=args.checkpoint.resolve()
args.out=args.out.resolve()
args.out.mkdir(parents=True,exist_ok=False)
config=root/'config/readiness'/('gait-gates.json' if args.kind=='gait' else 'office-suite.json')
cfg=json.loads(config.read_text())
if args.kind=='gait' and args.ground_torsion_radius is None:
    args.ground_torsion_radius=cfg['nominal_ground_torsion_radius_m']
source=[p for directory in ('src','sim','config','patches') for p in (root/directory).rglob('*')
        if p.is_file() and '__pycache__' not in p.parts and p.suffix not in ('.pyc',)]
source.extend([Path(__file__),args.checkpoint])
def manifest():
    return {str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in source}
frozen=manifest()
(args.out/'freeze.json').write_text(json.dumps({'files':frozen,'config':cfg,'kind':args.kind,
    'ground_torsion_radius':args.ground_torsion_radius,'ground_contact_offset':args.ground_contact_offset,
    'odometry':args.odometry,'lidar_profile':args.lidar_profile,'sam':args.sam,'api_proxy':args.api_proxy,
    'gait_gpu':args.gait_gpu},indent=2))
results=[]
if args.kind=='gait':
    cases=list(itertools.product(cfg['initial_yaws_rad'],cfg['threshold_heights_m']))
else:
    cases=list(itertools.product(cfg['tasks'],cfg['spawns_xyz'],cfg['yaws_rad']))
for i,case in enumerate(cases):
    if manifest()!=frozen:raise RuntimeError('Frozen files changed during suite; results are incomplete')
    name=f'{args.out.name}-{i:02d}'
    if args.kind=='gait':
        yaw,threshold=case
        run=args.out/name
        command=['sim/diagnose_go2.py','--yaw',str(yaw),'--threshold',str(threshold),'--out',str(run),'--gpu',str(args.gait_gpu)]
        if args.ground_torsion_radius is not None:
            command+=['--ground-torsion-radius',str(args.ground_torsion_radius)]
        if args.ground_contact_offset is not None:
            command+=['--ground-contact-offset',str(args.ground_contact_offset)]
    else:
        task,spawn,yaw=case
        run=root/'data/sim/office/runs'/name
        command=['sim/run_office.py','--name',name,'--task',task,'--spawn',*map(str,spawn),
                 '--yaw',str(yaw),'--max-seconds',str(cfg['max_seconds']),
                 '--odometry',args.odometry,'--lidar-profile',args.lidar_profile]
        if args.sam:command+=['--sam']
        if args.api_proxy:command+=['--api-proxy',args.api_proxy]
    command=['bash','scripts/with_gpu_compat.sh',str(root.parent/'envs/isaac51/bin/python'),'-u',*command,
             '--checkpoint',str(args.checkpoint)]
    with (args.out/f'{name}.log').open('w') as log:
        process=subprocess.Popen(command,cwd=root,env={**os.environ,'OMNI_KIT_ACCEPT_EULA':'YES'},
                                 stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
        # No second simulator is launched until this one has fully exited.
        returncode=process.wait()
    report=json.loads((run/'report.json').read_text()) if (run/'report.json').exists() else {'status':'missing_report'}
    results.append({'case':case,'run':str(run),'returncode':returncode,'report':report})
    (args.out/'episodes.json').write_text(json.dumps(results,indent=2))
    print(name,report.get('status'),report.get('passed',report.get('success')),flush=True)
    if returncode or report.get('status') in ('missing_report','error','interrupted'):
        raise RuntimeError('Infrastructure failure; suite incomplete, all evidence retained')
if args.kind=='gait':
    passed=all(r['report'].get('passed') is True for r in results)
else:
    reports=[r['report'] for r in results]
    passed=(len(reports)>=cfg['minimum_episodes'] and sum(r.get('success') is True for r in reports)>=cfg['minimum_successes']
            and sum(r.get('status')=='false_positive_stop' for r in reports)<=cfg['maximum_false_positive_stops']
            and sum(r.get('status')=='fallen' for r in reports)<=cfg['maximum_falls']
            and sum(r.get('contact_quality_passed') is not True for r in reports)<=cfg['maximum_nonfoot_physical_contact_episodes'])
summary={'kind':args.kind,'completed':len(results),'planned':len(cases),'passed':passed,'source_unchanged':manifest()==frozen}
summary['passed']=summary['passed'] and summary['source_unchanged']
(args.out/'summary.json').write_text(json.dumps(summary,indent=2))
print(json.dumps(summary),flush=True)
raise SystemExit(0 if summary['passed'] else 1)
