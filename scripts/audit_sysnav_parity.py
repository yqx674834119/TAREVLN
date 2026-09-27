"""Fail on unreviewed source/config drift; optionally verify a live run contract."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import yaml
from build_sysnav_parity_config import BASE, generate

ROOT=Path(__file__).resolve().parents[1]
TARE='src/exploration_planner/tare_planner/'
SEM='src/semantic_mapping/semantic_mapping/'
VLM='src/vlm_node/vlm_node/'
REASONS={
    TARE+'include/representation/representation.h':'Humble cv_bridge header compatibility only',
    TARE+'include/room_segmentation/room_segmentation_node.h':'Humble cv_bridge header compatibility only',
    TARE+'include/sensor_coverage_planner/sensor_coverage_planner_ground.h':'Humble cv_bridge header compatibility only',
    TARE+'src/grid_world/grid_world.cpp':'Empty/out-of-bounds mask memory guards; retained corrective patch',
    TARE+'src/local_coverage_planner/local_coverage_planner.cpp':'Shrinking random-sample queue bounds correction; retained upstream-bug correction, not bitwise equivalence',
    TARE+'src/sensor_coverage_planner/sensor_coverage_planner_ground.cpp':'Calibrated camera extrinsics, zero-clock startup guard and read-only status; official 12 m initial waypoint restored',
    TARE+'src/viewpoint_manager/viewpoint_manager.cpp':'Empty room-mask startup guard only',
    SEM+'cloud_image_fusion.py':'Actual camera extrinsic and reject out-of-FOV scan pixels',
    SEM+'detection_node.py':'Unique ROS node name, portable YOLOE backend, BGR input contract and preserve evidence',
    SEM+'semantic_map_new.py':'Correct original ObjectNodeList/bool/array ROS message contract; remove unused import',
    SEM+'semantic_mapping_node.py':'SAM asset path, RGB input contract, numeric message cast and preserve evidence',
    SEM+'utils.py':'Remove unused spaCy import',
    VLM+'constants.py':'Requested DeepSeek API provider',
    VLM+'deepseek_client.py':'Requested DeepSeek structured transport, budget and evidence log',
    VLM+'label_contract.py':'Normalize separator/case variants to actual detector identifiers; preserve unsupported names',
    VLM+'utils.py':'Calibrated simulated camera extrinsic',
    VLM+'vlm_reasoning_node.py':'Provider/identifier/message compatibility; optional early-stop dispatch default false preserves pinned disabled behavior, enabled only by an explicit experiment flag'
}
# These keys exist in upstream YAML but its nodes never declare/read them.
# Record the limitation; every other configured parameter must be effective.
UPSTREAM_UNUSED={
    'tare_planner_node': {'kDoorCollisionCheckPointNumThr','kDoorCollisionCheckRadius','kKeyposeGraphCollisionCheckPointNumThr','kKeyposeGraphCollisionCheckRadius'},
    'room_segmentation': {'savePcd'},
    'localPlanner': {'goalReachedThreshold','goalYawThreshold'},
    'pathFollower': {'goalYawGain'},
    'semantic_mapping_node': {'captioner_batch_size','image_processing_interval','vis_interval','visualize'},
    'detection_node': {'captioner_batch_size','detection_angular_state_time_bias','detection_linear_state_time_bias','image_processing_interval','target_object','vis_interval','visualize'}
}


def source_diff(source):
    untracked=subprocess.check_output(['git','-C',str(source),'ls-files','--others','--exclude-standard','--','src'],text=True).splitlines()
    if untracked:raise ValueError('Unreviewed untracked source files: '+str(untracked))
    paths=subprocess.check_output(['git','-C',str(source),'diff','--name-only',BASE],text=True).splitlines()
    unknown=set(paths)-set(REASONS)
    if unknown:raise ValueError('Unreviewed source files: '+str(sorted(unknown)))
    result={}
    for path in paths:
        diff=subprocess.check_output(['git','-C',str(source),'diff','--no-ext-diff',BASE,'--',path])
        result[path]=dict(patch_sha256=hashlib.sha256(diff).hexdigest(),reason=REASONS[path])
    return result


def check_runtime(runtime,config):
    failures=[]
    for node in config:
        actual=runtime['parameters'].get(node)
        if actual is None:
            failures.append('Missing live parameters: '+node)
            continue
        for key,value in config[node]['ros__parameters'].items():
            if key not in actual and key not in UPSTREAM_UNUSED.get(node,set()):
                failures.append(f'Missing effective parameter: {node}.{key}')
            elif key in actual and actual[key]!=value:
                failures.append(f'{node}.{key}: {actual[key]!r} != {value!r}')
    expected={'/state_estimation_at_scan':'sensor_scan','/sensor_scan':'sensor_scan',
              '/way_point':'tare_planner_node',
              '/path':'localPlanner','/cmd_vel':'pathFollower','/room_nodes_list':'room_segmentation'}
    expected={topic:{node} for topic,node in expected.items()}
    # Upstream room_segmentation advertises this topic but never publishes it.
    # Preserve that source behavior; navigationBoundary supplies the actual polygon.
    expected['/navigation_boundary']={'navigationBoundary','room_segmentation'}
    for topic,publishers in expected.items():
        pubs=runtime['topics'].get(topic,{}).get('publishers',[])
        if {p['node'] for p in pubs}!=publishers or len(pubs)!=len(publishers):
            failures.append(f'{topic} expected registered publishers {publishers}: {pubs}')
    for proc in runtime['processes']:
        if not proc['running']:failures.append('Node exited: '+proc['executable'])
        if any('/deps/sysnav/' in p for p in proc['mapped_project_files']):
            failures.append('Legacy workspace library loaded: '+proc['executable'])
    failures.extend(runtime.get('parameter_errors',[]))
    return failures


def check_instruction_contract(events,labels):
    instructions=[r for r in events if r['topic']=='/target_object_instruction' and r.get('target')]
    if not instructions:return ['No parsed task instruction recorded']
    failures=[]
    for row in instructions:
        for key in ('target','anchor'):
            value=row.get(key)
            if value and (value=='unknown' or value not in labels):
                failures.append(f'Instruction {key} does not match a detector identifier: {value!r}')
    return failures


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--source',type=Path,default=ROOT.parent/'deps/sysnav-full')
    ap.add_argument('--approve-reviewed-lock',action='store_true')
    ap.add_argument('--run',type=Path)
    ap.add_argument('--out',type=Path)
    args=ap.parse_args()
    differences=source_diff(args.source)
    lock=ROOT/'config/sysnav-source-review.json'
    if args.approve_reviewed_lock:
        lock.write_text(json.dumps(dict(upstream_revision=BASE,reviewed_changes=differences),indent=2)+'\n')
    if differences!=json.loads(lock.read_text())['reviewed_changes']:
        raise SystemExit('Source differs from the reviewed compatibility/calibration patch')
    config,manifest=generate(args.source,json.loads((ROOT/'config/sysnav-isaac-overrides.json').read_text()))
    if yaml.safe_load((ROOT/'config/sysnav-full-ros.yaml').read_text())!=config:
        raise SystemExit('Official launch/config inheritance has drifted')
    unchanged=[]
    for subtree in ('room_segmentation','representation','keypose_graph','planning_env','rolling_occupancy_grid','tsp_solver'):
        directory=TARE+'src/'+subtree
        if not subprocess.check_output(['git','-C',str(args.source),'diff','--name-only',BASE,'--',directory]).strip():
            unchanged.append(directory)
    failures=[]
    if args.run:
        report=json.loads((args.run/'report.json').read_text())
        if 'room_early_stop_enabled' in report:
            config['vlm_node']['ros__parameters']['enable_room_early_stop']=report['room_early_stop_enabled']
        failures=check_runtime(json.loads((args.run/'agent/runtime-contract.json').read_text()),config)
        vocabulary=yaml.safe_load((args.source/'src/semantic_mapping/semantic_mapping/config/objects.yaml').read_text())
        labels={p for item in vocabulary['prompts'].values() for p in item['prompts']}
        events=[json.loads(s) for s in (args.run/'agent/events.jsonl').read_text().splitlines()]
        failures.extend(check_instruction_contract(events,labels))
    result=dict(upstream_revision=BASE,source_review_passed=True,configuration_inheritance_passed=True,
                reviewed_changed_files=len(differences),byte_identical_source_directories=unchanged,
                runtime_checked=bool(args.run),runtime_contract_passed=not failures if args.run else None,
                runtime_failures=failures,claim='No unreviewed source or parameter drift; NOT a claim of bitwise algorithm or simulator equivalence')
    result['upstream_yaml_keys_not_effective']={k:sorted(v) for k,v in UPSTREAM_UNUSED.items()}
    if args.run:result['intentional_early_stop_intervention']=report.get('room_early_stop_enabled',False)
    if args.out:args.out.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))
    if failures:raise SystemExit(2)


if __name__=='__main__':main()
