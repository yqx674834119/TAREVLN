"""Frozen one-scene R2R-CE evaluation port for Habitat-Sim 0.3.3.

This is not an exact reproduction of Habitat 0.1.7 or any learned paper model.
The oracle follower is a privileged simulator diagnostic, never a VLN method.
"""
import argparse
import base64
import gzip
import hashlib
import io
import json
import math
import os
from pathlib import Path
import select
import subprocess
import sys
import time

import habitat_sim
import magnum as mn
from habitat_sim.utils.common import quat_from_coeffs
import numpy as np
from PIL import Image
from metrics import evaluate

ROOT = Path(__file__).resolve().parents[3]
ASSETS = ROOT/'assets/benchmarks'


def geodesic(sim, a, b):
    path = habitat_sim.ShortestPath()
    path.requested_start = a
    path.requested_end = b
    sim.pathfinder.find_path(path)
    return float(path.geodesic_distance)


def make_sim(scene):
    cfg = habitat_sim.SimulatorConfiguration()
    cfg.scene_id = str(ASSETS/'scene_datasets/mp3d'/scene/(scene+'.glb'))
    cfg.enable_physics = False
    cfg.allow_sliding = True
    cfg.gpu_device_id = 0
    agent_cfg = habitat_sim.agent.AgentConfiguration()
    agent_cfg.height, agent_cfg.radius = 1.5, 0.1
    sensors = []
    for name, kind, resolution in [('rgb', habitat_sim.SensorType.COLOR, [224, 224]),
                                    ('depth', habitat_sim.SensorType.DEPTH, [256, 256])]:
        sensor = habitat_sim.CameraSensorSpec()
        sensor.uuid, sensor.sensor_type, sensor.resolution = name, kind, resolution
        sensor.position, sensor.hfov = mn.Vector3(0, 1.25, 0), mn.Deg(90)
        sensors.append(sensor)
    agent_cfg.sensor_specifications = sensors
    agent_cfg.action_space = {
        k: habitat_sim.agent.ActionSpec(k, habitat_sim.agent.ActuationSpec(amount=v))
        for k, v in [('move_forward', .25), ('turn_left', 15), ('turn_right', 15)]
    }
    sim = habitat_sim.Simulator(habitat_sim.Configuration(cfg, [agent_cfg]))
    mesh = ASSETS/'scene_datasets/mp3d'/scene/(scene+'.navmesh')
    if not sim.pathfinder.load_nav_mesh(str(mesh)):
        raise RuntimeError('Official supplied navmesh could not be loaded')
    return sim


def jpeg(obs):
    stream = io.BytesIO()
    Image.fromarray(obs['rgb'][:, :, :3]).save(stream, format='JPEG', quality=90)
    return stream.getvalue()


def publish_live(out, obs, positions, step, action):
    image_path = out/'live.jpg.tmp'
    image_path.write_bytes(jpeg(obs))
    image_path.replace(out/'live.jpg')
    path = out/'live.json.tmp'
    path.write_text(json.dumps(dict(step=step, action=action, positions=positions)))
    path.replace(out/'live.json')


def call_policy(worker, episode, obs, step, last_action, collision):
    depth = obs['depth']
    values = []
    for x1, x2 in [(30, 85), (100, 156), (171, 226)]:
        patch = depth[100:180, x1:x2]
        valid = patch[np.isfinite(patch) & (patch > 0)]
        values.append(float(np.percentile(valid, 20)) if len(valid) else None)
    message = dict(instruction=episode['instruction']['instruction_text'],
                   rgb_jpeg=base64.b64encode(jpeg(obs)).decode(), depth_m=values,
                   last_action=last_action, collision=collision, step=step)
    worker.stdin.write(json.dumps(message)+'\n')
    worker.stdin.flush()
    ready, _, _ = select.select([worker.stdout], [], [], 150)
    if not ready:
        raise TimeoutError('Policy response timed out')
    result = json.loads(worker.stdout.readline())
    if 'error' in result:
        raise RuntimeError('Policy worker failed: '+result['error'])
    if result.get('action') not in {'stop', 'move_forward', 'turn_left', 'turn_right'}:
        raise ValueError('Invalid action')
    if type(result.get('repeat')) is not int or not 1 <= result['repeat'] <= 4:
        raise ValueError('Invalid repeat')
    return result


def run_episode(sim, ep, policy, args, truth):
    out = args.output/policy/str(ep['episode_id'])
    out.mkdir(parents=True, exist_ok=False)
    start = habitat_sim.AgentState()
    start.position = np.asarray(ep['start_position'], dtype=np.float32)
    start.rotation = quat_from_coeffs(ep['start_rotation'])
    sim.initialize_agent(0, start)
    sim.seed(args.seed)
    goal = np.asarray(ep['goals'][0]['position'], dtype=np.float32)
    initial = geodesic(sim, start.position, goal)
    if not math.isfinite(initial):
        raise RuntimeError('Unreachable official task endpoint')
    preflight = dict(start_navigable=bool(sim.pathfinder.is_navigable(start.position)),
                     goal_navigable=bool(sim.pathfinder.is_navigable(goal)),
                     computed_geodesic_m=initial,
                     dataset_geodesic_m=ep['info']['geodesic_distance'],
                     geodesic_delta_m=initial-ep['info']['geodesic_distance'])
    (out/'preflight.json').write_text(json.dumps(preflight, indent=2))
    obs = sim.get_sensor_observations()
    (out/'start.jpg').write_bytes(jpeg(obs))
    if not np.isfinite(obs['depth']).all() or np.std(obs['rgb'][:, :, :3]) < 1:
        raise RuntimeError('Blank or invalid sensor frame')
    positions, distances, events = [start.position.tolist()], [initial], []
    rng = np.random.RandomState(args.seed+int(ep['episode_id']))
    turns = int(rng.randint(0, 25))
    follower = sim.make_greedy_follower(goal_radius=2.8, stop_key='stop',
                    forward_key='move_forward', left_key='turn_left', right_key='turn_right')
    worker, worker_log = None, None
    if policy == 'vlm':
        worker_log = (out/'worker.stderr').open('w')
        worker = subprocess.Popen([sys.executable, str(Path(__file__).with_name('vlm_worker.py')),
                                   str(out/'policy')], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                  stderr=worker_log, text=True, bufsize=1, cwd=out)
    stopped, collision, last_action, decisions = False, False, None, 0
    termination, error, step = 'max_steps', None, 0
    t0 = time.time()
    publish_live(out, obs, positions, step, 'initializing')
    try:
        while step < args.max_steps:
            if (args.output/'STOP').exists():
                termination = 'operator_stopped'
                break
            if policy == 'stop':
                action, repeat = 'stop', 1
            elif policy == 'random':
                action, repeat = str(rng.choice(['stop','move_forward','turn_left','turn_right'], p=[.02,.68,.15,.15])), 1
            elif policy == 'handcrafted':
                action, repeat = ('turn_right' if step < turns else 'move_forward' if step < turns+37 else 'stop'), 1
            elif policy == 'oracle':
                action, repeat = follower.next_action_along(goal), 1
            else:
                decision = call_policy(worker, ep, obs, step, last_action, collision)
                action, repeat = decision['action'], decision['repeat']
                decisions += 1
                with (out/'decisions.jsonl').open('a') as f:
                    f.write(json.dumps(dict(step=step, **decision))+'\n')
                (out/f'view-{decisions:03d}.jpg').write_bytes(jpeg(obs))
            for _ in range(min(repeat, args.max_steps-step)):
                step += 1
                if action == 'stop':
                    stopped, termination = True, 'stop'
                    events.append(dict(step=step, action=action, collision=False))
                    break
                obs = sim.step(action)
                collision = bool(obs.get('collided', False))
                position = sim.get_agent(0).get_state().position.tolist()
                positions.append(position)
                distances.append(geodesic(sim, position, goal))
                events.append(dict(step=step, action=action, collision=collision))
                last_action = action
                publish_live(out, obs, positions, step, action)
            if stopped:
                break
    except Exception as e:
        termination, error = 'runtime_error', type(e).__name__+': '+str(e)[:160]
    finally:
        if worker:
            worker.stdin.close()
            try:
                worker.wait(timeout=5)
            except subprocess.TimeoutExpired:
                worker.terminate()
                worker.wait(timeout=5)
            worker_log.close()
    (out/'end.jpg').write_bytes(jpeg(obs))
    np.save(out/'end-depth.npy', obs['depth'])
    gt = truth.get(str(ep['episode_id']), {}).get('locations')
    result = dict(episode_id=ep['episode_id'], trajectory_id=ep['trajectory_id'],
                  policy=policy, instruction=ep['instruction']['instruction_text'],
                  **evaluate(positions, distances, stopped, initial, gt),
                  steps=step, decisions=decisions, collisions=sum(x['collision'] for x in events),
                  termination=termination, error=error, elapsed_s=time.time()-t0,
                  privileged=policy=='oracle', preflight=preflight)
    (out/'trajectory.json').write_text(json.dumps(dict(positions=positions, goal_distances=distances, actions=events)))
    (out/'result.json').write_text(json.dumps(result, indent=2))
    print(json.dumps({k:result[k] for k in ['episode_id','policy','sr','ne_m','steps','termination']}), flush=True)
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--scene', default='x8F5xyUWy9e')
    p.add_argument('--policies', nargs='+', choices=['stop','random','handcrafted','oracle','vlm'], default=['stop','random','handcrafted','oracle'])
    p.add_argument('--limit', type=int, default=1, help='0 means all scene episodes')
    p.add_argument('--offset', type=int, default=0, help='Deterministic offset in sorted episode IDs')
    p.add_argument('--episode-id', type=int, help='Select one exact official episode (interactive mode)')
    p.add_argument('--max-steps', type=int, default=500)
    p.add_argument('--seed', type=int, default=20260928)
    p.add_argument('--output', required=True, type=Path)
    args = p.parse_args()
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=False)
    tasks = ASSETS/'tasks/R2R_VLNCE_v1-3/val_unseen/val_unseen.json.gz'
    all_eps = json.load(gzip.open(tasks, 'rt'))['episodes']
    eps = sorted([e for e in all_eps if e['scene_id'].split('/')[1] == args.scene], key=lambda e:int(e['episode_id']))
    if args.episode_id is not None:
        eps = [e for e in eps if int(e['episode_id'])==args.episode_id]
    eps = eps[args.offset:]
    if args.limit:
        eps = eps[:args.limit]
    if not eps:
        raise ValueError('No matching episodes')
    truth_path = ASSETS/'tasks/R2R_VLNCE_v1-3_preprocessed/val_unseen/val_unseen_gt.json.gz'
    truth = json.load(gzip.open(truth_path, 'rt')) if truth_path.exists() else {}
    manifest = dict(scene=args.scene, split='val_unseen', task_version='R2R_VLNCE_v1-3',
        episodes=[e['episode_id'] for e in eps], selection='smallest scene by bytes; episode_id ascending; fixed before outcomes',
        habitat_sim_version=habitat_sim.__version__, seed=args.seed, max_steps=args.max_steps,
        policies=args.policies, forward_m=.25, turn_degrees=15, rgb=[224,224], depth=[256,256],
        vlm_call_limit=130, vlm_model='deepseek-flash', interactive_episode_id=args.episode_id,
        sensor_height_m=1.25, hfov_degrees=90, radius_m=.1, height_m=1.5, allow_sliding=True,
        success='STOP and final geodesic distance < 3.0m',
        task_sha256=hashlib.sha256(tasks.read_bytes()).hexdigest(),
        dense_gt_sha256=hashlib.sha256(truth_path.read_bytes()).hexdigest() if truth else None,
        limitations=['Habitat-Sim 0.3.3 port, not original 0.1.7 reproduction',
                      'Kinematic navmesh actions, not Go2 physics or TARE',
                      'Policy API/process separation, not filesystem security sandbox',
                      'VLM depth summaries in metric units; not raw normalized depth tensor'])
    (args.output/'manifest.json').write_text(json.dumps(manifest, indent=2))
    source = args.output/'source'
    source.mkdir()
    import shutil
    snapshots = [Path(__file__), Path(__file__).with_name('metrics.py'),
                 Path(__file__).with_name('vlm_worker.py'),
                 ROOT/'cross-building-floor-nav/src/vln_core/vision_api.py']
    for path in snapshots:
        shutil.copyfile(path,source/path.name)
    sim = make_sim(args.scene)
    results = []
    try:
        for policy in args.policies:
            for ep in eps:
                results.append(run_episode(sim, ep, policy, args, truth))
                (args.output/'results.json').write_text(json.dumps(results, indent=2))
    finally:
        sim.close()
    summary = {}
    for policy in args.policies:
        rows = [r for r in results if r['policy']==policy]
        keys = ['ne_m','sr','osr','spl','path_length_m','steps','ndtw_exact','sdtw_exact','ndtw_fast','sdtw_fast']
        summary[policy] = dict(n=len(rows), errors=sum(r['error'] is not None for r in rows),
             **{k:float(np.mean([r[k] for r in rows])) for k in keys if all(k in r for r in rows)})
    (args.output/'summary.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    main()
