"""Evaluator-only task loading and explicit Habitat/Isaac coordinates."""
import gzip
import json
import math
import re
import numpy as np

HABITAT_TO_ISAAC=np.array([[1.,0.,0.],[0.,0.,-1.],[0.,1.,0.]])


def habitat_to_isaac(position):
    return (HABITAT_TO_ISAAC@np.asarray(position,dtype=float)).tolist()


def isaac_to_habitat(position):
    return (HABITAT_TO_ISAAC.T@np.asarray(position,dtype=float)).tolist()


def spawn_yaw(rotation_xyzw):
    # Habitat's agent looks along local -Z; Go2 looks along local +X.
    x,y,z,w=np.asarray(rotation_xyzw,dtype=float)/np.linalg.norm(rotation_xyzw)
    forward=np.array([-2*(x*z+y*w),-2*(y*z-x*w),-(1-2*(x*x+y*y))])
    direction=HABITAT_TO_ISAAC@forward
    return math.atan2(direction[1],direction[0])


def load_episode(root,scene,episode_id):
    if not re.fullmatch(r'[A-Za-z0-9]{11}',scene):raise ValueError('Invalid MP3D scene id')
    path=root/'assets/benchmarks/tasks/R2R_VLNCE_v1-3/val_unseen/val_unseen.json.gz'
    with gzip.open(path,'rt') as f:episodes=json.load(f)['episodes']
    episode=next((e for e in episodes if str(e['episode_id'])==str(episode_id)),None)
    if episode is None or episode['scene_id'].split('/')[1]!=scene:
        raise ValueError('Episode does not belong to requested scene')
    spawn=habitat_to_isaac(episode['start_position']);spawn[2]+=.4
    return dict(episode=episode,spawn=spawn,yaw=spawn_yaw(episode['start_rotation']),
                instruction=episode['instruction']['instruction_text'],task_file=str(path))
