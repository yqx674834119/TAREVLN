"""Policy worker: observation-only JSON input, bounded discrete actions output.

This is a process/API separation, not an OS filesystem security sandbox.
No evaluator coordinates, goals, reference paths or navigation graph are sent.
"""
import base64
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]/'src'))
from vln_core.vision_api import VisionAPI

api = VisionAPI(Path(sys.argv[1]), max_calls=130)
history = []
allowed = {'instruction', 'rgb_jpeg', 'depth_m', 'last_action', 'collision', 'step'}
for line in sys.stdin:
    try:
        observation = json.loads(line)
        if set(observation) != allowed:
            raise ValueError('Unexpected policy input schema')
        frame = base64.b64decode(observation.pop('rgb_jpeg'))
        prompt = (
            'You control a robot camera in a continuous indoor navigation task. Follow the exact route instruction '
            'using the current RGB image and observation history. You do not know the goal coordinates or map. '
            'Actions: move_forward travels 0.25 metres; turn_left and turn_right rotate 15 degrees; stop ends '
            'the task permanently. Return JSON {"action": "move_forward|turn_left|turn_right|stop", '
            '"repeat": 1 to 4, "reason": "visible evidence and route progress"}. '
            'Use repeat=1 near the destination or obstacles; repeat up to 4 on clear straight routes. '
            'If blocked, inspect a different heading. Turn enough to view route landmarks. Stop only when the '
            'instruction endpoint is visually supported. Do not assume a passed landmark is the destination. '
            'Depth values are metres for left/centre/right lower-middle image patches, not goal distance.\n'
            + json.dumps(observation) + '\nRecent decisions: ' + json.dumps(history[-20:])
        )
        result = api.ask(prompt, [frame])
        action = result.get('action')
        repeat = result.get('repeat', 1)
        if action not in {'move_forward', 'turn_left', 'turn_right', 'stop'}:
            raise ValueError('Invalid action')
        if type(repeat) is not int or not 1 <= repeat <= 4:
            raise ValueError('Invalid repeat')
        result = dict(action=action, repeat=1 if action == 'stop' else repeat,
                      reason=str(result.get('reason', ''))[:1500])
        history.append(dict(step=observation['step'], **result))
        print(json.dumps(result), flush=True)
    except Exception as e:
        print(json.dumps({'error': type(e).__name__}), flush=True)
        break
