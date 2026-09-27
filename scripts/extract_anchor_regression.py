"""Extract fixed, visually reviewed RGB-D patches; no oracle/scene lookup."""
import base64
import hashlib
import json
from pathlib import Path
import zlib
import numpy as np

root = Path(__file__).resolve().parents[1]
run = root / "data/sim/office/runs/office-lounge-005"
spec = [(9200, 420, 300, "first_tabletop", True),
        (9200, 420, 245, "first_wall_above_table", False),
        (9200, 420, 370, "first_wall_below_table", False),
        (9200, 190, 445, "first_floor", False),
        (19600, 265, 276, "second_tabletop", True),
        (19600, 320, 240, "second_accepted_wrong_wall", False),
        (19600, 285, 340, "second_wall_below_table", False),
        (19600, 300, 410, "second_floor", False)]
samples = []
for step, u, v, label, expected in spec:
    source = run / f"depth_{step:07d}_0.npz"
    frame = np.load(source)
    a = frame["depth"][v-16:v+17, u-16:u+17]
    mm = (np.clip(np.nan_to_num(a, nan=0., posinf=0., neginf=0.),0,65.535)*1000).astype('<u2')
    w,x,y,z = frame["quaternion"].tolist()
    rotation = [[1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w)],
                [2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w)],
                [2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)]]
    position = frame["position"].tolist()
    samples.append({"name": label, "expected_valid": expected,
                    "source": str(source.relative_to(root)), "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                    "rgb_source": f"rgb_{step:07d}_0.jpg", "original_pixel": [u,v],
                    "review": "assistant visual review of recorded RGB; development regression, not human-annotated benchmark",
                    "robot_position": [position[0],position[1],position[2]-.45],
                    "check": {"anchor_view":"front","anchor_pixel_xy":[16,16]},
                    "view": {"label":"front","image_size":[33,33],"depth_mm":base64.b64encode(zlib.compress(mm.tobytes())).decode(),
                             "position":position,"rotation":rotation,"intrinsics":[320.,320.,320.-u+16,240.-v+16]}})
destination = root / "tests/fixtures/office_anchor_patches.json"
destination.parent.mkdir(parents=True, exist_ok=True)
destination.write_text(json.dumps(samples,indent=2))
print(destination)
