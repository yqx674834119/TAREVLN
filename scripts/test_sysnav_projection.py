"""Independent ray/geometry calibration checks, not a simulation acceptance test."""
from pathlib import Path
import sys
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'sim'))
from sysnav_panorama import Panorama, cube_rotations

u,v=np.meshgrid(np.arange(640),np.arange(640))
rays=np.stack((np.ones_like(u),(320-u)/320,(320-v)/320),axis=-1)
rays=rays/np.linalg.norm(rays,axis=-1,keepdims=True)
faces=[np.round((rays@rotation.T+1)*127.5).astype(np.uint8) for rotation in cube_rotations()]
image=Panorama([320,320,320,320]).stitch(faces)
# RGB encodes body-frame xyz; compare against independently calculated landmarks.
errors=[]
for azimuth in np.linspace(-170,170,35):
    for elevation in (-50,-25,0,25,50):
        a,e=np.deg2rad([azimuth,elevation])
        expected=np.array([np.cos(e)*np.cos(a),np.cos(e)*np.sin(a),np.sin(e)])
        px=round(961-azimuth*1920/360);py=round(321-elevation*1920/360)
        actual=image[py,px].astype(float)/127.5-1
        errors.append(np.linalg.norm(actual-expected))
assert max(errors)<.012,max(errors)
print({'landmarks':len(errors),'max_unit_ray_error':max(errors),'shape':image.shape})
