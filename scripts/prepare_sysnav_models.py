"""Prepare the exact upstream detector vocabulary once, outside the sandbox."""
import hashlib
import json
import os
from pathlib import Path
import torch
import yaml
from ultralytics import YOLOE

root=Path(__file__).resolve().parents[2]
output=root/'assets/sysnav'
os.chdir(output)
config=root/'deps/sysnav-full/src/semantic_mapping/semantic_mapping/config/objects.yaml'
names=[s for value in yaml.safe_load(config.read_text())['prompts'].values() for s in value['prompts']]
model=YOLOE(str(output/'yoloe-26x-seg.pt')).to('cuda:0')
embeddings=model.get_text_pe(names)
torch.save(embeddings.cpu(),output/'text-embeddings.pt')
model.set_classes(names,embeddings)
# Exercise the predictor now so inference cannot silently download dependencies.
import numpy as np
model.predict(np.zeros((640,1920,3),dtype=np.uint8),imgsz=(640,1920),device=0,verbose=False)
paths=[output/'yoloe-26x-seg.pt',output/'text-embeddings.pt',root/'assets/sam/sam2.1_hiera_base_plus.pt',config]
(output/'provenance.json').write_text(json.dumps({'model':'YOLOE-26x','classes':names,
    'files':{str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
    'sam2':'2.1 hiera base plus','torch':torch.__version__},indent=2))
print('SYSNAV_MODELS_READY',flush=True)
