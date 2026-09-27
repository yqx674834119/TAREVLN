"""Network-isolated SAM2 service. Receives only images and box prompts."""
import argparse
import json
import os
from pathlib import Path
import socket
from vln_perception.sam2_masks import Sam2Masks

ap=argparse.ArgumentParser()
ap.add_argument('--checkpoint',required=True)
ap.add_argument('--deny-probe',required=True)
args=ap.parse_args()
if Path(args.deny_probe).exists():raise RuntimeError('Evaluator files visible to SAM')
model=Sam2Masks(args.checkpoint)
server=socket.socket(socket.AF_UNIX,socket.SOCK_STREAM)
server.bind('/ipc/sam.sock');os.chmod('/ipc/sam.sock',0o600);server.listen(2)
print(json.dumps({'ready':True,'evaluator_visible':False,'network':'isolated',
                  'input':'RGB JPEG and image-space boxes only','model':'SAM2.1 large',
                  'optional_cuda_postprocessing':False}),flush=True)
index=0
while True:
    connection,_=server.accept()
    with connection:
        connection.settimeout(30)
        try:
            with connection.makefile('rb') as stream:
                line=stream.readline(8*1024*1024+1)
            if len(line)>8*1024*1024 or not line.endswith(b'\n'):
                raise ValueError('Incomplete or oversized SAM request')
            frame=json.loads(line)
            result={'valid':True,**model.segment(frame)}
            Path(f'/output/request-{index:05d}.json').write_text(json.dumps({'request':frame,'result':result}))
            index+=1
        except Exception as error:
            result={'valid':False,'error':type(error).__name__+': '+str(error)}
        connection.sendall(json.dumps(result,allow_nan=False).encode()+b'\n')
