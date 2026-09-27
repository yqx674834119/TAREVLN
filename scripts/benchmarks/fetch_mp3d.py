"""Prioritized, hash-checked ModelScope scene download; credentials never logged."""
import concurrent.futures
import hashlib
import json
import os
from pathlib import Path
import time

root=Path(__file__).resolve().parents[3]
os.environ['MODELSCOPE_HOME']=str(root/'secrets/modelscope')
os.environ['MODELSCOPE_CACHE']=str(root/'cache/modelscope')
from modelscope.hub.file_download import dataset_file_download
key=root/'secrets/modelscope-api-token'
if key.stat().st_mode & 0o077:raise PermissionError('Credential file must have mode 0600')
token=key.read_text().strip()
assets=root/'assets/benchmarks'
metadata=json.loads((assets/'modelscope-mp3d-files.json').read_text())
first=json.loads((assets/'first-scene.json').read_text())['scene_id']
scenes={f['Path'].split('/')[0] for f in metadata if f['Path'].endswith('.glb') and len(f['Path'].split('/'))==2}
files=[f for f in metadata if f['Type']=='blob' and f['Path'].split('/')[0] in scenes
       and f['Path'].endswith(('.glb','.house','.navmesh','_semantic.ply'))]
target=assets/'scene_datasets/mp3d';target.mkdir(parents=True,exist_ok=True)
def fetch(record):
    path=record['Path'];result={'path':path,'expected_bytes':record['Size'],'expected_sha256':record['Sha256']}
    try:
        if not (target/path).resolve().is_relative_to(target.resolve()):raise ValueError('Unsafe dataset path')
        for attempt in range(3):
            try:
                local=Path(dataset_file_download('Mrakas/mp3d',path,revision='master',local_dir=str(target),token=token))
                break
            except Exception:
                if attempt==2:raise
                time.sleep(2**attempt)
        digest=hashlib.sha256()
        with local.open('rb') as stream:
            while chunk:=stream.read(4*1024*1024):digest.update(chunk)
        result.update(bytes=local.stat().st_size,sha256=digest.hexdigest())
        if result['bytes']!=record['Size'] or (record['Sha256'] and result['sha256']!=record['Sha256']):
            raise ValueError('Downloaded file differs from mirror metadata')
        result['status']='verified'
    except Exception as error:result.update(status='error',error=type(error).__name__)
    return result
priority=[f for f in files if f['Path'].split('/')[0]==first and f['Path'].endswith(('.glb','.navmesh','.house'))]
remaining=[f for f in files if f not in priority]
results=[]
for phase,batch in [('first',priority),('remaining',remaining)]:
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        for result in pool.map(fetch,batch):
            results.append(result)
            temporary=assets/'mp3d-download-status.tmp'
            temporary.write_text(json.dumps({'phase':phase,'first_scene':first,'completed':len(results),'total':len(files),'files':results},indent=2))
            temporary.replace(assets/'mp3d-download-status.json')
            print(result['status'],result['path'],flush=True)
    if phase=='first':
        (assets/'first-scene-ready.json').write_text(json.dumps({'scene_id':first,'ready':all(r['status']=='verified' for r in results),'files':results},indent=2))
(assets/'mp3d-download-complete.json').write_text(json.dumps({'complete':all(r['status']=='verified' for r in results),'files':len(results),'bytes':sum(r.get('bytes',0) for r in results)},indent=2))
