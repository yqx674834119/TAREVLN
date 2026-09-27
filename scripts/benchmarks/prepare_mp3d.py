"""Authenticate privately, inventory the mirror, select a scene before outcomes."""
import gzip
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT/'assets/benchmarks'
os.environ['MODELSCOPE_HOME'] = str(ROOT/'secrets/modelscope')
os.environ['MODELSCOPE_CACHE'] = str(ROOT/'cache/modelscope')
from modelscope.hub.api import HubApi


def main():
    key = ROOT/'secrets/modelscope-api-token'
    if key.stat().st_mode & 0o077:
        raise PermissionError('Credential file must have mode 0600')
    token = key.read_text().strip()
    api = HubApi()
    api.login(token)  # Never print returned Git credentials or cookies.
    records = []
    for page in range(1, 1000):
        batch = api.get_dataset_files('Mrakas/mp3d', revision='master', root_path='/',
            recursive=True, page_number=page, page_size=100, token=token)
        if not batch:
            break
        records.extend(batch)
        if len(batch) < 100:
            break
    else:
        raise RuntimeError('Inventory pagination did not terminate')
    records = list({x['Path']: x for x in records}.values())
    (OUT/'modelscope-mp3d-files.json').write_text(json.dumps(records, indent=2))
    tasks = json.load(gzip.open(OUT/'tasks/R2R_VLNCE_v1-3/val_unseen/val_unseen.json.gz', 'rt'))['episodes']
    scenes = {e['scene_id'].split('/')[1] for e in tasks}
    candidates = [f for f in records if f['Type']=='blob' and f['Path'].endswith('.glb') and f['Path'].split('/')[0] in scenes]
    first = min(candidates, key=lambda f:(f['Size'], f['Path']))
    scene = first['Path'].split('/')[0]
    result = dict(scene_id=scene, glb=first, split='val_unseen',
        reason='Smallest downloadable scene in official R2R-CE val_unseen; selected by byte size before running any policy',
        episodes=sum(scene in e['scene_id'] for e in tasks))
    (OUT/'first-scene.json').write_text(json.dumps(result, indent=2))
    print('Authenticated; selected', scene, 'episodes', result['episodes'])


if __name__ == '__main__':
    main()
