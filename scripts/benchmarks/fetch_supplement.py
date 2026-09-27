"""Fetch author-hosted ETPNav assets and original R2R / dense R2R-CE truth.

Credentials stay outside the repository; downloaded Python/pickle/weights are
stored as data and are never imported or executed by this acquisition script.
"""
import concurrent.futures
import hashlib
import json
import os
from pathlib import Path
import time
import urllib.request
import zipfile

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / 'assets/benchmarks'
os.environ['MODELSCOPE_HOME'] = str(ROOT / 'secrets/modelscope')
os.environ['MODELSCOPE_CACHE'] = str(ROOT / 'cache/modelscope')
from modelscope.hub.api import HubApi
from modelscope.hub.file_download import model_file_download


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda: f.read(1024*1024), b''):
            h.update(b)
    return h.hexdigest()


def public_tasks():
    records = []
    links = {
        'R2R_train.json': 'https://www.dropbox.com/s/hh5qec8o5urcztn/R2R_train.json?dl=1',
        'R2R_val_seen.json': 'https://www.dropbox.com/s/8ye4gqce7v8yzdm/R2R_val_seen.json?dl=1',
        'R2R_val_unseen.json': 'https://www.dropbox.com/s/p6hlckr70a07wka/R2R_val_unseen.json?dl=1',
        'R2R_test.json': 'https://www.dropbox.com/s/w4pnbwqamwzdwd1/R2R_test.json?dl=1',
        'R2R_VLNCE_v1-3_preprocessed.zip': 'https://drive.google.com/uc?id=1fo8F4NKgZDH-bPSdVU3cONAkt5EW-tyr',
    }
    for name, url in links.items():
        path = OUT / 'tasks' / name
        try:
            if not path.exists():
                if name.endswith('.zip'):
                    import gdown
                    result = gdown.download(url, str(path)+'.part', quiet=True)
                    if not result:
                        raise RuntimeError('Drive returned no file')
                else:
                    with urllib.request.urlopen(url, timeout=90) as r, open(str(path)+'.part', 'wb') as f:
                        for b in iter(lambda: r.read(1024*1024), b''):
                            f.write(b)
                Path(str(path)+'.part').replace(path)
            if name.endswith('.zip'):
                with zipfile.ZipFile(path) as z:
                    if z.testzip():
                        raise ValueError('CRC mismatch')
                    target = path.parent.resolve()
                    for m in z.infolist():
                        if not (target/m.filename).resolve().is_relative_to(target):
                            raise ValueError('Unsafe archive path')
                    z.extractall(target)
            else:
                with path.open() as f:
                    data = json.load(f)
                if not isinstance(data, list):
                    raise ValueError('Not an R2R episode list')
            records.append(dict(name=name, url=url, bytes=path.stat().st_size,
                                sha256=digest(path), status='verified'))
        except Exception as e:
            records.append(dict(name=name, status='failed', error=type(e).__name__))
        (OUT/'supplement-public-status.json').write_text(json.dumps(records, indent=2))
        print(name, records[-1]['status'], flush=True)


def etpnav_assets():
    token_path = ROOT/'secrets/modelscope-api-token'
    if token_path.stat().st_mode & 0o077:
        raise PermissionError('Credential file must be private')
    token = token_path.read_text().strip()
    api = HubApi()
    files = api.get_model_files('admagic/ETPNav', recursive=True, token=token)
    files = [x for x in files if x.get('Size', 0) > 0]
    (OUT/'etpnav-model-metadata.json').write_text(json.dumps(files, indent=2))
    # Small task annotations first, then inference assets, then pretraining.
    files.sort(key=lambda x: (not x['Path'].startswith('datasets/'), x['Size']))
    records = []

    def fetch(f):
        for attempt in range(3):
            try:
                target = (OUT/'etpnav').resolve()
                if not (target/f['Path']).resolve().is_relative_to(target):
                    raise ValueError('Unsafe asset path')
                path = Path(model_file_download('admagic/ETPNav', f['Path'],
                    local_dir=str(OUT/'etpnav'), token=token))
                sha = digest(path)
                expected_sha = f.get('Sha256') or f.get('SHA256')
                if path.stat().st_size != f['Size'] or (expected_sha and sha != expected_sha):
                    raise ValueError('Integrity mismatch')
                return dict(path=f['Path'], bytes=f['Size'], sha256=sha,
                            expected_sha256=expected_sha, status='verified')
            except Exception as e:
                if attempt == 2:
                    return dict(path=f['Path'], status='failed', error=type(e).__name__)
                time.sleep(2)

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        for r in pool.map(fetch, files):
            records.append(r)
            status = dict(completed=len(records), total=len(files), files=records)
            tmp = OUT/'etpnav-download-status.tmp'
            tmp.write_text(json.dumps(status, indent=2))
            tmp.replace(OUT/'etpnav-download-status.json')
            print('ETPNav', r['path'], r['status'], flush=True)
    (OUT/'etpnav-download-complete.json').write_text(json.dumps(status, indent=2))


if __name__ == '__main__':
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        jobs = [pool.submit(public_tasks), pool.submit(etpnav_assets)]
        for j in jobs:
            j.result()
