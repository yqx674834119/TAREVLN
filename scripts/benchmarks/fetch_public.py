"""Background public-source acquisition. No model weights or repo code executed."""
import concurrent.futures
import hashlib
import json
from pathlib import Path
import subprocess
import time
import urllib.request

project=Path(__file__).resolve().parents[2]
root=project.parent
out=root/'assets/benchmarks';out.mkdir(parents=True,exist_ok=True)
papers=json.loads((project/'config/benchmarks/papers.json').read_text())['papers']
def fetch_file(url,destination):
    destination.parent.mkdir(parents=True,exist_ok=True)
    if not destination.exists():
        temporary=destination.with_suffix(destination.suffix+'.part')
        with urllib.request.urlopen(url,timeout=90) as response,temporary.open('wb') as stream:
            while block:=response.read(1024*1024):stream.write(block)
        temporary.replace(destination)
    return {'url':url,'path':str(destination),'bytes':destination.stat().st_size,
            'sha256':hashlib.sha256(destination.read_bytes()).hexdigest()}
def fetch_paper(paper):
    result={'id':paper['id']}
    try:
        destination=root/'deps/benchmarks'/paper['id']
        if not destination.exists():
            subprocess.run(['git','-c','http.proxy=','-c','https.proxy=','clone','--depth','1',
                            'https://github.com/'+paper['repo']+'.git',str(destination)],check=True)
        result['repository']={'path':str(destination),'url':'https://github.com/'+paper['repo'],
            'commit':subprocess.check_output(['git','-C',str(destination),'rev-parse','HEAD'],text=True).strip()}
        result['paper']=fetch_file(paper['pdf'],out/'papers'/(paper['id']+'.pdf'))
        result['status']='downloaded'
    except Exception as error:result.update(status='error',error=type(error).__name__+': '+str(error)[:300])
    return result
results=[]
with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
    for result in pool.map(fetch_paper,papers):
        results.append(result);print(result['id'],result['status'],flush=True)
        (out/'public-acquisition.json').write_text(json.dumps(results,indent=2))
try:
    result=fetch_file('https://drive.google.com/uc?id=1T9SjqZWyR2PCLSXYkFckfDeIs6Un0Rjm',out/'tasks/R2R_VLNCE_v1-3.zip')
    import zipfile
    with zipfile.ZipFile(result['path']) as archive:
        bad=archive.testzip()
        if bad:raise ValueError('CRC error: '+bad)
        # Dataset paths only; never extract traversal paths or executable code.
        target=(out/'tasks').resolve()
        for member in archive.infolist():
            if not (target/member.filename).resolve().is_relative_to(target):raise ValueError('Unsafe archive path')
        archive.extractall(target)
    result['status']='downloaded_and_crc_verified'
except Exception as error:result={'status':'error','error':type(error).__name__+': '+str(error)[:300]}
(out/'r2rce-acquisition.json').write_text(json.dumps(result,indent=2));print('R2R-CE',result['status'],flush=True)
