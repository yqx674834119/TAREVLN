"""Launch a model-only GPU sandbox, without scene files or API credentials."""
import hashlib
import json
from pathlib import Path
import select
import subprocess


class SamProcess:
    def __init__(self,project,out,gpu=2):
        project,out=Path(project).resolve(),Path(out).resolve()
        root=project.parent
        self.ipc=out/'ipc';self.ipc.mkdir(parents=True,exist_ok=False)
        checkpoint=(root/'assets/sam/sam2.1_hiera_large.pt').resolve(strict=True)
        checkpoint_hash=hashlib.sha256(checkpoint.read_bytes()).hexdigest()
        provenance=json.loads((project/'config/sam2-provenance.json').read_text())
        if checkpoint_hash!=provenance['checkpoint_sha256']:
            raise ValueError('SAM checkpoint differs from the recorded model')
        runtime=root/'envs/isaac51'
        torch=root/'envs/torch-cu128'
        source=root/'deps/sam2'
        deps=root/'envs/sam2-deps'
        driver=root/'toolchains/nvidia-580.173.02/lib'
        command=['bwrap','--unshare-all','--die-with-parent','--new-session','--clearenv',
                 '--ro-bind','/usr','/usr','--ro-bind','/lib','/lib','--ro-bind','/lib64','/lib64',
                 '--proc','/proc','--dev-bind','/dev','/dev','--tmpfs','/tmp','--dir','/tmp/home']
        for path in (runtime,torch,source,deps,driver,root/'toolchains/python'):command+=['--ro-bind',str(path),str(path)]
        command+=['--ro-bind',str(checkpoint),'/model.pt',
                  '--ro-bind',str(project/'src/vln_perception'),'/app/vln_perception',
                  '--ro-bind',str(project/'scripts/sam_worker.py'),'/worker.py',
                  '--bind',str(self.ipc),'/ipc','--bind',str(out),'/output']
        env={'PYTHONPATH':':'.join(map(str,[torch,source,deps]))+':/app','HOME':'/tmp/home',
             'PYTHONNOUSERSITE':'1','HF_HUB_OFFLINE':'1','CUDA_VISIBLE_DEVICES':str(gpu),
             'LD_LIBRARY_PATH':str(driver),'OMP_NUM_THREADS':'2','OPENBLAS_NUM_THREADS':'2'}
        for name,value in env.items():command+=['--setenv',name,value]
        command += [str(runtime/'bin/python'),'-u','/worker.py','--checkpoint','/model.pt',
                    '--deny-probe',str(project/'data/sim/office/private')]
        self.log=(out/'worker.log').open('w')
        self.child=subprocess.Popen(command,stdout=subprocess.PIPE,stderr=self.log,text=True)
        try:
            if not select.select([self.child.stdout],[],[],90)[0]:raise TimeoutError('SAM model startup timed out')
            line=self.child.stdout.readline()
            self.isolation=json.loads(line)
            if self.isolation.get('ready') is not True or self.isolation.get('evaluator_visible') is not False:
                raise RuntimeError('SAM sandbox did not establish isolation')
            self.isolation.update(checkpoint_sha256=checkpoint_hash,gpu=gpu,runtime=provenance['runtime'])
            (out/'isolation.json').write_text(json.dumps(self.isolation,indent=2))
        except BaseException:
            self.close();raise

    def close(self):
        if self.child.poll() is None:
            self.child.terminate()
            try:self.child.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.child.kill();self.child.wait(timeout=5)
        self.child.stdout.close();self.log.close()
