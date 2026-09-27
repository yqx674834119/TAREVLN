"""Loopback-only simulation console. Launches only its own allowlisted jobs."""
from __future__ import annotations
import argparse
import gzip
import json
import mimetypes
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, quote, urlparse
import uuid

PROJECT = Path(__file__).resolve().parents[2]
ROOT = PROJECT.parent
sys.path.insert(0, str(PROJECT/'src'))
from vln_policy.office_tasks import TASKS


def read_json(path, default=None):
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return {} if default is None else default


def read_lines(path):
    if not path.exists():
        return []
    result = []
    with path.open() as f:
        for line in f:
            try:
                result.append(json.loads(line))
            except ValueError:
                continue  # A currently appended last line may be incomplete.
    return result


def public_decision(value):
    # Only short action explanations and visible evidence, never hidden model reasoning.
    keys = ('action','reason','evidence','confidence','candidate_id','selected_view','repeat',
            'target_visible','inside_target_area')
    return {k:value[k] for k in keys if k in value}


class Console:
    def __init__(self, root=ROOT, project=PROJECT):
        self.root, self.project = Path(root), Path(project)
        self.jobs = {}
        self.records = {}
        self.lock = threading.Lock()
        self.cache = {}
        self.task_path = self.root/'assets/benchmarks/tasks/R2R_VLNCE_v1-3/val_unseen/val_unseen.json.gz'
        self.episodes = {}
        if self.task_path.exists():
            with gzip.open(self.task_path, 'rt') as f:
                for e in json.load(f)['episodes']:
                    if e['scene_id'].split('/')[1] == 'x8F5xyUWy9e':
                        self.episodes[str(e['episode_id'])] = e

    def catalog(self):
        return dict(tasks=[dict(id='office:'+k, kind='office', title={'lounge':'寻找等候休息区','meeting':'寻找会议区'}[k],
                               instruction=v['instruction'], label='Isaac · Go2 · TARE + VLM') for k,v in TASKS.items()] +
                   [dict(id='habitat:'+k, kind='habitat', title='R2R-CE · Episode '+k,
                         instruction=e['instruction']['instruction_text'], label='Habitat · 导航代理')
                    for k,e in sorted(self.episodes.items(), key=lambda x:int(x[0]))],
                    downloads=self.downloads())

    def downloads(self):
        result = {}
        for key in ('mp3d','etpnav'):
            x = read_json(self.root/f'assets/benchmarks/{key}-download-status.json')
            result[key] = dict(completed=x.get('completed',0), total=x.get('total',0),
                               failures=sum(r.get('status')!='verified' for r in x.get('files',[])))
        return result

    def scan(self):
        records = {}
        office = self.project/'data/sim/office/runs'
        if office.exists():
            directories = sorted(office.iterdir(), key=lambda p:p.stat().st_mtime, reverse=True)[:24]
            for p in directories:
                if not p.is_dir() or not (p/'report.json').exists():
                    continue
                report = read_json(p/'report.json')
                records['office:'+p.name] = dict(id='office:'+p.name,kind='office',path=p,
                    name=p.name, task_id='office:'+report.get('task','lounge'), status=report.get('status','unknown'),
                    success=report.get('success',False), stamp=p.stat().st_mtime)
        for p in (self.root/'outputs/benchmarks').glob('*/vlm/*'):
            if not p.is_dir() or not (p/'preflight.json').exists():
                continue
            result = read_json(p/'result.json')
            identifier = 'habitat:'+p.parents[1].name+':'+p.name
            records[identifier] = dict(id=identifier,kind='habitat',path=p,
                name=p.parents[1].name+' / '+p.name,task_id='habitat:'+p.name,
                status=result.get('termination','running'), success=bool(result.get('sr',False)), stamp=p.stat().st_mtime)
        for job in self.jobs.values():
            p = job['process']
            record = records.get(job['record_id'])
            if not record:
                record = dict(id=job['record_id'],kind=job['kind'],path=job['record_path'],name=job['name'],
                    task_id=job['task_id'],status='starting',success=False,stamp=job['started'])
                records[job['record_id']] = record
            record['owned'] = True
            record['job_id'] = job['id']
            record['active'] = p.poll() is None
            if record['active']:
                record['status'] = 'stopping' if job.get('stop_requested') else 'running'
            elif record['status'] in ('starting','running','stopping'):
                record['status'] = 'operator_stopped' if job.get('stop_requested') else 'process_exit_'+str(p.returncode)
        self.records = records
        return sorted([{k:v for k,v in r.items() if k!='path'} for r in records.values()], key=lambda x:x['stamp'], reverse=True)

    def record(self, identifier):
        self.scan()
        if identifier not in self.records:
            raise ValueError('Unknown run')
        return self.records[identifier]

    def media(self, identifier, relative):
        record = self.record(identifier)
        root = record['path'].resolve()
        path = (root/relative).resolve()
        if not path.is_relative_to(root) or path.suffix.lower() not in ('.jpg','.jpeg','.png'):
            raise ValueError('Media path is not allowed')
        if not path.is_file():
            raise FileNotFoundError('Frame unavailable')
        return path

    def image_url(self, record, path):
        return '/media?id='+quote(record['id'],safe='')+'&file='+quote(str(path.relative_to(record['path'])),safe='')

    def detail(self, identifier):
        r = self.record(identifier)
        now = time.monotonic()
        cache = self.cache.get(identifier)
        if cache and now-cache[0] < 1.0:
            return {**cache[1], 'active':r.get('active',False), 'status':r['status']}
        p = r['path']
        frames, trajectory, decisions = [], [], []
        if r['kind']=='office':
            report = read_json(p/'report.json')
            rows = read_lines(p/'trajectory.jsonl')
            for row in rows[::max(1,len(rows)//1800)]:
                trajectory.append(dict(t=row['time'],x=row['position'][0],y=row['position'][1],speed=row.get('speed')))
            for path in sorted(p.glob('chase_*.jpg')):
                index = int(path.stem.split('_')[1])
                front = p/f'rgb_{index:07d}_0.jpg'
                frames.append(dict(t=round(index*.005+.015,3),chase=self.image_url(r,path),
                                   rgb=self.image_url(r,front) if front.exists() else None))
            for path in sorted((p/'agent/vlm').glob('call-*/response.json')):
                d = read_json(path)
                context = d.get('context',{})
                result = d.get('result',{})
                decisions.append(dict(index=len(decisions)+1,t=context.get('simulation_time',context.get('time',0)),
                    purpose=context.get('purpose','decision'),latency=d.get('latency_seconds'),
                    **public_decision(result)))
            current = rows[-1] if rows else {}
            progress = dict(time=current.get('time',0), speed=current.get('speed'),
                action=current.get('action',{}).get('reason'),
                vlm_calls=current.get('action',{}).get('vlm_calls',len(decisions)),
                path_nodes=current.get('action',{}).get('path_nodes'))
            result = {k:report.get(k) for k in ['success','status','simulation_seconds','wall_seconds','distance_m',
                'contact_quality_passed','odometry','surface_geometry','physical_hardware_ready'] if k in report}
        else:
            report = read_json(p/'result.json')
            raw = read_json(p/'trajectory.json')
            live = read_json(p/'live.json')
            rows = read_lines(p/'decisions.jsonl')
            points = raw.get('positions',live.get('positions',[]))
            trajectory = [dict(t=i,x=point[0],y=point[2]) for i,point in enumerate(points)]
            start = p/'start.jpg'
            if start.exists():
                frames.append(dict(t=0,rgb=self.image_url(r,start),chase=None))
            for i,row in enumerate(rows,1):
                path = p/f'view-{i:03d}.jpg'
                if path.exists():
                    frames.append(dict(t=row['step'],rgb=self.image_url(r,path),chase=None))
                decisions.append(dict(index=i,t=row['step'],purpose='navigation_decision',**public_decision(row)))
            end = p/'end.jpg'
            if end.exists():
                frames.append(dict(t=report.get('steps',0),rgb=self.image_url(r,end),chase=None))
            elif (p/'live.jpg').exists():
                frames.append(dict(t=live.get('step',0),rgb=self.image_url(r,p/'live.jpg')+'&v='+str(live.get('step',0)),chase=None))
            progress = dict(time=report.get('steps',live.get('step',rows[-1]['step'] if rows else 0)),
                vlm_calls=report.get('decisions',len(rows)),action=live.get('action'),collisions=report.get('collisions'))
            result = {k:report[k] for k in ['sr','ne_m','osr','spl','ndtw_fast','steps','collisions','termination','error'] if k in report}
        frames.sort(key=lambda f:f['t'])
        result = dict(**{k:v for k,v in r.items() if k!='path'}, frames=frames,trajectory=trajectory,
            decisions=decisions,progress=progress,result=result,
            semantics='Isaac PhysX · articulated Go2 · native TARE + VLM' if r['kind']=='office' else 'Habitat-Sim · kinematic navigation agent · VLM baseline',
            unit='s' if r['kind']=='office' else 'step')
        self.cache[identifier] = (now,result)
        return result

    def command(self, task_id, duration, name):
        base = ['bash', str(self.project/'scripts/with_gpu_compat.sh')]
        if task_id.startswith('office:') and task_id[7:] in TASKS:
            if duration not in (20,30,60,120,300):
                raise ValueError('Unsupported duration')
            checkpoint = self.project/'data/readiness/training/go2-control-v5-rails/model_3595.pt'
            if not checkpoint.exists():
                raise ValueError('Recorded Go2 checkpoint unavailable')
            command = base+[str(self.root/'envs/isaac51/bin/python'),'-u','sim/run_office.py',
                '--name',name,'--task',task_id[7:],'--max-seconds',str(duration),'--checkpoint',str(checkpoint),
                '--odometry','gicp','--gicp-correspondence','.5','--lidar-profile','uniform64',
                '--sam','--surface-geometry','metric_plane','--api-proxy','http://127.0.0.1:18119']
            record_path = self.project/'data/sim/office/runs'/name
            return command, record_path, 'office:'+name, record_path
        if task_id.startswith('habitat:') and task_id[8:] in self.episodes:
            output = self.root/'outputs/benchmarks'/name
            command = base+[str(self.root/'envs/habitat-bench/bin/python'),'-u','scripts/benchmarks/run_habitat.py',
                '--policies','vlm','--episode-id',task_id[8:],'--output',str(output)]
            record_path = output/'vlm'/task_id[8:]
            return command, record_path, 'habitat:'+name+':'+task_id[8:], output
        raise ValueError('Unknown task')

    def start(self, task_id, duration):
        with self.lock:
            if any(j['process'].poll() is None for j in self.jobs.values()):
                raise ValueError('A dashboard simulation is already active')
            name = 'ui-'+time.strftime('%Y%m%d-%H%M%S')+'-'+uuid.uuid4().hex[:5]
            command,path,record_id,stop_root = self.command(task_id,duration,name)
            log_path = self.root/'logs'/('dashboard-'+name+'.log')
            log_path.parent.mkdir(parents=True,exist_ok=True)
            with log_path.open('w') as log:
                process = subprocess.Popen(command,cwd=self.project,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,
                    env={**os.environ,'OMNI_KIT_ACCEPT_EULA':'YES','HABITAT_SIM_LOG':'quiet','MAGNUM_LOG':'quiet',
                         'HTTPS_PROXY':'http://127.0.0.1:18119'},start_new_session=True)
            job = dict(id=name,name=name,task_id=task_id,kind=task_id.split(':')[0],process=process,
                record_path=path,record_id=record_id,stop_root=stop_root,started=time.time())
            self.jobs[name] = job
            return dict(job_id=name,record_id=record_id)

    def stop(self, job_id):
        with self.lock:
            job = self.jobs.get(job_id)
            if not job or job['process'].poll() is not None:
                raise ValueError('No active dashboard-owned job')
            job['stop_requested'] = True
            if job['stop_root'].exists():
                (job['stop_root']/'STOP').write_text('Requested in simulation console\n')
            else:
                job['process'].send_signal(signal.SIGINT)
            def enforce_stop():
                try:
                    job['process'].wait(timeout=30)
                except subprocess.TimeoutExpired:
                    # Only signal the process group created by this console.
                    try:
                        os.killpg(job['process'].pid,signal.SIGINT)
                    except ProcessLookupError:
                        pass
            threading.Thread(target=enforce_stop,daemon=True).start()
            return dict(stopping=True)


class Handler(BaseHTTPRequestHandler):
    server_version = 'VLNSimulationConsole/1'

    def log_message(self, fmt, *args):
        if args and str(args[0]).startswith('POST'):
            super().log_message(fmt,*args)

    def trusted_host(self):
        return self.headers.get('Host') in ('127.0.0.1:'+str(self.server.server_port),'localhost:'+str(self.server.server_port))

    def send(self, value, status=200, content_type='application/json; charset=utf-8'):
        content = json.dumps(value,ensure_ascii=False).encode() if not isinstance(value,bytes) else value
        self.send_response(status)
        self.send_header('Content-Type',content_type)
        self.send_header('Content-Length',str(len(content)))
        self.send_header('Cache-Control','no-store')
        self.send_header('X-Content-Type-Options','nosniff')
        self.end_headers()
        try:
            self.wfile.write(content)
        except (BrokenPipeError,ConnectionResetError):
            pass

    def do_GET(self):
        if not self.trusted_host():
            return self.send({'error':'Invalid Host'},403)
        url = urlparse(self.path)
        query = parse_qs(url.query)
        try:
            console = self.server.console
            if url.path=='/api/catalog':
                return self.send(console.catalog())
            if url.path=='/api/runs':
                return self.send(console.scan())
            if url.path=='/api/run':
                return self.send(console.detail(query.get('id',[''])[0]))
            if url.path=='/media':
                path = console.media(query.get('id',[''])[0],query.get('file',[''])[0])
                return self.send(path.read_bytes(),content_type=mimetypes.guess_type(str(path))[0])
            files = {'/':'index.html','/app.js':'app.js','/style.css':'style.css'}
            if url.path in files:
                path = PROJECT/'frontend'/files[url.path]
                return self.send(path.read_bytes(),content_type=mimetypes.guess_type(str(path))[0]+'; charset=utf-8')
            return self.send({'error':'Not found'},404)
        except (ValueError,FileNotFoundError) as error:
            return self.send({'error':str(error)},400)

    def do_POST(self):
        origin = self.headers.get('Origin')
        if not self.trusted_host() or self.headers.get('X-VLN-Request')!='dashboard' or (origin and origin!='http://'+self.headers.get('Host','')):
            return self.send({'error':'Same-origin dashboard request required'},403)
        try:
            size = int(self.headers.get('Content-Length','0'))
            if not 0 < size <= 4096:
                raise ValueError('Invalid request size')
            body = json.loads(self.rfile.read(size))
            if self.path=='/api/start':
                return self.send(self.server.console.start(str(body.get('task_id','')),body.get('duration',120)))
            if self.path=='/api/stop':
                return self.send(self.server.console.stop(str(body.get('job_id',''))))
            return self.send({'error':'Not found'},404)
        except (ValueError,TypeError) as error:
            return self.send({'error':str(error)},400)


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--port',type=int,default=8878)
    args = parser.parse_args()
    server = ThreadingHTTPServer(('127.0.0.1',args.port),Handler)
    server.console = Console()
    print('VLN console on loopback port',args.port,flush=True)
    server.serve_forever()
