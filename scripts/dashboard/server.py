"""Simulation console. Launches only its own allowlisted jobs."""
from __future__ import annotations
import argparse
import gzip
import json
import mimetypes
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, quote, urlparse
import uuid
import math
import re


API_PROXY_DEFAULT = 'http://127.0.0.1:18119'

PROJECT = Path(__file__).resolve().parents[2]
ROOT = PROJECT.parent
sys.path.insert(0, str(PROJECT/'src'))
from vln_policy.office_tasks import TASKS

from vln_policy.sysnav_tasks import TASKS as SYSNAV_TASKS


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
        self.api_proxy = os.environ.get('VLN_API_PROXY', API_PROXY_DEFAULT)
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

    def ensure_api_relay(self):
        """Fail before Isaac starts when the remote-to-local relay is absent.

        The 5090 host cannot complete DeepSeek TLS directly. The relay is a
        loopback endpoint reached through an SSH reverse forward established
        by the operator on the workstation. Launching a simulation without it
        only produces delayed APIConnectionError messages in the VLM node.
        """
        endpoint = urlparse(self.api_proxy)
        if (endpoint.scheme != 'http' or endpoint.hostname != '127.0.0.1'
                or endpoint.port != 18119 or endpoint.username
                or endpoint.password or endpoint.path or endpoint.query
                or endpoint.fragment):
            raise ValueError('VLN_API_PROXY must be http://127.0.0.1:18119')
        try:
            with socket.create_connection((endpoint.hostname, endpoint.port), timeout=1.5):
                return
        except OSError:
            raise ValueError(
                'DeepSeek relay unavailable at 127.0.0.1:18119. Start the local relay and SSH reverse tunnel: '
                'python3 scripts/api_connect_relay.py; '
                'ssh -fN -o ExitOnForwardFailure=yes -R '
                '127.0.0.1:18119:127.0.0.1:18119 5090') from None

    def catalog(self):
        return dict(tasks=[dict(id='sysnav:'+k,kind='office',title=v[0],instruction=v[1],
                               label='Isaac · Go2 · 完整 SysNav · DeepSeek') for k,v in SYSNAV_TASKS.items()]+
                   [dict(id='isaacmp3d:'+k,kind='office',title='Isaac MP3D · R2R Episode '+k,
                         instruction=e['instruction']['instruction_text'],label='Isaac 迁移评测 · Go2 · 完整 SysNav')
                    for k,e in sorted(self.episodes.items(),key=lambda x:int(x[0]))
                    if (self.root/'assets/mp3d-isaac/x8F5xyUWy9e/scene.usda').exists()]+
                   [dict(id='office:'+k, kind='office', title={'lounge':'寻找等候休息区','meeting':'寻找会议区'}[k],
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
                    name=p.name, task_id=('isaacmp3d:'+str(report['episode_id']) if report.get('mode')=='sysnav_full_mp3d' else
                        ('sysnav:' if report.get('mode')=='sysnav_full' else 'office:')+report.get('task','lounge')), status=report.get('status','unknown'),
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

    def planning(self,identifier,timestamp):
        if not math.isfinite(timestamp) or timestamp<0:raise ValueError('Invalid planning time')
        record=self.record(identifier)
        root=record['path']/'agent/planning'
        entries=[r for r in read_lines(root/'index.jsonl') if r['time']<=timestamp+.001]
        if not entries:return dict(available=False,reason='该时刻没有记录原生 TARE 规划数据')
        row=max(entries,key=lambda r:r['time'])
        if not re.fullmatch(r'plan-\d{10}\.json',row['file']):raise ValueError('Invalid planning frame')
        path=root/row['file']
        if path.is_symlink():raise ValueError('Planning symlinks are not allowed')
        result=read_json(path)
        if not result:return dict(available=False,reason='规划帧尚未完成写入')
        trail=[r['robot']['position'] for r in entries]
        history=[]
        for r in entries:
            points=r.get('waypoint')
            if points and (not history or points[0]!=history[-1]):history.append(points[0])
        planning_time=result.get('time', timestamp)
        return dict(available=True,**result,trail=trail[::max(1,len(trail)//1500)],
                    waypoint_history=history[-250:], objects=self.object_state(record, planning_time),
                    **self.room_state(record, entries, planning_time))

    def room_state(self, record, entries, timestamp):
        """Return downsampled native /room_mask data and room-node labels.

        The ROS mask is a 0.1 m odometry-grid image.  It is intentionally
        cropped and sampled here so the browser never receives the 3000x3000
        source image.  No evaluator scene or goal files are read.
        """
        try:
            import numpy as np
        except ImportError:
            np = None
        planning_root = record['path']/'agent/planning'
        mask_rows=[row for row in entries if row.get('room_mask_file')]
        if not mask_rows:
            # Older records wrote the NPZ but omitted its name from index.jsonl.
            # Recover that observational artifact by its stamped filename.
            for candidate in planning_root.glob('rooms-*.npz'):
                match=re.fullmatch(r'rooms-(\d{10})\.npz',candidate.name)
                if match:
                    t=int(match.group(1))/1000
                    if t<=timestamp+.001:mask_rows.append({'time':t,'room_mask_file':candidate.name})
        mask_info=None
        source_mask=None
        mask_resolution=.1
        mask_origin=None
        if mask_rows and np is not None:
            row=max(mask_rows,key=lambda item:item['time'])
            name=row.get('room_mask_file','')
            if re.fullmatch(r'rooms-\d{10}\.npz',name):
                path=planning_root/name
                if path.is_file() and not path.is_symlink():
                    try:
                        with np.load(path,allow_pickle=False) as data:
                            source_mask=np.asarray(data['mask'])
                            if 'resolution' in data:
                                mask_resolution=float(np.asarray(data['resolution']).reshape(-1)[0])
                            if 'origin' in data:
                                values=np.asarray(data['origin']).reshape(-1)
                                if len(values)>=2:mask_origin=[float(values[0]),float(values[1])]
                        if source_mask.ndim==2 and source_mask.size<=20_000_000:
                            source_mask=source_mask.astype(np.int32,copy=False)
                    except (OSError,ValueError,KeyError):
                        source_mask=None
        if source_mask is not None:
            valid=source_mask>0
            rr,cc=np.nonzero(valid)
            if len(rr):
                # Coordinates follow the native room-segmentation shift.
                r0,r1=max(0,int(rr.min())-4),min(source_mask.shape[0],int(rr.max())+5)
                c0,c1=max(0,int(cc.min())-4),min(source_mask.shape[1],int(cc.max())+5)
                max_side=280
                nr=min(max_side,r1-r0); nc=min(max_side,c1-c0)
                rows=np.linspace(r0,r1-1,nr,dtype=np.int32)
                cols=np.linspace(c0,c1-1,nc,dtype=np.int32)
                sampled=source_mask[np.ix_(rows,cols)]
                room_info=max(mask_rows,key=lambda item:item['time'])
                mask_origin=mask_origin or [-source_mask.shape[0]*mask_resolution/2,
                                             -source_mask.shape[1]*mask_resolution/2]
                mask_info=dict(time=room_info['time'],ids=sampled.tolist(),
                    width=int(nc),height=int(nr),
                    origin=[mask_origin[0]+r0*mask_resolution,
                            mask_origin[1]+c0*mask_resolution],
                    resolution=[(r1-r0)/max(nr,1)*mask_resolution,
                                (c1-c0)/max(nc,1)*mask_resolution],
                    source_shape=[int(source_mask.shape[0]),int(source_mask.shape[1])],
                    source='/room_mask',nonzero_ids=sorted(int(x) for x in np.unique(sampled) if x>0))
        events=read_lines(record['path']/'agent/events.jsonl')
        room_snapshots=[e for e in events if e.get('topic')=='/room_nodes_list' and e.get('time',0)<=timestamp+.001]
        rooms=[]
        if room_snapshots:
            rooms=[dict(room) for room in room_snapshots[-1].get('rooms',[])]
        labels={}
        for event in events:
            if event.get('topic')=='/room_type_answer' and event.get('time',0)<=timestamp+.001:
                if event.get('id') is not None and event.get('label'):
                    labels[str(event['id'])]=event['label']
        for room in rooms:
            if str(room.get('id')) in labels:room['label']=labels[str(room['id'])]
        current_id=None
        if mask_info and rooms:
            robot=(max(entries,key=lambda item:item['time']).get('robot') or {}).get('position',[])
            if len(robot)>=2:
                origin=mask_origin or [-source_mask.shape[0]*mask_resolution/2,
                                       -source_mask.shape[1]*mask_resolution/2]
                x=int(round((robot[0]-origin[0])/mask_resolution))
                y=int(round((robot[1]-origin[1])/mask_resolution))
                if 0<=x<source_mask.shape[0] and 0<=y<source_mask.shape[1]:
                    value=int(source_mask[x,y])
                    current_id=value if value>0 else None
        return dict(room_mask=mask_info,rooms=rooms,current_room_id=current_id)

    def object_state(self, record, timestamp):
        events=read_lines(record['path']/'agent/events.jsonl')
        snapshots=[e for e in events if e.get('topic')=='/object_nodes_list' and e.get('time',0)<=timestamp+.001]
        if not snapshots:
            return []
        objects=[]
        for item in snapshots[-1].get('objects',[]):
            try:
                xyz=[float(v) for v in item.get('xyz',[])[:3]]
                if len(xyz)==3 and all(math.isfinite(v) for v in xyz):
                    objects.append(dict(id=item.get('id'),label=str(item.get('label','unknown')),xyz=xyz))
            except (TypeError,ValueError):
                continue
        return objects

    def runtime_metrics(self, record, timestamp):
        rows=read_lines(record['path']/'runtime_metrics.jsonl')
        eligible=[row for row in rows if row.get('time',timestamp)<=timestamp+.001]
        return eligible[-1] if eligible else None

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
                if report.get('mode','').startswith('sysnav_full'):
                    front = p/f'panorama_{index:07d}.jpg'
                frame=dict(t=round(index*.005+.015,3),chase=self.image_url(r,path),
                           rgb=self.image_url(r,front) if front.exists() else None)
                frames.append(frame)
            visuals=read_lines(p/'agent/visuals.jsonl')
            for frame in frames:
                for kind in ('yolo','sam'):
                    candidates=[v for v in visuals if v.get('kind')==kind and
                                0<=v.get('time',-1)<=frame['t']+.35 and
                                abs(v.get('time',-1)-frame['t'])<=1.5]
                    if not candidates:continue
                    visual=max(candidates,key=lambda v:v['time'])
                    # Paths in this index are relative to the isolated agent output.
                    relative=visual.get('file','')
                    if not re.fullmatch(r'visuals/(yolo|sam)-\d{10}\.jpg',relative):continue
                    visual_path=p/'agent'/relative
                    if not visual_path.is_file() or visual_path.is_symlink():continue
                    frame[kind]=self.image_url(r,visual_path)
                    frame[kind+'_meta']=dict(time=visual['time'],topic=visual['topic'],
                                             age=round(frame['t']-visual['time'],3))
            for path in sorted((p/'agent/vlm').glob('call-*/response.json')):
                d = read_json(path)
                context = d.get('context',{})
                result = d.get('result',{})
                decisions.append(dict(index=len(decisions)+1,t=context.get('simulation_time',context.get('time',0)),
                    purpose=context.get('purpose','decision'),latency=d.get('latency_seconds'),
                    **public_decision(result)))
            for call in read_lines(p/'agent/api.jsonl'):
                result=call.get('result',{})
                summary={k:v for k,v in result.items() if k not in ('steps','reason')}
                request_images=[]
                for relative in call.get('request_images',[]):
                    if not isinstance(relative,str) or not re.fullmatch(r'api_requests/call-\d{4,}-\d{2}\.jpg',relative):
                        continue
                    image_path=p/'agent'/relative
                    if image_path.is_file() and not image_path.is_symlink():
                        request_images.append(self.image_url(r,image_path))
                decisions.append(dict(index=call['id'],t=call.get('time') or 0,
                    purpose=call['purpose'],latency=call.get('seconds'),
                    evidence=json.dumps(summary,ensure_ascii=False),
                    request_text=call.get('request_text',''), request_images=request_images,
                    api_status=call.get('status'),error_type=call.get('error_type'),
                    reason=result.get('reason', '已返回结构化结果' if call['status']=='ok' else 'API 调用失败：'+call.get('error_type','unknown'))))
            current = rows[-1] if rows else {}
            runtime_metrics=self.runtime_metrics(r,current.get('time',0))
            progress = dict(time=current.get('time',0), speed=current.get('speed'),
                action=current.get('action',{}).get('reason'),
                vlm_calls=current.get('action',{}).get('vlm_calls',len(decisions)),
                path_nodes=current.get('action',{}).get('path_nodes'),
                modules=current.get('action',{}).get('counts',{}),
                rooms=current.get('action',{}).get('rooms'),objects=current.get('action',{}).get('objects'),
                runtime_metrics=runtime_metrics)
            result = {k:report.get(k) for k in ['success','status','simulation_seconds','wall_seconds','distance_m',
                'contact_quality_passed','odometry','surface_geometry','physical_hardware_ready',
                'acceptance_passed','evaluation_definition','independent_object_evaluation','instruction',
                'benchmark_result','protocol','scene','episode_id'] if k in report}
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
            planning_available=(p/'agent/planning/index.jsonl').exists(),
            semantics=('Isaac PhysX · Go2 · 完整 SysNav · DeepSeek · 全景 RGB'+(' · MP3D 迁移协议' if report.get('mode')=='sysnav_full_mp3d' else '') if report.get('mode','').startswith('sysnav_full') else 'Isaac PhysX · articulated Go2 · TARE + custom VLM') if r['kind']=='office' else 'Habitat-Sim · kinematic navigation agent · VLM baseline',
            unit='s' if r['kind']=='office' else 'step')
        self.cache[identifier] = (now,result)
        return result

    def command(self, task_id, duration, name, gui=False):
        if isinstance(duration,bool) or duration not in (-1,20,30,60,120,300):
            raise ValueError('Unsupported duration')
        duration=int(duration)
        base = ['bash', str(self.project/'scripts/with_gpu_compat.sh')]
        if task_id.startswith('isaacmp3d:') and task_id[10:] in self.episodes:
            if duration not in (-1,20,30,60,120,300):raise ValueError('Unsupported duration')
            if not (self.root/'assets/mp3d-isaac/x8F5xyUWy9e/scene.usda').exists():
                raise ValueError('Converted MP3D scene unavailable')
            command=base+['python3','-u','scripts/benchmarks/run_isaac_mp3d.py','--name',name,
                '--episode-id',task_id[10:],'--max-seconds',str(duration)]
            if gui: command+=['--gui']
            record_path=self.project/'data/sim/office/runs'/name
            return command,record_path,'office:'+name,record_path
        if task_id.startswith('sysnav:') and task_id[7:] in SYSNAV_TASKS:
            if duration != -1 and duration not in (-1,20,30,60,120,300):
                raise ValueError('Unsupported duration')
            checkpoint=self.project/'data/readiness/training/go2-control-v5-rails/model_3595.pt'
            if not checkpoint.exists() or not (self.root/'assets/sysnav/provenance.json').exists():
                raise ValueError('Full SysNav models or Go2 checkpoint unavailable')
            command=base+[str(self.root/'envs/isaac51/bin/python'),'-u','sim/run_sysnav_full.py',
                '--name',name,'--task',task_id[7:],'--max-seconds',str(duration),'--checkpoint',str(checkpoint),
                '--odometry','gicp','--lidar-profile','uniform64','--api-proxy',self.api_proxy,
                '--isaac-gpu','0','--vision-gpu','1','--sam-gpu','2','--vlm-gpu','3']
            if gui: command += ['--gui']
            if duration==-1: command+=['--api-budget','-1']
            elif task_id=='sysnav:meeting_crossroom':command+=['--api-budget','600']
            record_path=self.project/'data/sim/office/runs'/name
            return command,record_path,'office:'+name,record_path
        if task_id.startswith('office:') and task_id[7:] in TASKS:
            if duration not in (-1,20,30,60,120,300):
                raise ValueError('Unsupported duration')
            checkpoint = self.project/'data/readiness/training/go2-control-v5-rails/model_3595.pt'
            if not checkpoint.exists():
                raise ValueError('Recorded Go2 checkpoint unavailable')
            command = base+[str(self.root/'envs/isaac51/bin/python'),'-u','sim/run_office.py',
                '--name',name,'--task',task_id[7:],'--max-seconds',str(duration),'--checkpoint',str(checkpoint),
                '--odometry','gicp','--gicp-correspondence','.5','--lidar-profile','uniform64',
                '--sam','--surface-geometry','metric_plane','--api-proxy',self.api_proxy]
            if gui: command+=['--gui']
            record_path = self.project/'data/sim/office/runs'/name
            return command, record_path, 'office:'+name, record_path
        if task_id.startswith('habitat:') and task_id[8:] in self.episodes:
            if duration==-1 or gui: raise ValueError('Habitat benchmark uses its fixed action budget')
            output = self.root/'outputs/benchmarks'/name
            command = base+[str(self.root/'envs/habitat-bench/bin/python'),'-u','scripts/benchmarks/run_habitat.py',
                '--policies','vlm','--episode-id',task_id[8:],'--output',str(output)]
            record_path = output/'vlm'/task_id[8:]
            return command, record_path, 'habitat:'+name+':'+task_id[8:], output
        raise ValueError('Unknown task')

    def start(self, task_id, duration, gui=False):
        if not isinstance(gui,bool): raise ValueError('Invalid GUI flag')
        with self.lock:
            if any(j['process'].poll() is None for j in self.jobs.values()):
                raise ValueError('A dashboard simulation is already active')
            if gui and not task_id.startswith(('sysnav:','isaacmp3d:','office:')):
                raise ValueError('GUI requires an Isaac task')
            if gui and not (os.environ.get('DISPLAY') or os.environ.get('WAYLAND_DISPLAY')):
                raise ValueError('Isaac GUI requires DISPLAY or WAYLAND_DISPLAY; connect X/VNC first.')
            if gui:
                try:
                    probe=subprocess.run(['xdpyinfo'],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,timeout=3)
                except (OSError,subprocess.TimeoutExpired):
                    raise ValueError('Cannot access Isaac desktop display') from None
                if probe.returncode: raise ValueError('Cannot access Isaac desktop display')
            self.ensure_api_relay()
            name = 'ui-'+time.strftime('%Y%m%d-%H%M%S')+'-'+uuid.uuid4().hex[:5]
            command,path,record_id,stop_root = self.command(task_id,duration,name,gui=gui)
            log_path = self.root/'logs'/('dashboard-'+name+'.log')
            log_path.parent.mkdir(parents=True,exist_ok=True)
            with log_path.open('w') as log:
                process = subprocess.Popen(command,cwd=self.project,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,
                    env={**os.environ,'OMNI_KIT_ACCEPT_EULA':'YES','HABITAT_SIM_LOG':'quiet','MAGNUM_LOG':'quiet',
                         'HTTPS_PROXY':self.api_proxy},start_new_session=True)
            job = dict(id=name,name=name,task_id=task_id,kind='office' if task_id.startswith(('sysnav:','isaacmp3d:')) else task_id.split(':')[0],process=process,
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
        address = self.connection.getsockname()[0]
        allowed = {address+':'+str(self.server.server_port)}
        if address == '127.0.0.1':
            allowed.add('localhost:'+str(self.server.server_port))
        return self.headers.get('Host') in allowed

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
            if url.path=='/api/planning':
                return self.send(console.planning(query.get('id',[''])[0],float(query.get('time',['0'])[0])))
            if url.path=='/media':
                path = console.media(query.get('id',[''])[0],query.get('file',[''])[0])
                return self.send(path.read_bytes(),content_type=mimetypes.guess_type(str(path))[0])
            files = {'/':'index.html','/app.js':'app.js','/planning.js':'planning.js','/style.css':'style.css'}
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
                return self.send(self.server.console.start(str(body.get('task_id','')),body.get('duration',120),gui=body.get('gui',False)))
            if self.path=='/api/stop':
                return self.send(self.server.console.stop(str(body.get('job_id',''))))
            return self.send({'error':'Not found'},404)
        except (ValueError,TypeError) as error:
            return self.send({'error':str(error)},400)


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--port',type=int,default=8878)
    parser.add_argument('--lan-ip',help='Also listen on this LAN IPv4 address')
    args = parser.parse_args()
    addresses = ['127.0.0.1']
    if args.lan_ip and args.lan_ip not in addresses:
        addresses.append(args.lan_ip)
    console = Console()
    servers = []
    try:
        for address in addresses:
            server = ThreadingHTTPServer((address,args.port),Handler)
            server.console = console
            servers.append(server)
        for server in servers[1:]:
            threading.Thread(target=server.serve_forever,daemon=True).start()
        print('VLN console listening on '+', '.join(addresses)+':'+str(args.port),flush=True)
        servers[0].serve_forever()
    finally:
        for server in servers:
            server.server_close()
