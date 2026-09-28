"""Wall-clock profiling; never substitutes simulated sensor observations."""
from collections import defaultdict
import json
from pathlib import Path
import time


class JsonTail:
    def __init__(self, path):
        self.path=Path(path); self.offset=0

    def read(self):
        rows=[]
        try:
            with self.path.open() as stream:
                stream.seek(self.offset)
                while True:
                    start=stream.tell(); line=stream.readline()
                    if not line or not line.endswith('\n'):
                        self.offset=start; break
                    self.offset=stream.tell()
                    try: rows.append(json.loads(line))
                    except ValueError: pass
        except OSError: pass
        return rows


class RuntimeMetrics:
    def __init__(self, output, sim_start, gui=False):
        self.start=time.perf_counter(); self.sim_start=sim_start; self.gui=gui
        self.previous=(self.start,sim_start,0,0)
        self.stages=defaultdict(float); self.renders=0; self.viewport_frames=0
        self.latest={}; self.api_calls=0; self.api_seconds=0.
        self.api=JsonTail(output/'agent/api.jsonl')
        self.perception={k:JsonTail(output/f'agent/{k}_timing.jsonl') for k in ('yolo','sam')}
        self.stream=(output/'runtime_metrics.jsonl').open('w',buffering=1)

    def add(self, stage, started):
        self.stages[stage]+=time.perf_counter()-started

    def sample(self, sim_time, steps, worker_ms=None):
        for row in self.api.read():
            if isinstance(row.get('seconds'),(int,float)):
                self.api_calls+=1; self.api_seconds+=row['seconds']
                self.latest['vlm_api_last_ms']=row['seconds']*1000
                self.latest['vlm_api_last_status']=row.get('status')
        for name, tail in self.perception.items():
            for row in tail.read():
                self.latest[name+'_last_ms']=row.get('milliseconds')
                self.latest[name+'_source_time']=row.get('time')
        now=time.perf_counter(); wall=max(now-self.start,1e-9)
        previous_wall,previous_sim,previous_render,previous_viewport=self.previous
        interval=max(now-previous_wall,1e-9)
        self.latest.update(time=sim_time, wall_seconds=wall, realtime_factor=(sim_time-self.sim_start)/wall,
            interval_realtime_factor=(sim_time-previous_sim)/interval,physics_steps=steps,physics_hz=steps/wall,
            render_calls=self.renders,render_fps=self.renders/wall,
            interval_render_fps=(self.renders-previous_render)/interval,
            viewport_fps=self.viewport_frames/wall if self.gui else None,
            interval_viewport_fps=(self.viewport_frames-previous_viewport)/interval if self.gui else None,
            gui=self.gui,worker_latency_ms=worker_ms,vlm_api_calls=self.api_calls,
            vlm_api_avg_ms=self.api_seconds/self.api_calls*1000 if self.api_calls else None,
            stage_wall_seconds=dict(self.stages),
            stage_wall_fraction={key:value/wall for key,value in self.stages.items()})
        # A rolling window stable enough to display at a 2 s web refresh.
        if interval>=2: self.previous=(now,sim_time,self.renders,self.viewport_frames)
        self.stream.write(json.dumps(self.latest,separators=(',',':'))+'\n')
        return dict(self.latest)

    def close(self): self.stream.close()
