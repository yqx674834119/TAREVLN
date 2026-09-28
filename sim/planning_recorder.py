"""Read-only recorder: native TARE geometry, source timestamps, and frames."""
from collections import deque
import json
from pathlib import Path
import threading
import numpy as np
from geometry_msgs.msg import PointStamped
from nav_msgs.msg import Path as ROSPath
from sensor_msgs.msg import PointCloud2, Image
from vln_core.planning_geometry import map_points,bounded_cloud


def stamp(message):
    return message.header.stamp.sec+message.header.stamp.nanosec/1e9


def cloud_xyz(message):
    fields={f.name:f for f in message.fields}
    endian='>' if message.is_bigendian else '<'
    formats={7:'f4',8:'f8'}
    if any(fields[k].datatype not in formats for k in ('x','y','z')):
        raise ValueError('Unsupported point datatype')
    dtype=np.dtype({'names':['x','y','z'],'formats':[endian+formats[fields[k].datatype] for k in ('x','y','z')],
                    'offsets':[fields[k].offset for k in ('x','y','z')],'itemsize':message.point_step})
    a=np.ndarray((message.height,message.width),dtype=dtype,buffer=bytes(message.data),
                 strides=(message.row_step,message.point_step))
    return np.stack([a[k].reshape(-1) for k in ('x','y','z')],axis=1)


class PlanningRecorder:
    def __init__(self,node,output,room_config):
        self.output=Path(output)/'planning';self.output.mkdir(exist_ok=True)
        self.index=(self.output/'index.jsonl').open('w',buffering=1)
        self.poses=deque(maxlen=150);self.layers={};self.lock=threading.RLock()
        self.last=-100.;self.rejected=0
        self.room_mask=None;self.mask_last_saved=-1.
        self.room_resolution=float(room_config['room_resolution'])
        self.room_shape=(int(room_config['room_x']),int(room_config['room_y']))
        topics={'free':'/freespace_cloud','candidates':'/viewpoint_vis_cloud',
                'obstacles':'/collision_cloud','free_paths':'/free_paths',
                'doors':'/door_cloud','room_map':'/room_map_cloud'}
        for key,topic in topics.items():
            node.create_subscription(PointCloud2,topic,lambda msg,k=key,t=topic:self.cloud(k,t,msg),2)
        for key,topic in {'path':'/path','coverage_path':'/local_path'}.items():
            node.create_subscription(ROSPath,topic,lambda msg,k=key,t=topic:self.path(k,t,msg),2)
        node.create_subscription(PointStamped,'/way_point',self.waypoint,5)
        node.create_subscription(Image,'/room_mask',self.mask,2)

    def mask(self,msg):
        if msg.encoding!='32SC1' or (msg.height,msg.width)!=self.room_shape:
            return
        dtype='>i4' if msg.is_bigendian else '<i4'
        a=np.ndarray((msg.height,msg.width),dtype=dtype,buffer=bytes(msg.data),strides=(msg.step,4)).copy()
        with self.lock:self.room_mask=(stamp(msg),a)

    def pose_for(self,t):
        with self.lock:
            if not self.poses:return None
            p=min(self.poses,key=lambda p:abs(p['time']-t))
            return p if abs(p['time']-t)<.21 else None

    def store(self,key,topic,msg,points,cloud=False):
        try:
            t=stamp(msg)
            a=map_points(points,msg.header.frame_id,self.pose_for(t))
            points=bounded_cloud(a) if cloud else np.round(a,3).tolist()
        except (ValueError,KeyError,TypeError):
            self.rejected+=1;return
        with self.lock:
            self.layers[key]=dict(topic=topic,time=t,source_frame=msg.header.frame_id,points=points)

    def cloud(self,key,topic,msg):
        try:
            points=cloud_xyz(msg)
        except (ValueError,KeyError,TypeError,BufferError):
            self.rejected+=1
            return
        self.store(key,topic,msg,points,True)

    def path(self,key,topic,msg):
        self.store(key,topic,msg,[[p.pose.position.x,p.pose.position.y,p.pose.position.z] for p in msg.poses])

    def waypoint(self,msg):
        self.store('waypoint','/way_point',msg,[[msg.point.x,msg.point.y,msg.point.z]])

    def observe(self,obs):
        with self.lock:
            self.poses.append({k:obs[k] for k in ('time','position','orientation','yaw')})

    def save(self,t):
        if t-self.last<.5:return
        with self.lock:
            if not self.poses:return
            # A source layer must never be newer than the recorded robot clock.
            layers={k:v for k,v in self.layers.items() if v['time']<=t+.001 and t-v['time']<3.}
            row=dict(time=t,frame='map',robot=self.poses[-1],layers=layers,rejected_frames=self.rejected)
            if self.room_mask and self.mask_last_saved<self.room_mask[0]<=t+.001:
                mask_t,mask=self.room_mask
                mask_name=f'rooms-{int(mask_t*1000):010d}.npz'
                origin=-np.asarray(mask.shape)*self.room_resolution/2
                temporary=self.output/(mask_name+'.tmp')
                with temporary.open('wb') as f:
                    np.savez_compressed(f,mask=mask,time=mask_t,resolution=self.room_resolution,origin=origin)
                temporary.replace(self.output/mask_name)
                self.mask_last_saved=mask_t
                row['room_mask_file']=mask_name
            name=f'plan-{int(t*1000):010d}.json'
            (self.output/name).write_text(json.dumps(row,separators=(',',':')))
            index_row=dict(time=t,file=name,robot=row['robot'],
                waypoint=layers.get('waypoint',{}).get('points',[]),
                counts={k:len(v['points']) for k,v in layers.items()})
            if 'room_mask_file' in row:
                index_row['room_mask_file']=row['room_mask_file']
            self.index.write(json.dumps(index_row)+'\n')
            self.last=t
