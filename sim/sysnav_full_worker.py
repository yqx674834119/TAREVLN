"""Sensor transport and supervision for the actual upstream SysNav ROS nodes.

No policy logic, goal coordinates, scene graph, or evaluator assets are here.
All motion is produced by native TARE -> localPlanner -> pathFollower.
"""
import argparse
import base64
from collections import Counter
import json
from pathlib import Path
import subprocess
import sys
import threading
import time

import cv2
import numpy as np
import rclpy
from std_msgs.msg import String
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from tare_planner.msg import (RoomNodeList, ObjectNodeList, DetectionResult, RoomType,
    NavigationQuery, VlmAnswer, TargetObjectInstruction, TargetObject, ViewpointRep,
    TargetObjectWithSpatial, ObjectType)
from office_ros_worker import Bridge as SensorBridge
from planning_recorder import PlanningRecorder
from vln_policy.sysnav_tasks import TASKS as PUBLIC_TASKS
TASKS = {key:value[1] for key,value in PUBLIC_TASKS.items()}


class Bridge(SensorBridge):
    def __init__(self, output):
        super().__init__()
        self.images = self.create_publisher(Image, '/camera/image', 5)
        self.keyboard = self.create_publisher(String, '/keyboard_input', 5)
        self.cv = CvBridge()
        self.events = (output/'events.jsonl').open('w', buffering=1)
        self.counts = Counter(self.counts)
        self.status, self.instruction = {}, None
        self.objects, self.rooms = {}, {}
        self.planning=PlanningRecorder(self,output)
        for topic, kind in [
            ('/room_nodes_list',RoomNodeList), ('/object_nodes_list',ObjectNodeList),
            ('/detection_result',DetectionResult), ('/room_type_query',RoomType),
            ('/room_type_answer',RoomType), ('/room_navigation_query',NavigationQuery),
            ('/room_navigation_answer',VlmAnswer), ('/target_object_instruction',TargetObjectInstruction),
            ('/target_object_query',TargetObject), ('/target_object_answer',TargetObject),
            ('/target_object_spatial_query',TargetObjectWithSpatial),
            ('/object_type_query',ObjectType), ('/object_type_answer',ObjectType),
            ('/anchor_object_query',TargetObject), ('/anchor_object_answer',TargetObject),
            ('/viewpoint_rep_header',ViewpointRep), ('/sysnav/status',String)]:
            self.create_subscription(kind,topic,lambda msg,t=topic:self.record(t,msg),50)

    def record(self, topic, msg):
        key=topic.strip('/')
        self.counts[key]+=1
        row={'topic':topic,'time':self.get_clock().now().nanoseconds/1e9}
        if topic=='/sysnav/status':
            self.status=json.loads(msg.data)
            return
        if topic=='/target_object_instruction':
            self.instruction=msg.target_object or None
            row.update(target=msg.target_object,room=msg.room_condition,attribute=msg.attribute_condition,spatial=msg.spatial_condition)
        elif topic=='/room_nodes_list':
            for node in msg.nodes:
                self.rooms[node.id]={'id':node.id,'area':node.area,'neighbors':list(node.neighbors)}
            row['rooms']=list(self.rooms.values())
        elif topic=='/object_nodes_list':
            for node in msg.nodes:
                for oid in node.object_id:
                    if node.status:
                        self.objects[oid]={'id':oid,'label':node.label,'xyz':[node.position.x,node.position.y,node.position.z]}
                    else:
                        self.objects.pop(oid,None)
            row['objects']=list(self.objects.values())
        elif topic=='/detection_result':
            row.update(labels=list(msg.label),ids=list(msg.track_id),stamp=msg.header.stamp.sec+msg.header.stamp.nanosec/1e9)
        elif isinstance(msg,TargetObject):
            row.update(id=msg.object_id,label=msg.object_label,is_target=msg.is_target)
        elif isinstance(msg,RoomType):
            row.update(id=msg.room_id,label=msg.room_type)
        elif isinstance(msg,NavigationQuery):
            row['query']=json.loads(msg.json)
        elif isinstance(msg,VlmAnswer):
            row.update(room_id=msg.room_id)
        elif isinstance(msg,ViewpointRep):
            row.update(viewpoint_id=msg.viewpoint_id)
        self.events.write(json.dumps(row)+'\n')

    def ingest(self,obs):
        self.planning.observe(obs)
        stamp=super().ingest(obs)
        if 'panorama' in obs:
            arr=np.frombuffer(base64.b64decode(obs['panorama']),dtype=np.uint8)
            image=cv2.imdecode(arr,cv2.IMREAD_COLOR)
            if image.shape != (640,1920,3):
                raise ValueError('Invalid SysNav panorama calibration')
            msg=self.cv.cv2_to_imgmsg(image,encoding='bgr8')
            msg.header.stamp=stamp
            msg.header.frame_id='camera'
            self.images.publish(msg)
            self.counts['camera']+=1
        return stamp


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--source',type=Path,required=True)
    ap.add_argument('--config',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--task',choices=TASKS,default='lounge')
    ap.add_argument('--instruction',help='Public language instruction only; no goal coordinates')
    args=ap.parse_args()
    forbidden=Path('/home/yqx/VLN/cross-building-floor-nav/data/sim/office/private/oracle.json')
    if forbidden.exists():
        raise RuntimeError('Evaluator data is visible to the policy')
    instruction=args.instruction or TASKS[args.task]
    if not 0<len(instruction)<=8192:raise ValueError('Invalid language instruction')
    (args.output/'isolation.json').write_text(json.dumps({'oracle_visible':False,'scene_visible':False,'stack':'native SysNav','instruction':instruction}))
    children,logs=[],[]
    rclpy.init()
    bridge=Bridge(args.output)
    threading.Thread(target=rclpy.spin,args=(bridge,),daemon=True).start()
    try:
        specs=[('tare_planner','tare_planner_node'),('tare_planner','room_segmentation'),
               ('terrain_analysis','terrainAnalysis'),('terrain_analysis_ext','terrainAnalysisExt'),
               ('local_planner','localPlanner'),('local_planner','pathFollower'),
               ('semantic_mapping','detection_node'),('semantic_mapping','semantic_mapping_node'),
               ('vlm_node','vlm_reasoning_node')]
        for package,executable in specs:
            log=(args.output/(executable+'.log')).open('w');logs.append(log)
            if package in ('semantic_mapping','vlm_node'):
                entry={'detection_node':'semantic_mapping.detection_node',
                       'semantic_mapping_node':'semantic_mapping.semantic_mapping_node',
                       'vlm_reasoning_node':'vlm_node.vlm_reasoning_node'}[executable]
                cmd=[sys.executable,'-u','-c',f'from {entry} import main; main()']
            else:
                cmd=[str(args.source/'install'/package/'lib'/package/executable)]
            cmd+=['--ros-args','--params-file',str(args.config),'-p','use_sim_time:=true']
            if executable=='room_segmentation':
                cmd+=['-r','__node:=room_segmentation']
            if executable=='localPlanner':
                cmd+=['-p','pathFolder:='+str(args.source/'src/base_autonomy/local_planner/paths')]
            children.append(subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT))
        sent=False
        trace=(args.output/'bridge.jsonl').open('w',buffering=1)
        for line in sys.stdin:
            obs=json.loads(line);bridge.ingest(obs)
            time.sleep(.03)
            for p,(_,name) in zip(children,specs):
                if p.poll() is not None:
                    raise RuntimeError(f'{name} exited with {p.returncode}; inspect node log')
            nodes=set(bridge.get_node_names())
            ready={'vlm_node','semantic_mapping_node','detection_node'}.issubset(nodes)
            if ready and not sent and bridge.counts['detection_result']>0:
                bridge.keyboard.publish(String(data=instruction));sent=True
            stop=bool(bridge.status.get('stop') and 0<=obs['time']-bridge.status.get('time',-100)<2)
            command=list(bridge.command)
            reason='native_sysnav_path_follower'
            if not bridge.instruction:
                command=[0.,0.,0.];reason='waiting_for_native_instruction'
            elif not 0<=obs['time']-bridge.command_time<.5:
                command=[0.,0.,0.];reason='stale_cmd_vel'
            if stop:
                command=[0.,0.,0.];reason='native_sysnav_target_reached'
            answer={'cmd_vel':command,'reason':reason,'stop':stop,'counts':dict(bridge.counts),
                    'sysnav':bridge.status,'instruction':bridge.instruction,'rooms':len(bridge.rooms),'objects':len(bridge.objects)}
            trace.write(json.dumps({'time':obs['time'],**answer})+'\n')
            bridge.planning.save(obs['time'])
            print(json.dumps(answer),flush=True)
    finally:
        for p in children:
            if p.poll() is None:p.terminate()
        for p in children:
            try:p.wait(timeout=8)
            except subprocess.TimeoutExpired:p.kill();p.wait()
        for log in logs:log.close()
        if rclpy.ok():rclpy.shutdown()


if __name__=='__main__':main()
