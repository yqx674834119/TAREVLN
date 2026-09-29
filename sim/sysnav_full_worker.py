"""Sensor transport and supervision for the actual upstream SysNav ROS nodes.

No policy logic, goal coordinates, scene graph, or evaluator assets are here.
All motion is produced by native TARE -> localPlanner -> pathFollower.
"""
import argparse
import base64
from collections import Counter
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
import yaml

import cv2
import numpy as np
import rclpy
from std_msgs.msg import String
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from tare_planner.msg import (RoomNodeList, ObjectNodeList, DetectionResult, RoomType,
    NavigationQuery, VlmAnswer, TargetObjectInstruction, TargetObject, ViewpointRep,
    TargetObjectWithSpatial, ObjectType, RoomEarlyStop1)
from sysnav_sensor_bridge import Bridge as SensorBridge
from planning_recorder import PlanningRecorder
from vln_core.sysnav_sensor_contract import SensorObservationAdapter
from vln_policy.sysnav_tasks import TASKS as PUBLIC_TASKS
TASKS = {key:value[1] for key,value in PUBLIC_TASKS.items()}


class Bridge(SensorBridge):
    def __init__(self, output, source, config):
        super().__init__()
        self.images = self.create_publisher(Image, '/camera/image', 5)
        self.keyboard = self.create_publisher(String, '/keyboard_input', 5)
        self.cv = CvBridge()
        self.visual_dir = output/'visuals'
        self.visual_dir.mkdir(parents=True, exist_ok=True)
        self.visual_index = (output/'visuals.jsonl').open('w', buffering=1)
        self.visual_seen = set()
        self.events = (output/'events.jsonl').open('w', buffering=1)
        self.counts = Counter(self.counts)
        self.status, self.instruction = {}, None
        vocabulary=yaml.safe_load((source/'src/semantic_mapping/semantic_mapping/config/objects.yaml').read_text())
        self.supported_labels={p for value in vocabulary['prompts'].values() for p in value['prompts']}-{'unknown'}
        self.instruction_contract_error=None
        self.objects, self.rooms = {}, {}
        room_config=yaml.safe_load(config.read_text())['room_segmentation']['ros__parameters']
        self.planning=PlanningRecorder(self,output,room_config)
        self.sensor_adapter=SensorObservationAdapter()
        for topic, kind in [
            ('/room_nodes_list',RoomNodeList), ('/object_nodes_list',ObjectNodeList),
            ('/detection_result',DetectionResult), ('/room_type_query',RoomType),
            ('/room_type_answer',RoomType), ('/room_navigation_query',NavigationQuery),
            ('/room_early_stop_1',RoomEarlyStop1),
            ('/room_navigation_answer',VlmAnswer), ('/target_object_instruction',TargetObjectInstruction),
            ('/target_object_query',TargetObject), ('/target_object_answer',TargetObject),
            ('/target_object_spatial_query',TargetObjectWithSpatial),
            ('/object_type_query',ObjectType), ('/object_type_answer',ObjectType),
            ('/anchor_object_query',TargetObject), ('/anchor_object_answer',TargetObject),
            ('/viewpoint_rep_header',ViewpointRep), ('/sysnav/status',String)]:
            self.create_subscription(kind,topic,lambda msg,t=topic:self.record(t,msg),50)
        # These are native semantic_mapping outputs.  Keeping them as image
        # artifacts lets the dashboard show exactly what YOLO and SAM saw,
        # without running a second detector in the UI process.
        self.create_subscription(Image, '/annotated_image_detection',
                                 lambda msg:self.record_visual('yolo', msg), 10)
        self.create_subscription(Image, '/annotated_image',
                                 lambda msg:self.record_visual('sam', msg), 10)

    def record_visual(self, kind, msg):
        try:
            received = self.get_clock().now().nanoseconds/1e9
            t = msg.header.stamp.sec + msg.header.stamp.nanosec/1e9
            name = f'{kind}-{int(round(t*1000)):010d}.jpg'
            if name in self.visual_seen:return
            image = self.cv.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            if image is None or image.ndim != 3 or image.shape[2] != 3:
                return
            path = self.visual_dir/name
            ok, encoded = cv2.imencode('.jpg', image, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
            if not ok:return
            temporary=path.with_suffix('.tmp')
            temporary.write_bytes(encoded.tobytes())
            temporary.replace(path)
            row = {'kind':kind, 'topic':'/annotated_image_detection' if kind == 'yolo' else '/annotated_image',
                   'time':float(t), 'received_time':received, 'file':str(Path('visuals')/name),
                   'width':int(image.shape[1]), 'height':int(image.shape[0])}
            self.visual_index.write(json.dumps(row, separators=(',',':'))+'\n')
            self.visual_seen.add(name)
            self.counts[kind+'_frames']+=1
        except (OSError, ValueError, TypeError, cv2.error) as error:
            # Visualization is observational and must never stop navigation.
            self.get_logger().warning(f'Cannot record {kind} image: {type(error).__name__}')
            return

    def record(self, topic, msg):
        key=topic.strip('/')
        self.counts[key]+=1
        row={'topic':topic,'time':self.get_clock().now().nanoseconds/1e9}
        if topic=='/sysnav/status':
            self.status=json.loads(msg.data)
            return
        if topic=='/target_object_instruction':
            self.instruction=msg.target_object or None
            row.update(target=msg.target_object,anchor=msg.anchor_object,room=msg.room_condition,attribute=msg.attribute_condition,spatial=msg.spatial_condition)
            if self.instruction:
                unsupported=[label for label in (msg.target_object,msg.anchor_object) if label and label not in self.supported_labels]
                self.instruction_contract_error=('Unsupported detector identifier(s): '+str(unsupported)) if unsupported else None
                row['detector_contract_passed']=not unsupported
        elif topic=='/room_nodes_list':
            # Native RoomNodeList is a full snapshot, not append-only history.
            self.rooms={node.id:{'id':node.id,'area':node.area,'neighbors':list(node.neighbors),
                'connected':node.is_connected,'centroid':[node.centroid.x,node.centroid.y,node.centroid.z]}
                for node in msg.nodes}
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
            boxes=[]
            for x1,y1,x2,y2 in zip(msg.x1,msg.y1,msg.x2,msg.y2):
                boxes.append([float(x1),float(y1),float(x2),float(y2)])
            row.update(labels=list(msg.label),ids=list(msg.track_id),boxes=boxes,
                       confidence=[float(x) for x in msg.confidence],
                       stamp=msg.header.stamp.sec+msg.header.stamp.nanosec/1e9)
        elif isinstance(msg,TargetObject):
            row.update(id=msg.object_id,label=msg.object_label,is_target=msg.is_target)
        elif isinstance(msg,RoomType):
            row.update(id=msg.room_id,label=msg.room_type,anchor=[msg.anchor_point.x,msg.anchor_point.y,msg.anchor_point.z],in_room=msg.in_room)
        elif isinstance(msg,NavigationQuery):
            row['query']=json.loads(msg.json)
        elif isinstance(msg,RoomEarlyStop1):
            row.update(nearby=msg.room_id_1,current=msg.room_id_2,enter_wrong_room=msg.enter_wrong_room,
                       nearby_anchor=[msg.anchor_point_1.x,msg.anchor_point_1.y,msg.anchor_point_1.z])
        elif isinstance(msg,VlmAnswer):
            row.update(room_id=msg.room_id,answer_type=msg.answer_type,anchor=[msg.anchor_point.x,msg.anchor_point.y,msg.anchor_point.z])
        elif isinstance(msg,ViewpointRep):
            row.update(viewpoint_id=msg.viewpoint_id)
        self.events.write(json.dumps(row)+'\n')

    def ingest(self,obs):
        obs=self.sensor_adapter.convert(obs)
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
    ap.add_argument('--boundary',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--task',choices=TASKS,default='lounge')
    ap.add_argument('--instruction',help='Public language instruction only; no goal coordinates')
    ap.add_argument('--enable-room-early-stop',action='store_true')
    args=ap.parse_args()
    forbidden=Path('/home/yqx/VLN/cross-building-floor-nav/data/sim/office/private/oracle.json')
    if forbidden.exists():
        raise RuntimeError('Evaluator data is visible to the policy')
    instruction=args.instruction or TASKS[args.task]
    if not 0<len(instruction)<=8192:raise ValueError('Invalid language instruction')
    (args.output/'isolation.json').write_text(json.dumps({'oracle_visible':False,'scene_visible':False,'stack':'native SysNav','instruction':instruction}))
    children,logs=[],[]
    rclpy.init()
    bridge=Bridge(args.output,args.source,args.config)
    spin_thread=threading.Thread(target=rclpy.spin,args=(bridge,),daemon=True)
    spin_thread.start()
    try:
        specs=[('sensor_scan_generation','sensorScanGeneration'),('tare_planner','navigationBoundary'),
               ('tare_planner','tare_planner_node'),('tare_planner','room_segmentation'),
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
            if executable=='navigationBoundary':
                cmd+=['-p','boundary_file_dir:='+str(args.boundary)]
            if executable=='vlm_reasoning_node':
                cmd+=['-p','enable_room_early_stop:='+str(args.enable_room_early_stop).lower()]
            child_env=os.environ.copy()
            gpu_by_node={'detection_node':'SYSNAV_DETECTION_GPU',
                         'semantic_mapping_node':'SYSNAV_SEMANTIC_GPU',
                         'vlm_reasoning_node':'SYSNAV_VLM_GPU'}
            gpu_name=gpu_by_node.get(executable)
            if gpu_name:
                child_env['CUDA_VISIBLE_DEVICES']=child_env.get(gpu_name,child_env.get('CUDA_VISIBLE_DEVICES','0'))
            children.append(subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT,env=child_env))
        sent=False
        runtime_saved=False
        trace=(args.output/'bridge.jsonl').open('w',buffering=1)
        for line in sys.stdin:
            obs=json.loads(line);bridge.ingest(obs)
            time.sleep(.03)
            if bridge.instruction_contract_error:
                raise RuntimeError(bridge.instruction_contract_error)
            for p,(_,name) in zip(children,specs):
                if p.poll() is not None:
                    raise RuntimeError(f'{name} exited with {p.returncode}; inspect node log')
            nodes=set(bridge.get_node_names())
            ready={'vlm_node','semantic_mapping_node','detection_node'}.issubset(nodes)
            if ready and not sent and bridge.counts['detection_result']>0:
                bridge.keyboard.publish(String(data=instruction));sent=True
            if sent and not runtime_saved and bridge.counts['sensor_scans']>0:
                from sysnav_runtime_audit import capture_runtime
                capture_runtime(bridge,args.output,args.source,specs,children)
                runtime_saved=True
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
        spin_thread.join(timeout=5)
        bridge.visual_index.close()


if __name__=='__main__':main()
