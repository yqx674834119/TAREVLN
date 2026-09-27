"""Real ROS/DDS fault injection with synthetic messages, no robot connection."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import struct
import sys
import time
import rclpy
from rclpy.qos import QoSProfile, ReliabilityPolicy
from geometry_msgs.msg import TwistStamped
from nav_msgs.msg import Odometry
from sensor_msgs.msg import PointCloud2,PointField
from std_msgs.msg import Bool,String
from std_srvs.srv import SetBool

ap=argparse.ArgumentParser()
ap.add_argument('--out',type=Path,required=True)
args=ap.parse_args()
if os.environ.get('ROS_DOMAIN_ID')!='94' or os.environ.get('ROS_LOCALHOST_ONLY')!='1':
    raise SystemExit('This dry-run bench requires isolated localhost ROS domain 94')
args.out.mkdir(parents=True,exist_ok=False)
root=Path(__file__).resolve().parents[1]
log=(args.out/'gate.log').open('w')
child=subprocess.Popen([sys.executable,'-u',str(root/'sim/ros_motion_gate.py')],stdout=log,stderr=subprocess.STDOUT)
rclpy.init()
node=rclpy.create_node('readiness_fault_injector')
qos=QoSProfile(depth=1,reliability=ReliabilityPolicy.BEST_EFFORT)
pub={name:node.create_publisher(cls,'/readiness/'+name,qos) for name,cls in
     [('input_cmd_vel',TwistStamped),('odometry',Odometry),('scan',PointCloud2),('localization_healthy',Bool),('robot_link',Bool),('deadman',Bool),('estop',Bool)]}
records=[]
phase='startup'
node.create_subscription(String,'/readiness/guard_status',lambda m:records.append({'received_at':time.monotonic(),'phase':phase,**json.loads(m.data)}),qos)
arm=node.create_client(SetBool,'/readiness/arm')

def pump(duration, fault=None):
    deadline=time.monotonic()+duration
    next_send=0.
    while time.monotonic()<deadline:
        if child.poll() is not None:raise RuntimeError('Gate process exited')
        if time.monotonic()>=next_send:
            next_send=time.monotonic()+.04
            stamp=node.get_clock().now().to_msg()
            cmd=TwistStamped();cmd.header.stamp=stamp;cmd.header.frame_id='vehicle'
            cmd.twist.linear.x=.2;cmd.twist.angular.z=.2
            if fault=='invalid_command':cmd.twist.linear.x=float('nan')
            if fault!='stale_command':pub['input_cmd_vel'].publish(cmd)
            odom=Odometry();odom.header.stamp=stamp;odom.header.frame_id='map';odom.child_frame_id='vehicle';odom.pose.pose.orientation.w=1.
            if fault=='invalid_pose':odom.pose.pose.position.x=float('nan')
            pub['odometry'].publish(odom)
            scan=PointCloud2();scan.header.stamp=stamp;scan.header.frame_id='map';scan.width=20;scan.height=1;scan.point_step=16;scan.row_step=320
            scan.fields=[PointField(name=k,offset=i*4,datatype=7,count=1) for i,k in enumerate(('x','y','z'))]
            scan.data=b''.join(struct.pack('<ffff',i*.1,1.,0.,0.) for i in range(20))
            if fault=='empty_scan':scan.data=bytes(320)
            if fault=='invalid_scan_fields':scan.fields=[]
            if fault=='nonfinite_scan':scan.data=struct.pack('<ffff',float('nan'),1.,0.,0.)*20
            pub['scan'].publish(scan)
            for name in ['localization_healthy','robot_link','deadman']:
                if fault=='disconnected' and name=='robot_link':continue
                pub[name].publish(Bool(data=not(fault=='deadman_released' and name=='deadman')))
            if fault=='estop':pub['estop'].publish(Bool(data=True))
        rclpy.spin_once(node,timeout_sec=.005)

def request_arm():
    response=arm.call_async(SetBool.Request(data=True))
    deadline=time.monotonic()+2
    while not response.done() and time.monotonic()<deadline:rclpy.spin_once(node,timeout_sec=.01)
    if not response.done() or not response.result().success:raise RuntimeError('Arm preconditions failed')

try:
    if not arm.wait_for_service(timeout_sec=10):raise RuntimeError('Gate service unavailable')
    pump(.6)
    reports=[]
    for fault in ['stale_command','disconnected','invalid_pose','invalid_command','empty_scan','invalid_scan_fields','nonfinite_scan','deadman_released','estop']:
        phase='healthy_before_'+fault
        pump(.4);request_arm();pump(.4)
        healthy=[x for x in records if x['phase']==phase]
        if not any(any(abs(v)>.01 for v in x['output']) for x in healthy):raise RuntimeError('No healthy nonzero dry-run output')
        phase=fault;started=time.monotonic();pump(.6,fault)
        rows=[x for x in records if x['phase']==fault]
        zero=[x for x in rows if not x['armed'] and all(v==0 for v in x['output'])]
        latency=zero[0]['received_at']-started if zero else None
        passed=latency is not None and latency<.40 and all(all(v==0 for v in x['output']) for x in rows if x['received_at']>started+.40)
        reports.append({'fault':fault,'stop_latency_s':latency,'passed':passed})
        phase='recovery_'+fault;pump(.3)
        recovery=[x for x in records if x['phase']==phase]
        if any(x['armed'] or any(x['output']) for x in recovery):raise RuntimeError('Unexpected automatic rearm')
    report={'scope':'ROS 2 synthetic fault injection; no physical hardware or transport', 'passed':all(x['passed'] for x in reports),'cases':reports}
    (args.out/'report.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report),flush=True)
finally:
    (args.out/'trace.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in records))
    child.terminate()
    try:child.wait(timeout=5)
    except subprocess.TimeoutExpired:child.kill();child.wait()
    node.destroy_node();rclpy.shutdown();log.close()
