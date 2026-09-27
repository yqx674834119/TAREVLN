"""Transport-only Isaac interface for the official SysNav simulation graph.

The native sensorScanGeneration node owns scan synchronization and sensor_scan.
This bridge contains no mission planner, semantic decisions, or custom goals.
"""
import math
import numpy as np
import rclpy
from rclpy.node import Node
from builtin_interfaces.msg import Time
from geometry_msgs.msg import PointStamped, TransformStamped, TwistStamped
from nav_msgs.msg import Odometry, Path
from rosgraph_msgs.msg import Clock
from sensor_msgs.msg import PointCloud2, PointField
from std_msgs.msg import Header
from tf2_ros import TransformBroadcaster, StaticTransformBroadcaster
from vln_core.sysnav_sensor_contract import LIDAR_IN_BASE


class Bridge(Node):
    def __init__(self):
        super().__init__('isaac_sysnav_bridge',parameter_overrides=[rclpy.parameter.Parameter('use_sim_time',value=True)])
        self.clock_pub=self.create_publisher(Clock,'/clock',10)
        self.odom_pub=self.create_publisher(Odometry,'/state_estimation',10)
        self.cloud_pub=self.create_publisher(PointCloud2,'/registered_scan',5)
        self.tf=TransformBroadcaster(self)
        self.static_tf=StaticTransformBroadcaster(self)
        base=TransformStamped(header=Header(frame_id='sensor'),child_frame_id='base_link')
        base.transform.translation.z=-float(LIDAR_IN_BASE[2])
        base.transform.rotation.w=1.
        camera=TransformStamped(header=Header(frame_id='sensor'),child_frame_id='camera')
        camera.transform.rotation.x=-.5
        camera.transform.rotation.y=.5
        camera.transform.rotation.z=-.5
        camera.transform.rotation.w=.5
        self.static_tf.sendTransform([base,camera])
        self.command,self.command_time=[0.,0.,0.],-100.
        self.last_pose_time=None
        self.waypoint,self.path_nodes=None,0
        self.counts=dict(waypoints=0,paths=0,cmd_vel=0,synchronized_scans=0,sensor_scans=0)
        self.create_subscription(TwistStamped,'/cmd_vel',self.on_command,10)
        self.create_subscription(PointStamped,'/way_point',self.on_waypoint,10)
        self.create_subscription(Path,'/path',self.on_path,10)
        self.create_subscription(Odometry,'/state_estimation_at_scan',lambda _:self.count('synchronized_scans'),10)
        self.create_subscription(PointCloud2,'/sensor_scan',lambda _:self.count('sensor_scans'),5)

    def count(self,key):self.counts[key]+=1

    def on_command(self,msg):
        self.command=[msg.twist.linear.x,msg.twist.linear.y,msg.twist.angular.z]
        self.command_time=msg.header.stamp.sec+msg.header.stamp.nanosec/1e9
        self.count('cmd_vel')

    def on_waypoint(self,msg):
        self.waypoint=[msg.point.x,msg.point.y,msg.point.z]
        self.count('waypoints')

    def on_path(self,msg):
        self.path_nodes=len(msg.poses)
        self.count('paths')

    def ingest(self,obs):
        if obs.get('pose_frame')!='sensor':
            raise ValueError('Native SysNav odometry must be at the LiDAR ray origin')
        stamp=Time(sec=int(obs['time']),nanosec=int((obs['time']%1)*1e9))
        self.clock_pub.publish(Clock(clock=stamp))
        pose_time=obs.get('pose_time',obs['time'])
        pose_stamp=Time(sec=int(pose_time),nanosec=int((pose_time%1)*1e9))
        if pose_time!=self.last_pose_time:
            odom=Odometry(header=Header(stamp=pose_stamp,frame_id='map'),child_frame_id='sensor')
            odom.pose.pose.position.x,odom.pose.pose.position.y,odom.pose.pose.position.z=obs['position']
            odom.pose.pose.orientation.w,odom.pose.pose.orientation.x,odom.pose.pose.orientation.y,odom.pose.pose.orientation.z=obs['orientation']
            odom.twist.twist.linear.x,odom.twist.twist.linear.y,odom.twist.twist.linear.z=obs['velocity']
            self.odom_pub.publish(odom)
            tf=TransformStamped(header=odom.header,child_frame_id='sensor')
            tf.transform.translation.x,tf.transform.translation.y,tf.transform.translation.z=obs['position']
            tf.transform.rotation=odom.pose.pose.orientation
            self.tf.sendTransform(tf)
            # Native localPlanner removes yaw only and uses the odometry Z.
            vehicle=TransformStamped(header=odom.header,child_frame_id='vehicle')
            vehicle.transform.translation=tf.transform.translation
            vehicle.transform.rotation.z=math.sin(obs['yaw']/2)
            vehicle.transform.rotation.w=math.cos(obs['yaw']/2)
            self.tf.sendTransform(vehicle)
            self.last_pose_time=pose_time
        if 'points' in obs:
            if abs(pose_time-obs['time'])>1e-5:
                raise ValueError('A registered scan must carry its matching pose timestamp')
            points=np.asarray(obs['points'],dtype='<f4').reshape(-1,4)
            cloud=PointCloud2(header=Header(stamp=stamp,frame_id='map'),height=1,width=len(points),
                fields=[PointField(name=n,offset=i*4,datatype=PointField.FLOAT32,count=1) for i,n in enumerate(['x','y','z','intensity'])],
                is_bigendian=False,point_step=16,row_step=16*len(points),is_dense=True,data=points.tobytes())
            self.cloud_pub.publish(cloud)
        return stamp
