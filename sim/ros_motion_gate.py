"""Independent ROS 2 dry-run velocity gate. Contains no robot transport."""
import json
import math
import struct
import time
import rclpy
from rclpy.node import Node
from rclpy.clock import Clock, ClockType
from rclpy.qos import QoSProfile, ReliabilityPolicy
from geometry_msgs.msg import TwistStamped
from nav_msgs.msg import Odometry
from sensor_msgs.msg import PointCloud2
from std_msgs.msg import Bool, String
from std_srvs.srv import SetBool
from vln_core.motion_guard import GuardConfig, MotionGuard


class VelocityGate(Node):
    def __init__(self):
        super().__init__('readiness_velocity_gate')
        self.guard=MotionGuard(GuardConfig(required_health=('pose','scan','localization','robot_link','operator_deadman')))
        self.sequence=0
        self.last_stamp=None
        qos=QoSProfile(depth=1,reliability=ReliabilityPolicy.BEST_EFFORT)
        self.output=self.create_publisher(TwistStamped,'/readiness/guarded_cmd_vel',qos)
        self.status=self.create_publisher(String,'/readiness/guard_status',qos)
        self.create_subscription(TwistStamped,'/readiness/input_cmd_vel',self.command,qos)
        self.create_subscription(Odometry,'/readiness/odometry',self.pose,qos)
        self.create_subscription(PointCloud2,'/readiness/scan',self.scan,qos)
        for topic,source in [('localization_healthy','localization'),('robot_link','robot_link'),('deadman','operator_deadman')]:
            self.create_subscription(Bool,'/readiness/'+topic,lambda m,s=source:self.guard.update_health(s,time.monotonic(),m.data),qos)
        self.create_subscription(Bool,'/readiness/estop',lambda m:self.guard.set_estop(m.data),qos)
        self.create_service(SetBool,'/readiness/arm',self.arm)
        self.create_timer(.02,self.tick,clock=Clock(clock_type=ClockType.STEADY_TIME))

    def fresh(self,header):
        stamp=header.stamp.sec+header.stamp.nanosec/1e9
        age=self.get_clock().now().nanoseconds/1e9-stamp
        return 0<=age<=self.guard.config.timeout_s,stamp

    def command(self,msg):
        fresh,stamp=self.fresh(msg.header)
        values=[msg.twist.linear.x,msg.twist.linear.y,msg.twist.angular.z]
        unsupported=[msg.twist.linear.z,msg.twist.angular.x,msg.twist.angular.y]
        if not fresh or msg.header.frame_id!='vehicle' or any(v!=0 for v in unsupported) or (self.last_stamp is not None and stamp<=self.last_stamp):
            self.guard.stop('invalid_command_header_or_axes')
            self.guard.command_time=None
            return
        self.last_stamp=stamp
        self.sequence+=1
        self.guard.update_command(values,time.monotonic(),self.sequence)

    def pose(self,msg):
        fresh,_=self.fresh(msg.header)
        p,q=msg.pose.pose.position,msg.pose.pose.orientation
        numbers=[p.x,p.y,p.z,q.w,q.x,q.y,q.z,*msg.pose.covariance]
        valid=(fresh and msg.header.frame_id=='map' and msg.child_frame_id=='vehicle'
               and all(math.isfinite(x) for x in numbers)
               and abs(sum(x*x for x in [q.w,q.x,q.y,q.z])-1)<.05)
        self.guard.update_health('pose',time.monotonic(),valid)

    def scan(self,msg):
        fresh,_=self.fresh(msg.header)
        valid=(fresh and msg.header.frame_id=='map' and msg.width*msg.height>=20
               and msg.point_step>=12 and len(msg.data)==msg.row_step*msg.height
               and msg.row_step>=msg.width*msg.point_step)
        fields={f.name:f for f in msg.fields}
        valid=valid and all(k in fields and fields[k].datatype==7 and fields[k].count==1
                            and 0<=fields[k].offset<=msg.point_step-4 for k in ('x','y','z'))
        if valid:
            # A packet header alone is not evidence of a usable point cloud.
            unpack=struct.Struct('>f' if msg.is_bigendian else '<f').unpack_from
            points=[]
            total=msg.width*msg.height
            indices=range(0,total,max(1,math.ceil(total/1024)))
            for index in indices:
                row,col=divmod(index,msg.width)
                offset=row*msg.row_step+col*msg.point_step
                p=tuple(unpack(msg.data,offset+fields[k].offset)[0] for k in ('x','y','z'))
                if all(math.isfinite(v) for v in p):points.append(p)
            valid=(len(points)>=20 and len(points)>=.9*len(indices)
                   and any(max(p[i] for p in points)-min(p[i] for p in points)>.1 for i in range(3)))
        self.guard.update_health('scan',time.monotonic(),valid)

    def arm(self,request,response):
        if not request.data:
            self.guard.stop('operator_disarmed')
            response.success=True
        else:
            response.success=self.guard.arm(time.monotonic(),operator_requested=True)
        response.message=self.guard.reason
        return response

    def tick(self):
        values=self.guard.tick(time.monotonic())
        message=TwistStamped()
        message.header.stamp=self.get_clock().now().to_msg()
        message.header.frame_id='vehicle'
        message.twist.linear.x,message.twist.linear.y,message.twist.angular.z=values
        self.output.publish(message)
        status=String()
        status.data=json.dumps({'scope':'dry_run_no_robot_transport','armed':self.guard.armed,'reason':self.guard.reason,'output':values})
        self.status.publish(status)


def main():
    rclpy.init()
    node=VelocityGate()
    try:rclpy.spin(node)
    finally:
        node.guard.stop('process_shutdown')
        node.tick()
        node.destroy_node()
        rclpy.shutdown()


if __name__=='__main__':main()
