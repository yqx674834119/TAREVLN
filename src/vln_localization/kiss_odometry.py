"""LiDAR-only odometry with explicit input and failure contracts.

KISS-ICP is odometry, not loop-closing SLAM. The motion checks below are basic
plausibility checks, not calibrated localization uncertainty or a safety proof.
"""
import math
import time
import numpy as np
from kiss_icp.config import load_config
from kiss_icp.kiss_icp import KissICP


class LidarOdometry:
    def __init__(self,voxel_size=.1,initial_threshold=.5,fixed_threshold=None,method='kiss',prediction='constant_velocity'):
        cfg=load_config(None)
        cfg.data.min_range=.4
        cfg.data.max_range=11.
        cfg.data.deskew=False  # Current simulator emits instantaneous scans.
        cfg.mapping.voxel_size=voxel_size
        cfg.registration.max_num_threads=2
        cfg.registration.max_num_iterations=100
        cfg.adaptive_threshold.initial_threshold=initial_threshold
        cfg.adaptive_threshold.fixed_threshold=fixed_threshold
        self.config=cfg.model_dump()
        self.config['method']=method
        if method=='gicp':
            from .gicp_mapping import GICPMapping
            self.icp=GICPMapping(voxel_size,prediction)
            self.config['gicp']={'keyframes':10,'correspondence_m':.5,'max_iterations':50,'threads':2,'prediction':prediction}
        elif method=='kiss':self.icp=KissICP(cfg)
        else:raise ValueError('Unknown odometry method')
        self.last_time=None
        self.last_pose=np.eye(4)
        self.failure=None

    def update(self,frame):
        started=time.perf_counter()
        if self.failure:return {'valid':False,'control_ready':False,'reason':self.failure,'latched':True}
        try:
            if set(frame)!={'time','points_base_xyz'}:
                raise ValueError('Only scan timestamp and points in the base frame are allowed')
            stamp=frame['time']
            if type(stamp) not in (float,int) or not math.isfinite(stamp):
                raise ValueError('Invalid scan timestamp')
            if self.last_time is not None and not 0<stamp-self.last_time<=.5:
                raise ValueError('Nonmonotonic timestamp or scan gap over 0.5 seconds')
            points=np.asarray(frame['points_base_xyz'],dtype=np.float64)
            if points.ndim!=2 or points.shape[1]!=3 or len(points)<100 or not np.isfinite(points).all():
                raise ValueError('Invalid or sparse point cloud')
            if np.linalg.matrix_rank(points-points.mean(axis=0),tol=.01)<3:
                raise ValueError('Insufficient three-dimensional scene structure')
            _,keypoints=self.icp.register_frame(points,np.zeros(len(points)))
            pose=self.icp.last_pose.copy()
            if not np.isfinite(pose).all() or len(keypoints)<50:
                raise ValueError('Invalid registration output or too few keypoints')
            if self.last_time is not None:
                dt=stamp-self.last_time
                delta=np.linalg.inv(self.last_pose)@pose
                angle=math.acos(float(np.clip((np.trace(delta[:3,:3])-1)/2,-1,1)))
                if np.linalg.norm(delta[:3,3])>2*dt or angle>3*dt:
                    raise ValueError('Pose jump exceeds simulation motion envelope')
            self.last_time=stamp
            self.last_pose=pose
            return {'valid':True,'control_ready':False,'quality':'unvalidated','time':stamp,'odom_from_base':pose.tolist(),'keypoints':len(keypoints),
                    'registration':getattr(self.icp,'diagnostics',{}),
                    'processing_seconds':time.perf_counter()-started,'scope':'lidar_odometry_no_loop_closure'}
        except (ValueError,TypeError,KeyError) as error:
            self.failure=str(error)
            return {'valid':False,'control_ready':False,'reason':self.failure,'latched':True}
