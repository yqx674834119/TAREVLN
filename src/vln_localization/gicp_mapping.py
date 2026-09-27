"""Observed-keyframe GICP map registration using the upstream small_gicp API."""
from collections import deque
import math
import numpy as np
import small_gicp


class GICPMapping:
    def __init__(self,voxel_size=.1,prediction='constant_velocity'):
        self.voxel_size=voxel_size
        if prediction not in ('constant_velocity','static'):raise ValueError('Invalid prediction')
        self.prediction=prediction
        self.last_pose=np.eye(4)
        self.last_delta=np.eye(4)
        self.keyframe_pose=np.eye(4)
        self.keyframes=deque(maxlen=10)
        self.diagnostics={}

    def register_frame(self,points,timestamps):
        points=points[(np.linalg.norm(points,axis=1)>.4)&(np.linalg.norm(points,axis=1)<11)]
        if len(points)<100:raise ValueError('Too few range-valid points')
        if self.keyframes:
            target=np.concatenate(self.keyframes)
            guess=self.last_pose@self.last_delta if self.prediction=='constant_velocity' else self.last_pose
            result=small_gicp.align(target,points,init_T_target_source=guess,
                registration_type='GICP',downsampling_resolution=self.voxel_size,
                max_correspondence_distance=.5,num_threads=2,max_iterations=50)
            eigen=np.linalg.eigvalsh(result.H)
            condition=float(eigen[-1]/max(eigen[0],1e-12))
            self.diagnostics={'converged':bool(result.converged),'iterations':int(result.iterations),
                              'inliers':int(result.num_inliers),'hessian_condition':condition}
            if not result.converged or result.num_inliers<100 or eigen[0]<=0 or condition>1e8:
                raise ValueError('Unconverged, sparse, or ill-conditioned GICP registration')
            pose=result.T_target_source
            self.last_delta=np.linalg.inv(self.last_pose)@pose
            self.last_pose=pose.copy()
        delta=np.linalg.inv(self.keyframe_pose)@self.last_pose
        angle=math.acos(float(np.clip((np.trace(delta[:3,:3])-1)/2,-1,1)))
        if not self.keyframes or np.linalg.norm(delta[:3,3])>.3 or angle>.2:
            cloud=points@self.last_pose[:3,:3].T+self.last_pose[:3,3]
            self.keyframes.append(cloud)
            self.keyframe_pose=self.last_pose.copy()
        return points,points
