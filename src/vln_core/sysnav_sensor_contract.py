"""Convert GICP's base pose to the physical ray origin used by native SysNav.

The native `vehicle` planning frame is yaw-only at sensor height. `base_link`
denotes the actual articulated body; these are deliberately distinct frames.
"""
import numpy as np
from .planning_geometry import rotation_wxyz

LIDAR_IN_BASE = np.array([0., 0., .45])


class SensorObservationAdapter:
    def __init__(self):
        self.previous = None
        self.velocity = np.zeros(3)

    def convert(self, obs):
        if obs.get('pose_frame', 'base_link') != 'base_link':
            raise ValueError('Expected a base pose; refusing a second sensor offset')
        result = dict(obs)
        rotation = rotation_wxyz(obs['orientation'])
        position = np.asarray(obs['position']) + rotation @ LIDAR_IN_BASE
        time = obs.get('pose_time', obs['time'])
        if self.previous is not None and time > self.previous[0]:
            self.velocity = rotation.T @ ((position-self.previous[1])/(time-self.previous[0]))
        elif self.previous is None:
            self.velocity = np.asarray(obs['velocity'])
        if self.previous is None or time > self.previous[0]:
            self.previous = (time, position.copy())
        result.update(position=position.tolist(), velocity=self.velocity.tolist(),
                      pose_frame='sensor', base_position=list(obs['position']))
        # Registered scan endpoints are already in map. Never translate them again.
        return result
