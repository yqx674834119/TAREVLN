"""Synchronized persistent cube cameras; gate render products between captures."""
import numpy as np
from isaacsim.sensors.camera import Camera
from isaacsim.core.utils.rotations import rot_matrix_to_quat
from sysnav_panorama import cube_rotations


def set_render_enabled(camera,enabled):
    # Pinned Isaac 5.1 Replicator render-product interface. pause() only pauses
    # callbacks and would leave expensive rendering active.
    camera._render_product.hydra_texture.set_updates_enabled(enabled)


class CubeCameraRig:
    def __init__(self,front,review):
        self.cameras=[front];self.review=review;self.rotations=cube_rotations()
        for i in range(1,6):
            camera=Camera(f'/World/CubeCamera{i}',frequency=-1,resolution=(640,640))
            camera.set_focal_length(2.4);camera.set_horizontal_aperture(4.8);camera.set_vertical_aperture(4.8)
            camera.set_clipping_range(.05,40)
            self.cameras.append(camera)

    def initialize(self):
        for camera in self.cameras[1:]:camera.initialize()
        self.set_enabled(False)

    def set_enabled(self,enabled):
        for camera in [*self.cameras,self.review]:set_render_enabled(camera,enabled)

    def capture(self,world,position,rotation,minimum_renders=4,maximum_renders=40):
        origin=position+rotation@np.array([0.,0.,.45])
        for camera,relative in zip(self.cameras,self.rotations):
            camera.set_world_pose(origin,rot_matrix_to_quat(rotation@relative),camera_axes='world')
        self.set_enabled(True)
        try:
            for count in range(1,maximum_renders+1):
                world.render()
                if count<minimum_renders:continue
                images=[c.get_rgba() for c in [*self.cameras,self.review]]
                if all(a is not None and a.ndim==3 and a.shape[2]>=3 and a.size for a in images):
                    return [a[:,:,:3].copy() for a in images[:6]],images[-1][:,:,:3].copy(),count
            raise RuntimeError('Synchronized camera rig did not produce all seven observations')
        finally:self.set_enabled(False)
