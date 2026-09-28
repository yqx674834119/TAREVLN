"""Calibrated cube-camera to SysNav equirectangular image (body +X forward)."""
import numpy as np
import cv2


def cube_rotations():
    result = []
    for angle in (0, np.pi/2, np.pi, -np.pi/2):
        c, s = np.cos(angle), np.sin(angle)
        result.append(np.array([[c,-s,0],[s,c,0],[0,0,1.]]))
    # Optical forward is local +X, local +Y is image left, +Z is image up.
    result += [np.array([[0,0,-1],[0,1,0],[1,0,0.]]),
               np.array([[0,0,1],[0,1,0],[-1,0,0.]])]
    return result


class Panorama:
    def __init__(self, intrinsics, size=640, backend='atlas'):
        if backend not in ('reference','atlas'):raise ValueError('Invalid panorama backend')
        self.backend=backend;self.size=size
        u, v = np.meshgrid(np.arange(1920), np.arange(640))
        azimuth = (961-u)*(2*np.pi/1920)
        elevation = (321-v)*(2*np.pi/1920)
        rays = np.stack((np.cos(elevation)*np.cos(azimuth),
                         np.cos(elevation)*np.sin(azimuth), np.sin(elevation)), axis=-1)
        local = [rays @ r for r in cube_rotations()]
        face = np.argmax(np.stack([r[:,:,0] for r in local]), axis=0)
        self.maps = []
        fx, fy, cx, cy = intrinsics
        for i, ray in enumerate(local):
            z = np.maximum(ray[:,:,0], 1e-6)
            x = (cx-fx*ray[:,:,1]/z).astype(np.float32)
            y = (cy-fy*ray[:,:,2]/z).astype(np.float32)
            mask = face == i
            if np.any((x[mask]<-1)|(x[mask]>size)|(y[mask]<-1)|(y[mask]>size)):
                raise ValueError('Cube calibration leaves uncovered panorama pixels')
            self.maps.append((mask,x,y))

        # One remap into a six-face atlas. Quantize each local coordinate first,
        # exactly as OpenCV does, so adding a large atlas offset cannot change
        # float rounding at a 1/32 pixel interpolation boundary.
        atlas_x=np.zeros((640,1920),np.float32);atlas_y=np.zeros_like(atlas_x)
        for i,(mask,x,y) in enumerate(self.maps):
            atlas_x[mask]=np.rint(np.clip(x[mask],0,size-1)*32)/32
            atlas_y[mask]=np.rint(np.clip(y[mask],0,size-1)*32)/32+i*size
        self.atlas_maps=cv2.convertMaps(atlas_x,atlas_y,cv2.CV_16SC2)

    def stitch(self, images):
        if len(images) != 6:
            raise ValueError('A synchronized six-face observation is required')
        if self.backend=='atlas':
            if any(im.shape!=(self.size,self.size,3) or im.dtype!=np.uint8 for im in images):
                raise ValueError('Expected six calibrated uint8 RGB faces')
            return cv2.remap(np.concatenate(images,axis=0),*self.atlas_maps,
                             cv2.INTER_LINEAR,borderMode=cv2.BORDER_REPLICATE)
        result = np.zeros((640,1920,3), dtype=np.uint8)
        for image, (mask,x,y) in zip(images,self.maps):
            sample = cv2.remap(image,x,y,cv2.INTER_LINEAR,borderMode=cv2.BORDER_REPLICATE)
            result[mask] = sample[mask]
        return result
