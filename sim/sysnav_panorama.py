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
    def __init__(self, intrinsics, size=640):
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

    def stitch(self, images):
        if len(images) != 6:
            raise ValueError('A synchronized six-face observation is required')
        result = np.zeros((640,1920,3), dtype=np.uint8)
        for image, (mask,x,y) in zip(images,self.maps):
            sample = cv2.remap(image,x,y,cv2.INTER_LINEAR,borderMode=cv2.BORDER_REPLICATE)
            result[mask] = sample[mask]
        return result
