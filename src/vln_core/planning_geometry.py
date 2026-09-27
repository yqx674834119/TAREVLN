"""Frame-correct, bounded geometry for reviewing real planner messages."""
import numpy as np


def rotation_wxyz(q):
    w,x,y,z=np.asarray(q,dtype=float)/np.linalg.norm(q)
    return np.array([[1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w)],
                     [2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w)],
                     [2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)]])


def map_points(points,frame,pose):
    a=np.asarray(points,dtype=float).reshape(-1,3)
    if frame in ('map','/map'):return a
    if frame in ('vehicle','/vehicle') and pose is not None:
        # SysNav localPlanner explicitly removes yaw only, even on a tilted Go2.
        rotation=rotation_wxyz(pose['orientation'])
        yaw=np.arctan2(rotation[1,0],rotation[0,0])
        planar=np.array([[np.cos(yaw),-np.sin(yaw),0.],
                         [np.sin(yaw),np.cos(yaw),0.],[0.,0.,1.]])
        return a@planar.T+pose['position']
    raise ValueError('Missing pose or unsupported planner frame: '+frame)


def bounded_cloud(points,voxel=.15,limit=4500):
    a=np.asarray(points,dtype=float).reshape(-1,3)
    a=a[np.isfinite(a).all(axis=1)]
    if not len(a):return []
    _,indices=np.unique(np.floor(a/voxel).astype(np.int64),axis=0,return_index=True)
    a=a[np.sort(indices)]
    if len(a)>limit:a=a[np.linspace(0,len(a)-1,limit).astype(int)]
    return np.round(a,3).tolist()
