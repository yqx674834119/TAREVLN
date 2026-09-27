"""Collision-aware third-person camera, confined to evaluator/review rendering."""
import numpy as np


def unobstructed_eye(query,look,desired):
    look=np.asarray(look,dtype=float);desired=np.asarray(desired,dtype=float)
    delta=desired-look;length=float(np.linalg.norm(delta))
    if length<1e-6:return desired
    best=(0.,desired)
    for angle in (0.,np.pi/4,-np.pi/4,np.pi/2,-np.pi/2,3*np.pi/4,-3*np.pi/4,np.pi):
        c,s=np.cos(angle),np.sin(angle)
        direction=np.array([c*delta[0]-s*delta[1],s*delta[0]+c*delta[1],delta[2]])/length
        hits=[]
        def visit(hit,reverse=False):
            distance=length-float(hit.distance) if reverse else float(hit.distance)
            if not str(hit.collision).startswith('/World/Go2') and distance>.02:
                hits.append(distance)
            return True
        # Scanned surfaces may be one-sided in the scene query. Audit both
        # directions to keep a review camera out of the back of a wall.
        query.raycast_all(tuple(look),tuple(direction),length,visit)
        query.raycast_all(tuple(look+direction*length),tuple(-direction),length,lambda h:visit(h,True))
        clear=max(.05,min(length,min(hits)-.2)) if hits else length
        eye=look+direction*clear
        if clear>=min(1.7,length):return eye
        if clear>best[0]:best=(clear,eye)
    return best[1]
