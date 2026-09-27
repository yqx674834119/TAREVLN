"""Gravity-aligned plane evidence for surfaces with a narrow image footprint."""
import base64
import hashlib
import zlib


def observed_points(view,robot_position):
    import numpy as np
    width,height=view['image_size']
    depth=np.frombuffer(zlib.decompress(base64.b64decode(view['depth_mm'])),dtype='<u2').reshape(height,width)/1000.
    fx,fy,cx,cy=view['intrinsics']
    y,x=np.mgrid[:height,:width]
    rays=np.stack((depth,(cx-x)*depth/fx,(cy-y)*depth/fy),axis=-1)
    points=rays@np.asarray(view['rotation']).T+np.asarray(view['position'])
    distance=np.linalg.norm(points[:,:,:2]-np.asarray(robot_position)[:2],axis=-1)
    valid=((depth>0)&(depth<40)&(points[:,:,2]>=.2)&(points[:,:,2]<=1.2)&(distance>=.2)&(distance<=3.))
    return points,valid


def fit_horizontal_support(points,mask):
    import numpy as np
    import cv2
    cloud=points[mask.astype(bool)]
    if len(cloud)<60:return {'valid':False,'reason':'too few measured surface points'}
    centre=np.median(cloud,axis=0)
    covariance=np.cov((cloud-centre).T)
    values,vectors=np.linalg.eigh(covariance)
    normal=vectors[:,0]
    residual=np.abs((cloud-centre)@normal)
    hull=cv2.convexHull(cloud[:,:2].astype(np.float32))
    span=float(cv2.contourArea(hull))
    # A wall's height slice is still vertical. A single image scanline is
    # rank deficient. Neither may become a horizontal surface by cropping.
    valid=(values[1]>=.0002 and abs(normal[2])>=.94 and span>=.015
           and float(np.quantile(residual,.95))<=.008 and float(residual.max())<=.015)
    return {'valid':bool(valid),'reason':'measured horizontal plane' if valid else 'plane orientation, span, rank or residual rejected',
            'normal_up_cosine':float(abs(normal[2])),'eigenvalues_m2':values.tolist(),
            'residual_p95_m':float(np.quantile(residual,.95)),'maximum_residual_m':float(residual.max()),
            'support_span_xy_m2':span,'measured_points':len(cloud)}


def candidate_from_support(view,robot_position,points,component,certificate=None):
    import cv2
    import numpy as np
    certificate=certificate or fit_horizontal_support(points,component)
    if not certificate['valid']:return None
    yy,xx=np.where(component)
    clearance=cv2.distanceTransform(component.astype(np.uint8),cv2.DIST_L2,5)
    maximum=float(clearance.max())
    eligible=clearance[yy,xx]>=maximum*.75
    centre=np.median(points[yy,xx,:2],axis=0)
    distance=np.linalg.norm(points[yy,xx,:2]-centre,axis=1)
    distance[~eligible]=np.inf
    index=int(distance.argmin());u,v=int(xx[index]),int(yy[index])
    point=points[v,u].tolist()
    horizontal=float(np.linalg.norm(points[v,u,:2]-np.asarray(robot_position)[:2]))
    if not .2<=horizontal<=3. or not .2<=point[2]<=1.2:return None
    grounding={'valid':True,'reason':'near observed mask-supported horizontal plane','mode':'observed_mask_plane',
               'anchor_world_xyz':point,'horizontal_distance_m':horizontal,'anchor_pixel_xy':[u,v],
               'anchor_view':view['label'],'surface':certificate}
    support={'mask_bits':base64.b64encode(zlib.compress(np.packbits(component.ravel(),bitorder='little').tobytes())).decode(),
             'image_size':view['image_size'],'image_sha256':hashlib.sha256(base64.b64decode(view['jpeg'])).hexdigest(),
             'geometry_method':'metric_plane','source':'metric_geometry'}
    return {'view':view['label'],'pixel_xy':[u,v],'bbox_xyxy':[int(xx.min()),int(yy.min()),int(xx.max()+1),int(yy.max()+1)],
            'component_pixels':len(xx),'clearance_px':float(clearance[v,u]),'anchor_xyz':point,'grounding':grounding,
            'geometry_method':'metric_plane','support':support}


def metric_surface_candidates(views,robot_position,maximum_per_view=8):
    import cv2
    import numpy as np
    candidates=[]
    for view in views:
        points,valid=observed_points(view,robot_position)
        heights=points[:,:,2]
        counts,edges=np.histogram(heights[valid],bins=100,range=(.2,1.2))
        accepted=[]
        for index in np.argsort(counts)[::-1]:
            if counts[index]<60:break
            level=(edges[index]+edges[index+1])/2
            band=(valid&(np.abs(heights-level)<.0075)).astype(np.uint8)
            count,labels,stats,_=cv2.connectedComponentsWithStats(band,connectivity=8)
            for label in range(1,count):
                if stats[label,cv2.CC_STAT_AREA]<60:continue
                component=(labels==label).astype(np.uint8)
                certificate=fit_horizontal_support(points,component)
                if not certificate['valid']:continue
                candidate=candidate_from_support(view,robot_position,points,component,certificate)
                if candidate is None:continue
                yy,xx=np.where(component)
                voxels={tuple(v) for v in np.unique(np.floor(points[yy,xx,:2]/.1).astype(np.int32),axis=0)}
                duplicate=any(abs(candidate['anchor_xyz'][2]-old['anchor_xyz'][2])<.025 and
                              len(voxels&old_voxels)>=.6*min(len(voxels),len(old_voxels)) for old,old_voxels in accepted)
                if not duplicate:accepted.append((candidate,voxels))
        # Rank by measured metric span, not projected pixel count. A large table
        # viewed at a grazing angle may occupy fewer pixels than a nearby seat.
        accepted.sort(key=lambda item:item[0]['grounding']['surface']['support_span_xy_m2'],reverse=True)
        candidates.extend(c for c,_ in accepted[:maximum_per_view])
    for index,candidate in enumerate(candidates):candidate['id']=f'S{index}'
    return candidates
