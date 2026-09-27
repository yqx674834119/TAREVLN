"""Geometric surface proposals; appearance and task identity remain VLM decisions."""
import base64
import io
import json
import zlib
from .office_grounding import ground_visual_anchor


def observed_surface_geometry(view,robot_position):
    import numpy as np
    width,height=view['image_size']
    depth=np.frombuffer(zlib.decompress(base64.b64decode(view['depth_mm'])),dtype='<u2').reshape(height,width)/1000.
    fx,fy,cx,cy=view['intrinsics']
    y,x=np.mgrid[:height,:width]
    rays=np.stack((depth,(cx-x)*depth/fx,(cy-y)*depth/fy),axis=-1)
    points=rays@np.asarray(view['rotation']).T+np.asarray(view['position'])
    normal=np.cross(points[2:-2,4:]-points[2:-2,:-4],points[4:,2:-2]-points[:-4,2:-2])
    norm=np.linalg.norm(normal,axis=-1)
    up=np.zeros((height,width))
    up[2:-2,2:-2]=np.divide(np.abs(normal[:,:,2]),norm,out=np.zeros_like(norm),where=norm>1e-8)
    distance=np.linalg.norm(points[:,:,:2]-np.asarray(robot_position)[:2],axis=-1)
    mask=((depth>0)&(depth<40)&(points[:,:,2]>=.2)&(points[:,:,2]<=1.2)
          &(distance>=.2)&(distance<=3.)&(up>.94)).astype(np.uint8)
    return points,mask


def horizontal_surface_candidates(views, robot_position, maximum_per_view=6):
    if not views:return []
    import cv2
    import numpy as np
    candidates=[]
    for view in views:
        points,mask=observed_surface_geometry(view,robot_position)
        count,labels,stats,_=cv2.connectedComponentsWithStats(mask,connectivity=8)
        accepted=[]
        for label in sorted(range(1,count),key=lambda n:int(stats[n,cv2.CC_STAT_AREA]),reverse=True):
            if stats[label,cv2.CC_STAT_AREA]<60:continue
            component=(labels==label).astype(np.uint8)
            clearance=cv2.distanceTransform(component,cv2.DIST_L2,5)
            # The most interior observed pixel avoids asking a model to guess a
            # tabletop coordinate near a depth discontinuity or a furniture leg.
            v,u=np.unravel_index(np.argmax(clearance),clearance.shape)
            if clearance[v,u]<3:continue
            anchor={'anchor_view':view['label'],'anchor_pixel_xy':[int(u),int(v)]}
            grounding=ground_visual_anchor(anchor,[view],robot_position)
            if not grounding['valid']:continue
            yy,xx=np.where(component)
            voxels=np.unique(np.floor(points[yy,xx,:2]/.1).astype(np.int32),axis=0)
            if len(voxels)<4:continue
            accepted.append({'view':view['label'],'pixel_xy':[int(u),int(v)],
                'bbox_xyxy':[int(xx.min()),int(yy.min()),int(xx.max()+1),int(yy.max()+1)],
                'component_pixels':len(xx),'clearance_px':float(clearance[v,u]),
                'anchor_xyz':grounding['anchor_world_xyz'],'grounding':grounding})
            if len(accepted)>=maximum_per_view:break
        candidates.extend(accepted)
    for index,candidate in enumerate(candidates):candidate['id']=f'S{index}'
    return candidates


def refine_surface_candidates(segmenter,views,robot_position,candidates):
    """Intersect SAM support with observed horizontal geometry, never fill holes."""
    import cv2
    import numpy as np
    from vln_perception.sam2_masks import decode_mask
    accepted=[]
    for view in views:
        grouped=[c for c in candidates if c['view']==view['label']]
        if not grouped:continue
        width,height=view['image_size']
        boxes=[[max(0,c['bbox_xyxy'][0]-8),max(0,c['bbox_xyxy'][1]-8),
                min(width,c['bbox_xyxy'][2]+8),min(height,c['bbox_xyxy'][3]+8)] for c in grouped]
        response=segmenter.segment({'jpeg':view['jpeg'],'boxes_xyxy':boxes})
        if response['image_size']!=view['image_size'] or len(response['masks'])!=len(grouped):
            raise ValueError('SAM mask dimensions differ from sensor frame')
        points,horizontal=observed_surface_geometry(view,robot_position)
        for candidate,encoded,score in zip(grouped,response['masks'],response['scores']):
            mask=decode_mask(encoded,view['image_size'])
            # SAM may cover a whole chair/table. Keep only the measured height
            # slice of the seed surface; legs and adjacent lower seats are out.
            support=(mask & (horizontal>0) & (np.abs(points[:,:,2]-candidate['anchor_xyz'][2])<.04)).astype(np.uint8)
            count,labels,stats,_=cv2.connectedComponentsWithStats(support,connectivity=8)
            x0,y0,x1,y1=candidate['bbox_xyxy']
            overlap=np.bincount(labels[y0:y1,x0:x1].ravel(),minlength=count);overlap[0]=0
            if not overlap.any():continue
            label=int(overlap.argmax())
            if stats[label,cv2.CC_STAT_AREA]<60:continue
            component=(labels==label).astype(np.uint8)
            clearance=cv2.distanceTransform(component,cv2.DIST_L2,5)
            v,u=np.unravel_index(clearance.argmax(),clearance.shape)
            if clearance[v,u]<3:continue
            grounding=ground_visual_anchor({'anchor_view':view['label'],'anchor_pixel_xy':[int(u),int(v)]},[view],robot_position)
            if not grounding['valid']:continue
            yy,xx=np.where(component)
            voxels=np.unique(np.floor(points[yy,xx,:2]/.1).astype(np.int32),axis=0)
            if len(voxels)<4:continue
            accepted.append({**candidate,'pixel_xy':[int(u),int(v)],
                'bbox_xyxy':[int(xx.min()),int(yy.min()),int(xx.max()+1),int(yy.max()+1)],
                'component_pixels':len(xx),'clearance_px':float(clearance[v,u]),
                'anchor_xyz':grounding['anchor_world_xyz'],'grounding':grounding,
                'sam':{'predicted_mask_iou':float(score),'mask_pixels':int(mask.sum()),
                       'support_pixels':len(xx),'support_voxels_xy':voxels.tolist(),
                       'mask_bits':base64.b64encode(zlib.compress(np.packbits(component.ravel(),bitorder='little').tobytes())).decode(),
                       'image_size':view['image_size'],'image_sha256':response['image_sha256']}})
    return accepted


def annotated_surface_views(views,candidates):
    from PIL import Image,ImageDraw
    images=[]
    for view in views:
        image=Image.open(io.BytesIO(base64.b64decode(view['jpeg']))).convert('RGB')
        for candidate in candidates:
            if candidate['view']==view['label'] and candidate.get('sam'):
                from vln_perception.sam2_masks import decode_mask
                support=decode_mask(candidate['sam']['mask_bits'],view['image_size'])
                image.paste(Image.blend(image,Image.new('RGB',image.size,'#22c55e'),.35),
                            mask=Image.fromarray(support.astype('uint8')*255))
        draw=ImageDraw.Draw(image)
        draw.text((8,8),view['label'],fill='white',stroke_width=2,stroke_fill='black')
        for candidate in candidates:
            if candidate['view']!=view['label']:continue
            u,v=candidate['pixel_xy']
            draw.rectangle(candidate['bbox_xyxy'],outline='cyan',width=2)
            draw.ellipse((u-4,v-4,u+4,v+4),fill='red')
            draw.text((u+7,v-9),candidate['id'],fill='yellow',stroke_width=2,stroke_fill='black')
        buffer=io.BytesIO();image.save(buffer,format='JPEG',quality=90);images.append(buffer.getvalue())
    return images


def select_surface_anchor(api,verification_prompt,views,robot_position,context,segmenter=None):
    if not views:return None
    candidates=horizontal_surface_candidates(views,robot_position)
    if segmenter and candidates:
        candidates=refine_surface_candidates(segmenter,views,robot_position,candidates)
    if not candidates:return None
    images=annotated_surface_views(views,candidates)
    prompt=(verification_prompt+'\nFor the anchor, select surface_id from the S-prefixed markers below, or null if none is a visible TABLE TOP belonging to the requested furniture group. '
        'The red dot is an observed horizontal surface point; the cyan box bounds its depth component. Green tint, if present, shows the intersection of a SAM mask and valid horizontal depth measurements. '
        'A geometric candidate may belong to a chair seat/arm, cabinet, shelf, or other object; it is NOT a table label. '
        'Inspect the object containing the red dot. Do not choose a chair/cabinet merely because a table is nearby. '
        'Return the complete verification JSON plus surface_id. Pixel coordinates will be filled mechanically from that ID. '
        'The markers are for perception only, not movement destinations. Available surface IDs by view: '+
        json.dumps({v['label']:[c['id'] for c in candidates if c['view']==v['label']] for v in views}))
    check=api.ask(prompt,images,{'purpose':'surface_proposal_verification',**context,'surface_candidates':candidates})
    selected=next((c for c in candidates if c['id']==check.get('surface_id')),None)
    # Ignore any invented image/world coordinates in the language response.
    check['anchor_view']=selected['view'] if selected else None
    check['anchor_pixel_xy']=selected['pixel_xy'] if selected else None
    check['surface_proposal']={'candidates':len(candidates),'selected_id':selected['id'] if selected else None,
                               'valid_selection':selected is not None,'sam_refined':bool(segmenter),
                               'selected_support':selected.get('sam') if selected else None}
    return check
