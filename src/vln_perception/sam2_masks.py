"""SAM2 image/box segmentation following SysNav's mask prediction interface."""
import base64
import hashlib
import io
import time
import zlib


class Sam2Masks:
    def __init__(self,checkpoint,config='configs/sam2.1/sam2.1_hiera_l.yaml',device='cuda:0'):
        import torch
        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor
        self.torch=torch
        self.device=device
        # Optional compiled hole/sprinkle postprocessing is disabled explicitly;
        # the model runs the same image encoder and prompt/mask decoder.
        model=build_sam2(config,str(checkpoint),device=device,apply_postprocessing=False)
        self.predictor=SAM2ImagePredictor(model)
        self.image_hash=None

    def segment(self,frame):
        import numpy as np
        from PIL import Image
        if set(frame)!={'jpeg','boxes_xyxy'}:
            raise ValueError('SAM accepts only RGB JPEG and image-space boxes')
        encoded=base64.b64decode(frame['jpeg'],validate=True)
        if len(encoded)>4*1024*1024:raise ValueError('Image payload too large')
        image=Image.open(io.BytesIO(encoded)).convert('RGB')
        width,height=image.size
        if max(width,height)>2048:raise ValueError('Image dimensions too large')
        boxes=np.asarray(frame['boxes_xyxy'],dtype=np.float32)
        if boxes.ndim!=2 or boxes.shape[1]!=4 or not 1<=len(boxes)<=24 or not np.isfinite(boxes).all():
            raise ValueError('Invalid SAM box prompts')
        if (np.any(boxes[:,:2]<0) or np.any(boxes[:,2]>width) or np.any(boxes[:,3]>height)
            or np.any(boxes[:,2:]<=boxes[:,:2])):raise ValueError('SAM box outside image')
        started=time.perf_counter()
        digest=hashlib.sha256(encoded).hexdigest()
        torch=self.torch
        with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
            if self.image_hash!=digest:
                self.predictor.set_image(np.array(image,copy=True))
                self.image_hash=digest
            masks,scores,_=self.predictor.predict(point_coords=None,point_labels=None,box=boxes,multimask_output=False)
            torch.cuda.synchronize()
        masks=np.asarray(masks,dtype=bool)
        if masks.ndim==2:masks=masks[None]
        if masks.ndim==4:masks=masks[:,0]
        scores=np.asarray(scores).reshape(-1)
        if masks.shape!=(len(boxes),height,width) or len(scores)!=len(boxes) or not np.isfinite(scores).all():
            raise ValueError('Invalid SAM output shape or scores')
        return {'image_sha256':digest,'image_size':[width,height],
                'masks':[base64.b64encode(zlib.compress(np.packbits(mask.ravel(),bitorder='little').tobytes())).decode() for mask in masks],
                'scores':scores.tolist(),'mask_pixels':masks.sum(axis=(1,2)).tolist(),
                'processing_seconds':time.perf_counter()-started,
                'scope':'SAM2 instance/part masks from RGB and geometric box prompts; no semantic labels or 3D pose supplied'}


def decode_mask(encoded,image_size):
    import numpy as np
    width,height=image_size
    packed=np.frombuffer(zlib.decompress(base64.b64decode(encoded)),dtype=np.uint8)
    if len(packed)!=(width*height+7)//8:raise ValueError('Invalid packed SAM mask length')
    return np.unpackbits(packed,bitorder='little')[:width*height].reshape(height,width).astype(bool)
