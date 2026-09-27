import base64
import io
import unittest
from unittest.mock import Mock
import zlib
try:
    import cv2
    import numpy as np
    from PIL import Image
except ImportError:
    np=None
from vln_policy.office_candidates import horizontal_surface_candidates,select_surface_anchor,refine_surface_candidates
from vln_policy.office_surface import verify_table_surface


@unittest.skipIf(np is None,'RGB-D numerical dependencies required')
class SurfaceCandidateTests(unittest.TestCase):
    def view(self,surface_height=.45):
        h,w=240,320
        v,u=np.mgrid[:h,:w]
        depth=np.zeros((h,w))
        below=v>h/2
        depth[below]=(.9-surface_height)*160/(v[below]-h/2)
        lateral=(w/2-u)*depth/160
        depth[(depth<1.)|(depth>2.2)|(np.abs(lateral)>.5)]=0
        mm=(depth*1000).astype('<u2')
        buffer=io.BytesIO();Image.new('RGB',(w,h)).save(buffer,format='JPEG')
        return {'label':'front','image_size':[w,h],'intrinsics':[160.,160.,160.,120.],
                'position':[0.,0.,.9],'rotation':np.eye(3).tolist(),
                'jpeg':base64.b64encode(buffer.getvalue()).decode(),
                'depth_mm':base64.b64encode(zlib.compress(mm.tobytes())).decode()}

    def test_surface_pixel_is_metric_and_inside_observed_patch(self):
        candidates=horizontal_surface_candidates([self.view()],[0.,0.,.3])
        self.assertEqual(len(candidates),1)
        self.assertTrue(candidates[0]['grounding']['valid'])
        self.assertGreater(candidates[0]['clearance_px'],7)
        self.assertAlmostEqual(candidates[0]['anchor_xyz'][2],.45,places=2)

    def test_floor_is_not_a_furniture_candidate(self):
        self.assertEqual(horizontal_surface_candidates([self.view(0.)],[0.,0.,.3]),[])

    def test_unknown_id_cannot_supply_invented_pixel(self):
        api=Mock();api.ask.return_value={'surface_id':'S999','anchor_view':'front','anchor_pixel_xy':[1,2]}
        check=select_surface_anchor(api,'verify',[self.view()],[0.,0.,.3],{})
        self.assertIsNone(check['anchor_pixel_xy'])
        self.assertFalse(check['surface_proposal']['valid_selection'])

    def test_valid_id_uses_measured_pixel_instead_of_model_coordinates(self):
        view=self.view();candidates=horizontal_surface_candidates([view],[0.,0.,.3])
        api=Mock();api.ask.return_value={'surface_id':'S0','anchor_view':'rear','anchor_pixel_xy':[1,2]}
        check=select_surface_anchor(api,'verify',[view],[0.,0.,.3],{})
        self.assertEqual(check['anchor_view'],'front')
        self.assertEqual(check['anchor_pixel_xy'],candidates[0]['pixel_xy'])

    def test_identity_format_failure_is_not_counted_as_correct_classification(self):
        view=self.view();candidate=horizontal_surface_candidates([view],[0.,0.,.3])[0]
        api=Mock();api.ask.side_effect=[{'type':'json_object'},{'type':'json_object'}]
        result=verify_table_surface(api,{},candidate['grounding'],[view],{})
        self.assertFalse(result['valid'])
        self.assertFalse(result['format_valid'])
        self.assertEqual(api.ask.call_count,2)

    def test_sam_empty_mask_cannot_be_replaced_by_unmasked_depth(self):
        view=self.view();candidates=horizontal_surface_candidates([view],[0.,0.,.3])
        mask=np.zeros((240,320),dtype='uint8')
        encoded=base64.b64encode(zlib.compress(np.packbits(mask.ravel(),bitorder='little').tobytes())).decode()
        segmenter=Mock();segmenter.segment.return_value={'image_size':[320,240],'masks':[encoded],'scores':[.99],'image_sha256':'test'}
        self.assertEqual(refine_surface_candidates(segmenter,[view],[0.,0.,.3],candidates),[])

    def test_sam_mask_must_intersect_real_depth_not_hallucinate_surface(self):
        view=self.view();candidates=horizontal_surface_candidates([view],[0.,0.,.3])
        mask=np.zeros((240,320),dtype='uint8');mask[:80,:]=1
        encoded=base64.b64encode(zlib.compress(np.packbits(mask.ravel(),bitorder='little').tobytes())).decode()
        segmenter=Mock();segmenter.segment.return_value={'image_size':[320,240],'masks':[encoded],'scores':[.99],'image_sha256':'test'}
        self.assertEqual(refine_surface_candidates(segmenter,[view],[0.,0.,.3],candidates),[])
