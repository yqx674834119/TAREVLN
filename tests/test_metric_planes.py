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
from vln_policy.office_planes import fit_horizontal_support,metric_surface_candidates
from vln_policy.office_candidates import horizontal_surface_candidates,select_surface_anchor,refine_surface_candidates
from vln_policy.office_surface import verify_table_surface


@unittest.skipIf(np is None,'RGB-D numerical dependencies required')
class MetricPlaneTests(unittest.TestCase):
    def thin_table(self):
        h,w=240,320
        v,u=np.mgrid[:h,:w]
        depth=np.zeros((h,w));below=v>120
        depth[below]=.03*160/(v[below]-120)
        depth[(depth<1)|(depth>2.2)|(np.abs((160-u)*depth/160)>.5)]=0
        buffer=io.BytesIO();Image.new('RGB',(w,h)).save(buffer,format='JPEG')
        return {'label':'front','image_size':[w,h],'intrinsics':[160.,160.,160.,120.],
                'position':[0.,0.,.78],'rotation':np.eye(3).tolist(),
                'jpeg':base64.b64encode(buffer.getvalue()).decode(),
                'depth_mm':base64.b64encode(zlib.compress((depth*1000).astype('<u2').tobytes())).decode()}

    def test_two_pixel_rows_can_support_a_measured_plane(self):
        view=self.thin_table()
        self.assertEqual(horizontal_surface_candidates([view],[0,0,.33]),[])
        candidates=metric_surface_candidates([view],[0,0,.33])
        self.assertEqual(len(candidates),1)
        c=candidates[0]
        self.assertEqual(c['bbox_xyxy'][3]-c['bbox_xyxy'][1],2)
        self.assertAlmostEqual(c['anchor_xyz'][2],.75,places=3)
        self.assertGreater(c['grounding']['surface']['support_span_xy_m2'],.1)
        self.assertLessEqual(c['clearance_px'],1)

    def test_wall_height_slice_is_not_a_table(self):
        y,z=np.meshgrid(np.linspace(-.6,.6,80),np.linspace(.74,.76,5))
        points=np.stack((np.ones_like(y),y,z),axis=-1)
        self.assertFalse(fit_horizontal_support(points,np.ones(y.shape,bool))['valid'])

    def test_single_scanline_is_rank_deficient(self):
        x=np.linspace(1,2,100)
        points=np.stack((x,np.zeros(100),np.full(100,.75)),axis=-1)[None,:,:]
        self.assertFalse(fit_horizontal_support(points,np.ones((1,100),bool))['valid'])

    def test_rough_height_band_is_not_a_plane(self):
        x,y=np.meshgrid(np.linspace(1,2,25),np.linspace(-.5,.5,25))
        z=.75+.04*np.random.default_rng(13).uniform(-1,1,x.shape)
        self.assertFalse(fit_horizontal_support(np.stack((x,y,z),axis=-1),np.ones(x.shape,bool))['valid'])

    def test_missing_depth_cannot_support_a_plane(self):
        view=self.thin_table();view['depth_mm']=base64.b64encode(zlib.compress(bytes(240*320*2))).decode()
        self.assertEqual(metric_surface_candidates([view],[0,0,.33]),[])

    def test_sam_refinement_keeps_measured_thin_plane_and_rejects_empty_mask(self):
        view=self.thin_table();candidates=metric_surface_candidates([view],[0,0,.33])
        segmenter=Mock()
        for fill,expected in ((255,1),(0,0)):
            encoded=base64.b64encode(zlib.compress(bytes([fill])*(240*320//8))).decode()
            segmenter.segment.return_value={'image_size':[320,240],'masks':[encoded],'scores':[.99]}
            refined=refine_surface_candidates(segmenter,[view],[0,0,.33],candidates)
            self.assertEqual(len(refined),expected)
            if refined:self.assertLessEqual(refined[0]['component_pixels'],candidates[0]['component_pixels'])

    def test_individual_workdesk_cannot_complete_shared_meeting_table(self):
        view=self.thin_table();api=Mock();api.ask.return_value={'surface_id':'S0'}
        check=select_surface_anchor(api,'verify',[view],[0,0,.33],{'surface_geometry':'metric_plane'})
        grounding=check.pop('_measured_grounding')
        for role,expected in (('individual_workdesk',False),('unknown',False),('shared_meeting',True)):
            api.ask.return_value={'surface_type':'tabletop','table_role':role,'confidence':.99,'evidence':'visible group'}
            self.assertEqual(verify_table_surface(api,check,grounding,[view],{'task':'meeting'})['valid'],expected)

    def test_geometry_cannot_bypass_chair_identity_or_frame_matching(self):
        view=self.thin_table();api=Mock()
        api.ask.return_value={'surface_id':'S0','_measured_grounding':{'valid':True,'anchor_world_xyz':[9,9,9]}}
        check=select_surface_anchor(api,'verify',[view],[0,0,.33],{'surface_geometry':'metric_plane'})
        grounding=check.pop('_measured_grounding')
        self.assertAlmostEqual(grounding['anchor_world_xyz'][2],.75,places=3)
        api.ask.return_value={'surface_type':'chair_seat','confidence':.99,'evidence':'seat'}
        result=verify_table_surface(api,check,grounding,[view],{})
        self.assertFalse(result['valid']);self.assertEqual(result['reason'],'visual surface identity rejected')
        check['surface_proposal']['selected_support']['image_sha256']='another-frame'
        api.reset_mock()
        self.assertFalse(verify_table_surface(api,check,grounding,[view],{})['valid'])
        api.ask.assert_not_called()
