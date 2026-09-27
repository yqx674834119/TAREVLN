import array
import base64
import json
from pathlib import Path
import sys
import unittest
import zlib
from vln_policy.office_grounding import ground_visual_anchor


class MetricGroundingTests(unittest.TestCase):
    def view(self, distance):
        # A horizontal tabletop 0.5 m below the camera, not a constant-depth wall.
        fy = distance*46/.5
        pixels = array.array('H', [int(.5*fy/(v+30)*1000) for v in range(33) for _ in range(33)])
        if sys.byteorder != 'little': pixels.byteswap()
        return {'label': 'front', 'image_size': [33,33], 'depth_mm': base64.b64encode(zlib.compress(pixels.tobytes())).decode(),
                'intrinsics': [184.,fy or 1.,16.,-30.], 'position': [0.,0.,1.],
                'rotation': [[1,0,0],[0,1,0],[0,0,1]]}

    def test_far_visual_match_cannot_be_arrival(self):
        check={'anchor_view': 'front', 'anchor_pixel_xy': [16,16]}
        far=ground_visual_anchor(check,[self.view(5.)],[0.,0.,.2])
        self.assertFalse(far['valid'])
        self.assertEqual(far['horizontal_distance_m'],5.)
        near=ground_visual_anchor(check,[self.view(2.)],[0.,0.,.2])
        self.assertTrue(near['valid'])
        self.assertEqual(near['anchor_world_xyz'],[2.,0.,.5])

    def test_missing_out_of_frame_and_empty_depth_fail_closed(self):
        for check,views in [({},[]),({'anchor_view':'front','anchor_pixel_xy':[-1,2]},[self.view(2.)]),
                            ({'anchor_view':'front','anchor_pixel_xy':[16,16]},[self.view(0.)])]:
            self.assertFalse(ground_visual_anchor(check,views,[0.,0.,.2])['valid'])

    def test_world_rotation_and_pixel_ray_are_applied(self):
        view=self.view(2.)
        view['rotation']=[[0,-1,0],[1,0,0],[0,0,1]]
        result=ground_visual_anchor({'anchor_view':'front','anchor_pixel_xy':[17,16]},[view],[0.,0.,.2])
        self.assertEqual(result['anchor_world_xyz'],[2/184,2.,.5])

    def test_nearby_floor_is_not_a_table_anchor(self):
        view=self.view(1.)
        view['position'][2]=.5
        result=ground_visual_anchor({'anchor_view':'front','anchor_pixel_xy':[16,16]},[view],[0.,0.,.2])
        self.assertFalse(result['valid'])
        self.assertIn('floor',result['reason'])

    def test_fixed_recorded_rgbd_regression(self):
        samples=json.loads((Path(__file__).parent/'fixtures/office_anchor_patches.json').read_text())
        for sample in samples:
            with self.subTest(sample=sample['name']):
                result=ground_visual_anchor(sample['check'],[sample['view']],sample['robot_position'])
                self.assertEqual(result['valid'], sample['expected_valid'], result)

    def test_vertical_wall_at_valid_height_is_not_tabletop(self):
        view=self.view(2.)
        pixels=array.array('H',[2000]*(33*33))
        if sys.byteorder != 'little': pixels.byteswap()
        view['depth_mm']=base64.b64encode(zlib.compress(pixels.tobytes())).decode()
        result=ground_visual_anchor({'anchor_view':'front','anchor_pixel_xy':[16,16]},[view],[0.,0.,.2])
        self.assertFalse(result['valid'])
        self.assertIn('horizontal',result['reason'])
