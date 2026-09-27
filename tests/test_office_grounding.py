import array
import base64
import sys
import unittest
import zlib
from vln_policy.office_grounding import ground_visual_anchor


class MetricGroundingTests(unittest.TestCase):
    def view(self, distance):
        pixels = array.array('H', [int(distance*1000)]*25)
        if sys.byteorder != 'little': pixels.byteswap()
        return {'label': 'front', 'image_size': [5,5], 'depth_mm': base64.b64encode(zlib.compress(pixels.tobytes())).decode(),
                'intrinsics': [2.,2.,2.,2.], 'position': [0.,0.,.5],
                'rotation': [[1,0,0],[0,1,0],[0,0,1]]}

    def test_far_visual_match_cannot_be_arrival(self):
        check={'anchor_view': 'front', 'anchor_pixel_xy': [2,2]}
        far=ground_visual_anchor(check,[self.view(5.)],[0.,0.,.2])
        self.assertFalse(far['valid'])
        self.assertEqual(far['horizontal_distance_m'],5.)
        near=ground_visual_anchor(check,[self.view(2.)],[0.,0.,.2])
        self.assertTrue(near['valid'])
        self.assertEqual(near['anchor_world_xyz'],[2.,0.,.5])

    def test_missing_out_of_frame_and_empty_depth_fail_closed(self):
        for check,views in [({},[]),({'anchor_view':'front','anchor_pixel_xy':[-1,2]},[self.view(2.)]),
                            ({'anchor_view':'front','anchor_pixel_xy':[2,2]},[self.view(0.)])]:
            self.assertFalse(ground_visual_anchor(check,views,[0.,0.,.2])['valid'])

    def test_world_rotation_and_pixel_ray_are_applied(self):
        view=self.view(2.)
        view['rotation']=[[0,-1,0],[1,0,0],[0,0,1]]
        result=ground_visual_anchor({'anchor_view':'front','anchor_pixel_xy':[3,2]},[view],[0.,0.,.2])
        self.assertEqual(result['anchor_world_xyz'],[1.,2.,.5])

    def test_nearby_floor_is_not_a_table_anchor(self):
        view=self.view(1.)
        view['position'][2]=0.
        result=ground_visual_anchor({'anchor_view':'front','anchor_pixel_xy':[2,2]},[view],[0.,0.,.2])
        self.assertFalse(result['valid'])
        self.assertIn('floor',result['reason'])
