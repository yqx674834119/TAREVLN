import sys
from pathlib import Path
import unittest
from unittest.mock import Mock

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'sim'))
from sysnav_evaluation import evaluate_object_stop, visible_targets


class ObjectAcceptanceTest(unittest.TestCase):
    def setUp(self):
        self.definition = dict(base_z_range_m=[.05,.7], maximum_footprint_distance_m=1.,
            maximum_grounding_error_m=.4, requires_visible_target=True,
            targets=[dict(prim='/World/Chair', min=[2.,1.,0.], max=[3.,2.,.8])])

    def check(self, position=(1.5,1.5,.4), predicted=(2.5,1.5,.4), visible=('/World/Chair',)):
        return evaluate_object_stop(position,predicted,visible,self.definition)['passed']

    def test_visible_grounded_near_object(self):
        self.assertTrue(self.check())

    def test_near_wrong_object_is_rejected(self):
        self.assertFalse(self.check(predicted=[4.,1.5,.4]))
        self.assertFalse(self.check(predicted=None))

    def test_other_floor_and_far_stop_rejected(self):
        self.assertFalse(self.check(position=[2.5,1.5,3.4]))
        self.assertFalse(self.check(position=[.5,1.5,.4]))

    def test_hidden_target_rejected(self):
        self.assertFalse(self.check(visible=[]))
        query=Mock()
        query.raycast_closest.return_value={'hit':True,'collision':'/World/Wall/mesh'}
        self.assertEqual(visible_targets(query,[1.5,1.5,.85],self.definition),[])

    def test_first_hit_must_be_target_instance(self):
        query=Mock()
        query.raycast_closest.return_value={'hit':True,'collision':'/World/ChairOther/mesh'}
        self.assertEqual(visible_targets(query,[1.5,1.5,.85],self.definition),[])
        query.raycast_closest.return_value={'hit':True,'collision':'/World/Chair/mesh'}
        self.assertEqual(visible_targets(query,[1.5,1.5,.85],self.definition),['/World/Chair'])


if __name__ == '__main__':
    unittest.main()
