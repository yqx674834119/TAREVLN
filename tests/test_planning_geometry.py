import math
import unittest
import numpy as np
from vln_core.planning_geometry import map_points,bounded_cloud


class PlanningGeometryTest(unittest.TestCase):
    def test_vehicle_path_uses_full_pose(self):
        pose=dict(position=[4.,5.,.3],orientation=[math.sqrt(.5),0.,0.,math.sqrt(.5)])
        np.testing.assert_allclose(map_points([[2.,0.,0.]],'vehicle',pose),[[4.,7.,.3]],atol=1e-7)

    def test_unknown_frame_and_missing_pose_rejected(self):
        for frame in ('vehicle','camera','odom'):
            with self.assertRaises(ValueError):map_points([[0.,0.,0.]],frame,None)

    def test_cloud_drops_nonfinite_and_bounds_work(self):
        result=bounded_cloud([[0,0,0],[.01,0,0],[float('nan'),1,0],[1,1,1]],limit=1)
        self.assertEqual(result,[[0.,0.,0.]])


if __name__=='__main__':unittest.main()
