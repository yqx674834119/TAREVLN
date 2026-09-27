import unittest
try:
    import numpy as np
    from vln_localization.kiss_odometry import LidarOdometry
except ImportError:
    np=None


@unittest.skipIf(np is None,'Separate odometry runtime is required')
class OdometryContractTest(unittest.TestCase):
    def setUp(self):
        self.frame={'time':0.,'points_base_xyz':np.random.default_rng(9).uniform(-2,2,(300,3)).tolist()}

    def test_truth_fields_are_rejected_and_failure_is_latched(self):
        odom=LidarOdometry()
        self.assertFalse(odom.update(dict(self.frame,truth_pose=np.eye(4).tolist()))['valid'])
        self.assertFalse(odom.update(self.frame)['valid'])

    def test_numerical_registration_does_not_claim_control_readiness(self):
        result=LidarOdometry().update(self.frame)
        self.assertTrue(result['valid'])
        self.assertFalse(result['control_ready'])

    def test_duplicate_time_and_large_gaps_are_rejected(self):
        for stamp in (0.,-.1,.6):
            odom=LidarOdometry();self.assertTrue(odom.update(self.frame)['valid'])
            self.assertFalse(odom.update(dict(self.frame,time=stamp))['valid'])

    def test_nonfinite_and_degenerate_clouds_are_rejected(self):
        for cloud in ([[float('nan'),0,0]]*100,[[i/100.,0,0] for i in range(100)]):
            self.assertFalse(LidarOdometry().update(dict(self.frame,points_base_xyz=cloud))['valid'])

    def test_tighter_gicp_matching_tracks_a_known_small_translation(self):
        odom=LidarOdometry(method='gicp',gicp_correspondence=.15)
        self.assertTrue(odom.update(self.frame)['valid'])
        points=np.asarray(self.frame['points_base_xyz'])-np.array([.02,.01,0])
        result=odom.update({'time':.2,'points_base_xyz':points.tolist()})
        self.assertTrue(result['valid'])
        np.testing.assert_allclose(np.asarray(result['odom_from_base'])[:3,3],[.02,.01,0],atol=.003)
        self.assertFalse(result['control_ready'])
        self.assertTrue(np.isfinite(result['registration']['gicp_cost_per_inlier']))

    def test_invalid_gicp_match_distances_are_rejected(self):
        for distance in (0.,-.1,float('nan'),float('inf')):
            with self.assertRaises(ValueError):LidarOdometry(method='gicp',gicp_correspondence=distance)
