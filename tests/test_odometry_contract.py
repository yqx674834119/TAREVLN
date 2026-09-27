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
