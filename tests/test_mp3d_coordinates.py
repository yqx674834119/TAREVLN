import math
from pathlib import Path
import sys
import unittest
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'sim'))
from mp3d_episode import habitat_to_isaac,isaac_to_habitat,spawn_yaw


class MP3DCoordinatesTest(unittest.TestCase):
    def test_episode_start_and_roundtrip(self):
        point=[-.944907,.046108,5.948920]
        np.testing.assert_allclose(habitat_to_isaac(point),[-.944907,-5.948920,.046108])
        np.testing.assert_allclose(isaac_to_habitat(habitat_to_isaac(point)),point)

    def test_habitat_negative_z_forward_is_isaac_positive_y(self):
        self.assertAlmostEqual(spawn_yaw([0,0,0,1]),math.pi/2)

    def test_quarter_turn(self):
        self.assertAlmostEqual(abs(spawn_yaw([0,math.sqrt(.5),0,math.sqrt(.5)])),math.pi)


if __name__=='__main__':unittest.main()
