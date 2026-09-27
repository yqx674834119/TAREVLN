import importlib.util
from pathlib import Path
import unittest
from unittest.mock import Mock
import numpy as np

spec=importlib.util.spec_from_file_location('camera_readiness',Path(__file__).resolve().parents[1]/'sim/camera_readiness.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)


class CameraReadinessTest(unittest.TestCase):
    def test_delayed_render_returns_real_frame_without_physics_step(self):
        world=Mock();camera=Mock();frame=np.ones((2,2,4),dtype=np.uint8)
        camera.get_rgba.side_effect=[None,np.array([]),frame]
        actual,count=module.render_rgba(world,camera)
        self.assertIs(actual,frame)
        self.assertEqual(count,6)
        world.step.assert_not_called()

    def test_missing_sensor_fails_with_finite_budget(self):
        world=Mock();camera=Mock();camera.get_rgba.return_value=np.array([])
        with self.assertRaisesRegex(RuntimeError,'40 renders'):
            module.render_rgba(world,camera)
        self.assertEqual(world.render.call_count,40)

    def test_pose_change_gets_minimum_render_updates(self):
        world=Mock();camera=Mock();camera.get_rgba.return_value=np.ones((2,2,4))
        module.render_rgba(world,camera)
        self.assertEqual(world.render.call_count,4)
