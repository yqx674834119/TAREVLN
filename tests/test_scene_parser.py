import tempfile
import unittest
from pathlib import Path

from sim.generate_demo_scene import generate


class SceneParserTests(unittest.TestCase):
    def test_usd_scene_contains_physics_boxes(self):
        # The actual USD/Go2 load is a separate remote runtime acceptance.
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            generate(output)
            stage = (output / "demo_campus.usda").read_text()
            self.assertGreaterEqual(stage.count('def Cube "'), 50)
            self.assertIn('PhysicsCollisionAPI', stage)


if __name__ == "__main__":
    unittest.main()
