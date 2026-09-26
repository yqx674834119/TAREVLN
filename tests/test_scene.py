import json
import tempfile
import unittest
from pathlib import Path

from sim.generate_demo_scene import generate


class DemoSceneTests(unittest.TestCase):
    def test_a5f_b3f_and_hidden_connectors_are_separate(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            summary = generate(output)
            stage = (output / "demo_campus.usda").read_text()
            prior = json.loads((output / "prior.json").read_text())
            oracle = json.loads((output / "oracle.json").read_text())
            self.assertEqual(summary["slabs"], 8)
            self.assertIn('def Cube "A_F5_Slab"', stage)
            self.assertIn('def Cube "B_F3_Slab"', stage)
            self.assertEqual(prior["goal"]["floor"], 3)
            self.assertNotIn("connectors", prior)
            self.assertEqual(len(oracle["connectors"]), 5)


if __name__ == "__main__":
    unittest.main()
