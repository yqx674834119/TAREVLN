import unittest
import json
from importlib.util import find_spec

from sim.p1_spec import make_spec, public_prior
from sim.p1_interactions import Elevator, evaluate_stop


class P1ContractTests(unittest.TestCase):
    def test_elevator_travel_volume_is_not_cut_by_slabs(self):
        spec = make_spec()
        floors = [box for box in spec["boxes"] if box["name"].endswith("_Floor")]
        self.assertEqual(len(floors), 8)
        for lift in spec["lifts"]:
            for floor in floors:
                inside_xy = all(abs(lift[key] - floor["pos"][axis]) < floor["size"][axis] / 2
                                for axis, key in enumerate(("x", "y")))
                self.assertFalse(inside_xy)
        self.assertNotIn("lifts", public_prior())
        self.assertNotIn("signs", public_prior())
        self.assertNotIn("connections", public_prior())

    def test_optional_bridge_has_explicit_directed_capability_edges(self):
        plain, linked = make_spec(), make_spec(bridge=True)
        self.assertFalse(any(c["kind"] == "skybridge" for c in plain["connections"]))
        edges = [c for c in linked["connections"] if c["kind"] == "skybridge"]
        self.assertEqual({(e["from"], e["to"]) for e in edges}, {("A/F2", "B/F2"), ("B/F2", "A/F2")})
        self.assertTrue(all(e["directed"] and "walk" in e["capabilities"] for e in edges))

    def test_elevator_needs_approach_boarding_and_served_floor(self):
        e = Elevator(make_spec()["lifts"][0])
        self.assertFalse(e.call(5, [-22, 0, 12.4])["ok"])
        self.assertTrue(e.call(5, [-27, -6.5, 12.4])["ok"])
        for _ in range(201):
            e.tick(.005)
        self.assertEqual(e.state, "open")
        self.assertFalse(e.select(1, [-27, -8.1, 12.4])["ok"])
        self.assertFalse(e.select(9, [-27, -9.6, 12.4])["ok"])
        self.assertTrue(e.select(1, [-27, -9.6, 12.4])["ok"])
        history = []
        for _ in range(5000):
            e.tick(.005)
            if e.state == "moving":
                self.assertEqual(e.door_fraction, 0)
                history.append(e.z)
        self.assertEqual(e.floor, 1)
        self.assertEqual(e.state, "open")
        self.assertTrue(all(a >= b for a, b in zip(history, history[1:])))
        self.assertNotIn("floor", e.feedback())

    def test_call_from_another_floor_keeps_departure_door_closed(self):
        e = Elevator(make_spec()["lifts"][1])
        self.assertTrue(e.call(2, [27, -6.5, 3.4])["ok"])
        e.tick(.1)
        self.assertEqual(e.door_fraction, 0)
        for _ in range(1500):
            e.tick(.005)
        self.assertEqual((e.floor, e.state), (2, "open"))
        self.assertTrue(e.call(2, [27, -6.5, 3.4])["ok"])
        self.assertEqual(e.state, "open")

    def test_stop_requires_goal_floor_identity_outside_cabin_and_dwell(self):
        spec = make_spec()
        identity = {"building": "B", "floor": 3}
        self.assertTrue(evaluate_stop(spec, [22, 0, 6.4], 0, identity, 1.2)["success"])
        self.assertFalse(evaluate_stop(spec, [22, 0, 3.4], 0, identity, 1.2)["success"])
        self.assertFalse(evaluate_stop(spec, [22, 0, 6.4], 0, identity, .2)["success"])
        self.assertFalse(evaluate_stop(spec, [22, 0, 6.4], 0, {"building": "B", "floor": 2}, 1.2)["success"])
        result = evaluate_stop(spec, [27, -9.6, 6.4], 0, identity, 1.2)
        self.assertFalse(result["conditions"]["outside_elevator"])

    @unittest.skipUnless(find_spec("numpy"), "Isaac runtime regression: NumPy is optional for core tests")
    def test_simulator_scalar_results_can_be_written_as_json(self):
        import numpy as np
        result = evaluate_stop(make_spec(), np.array([22., 0., 6.2]), np.float32(.01),
                               {"building": "B", "floor": 3}, 1.1)
        self.assertTrue(json.loads(json.dumps(result))["success"])


if __name__ == "__main__":
    unittest.main()
