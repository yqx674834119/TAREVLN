import unittest

from vln_core.connections import ConnectionMemory, ConnectionState


class ConnectionMemoryTests(unittest.TestCase):
    def test_sighting_and_sign_are_not_a_verified_route(self):
        memory = ConnectionMemory()
        memory.sighting("lift", "A/F5", "rgb:1")
        memory.indication("lift", "A/F1", "panel:1", capability="elevator")
        self.assertIsNone(memory.verified_route("A/F5", "A/F1", {"elevator"}))
        memory.traversed("lift", "A/F1", "transition:1")
        self.assertIsNone(memory.verified_route("A/F5", "A/F1", {"elevator"}))
        with self.assertRaises(ValueError):
            memory.verified("lift", "A/F1", "transition:1")
        memory.verified("lift", "A/F1", "floor-sign:1")
        self.assertEqual(memory.verified_route("A/F5", "A/F1", {"elevator"}), ["lift"])
        self.assertIsNone(memory.verified_route("A/F1", "A/F5", {"elevator"}))
        self.assertIsNone(memory.verified_route("A/F5", "A/F1", {"walk"}))

    def test_wrong_sign_and_blocked_edge_change_route(self):
        memory = ConnectionMemory()
        memory.sighting("e1", "A/F5", "rgb:1")
        memory.indication("e1", "A/F1", "sign:1", capability="elevator")
        memory.traversed("e1", "A/F2", "transition:1")
        memory.verified("e1", "A/F2", "floor-sign:2")
        self.assertEqual(memory.edges["e1"].destination, "A/F2")
        self.assertEqual(memory.edges["e1"].state, ConnectionState.VERIFIED)
        self.assertIsNone(memory.verified_route("A/F5", "A/F1", {"elevator"}))
        memory.blocked("e1", "closure:1")
        self.assertIsNone(memory.verified_route("A/F5", "A/F2", {"elevator"}))


if __name__ == "__main__":
    unittest.main()
