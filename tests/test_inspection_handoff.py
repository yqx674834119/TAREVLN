import unittest
from vln_policy.inspection_handoff import InspectionHandoff


class InspectionHandoffTests(unittest.TestCase):
    def test_reached_position_is_held_until_next_sensor_frame(self):
        handoff=InspectionHandoff()
        self.assertFalse(handoff.waiting(9.))
        handoff.reached(9.2)
        self.assertTrue(handoff.waiting(9.3))
        # The recorded regression reached at 9.2 s, but queried again only at
        # 16 s after driving away. A 10 s frame must be eligible immediately.
        self.assertTrue(handoff.waiting(10.))
        handoff.captured()
        self.assertFalse(handoff.waiting(10.1))

    def test_missing_camera_cannot_pause_navigation_forever(self):
        handoff=InspectionHandoff();handoff.reached(9.2)
        self.assertFalse(handoff.waiting(13.3))
        self.assertIsNone(handoff.reached_at)

    def test_clock_reset_discards_old_inspection(self):
        handoff=InspectionHandoff();handoff.reached(9.2)
        self.assertFalse(handoff.waiting(0.))
