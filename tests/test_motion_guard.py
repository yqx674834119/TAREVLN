import math
import unittest
from vln_core.motion_guard import MotionGuard


class MotionGuardTests(unittest.TestCase):
    def ready(self):
        g=MotionGuard()
        for name in g.config.required_health:
            g.update_health(name,10.,True)
        g.update_command([.2,.1,.3],10.,0)
        return g

    def test_default_and_implicit_arm_never_move(self):
        g=self.ready()
        self.assertEqual(g.tick(10.),(0.,0.,0.))
        self.assertFalse(g.arm(10.))

    def test_missing_any_health_source_prevents_arming(self):
        for name in MotionGuard().config.required_health:
            g=self.ready();del g.health[name]
            self.assertFalse(g.arm(10.,operator_requested=True))

    def test_acceleration_and_absolute_bounds(self):
        g=self.ready();g.update_command([100,-100,100],10.01,1)
        self.assertTrue(g.arm(10.01,operator_requested=True))
        value=g.tick(10.11)
        for actual,expected in zip(value,[.04,-.04,.08]):self.assertAlmostEqual(actual,expected)

    def test_stale_input_stops_without_ramp_and_cannot_auto_resume(self):
        g=self.ready();g.arm(10.,operator_requested=True);g.tick(10.1)
        self.assertEqual(g.tick(10.3),(0.,0.,0.))
        for name in g.config.required_health:g.update_health(name,10.31,True)
        g.update_command([.2,0,0],10.31,1)
        self.assertEqual(g.tick(10.32),(0.,0.,0.))

    def test_estop_release_does_not_clear_latch(self):
        g=self.ready();g.arm(10.,operator_requested=True);g.tick(10.1)
        g.set_estop(True);g.set_estop(False)
        self.assertEqual(g.tick(10.15),(0.,0.,0.))
        self.assertFalse(g.arm(10.15,operator_requested=True))
        self.assertTrue(g.clear_estop(operator_acknowledged=True))
        self.assertFalse(g.armed)

    def test_nonfinite_replayed_or_backward_command_revokes_arm(self):
        for values,received,sequence in [([math.nan,0,0],10.1,1),([.2,0,0],10.1,0),([.2,0,0],9.9,1)]:
            g=self.ready();g.arm(10.,operator_requested=True)
            g.update_command(values,received,sequence)
            self.assertEqual(g.tick(10.1),(0.,0.,0.))
            self.assertFalse(g.armed)

    def test_invalid_pose_or_disconnected_transport_stops(self):
        for source in ['pose','robot_link']:
            g=self.ready();g.arm(10.,operator_requested=True);g.tick(10.1)
            g.update_health(source,10.11,False)
            self.assertEqual(g.tick(10.12),(0.,0.,0.))

    def test_stop_command_bypasses_ramp(self):
        g=self.ready();g.arm(10.,operator_requested=True);g.tick(10.1)
        g.update_command([0,0,0],10.11,1)
        self.assertEqual(g.tick(10.12),(0.,0.,0.))

    def test_clock_rewind_or_future_input_fails_closed(self):
        g=self.ready();g.arm(10.,operator_requested=True);g.tick(10.1)
        self.assertEqual(g.tick(9.),(0.,0.,0.))
        g=self.ready();self.assertFalse(g.arm(9.,operator_requested=True))

    def test_malformed_input_does_not_escape_the_stop_gate(self):
        for values in [None,{},'forward',[True,0,0]]:
            g=self.ready();g.arm(10.,operator_requested=True)
            g.update_command(values,10.1,1)
            self.assertEqual(g.tick(10.1),(0.,0.,0.))

    def test_watchdog_overrun_revokes_arm_even_with_fresh_inputs(self):
        g=self.ready();g.arm(10.,operator_requested=True)
        for name in g.config.required_health:g.update_health(name,11.,True)
        g.update_command([.2,0,0],11.,1)
        self.assertEqual(g.tick(11.),(0.,0.,0.))
        self.assertEqual(g.reason,'watchdog_tick_overrun')
