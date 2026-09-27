"""Simulation Go2 velocity adapter with a physical terminal standing mode."""
import numpy as np
from isaacsim.core.utils.types import ArticulationAction
from go2_policy import Go2Policy, DEFAULTS


class SysNavGo2Policy(Go2Policy):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.stop_start = None
        self.stop_steps = 0

    def request_terminal_stand(self):
        if self.stop_start is None:
            self.stop_start = self.robot.get_joint_positions()[self.indices].copy()

    def step(self, command, platform_velocity=0.0):
        if self.stop_start is None:
            return super().step(command, platform_velocity)
        # One-second blend, using torque-limited PD at the existing 200 Hz physics
        # rate. No pose/velocity reset, contact override, or kinematic body lock.
        self.stop_steps += 1
        alpha = min(self.stop_steps / 200., 1.)
        alpha = alpha * alpha * (3. - 2.*alpha)
        target = (1.-alpha)*self.stop_start + alpha*DEFAULTS
        q = self.robot.get_joint_positions()[self.indices]
        qd = self.robot.get_joint_velocities()[self.indices]
        torque = 25.*(target-q) - 1.5*qd
        upper = np.clip(23.5*(1.-qd/30.), 0., 23.5)
        lower = np.clip(23.5*(-1.-qd/30.), -23.5, 0.)
        self.robot.apply_action(ArticulationAction(joint_efforts=np.clip(torque,lower,upper),
                                                   joint_indices=self.indices))
