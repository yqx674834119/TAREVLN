"""Hardware-independent gate for a future manufacturer high-level velocity API.

No transport, robot connection, mode switch, or implicit arming lives here.
Every timestamp is the receiver's monotonic clock; ROS/simulation stamps cannot
extend a command's validity. A transport must call tick even without new input.
"""
from dataclasses import dataclass, field
import math


def _finite(value):
    return type(value) in (int,float) and math.isfinite(value)


@dataclass(frozen=True)
class GuardConfig:
    timeout_s: float = .25
    max_speed: tuple = (.3, .2, .5)
    acceleration: tuple = (.4, .4, .8)
    required_health: tuple = ("pose", "scan", "robot_link", "operator_deadman")


@dataclass
class MotionGuard:
    config: GuardConfig = field(default_factory=GuardConfig)
    armed: bool = False
    estop: bool = False
    reason: str = "disabled_by_default"
    health: dict = field(default_factory=dict)
    command: tuple = (0., 0., 0.)
    output: tuple = (0., 0., 0.)
    command_time: float | None = None
    sequence: int = -1
    last_tick: float | None = None

    def stop(self, reason):
        self.armed = False
        self.output = (0., 0., 0.)
        self.reason = reason

    def set_estop(self, pressed):
        if pressed:
            self.estop = True
            self.stop("estop")
        # Releasing a button does not clear the latch or rearm the robot.

    def clear_estop(self, *, operator_acknowledged=False):
        if not operator_acknowledged:
            return False
        self.estop = False
        self.stop("estop_reset_requires_new_arm")
        return True

    def update_health(self, source, received_at, valid):
        if source not in self.config.required_health:
            raise ValueError("Unknown health source")
        previous = self.health.get(source)
        if type(valid) is not bool or not _finite(received_at) or (previous and received_at <= previous[0]):
            self.health.pop(source, None)
            self.stop("invalid_health_update")
            return
        self.health[source] = (received_at, valid)
        if not valid:
            self.stop("invalid_"+source)

    def update_command(self, values, received_at, sequence):
        valid = (isinstance(values,(list,tuple)) and len(values) == 3 and all(_finite(v) for v in values)
                 and _finite(received_at) and type(sequence) is int and sequence > self.sequence
                 and (self.command_time is None or received_at > self.command_time))
        if not valid:
            self.command_time = None
            self.stop("invalid_or_replayed_command")
            return
        self.command = tuple(max(-limit,min(limit,v)) for v,limit in zip(values,self.config.max_speed))
        self.command_time, self.sequence = received_at, sequence

    def _fault(self, now):
        if not _finite(now) or (self.last_tick is not None and (not _finite(self.last_tick) or now < self.last_tick)):
            return "invalid_monotonic_clock"
        if self.armed and self.last_tick is not None and now-self.last_tick > self.config.timeout_s:
            return "watchdog_tick_overrun"
        if self.estop:
            return "estop"
        for name in self.config.required_health:
            timestamp, valid = self.health.get(name, (float('-inf'),False))
            if not valid or not 0 <= now-timestamp <= self.config.timeout_s:
                return "unhealthy_or_stale_"+name
        if self.command_time is None or not 0 <= now-self.command_time <= self.config.timeout_s:
            return "stale_command"
        return None

    def arm(self, now, *, operator_requested=False):
        fault = self._fault(now)
        if not operator_requested or fault:
            self.stop(fault or "explicit_arm_required")
            return False
        self.armed, self.last_tick, self.reason = True, now, "armed"
        self.output = (0.,0.,0.)
        return True

    def tick(self, now):
        fault = self._fault(now)
        if fault:
            self.stop(fault)
        if not self.armed:
            self.last_tick = now
            return self.output
        dt = min(now-self.last_tick, self.config.timeout_s)
        self.last_tick = now
        if all(v == 0 for v in self.command):
            self.output = (0.,0.,0.)  # Stop bypasses acceleration ramp.
        else:
            self.output = tuple(old+max(-a*dt,min(a*dt,new-old))
                                for old,new,a in zip(self.output,self.command,self.config.acceleration))
        self.reason = "command_accepted"
        return self.output
