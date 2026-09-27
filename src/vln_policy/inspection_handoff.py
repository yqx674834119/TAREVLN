"""Bounded pause to observe the position the semantic policy actually selected."""


class InspectionHandoff:
    def __init__(self,maximum_wait=4.):
        self.maximum_wait=maximum_wait
        self.reached_at=None

    def reached(self,stamp):
        self.reached_at=stamp

    def waiting(self,stamp):
        if self.reached_at is not None and not 0<=stamp-self.reached_at<=self.maximum_wait:
            self.reached_at=None
        return self.reached_at is not None

    def captured(self):
        self.reached_at=None
