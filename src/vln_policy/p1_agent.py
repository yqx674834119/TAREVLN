"""Deterministic P1 integration baseline driven by visible signs, lidar and odometry.

This module is not TARE or a VLM. It never imports simulator modules. It receives
only public priors and the sensor/skill-feedback whitelist over stdin/stdout.
"""
from __future__ import annotations

import heapq
import json
import math
import sys

from vln_core.connections import ConnectionMemory


def wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


class LocalGrid:
    resolution = .25

    def __init__(self):
        self.occupied = set()
        self.free = set()

    def cell(self, point):
        return tuple(round(float(v) / self.resolution) for v in point[:2])

    def update(self, xy, scan):
        observed_free, hits = set(), set()
        for angle, distance in scan:
            for step in range(int(max(0, distance - .15) / self.resolution) + 1):
                r = step * self.resolution
                observed_free.add(self.cell((xy[0] + r * math.cos(angle), xy[1] + r * math.sin(angle))))
            if distance < 11.95:
                hits.add(self.cell((xy[0] + distance * math.cos(angle), xy[1] + distance * math.sin(angle))))
        self.occupied.difference_update(observed_free)
        self.occupied.update(hits)
        self.free.update(observed_free - hits)

    def waypoint(self, xy, target):
        start, goal = self.cell(xy), self.cell(target)
        blocked = {(x + dx, y + dy) for x, y in self.occupied
                   for dx in range(-2, 3) for dy in range(-2, 3) if dx * dx + dy * dy <= 4}
        blocked.discard(start)
        if goal in blocked:
            candidates = [(goal[0] + dx, goal[1] + dy) for dx in range(-4, 5) for dy in range(-4, 5)]
            allowed = [p for p in candidates if p not in blocked]
            if not allowed:
                return None
            goal = min(allowed, key=lambda p: math.dist(p, goal) + .02 * math.dist(p, start))
        bounds = (min(start[0], goal[0]) - 32, max(start[0], goal[0]) + 32,
                  min(start[1], goal[1]) - 32, max(start[1], goal[1]) + 32)
        queue = [(0, 0, start)]
        best, parent = {start: 0}, {}
        count = 0
        while queue and count < 16000:
            _, cost, node = heapq.heappop(queue)
            if cost > best[node]:
                continue
            if node == goal:
                path = [node]
                while path[-1] in parent:
                    path.append(parent[path[-1]])
                path.reverse()
                point = path[min(3, len(path) - 1)]
                return [point[0] * self.resolution, point[1] * self.resolution]
            count += 1
            for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1)):
                new = (node[0] + dx, node[1] + dy)
                if new in blocked or not (bounds[0] <= new[0] <= bounds[1] and bounds[2] <= new[1] <= bounds[3]):
                    continue
                if dx and dy and ((node[0] + dx, node[1]) in blocked or (node[0], node[1] + dy) in blocked):
                    continue
                updated = cost + math.hypot(dx, dy) * (1 if new in self.free else 2.0)
                if updated < best.get(new, float("inf")):
                    best[new] = updated
                    parent[new] = node
                    heapq.heappush(queue, (updated + math.dist(new, goal), updated, new))
        return None


class Agent:
    def __init__(self, prior):
        self.prior = prior
        self.identity = dict(prior["known_start"])
        self.state = "seek_lift"
        self.memory = ConnectionMemory()
        self.grid = LocalGrid()
        self.target = None
        self.lift = None
        self.desired_floor = None
        self.gaze = None
        self.edge_id = None
        self.wait_ticks = 0
        self.trip_count = 0
        self.last_frame = None
        self.distance = 0.0
        self.last_xy = None
        self.search_ticks = 0
        self.search_state = None
        self.did_recenter = False

    @staticmethod
    def approach(sign, xy, stand_off=1.4):
        sx, sy = sign["point"][:2]
        d = max(.01, math.hypot(xy[0] - sx, xy[1] - sy))
        return [sx + stand_off * (xy[0] - sx) / d, sy + stand_off * (xy[1] - sy) / d]

    def navigation(self, pose, target):
        if math.dist(pose[:2], target) < .35:
            return [0, 0, 0], True
        point = self.grid.waypoint(pose[:2], target)
        if point is None:
            return [0, 0, .35], False
        angle = math.atan2(point[1] - pose[1], point[0] - pose[0])
        error = wrap(angle - pose[3])
        speed = min(.55, math.dist(pose[:2], target)) if abs(error) < .35 else 0
        return [speed, 0, max(-.7, min(.7, 1.4 * error))], False

    def search(self, pose):
        # A second public-map-derived observation position for large empty P1
        # rooms. Real cluttered buildings require frontier/TARE exploration.
        self.search_ticks += 1
        if self.search_ticks > 180 and not self.did_recenter:
            building = next(b for b in self.prior["buildings"] if b["id"] == self.identity["building"])
            command, reached = self.navigation(pose, building["center_xy"])
            self.did_recenter = reached
            return command
        return [0, 0, .5]

    def tick(self, observation):
        pose, signs = observation["odom"], observation["signs"]
        if self.search_state != self.state:
            self.search_ticks, self.did_recenter = 0, False
            self.search_state = self.state
        xy = pose[:2]
        if self.last_xy is not None:
            self.distance += math.dist(xy, self.last_xy)
        self.last_xy = list(xy)
        if self.state != "ride":
            self.grid.update(xy, observation["scan"])
        command, request, stop = [0, 0, 0], None, False
        feedback = observation.get("skill_feedback", {})
        lift_signs = [s for s in signs if s["text"]["kind"] == "lift"]
        if self.state == "seek_lift":
            if lift_signs:
                self.lift = min(lift_signs, key=lambda s: s["range_m"])
                text = self.lift["text"]
                self.desired_floor = self.prior["goal"]["floor"] if self.identity["building"] == self.prior["goal"]["building"] else text.get("link_B", 1)
                if self.desired_floor not in text["floors"]:
                    self.state = "unsupported_connection"
                else:
                    self.target = self.approach(self.lift, xy)
                    self.state = "approach_lift"
                    self.trip_count += 1
                    self.edge_id = f"trip_{self.trip_count}"
                    source = f'{self.identity["building"]}/F{self.identity["floor"]}'
                    self.memory.sighting(self.edge_id, source, self.lift["evidence"])
                    self.memory.indication(self.edge_id, f'{self.identity["building"]}/F{self.desired_floor}',
                                           self.lift["evidence"] + ":panel", "elevator_api_assisted")
            else:
                command = self.search(pose)
        elif self.state == "approach_lift":
            command, reached = self.navigation(pose, self.target)
            if reached:
                request = {"op": "call", "lift": self.lift["text"]["id"], "floor": self.identity["floor"]}
                self.state = "wait_open"
                self.wait_ticks = 0
        elif self.state == "wait_open":
            self.wait_ticks += 1
            sx, sy = self.lift["point"][:2]
            angle = math.atan2(sy - pose[1], sx - pose[0])
            command = [0, 0, max(-.5, min(.5, wrap(angle - pose[3])))]
            if feedback.get("door_open"):
                self.state = "find_cabin"
            elif self.wait_ticks > 500:
                self.state = "skill_timeout"
        elif self.state == "find_cabin":
            candidates = [s for s in signs if s["text"]["kind"] == "cabin" and s["text"].get("id") == self.lift["text"]["id"]]
            if candidates:
                self.target = self.approach(candidates[0], xy, 1.2)
                self.gaze = math.atan2(candidates[0]["point"][1] - pose[1], candidates[0]["point"][0] - pose[0])
                self.state = "board"
            else:
                sx, sy = self.lift["point"][:2]
                command = [0, 0, max(-.45, min(.45, wrap(math.atan2(sy - pose[1], sx - pose[0]) - pose[3])))]
        elif self.state == "board":
            command, reached = self.navigation(pose, self.target)
            if reached:
                request = {"op": "select", "lift": self.lift["text"]["id"], "floor": self.desired_floor}
                self.state = "ride"
                self.wait_ticks = 0
        elif self.state == "ride":
            self.wait_ticks += 1
            if feedback.get("last_result", {}).get("ok") is False:
                self.state = "skill_rejected"
            elif self.wait_ticks > 15 and feedback.get("door_open"):
                self.state = "verify_floor"
                self.grid = LocalGrid()
        elif self.state == "verify_floor":
            command = [0, 0, max(-.5, min(.5, wrap(self.gaze + math.pi - pose[3])))]
            candidates = [s for s in signs if s["text"]["kind"] == "floor"]
            if candidates:
                sign = min(candidates, key=lambda s: s["range_m"])
                identity = {key: sign["text"][key] for key in ("building", "floor")}
                if identity != {"building": self.identity["building"], "floor": self.desired_floor}:
                    self.state = "wrong_floor"
                else:
                    dest = f'{identity["building"]}/F{identity["floor"]}'
                    self.memory.traversed(self.edge_id, dest, f'transition:{observation["time"]}')
                    self.memory.verified(self.edge_id, dest, sign["evidence"])
                    self.identity = identity
                    self.target = self.approach(sign, xy, 1.0)
                    self.state = "exit_cabin"
        elif self.state == "exit_cabin":
            command, reached = self.navigation(pose, self.target)
            if reached:
                self.state = "go_goal" if self.identity["building"] == self.prior["goal"]["building"] else ("seek_bridge" if self.identity["floor"] == 2 else "exit_building")
        elif self.state == "seek_bridge":
            candidates = [s for s in signs if s["text"]["kind"] == "bridge" and s["text"].get("to") == "B"]
            if candidates:
                # The visible board indicates the crossing; OSM gives the destination building.
                self.target = [self.prior["buildings"][1]["center_xy"][0], candidates[0]["point"][1]]
                self.state = "cross_bridge"
            else:
                command = self.search(pose)
        elif self.state in {"exit_building", "cross_bridge"}:
            target = self.prior["outdoor_footways"][0]["from_xy"] if self.state == "exit_building" else self.target
            command, reached = self.navigation(pose, target)
            if reached:
                if self.state == "exit_building":
                    self.state = "outdoor"
                else:
                    self.state = "verify_entry"
                    self.desired_floor = 2
                    self.grid = LocalGrid()
        elif self.state == "outdoor":
            destination = next(b for b in self.prior["buildings"] if b["id"] == self.prior["goal"]["building"])
            command, reached = self.navigation(pose, destination["center_xy"])
            # Require a visible building-name board on approach, saved as evidence.
            for s in signs:
                if s["text"].get("kind") == "building" and s["text"].get("building") == destination["id"]:
                    self.last_frame = s["evidence"]
            if reached:
                if self.last_frame:
                    self.state = "verify_entry"
                    self.desired_floor = 1
                    self.grid = LocalGrid()
                else:
                    self.state = "building_identity_missing"
        elif self.state == "verify_entry":
            command = [0, 0, .5]
            candidates = [s for s in signs if s["text"].get("kind") == "floor"
                          and s["text"].get("building") == self.prior["goal"]["building"]]
            if candidates:
                sign = candidates[0]
                self.identity = {k: sign["text"][k] for k in ("building", "floor")}
                self.state = "seek_lift"
        elif self.state == "go_goal":
            destination = next(b for b in self.prior["buildings"] if b["id"] == self.prior["goal"]["building"])
            command, reached = self.navigation(pose, destination["center_xy"])
            if reached:
                self.state = "stop"
        elif self.state == "stop":
            stop = True
        else:
            stop = True
        # The released Go2 flat policy stalls on zero-translation yaw commands.
        # Use a small, lidar-checked arc to sustain stepping; no base pose edits.
        if abs(command[2]) > .12 and abs(command[0]) < .05 and not stop:
            front = min((d for a, d in observation["scan"] if abs(wrap(a - pose[3])) < .4), default=0)
            rear = min((d for a, d in observation["scan"] if abs(wrap(a - pose[3] - math.pi)) < .4), default=0)
            command[0] = .3 if front > 1.0 else (-.3 if rear > 1.0 else 0)
            command[2] = math.copysign(.7, command[2])
        return {"cmd_vel": command, "request": request, "stop": stop, "state": self.state,
                "identity": self.identity, "verified_edges": sum(e.state.value == "verified" for e in self.memory.edges.values()),
                "distance_m": self.distance}


def main():
    prior = json.loads(open("/input/prior.json").read())
    agent = Agent(prior)
    for line in sys.stdin:
        message = json.loads(line)
        if message.get("op") == "audit":
            probes = {}
            for path in message["paths"]:
                try:
                    open(path, "rb").read(1)
                    probes[path] = "READABLE"
                except OSError:
                    probes[path] = "DENIED"
            print(json.dumps({"audit": probes}), flush=True)
        else:
            print(json.dumps(agent.tick(message)), flush=True)


if __name__ == "__main__":
    main()
