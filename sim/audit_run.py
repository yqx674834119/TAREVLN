"""Independently re-evaluate a recorded P1 trajectory, including incomplete runs.

This never repairs or replaces the original runtime report. A post-hoc task
success cannot turn a crashed runtime into a cleanly completed experiment.
"""
import argparse
import json
from pathlib import Path
import math

from p1_interactions import evaluate_stop

parser = argparse.ArgumentParser()
parser.add_argument("run", type=Path)
parser.add_argument("--oracle", type=Path)
args = parser.parse_args()
oracle = args.oracle or args.run.parents[1] / "private/oracle.json"
spec = json.loads(oracle.read_text())
rows = [json.loads(line) for line in (args.run / "trajectory.jsonl").read_text().splitlines()]
observations = [json.loads(line) for line in (args.run / "observations.jsonl").read_text().splitlines()]
original = json.loads((args.run / "report.json").read_text())
last = rows[-1]
dwell = 0.0
for i in range(len(rows)-1, 0, -1):
    row = rows[i]
    if not row["action"].get("stop") or row["speed"] >= .06:
        break
    dwell += row["time"] - rows[i-1]["time"]
result = evaluate_stop(spec, last["position"], last["speed"], last["action"]["identity"], dwell)
last_selection = max((r["time"] for r in rows if (r["action"].get("request") or {}).get("op") == "select"), default=0)
arrival_frames = [o for o in observations if o["time"] > last_selection and any(
    s["text"].get("kind") == "floor" and s["text"].get("building") == spec["goal"]["building"]
    and s["text"].get("floor") == spec["goal"]["floor"] for s in o["signs"])]
result["conditions"]["arrival_visual_evidence"] = bool(arrival_frames)
result["success"] = all(result["conditions"].values())
clock_offsets = [o["camera_time"] - o["time"] for o in observations if o.get("camera_time") is not None]
active_clock_offsets = [o["camera_time"] - o["time"] for o in observations if o.get("camera_time") is not None and o["time"] >= 2]
signs = {(s["text"].get("building"), s["text"].get("floor")) for o in observations
         for s in o["signs"] if s["text"]["kind"] == "floor"}
transitions = []
for i, row in enumerate(rows):
    if i == 0 or row["action"]["state"] != rows[i-1]["action"]["state"]:
        transitions.append({"time": row["time"], "state": row["action"]["state"], "position": row["position"]})
result = {"task_conditions_passed": result["success"], "conditions": result["conditions"],
          "runtime_status": original["status"], "runtime_report_complete": original["status"] != "starting",
          "recorded_stop_dwell_seconds": dwell, "last_recorded_time": last["time"],
          "goal_identity_frames_after_last_selection": len(arrival_frames),
          "final_position": last["position"], "final_speed_m_s": last["speed"],
          "largest_recorded_position_step_m": max(math.dist(a["position"], b["position"]) for a, b in zip(rows, rows[1:])),
          "floor_identities_observed": sorted(signs), "trajectory_rows": len(rows), "observation_rows": len(observations),
          "camera_time_minus_physics_time_range_s": [min(clock_offsets), max(clock_offsets)] if clock_offsets else None,
          "active_camera_time_offset_range_s": [min(active_clock_offsets), max(active_clock_offsets)] if active_clock_offsets else None,
          "transitions": transitions}
(args.run / "independent_audit.json").write_text(json.dumps(result, indent=2))
print(json.dumps({k: v for k, v in result.items() if k != "transitions"}, indent=2))
