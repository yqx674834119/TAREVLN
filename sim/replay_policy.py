"""Check a policy change against recorded sensor inputs, without a simulator."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from vln_policy.p1_agent import Agent

parser = argparse.ArgumentParser()
parser.add_argument("run", type=Path)
args = parser.parse_args()
prior = json.loads((args.run.parents[1] / "public/prior.json").read_text())
agent = Agent(prior)
observations = [json.loads(line) for line in (args.run / "observations.jsonl").read_text().splitlines()]
rows = [json.loads(line) for line in (args.run / "trajectory.jsonl").read_text().splitlines()]
mismatches, checked = [], 0
for observation, row in zip(observations, rows):
    expected = row["action"]
    if expected["state"] == "warmup":
        continue
    actual = agent.tick(observation)
    checked += 1
    if any(actual[k] != expected[k] for k in ("state", "request", "cmd_vel", "identity", "verified_edges")):
        mismatches.append({"time": row["time"], "expected": expected, "actual": actual})
result = {"frames_checked": checked, "mismatch_count": len(mismatches), "first_mismatches": mismatches[:3],
          "passed": checked > 0 and not mismatches, "evidence_type": "offline policy replay, not physical execution"}
(args.run / "policy_replay.json").write_text(json.dumps(result, indent=2))
print(json.dumps(result, indent=2))
