"""Render plots and a camera replay from recorded simulation evidence only."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

parser = argparse.ArgumentParser()
parser.add_argument("run", type=Path)
args = parser.parse_args()
run = args.run
report = json.loads((run / "report.json").read_text())
spec = json.loads((run.parents[1] / "private/oracle.json").read_text())
rows = [json.loads(line) for line in (run / "trajectory.jsonl").read_text().splitlines()]
t = np.array([r["time"] for r in rows])
xyz = np.array([r["position"] for r in rows])
fig, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
ax = axes[0]
for label, x in (("A", -22), ("B", 22)):
    ax.add_patch(Rectangle((x-8, -8), 16, 16, facecolor="#dce5eb", edgecolor="#778b99", alpha=.6))
    ax.text(x, 7, label, ha="center", weight="bold")
ax.add_patch(Rectangle((-14, -1.5), 28, 3, facecolor="#e1e5dc", zorder=-1))
if spec.get("bridge"):
    ax.add_patch(Rectangle((-14, 4.75), 28, 2.5, facecolor="#e5dced", zorder=-1))
points = ax.scatter(xyz[:, 0], xyz[:, 1], c=xyz[:, 2], cmap="viridis", vmin=0, vmax=12.5, s=3)
ax.plot(xyz[0, 0], xyz[0, 1], "r^", label="start A5F")
ax.plot(xyz[-1, 0], xyz[-1, 1], "rx", label="final")
ax.set(xlabel="x (m)", ylabel="y (m)", title="Recorded Go2 trajectory (color = height)")
ax.set_aspect("equal")
ax.legend()
fig.colorbar(points, ax=ax, label="base z (m)", shrink=.6)
ax = axes[1]
ax.plot(t, xyz[:, 2], color="#006c80", label="Go2 body")
for lift, color in (("EA", "#ad5623"), ("EB", "#6753a0")):
    ax.plot(t, [r["lift_states"][lift]["z"] for r in rows], "--", color=color, label=lift+" cabin")
ax.set(xlabel="simulation time (s)", ylabel="height (m)", title="Continuous physical elevator travel")
ax.grid(alpha=.2)
ax.legend()
fig.suptitle(f"P1 / {run.name} / goal reached={report['success']} / undesired contact pairs={len(report.get('contact_pairs', []))}")
fig.savefig(run / "trajectory.png", dpi=150)
plt.close(fig)

# Select actual saved camera frames at changes of task state.
changes = []
previous = None
for row in rows:
    state = row["action"]["state"]
    if state != previous:
        changes.append((row["time"], state))
    previous = state
frames = sorted(run.glob("rgb_*.png"))
frame_times = [(int(p.stem.split("_")[1]) * .005 + .005, p) for p in frames]
selected = [c for c in changes if c[1] in {"board", "exit_cabin", "outdoor", "verify_entry", "go_goal", "stop"}]
if selected:
    fig, axes = plt.subplots((len(selected)+2)//3, 3, figsize=(15, 3.8*((len(selected)+2)//3)), squeeze=False)
    for ax in axes.flat:
        ax.axis("off")
    for ax, (time, state) in zip(axes.flat, selected):
        _, frame = min(frame_times, key=lambda item: abs(item[0] - time))
        ax.imshow(plt.imread(frame))
        ax.set_title(f"{time:.1f}s / {state}")
    fig.tight_layout()
    fig.savefig(run / "keyframes.jpg", dpi=100)
    plt.close(fig)

try:
    import imageio.v2 as imageio
    import imageio_ffmpeg
    # Uniform one-simulation-second sampling; play at 5x simulated speed.
    with imageio.get_writer(run / "camera_5x.mp4", fps=5, codec="libx264", quality=7) as writer:
        for second in range(int(t[-1])+1):
            _, frame = min(frame_times, key=lambda item: abs(item[0] - second))
            writer.append_data(imageio.imread(frame))
except ImportError:
    print("Video skipped: install imageio and imageio-ffmpeg. Plots are complete.")
print(json.dumps({"run": str(run), "status": report["status"], "success": report["success"]}))
