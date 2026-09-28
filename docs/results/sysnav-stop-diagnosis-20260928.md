# SysNav object-stop diagnosis (2026-09-28)

## Observed failure

Run `ui-20260928-184606-7f344` stopped at simulation time 28.5 s because the native
planner published `found=true, stop=true`. The bridge correctly held `cmd_vel` at
zero; this was a semantic terminal stop, not a ROS, Isaac, VLM, or CUDA crash.
The independent evaluator rejected the stop because the Go2 base was 1.0114 m
from the accepted chair footprint (the limit is 1.0 m). The reported object
position was already grounded (`grounding_error_m=0`), so the error was the
terminal approach condition.

The old TARE condition used a keypose-graph distance below 1 m. That graph can
under-estimate or remain stale relative to the observed object footprint. The
planner therefore sent the robot's current pose as the terminal waypoint while
the base was still just outside the acceptance region.

## Repair

The TARE object-stop gate now computes the planar distance from the robot base to
the observed object's 3-D bounding-box footprint. It requires that distance to
be at most 0.9 m, leaving 0.1 m for localization and controller error. The graph
distance remains diagnostic and is no longer a necessary condition when a valid
3-D box is available. Messages without a valid box use the legacy gate.

The upstream implementation is committed on `codex/isaac-system-parity` as
`ed3f2fff` (with the preceding parameterized box-gate commit `9a9fd234`), and
the project carries the reproducible patch at
`patches/sysnav-object-stop-box.patch`.

## Verification

- Native SysNav packages rebuilt successfully with `scripts/build_sysnav_full.sh`.
- Retry `ui-20260928-192302-27708` failed at Isaac camera initialization before
  navigation (`Camera produced no valid frame after 40 renders`); this is a
  startup/rendering failure unrelated to the planner patch.
- Retry `ui-20260928-192401-b7ac0` reached simulation 59.9 s without a stop;
  logs show the object footprint gate was approaching at 0.42--0.90 m but the
  graph-distance prerequisite still prevented stopping. This exposed and led to
  the second change.
- Final run `ui-20260928-193126-e2664` completed at simulation 35.5 s. The
  evaluator passed the chair target: footprint distance 0.6846 m, grounding
  error 0 m, visible target true, contact audit passed. Native bridge emitted
  `native_sysnav_target_reached` only after the footprint gate entered the 0.9 m
  band. It produced 168 synchronized scans, 166 paths, and 6.98 m of motion.

The older `ui-20260928-172056-ae2d5` run is a separate local-planner no-free-path
case and is not the cause of the latest lounge run.
