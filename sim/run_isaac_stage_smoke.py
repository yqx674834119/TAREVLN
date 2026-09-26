"""First Isaac Sim check for the generated USD stage, after EULA acceptance.

This does not load Go2 or execute a navigation task. The script is staged for
the high-fidelity installation and is not part of the MuJoCo acceptance gate.
"""

from __future__ import annotations

import json
import os
from pathlib import Path


if os.environ.get("OMNI_KIT_ACCEPT_EULA", "").upper() not in {"YES", "Y", "1"}:
    raise SystemExit("Accept the NVIDIA Omniverse EULA yourself before running Isaac Sim")

from isaacsim import SimulationApp  # noqa: E402 - Kit must start before USD imports


OUTPUT = Path(__file__).resolve().parents[1] / "data" / "sim" / "demo"
STAGE = OUTPUT / "demo_campus.usda"
if not STAGE.is_file():
    raise SystemExit(f"generate the scene first: {STAGE}")

app = SimulationApp({"headless": True, "open_usd": str(STAGE), "width": 640, "height": 480})
try:
    import omni.usd
    from isaacsim.core.utils.stage import is_stage_loading
    from pxr import UsdGeom

    for _ in range(120):
        app.update()
        if not is_stage_loading():
            break
    stage = omni.usd.get_context().get_stage()
    if stage is None:
        raise RuntimeError("Isaac Sim did not open the generated USD stage")
    cube_count = sum(prim.IsA(UsdGeom.Cube) for prim in stage.Traverse())
    if cube_count < 50:
        raise RuntimeError(f"Isaac Sim loaded only {cube_count} campus cubes")
    report = {"simulator": "Isaac Sim 5.1", "stage": str(STAGE), "cubes": cube_count,
              "scope": "USD stage load only; no Go2, elevator, or navigation acceptance"}
    (OUTPUT / "isaac_stage_smoke.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
finally:
    app.close()
