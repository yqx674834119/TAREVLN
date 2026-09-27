"""Prepare NVIDIA's Office asset for static collision and private evaluation."""
import argparse
import json
from pathlib import Path
from pxr import Gf, Sdf, Usd, UsdGeom, UsdLux, UsdPhysics

ap = argparse.ArgumentParser()
ap.add_argument("--allow-incomplete-preview", action="store_true")
args = ap.parse_args()
root = Path(__file__).resolve().parents[1]
cache = root.parent / "assets/usd-cache"
manifest = json.loads((cache / "office-download.json").read_text())
if not manifest["complete"] and not args.allow_incomplete_preview:
    raise SystemExit("Office dependencies are not complete; use preview only for authoring")
out = root / "data/sim/office/private"
out.mkdir(parents=True, exist_ok=True)
stage = Usd.Stage.CreateNew(str(out / "scene.usda"))
world = UsdGeom.Xform.Define(stage, "/World")
stage.SetDefaultPrim(world.GetPrim())
UsdGeom.SetStageMetersPerUnit(stage, 1)
UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
office = UsdGeom.Xform.Define(stage, "/World/Office")
office.GetPrim().GetReferences().AddReference(str(cache / "Isaac/Environments/Office/office.usd"))
# Make collision overrides explicit on the composed static meshes. Doors remain
# physical geometry. The office's pre-authored open/closed placement is retained.
while True:
    instances = [p for p in stage.Traverse() if p.IsInstance()]
    if not instances:
        break
    for prim in instances:
        prim.SetInstanceable(False)
count = 0
for prim in stage.Traverse():
    if prim.IsA(UsdGeom.Mesh):
        UsdPhysics.CollisionAPI.Apply(prim).CreateCollisionEnabledAttr(True)
        UsdPhysics.MeshCollisionAPI.Apply(prim).CreateApproximationAttr("none")
        count += 1
    if prim.IsA(UsdLux.DomeLight):
        # Constant ambient sky removes a rendering dependency on a remote HDR.
        UsdLux.DomeLight(prim).GetTextureFileAttr().Set(Sdf.AssetPath())
        UsdLux.DomeLight(prim).GetIntensityAttr().Set(400)
light = UsdLux.DomeLight.Define(stage, "/World/OfficeAmbient")
light.CreateIntensityAttr(350)
# The original office's enclosed ceiling blocks a dome. Model ceiling panels
# explicitly so vision does not depend on emissive-material render settings.
for x in [-21., -15., -8., 1.]:
    for y in [-8., 0., 8., 16., 24., 32., 40., 48., 56.]:
        panel = UsdLux.RectLight.Define(stage, f"/World/Lights/P{int(x+25)}_{int(y+10)}")
        panel.CreateWidthAttr(2.)
        panel.CreateHeightAttr(2.)
        panel.CreateIntensityAttr(12000.)
        UsdGeom.Xformable(panel).AddTranslateOp().Set(Gf.Vec3d(x,y,2.6))
scene = UsdPhysics.Scene.Define(stage, "/World/physicsScene")
scene.CreateGravityDirectionAttr(Gf.Vec3f(0, 0, -1))
scene.CreateGravityMagnitudeAttr(9.81)
stage.GetRootLayer().Save()
bb = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ["default", "render"], useExtentsHint=False)
inventory = []
for prim in office.GetPrim().GetChildren():
    bound = bb.ComputeWorldBound(prim).ComputeAlignedRange()
    if not bound.IsEmpty():
        inventory.append({"name": prim.GetName(), "min": list(bound.GetMin()), "max": list(bound.GetMax())})
(out / "asset-inventory.json").write_text(json.dumps(inventory, indent=2))
(out / "build.json").write_text(json.dumps({"asset": manifest["scene"], "dependencies_complete": manifest["complete"], "collision_meshes": count, "static_mesh_collisions": True}, indent=2))
evaluation = json.loads((root / "sim/fixtures/office-evaluation.json").read_text())
for task, oracle in evaluation["tasks"].items():
    name = "oracle.json" if task == "meeting" else f"oracle_{task}.json"
    (out / name).write_text(json.dumps(oracle, indent=2))
print(json.dumps({"collision_meshes": count, "inventory_items": len(inventory), "complete": manifest["complete"]}))
