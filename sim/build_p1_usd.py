"""Author P1 collision geometry, sliding doors, moving cabins and visible QR/text signs."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from pxr import Gf, Sdf, Usd, UsdGeom, UsdLux, UsdPhysics, UsdShade
from PIL import Image, ImageDraw, ImageFont
import qrcode

from p1_spec import write_spec


def add_box(stage, path, position, size, color, kinematic=False):
    prim = UsdGeom.Cube.Define(stage, path)
    prim.CreateSizeAttr(1.0)
    xf = UsdGeom.Xformable(prim)
    translate = xf.AddTranslateOp()
    translate.Set(Gf.Vec3d(*position))
    xf.AddScaleOp().Set(Gf.Vec3f(*size))
    prim.CreateDisplayColorAttr([Gf.Vec3f(*color)])
    UsdPhysics.CollisionAPI.Apply(prim.GetPrim())
    if kinematic:
        body = UsdPhysics.RigidBodyAPI.Apply(prim.GetPrim())
        body.CreateKinematicEnabledAttr(True)
    return prim, translate


def sign_texture(directory, name, payload, label):
    directory.mkdir(parents=True, exist_ok=True)
    code = qrcode.QRCode(error_correction=qrcode.constants.ERROR_CORRECT_L, box_size=10, border=4)
    code.add_data(json.dumps(payload, separators=(",", ":")))
    code.make(fit=True)
    image = code.make_image(fill_color="black", back_color="white").convert("RGB").resize((768, 768), Image.Resampling.NEAREST)
    canvas = Image.new("RGB", (768, 896), "white")
    canvas.paste(image, (0, 0))
    fontfile = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf")
    font = ImageFont.truetype(str(fontfile), 30) if fontfile.is_file() else ImageFont.load_default()
    draw = ImageDraw.Draw(canvas)
    draw.text((384, 810), label, fill="black", font=font, anchor="mm")
    path = directory / (name + ".png")
    canvas.save(path)
    return path


def add_sign(stage, root, sign):
    path = "/World/Signs/" + sign["name"]
    mesh = UsdGeom.Mesh.Define(stage, path)
    w, h = sign["size"], sign["size"] * 896 / 768
    # Plane normal is local +X, local +Y is image-right when viewed from +X.
    mesh.CreatePointsAttr([(0, -w / 2, -h / 2), (0, w / 2, -h / 2),
                          (0, w / 2, h / 2), (0, -w / 2, h / 2)])
    mesh.CreateFaceVertexCountsAttr([4])
    mesh.CreateFaceVertexIndicesAttr([0, 1, 2, 3])
    mesh.CreateDoubleSidedAttr(False)
    mesh.CreateSubdivisionSchemeAttr("none")
    uv = UsdGeom.PrimvarsAPI(mesh).CreatePrimvar("st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.faceVarying)
    uv.Set([(0, 0), (1, 0), (1, 1), (0, 1)])
    xf = UsdGeom.Xformable(mesh)
    xf.AddTranslateOp().Set(Gf.Vec3d(*sign["pos"]))
    xf.AddRotateZOp().Set(float(sign["yaw"]))
    texture = sign_texture(root / "textures", sign["name"], sign["payload"], sign["label"])
    material = UsdShade.Material.Define(stage, path + "/Material")
    surface = UsdShade.Shader.Define(stage, path + "/Material/Surface")
    surface.CreateIdAttr("UsdPreviewSurface")
    surface.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(1.0)
    tex = UsdShade.Shader.Define(stage, path + "/Material/Texture")
    tex.CreateIdAttr("UsdUVTexture")
    tex.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(str(texture))
    tex.CreateInput("sourceColorSpace", Sdf.ValueTypeNames.Token).Set("sRGB")
    tex.CreateOutput("rgb", Sdf.ValueTypeNames.Float3)
    reader = UsdShade.Shader.Define(stage, path + "/Material/UV")
    reader.CreateIdAttr("UsdPrimvarReader_float2")
    reader.CreateInput("varname", Sdf.ValueTypeNames.Token).Set("st")
    reader.CreateOutput("result", Sdf.ValueTypeNames.Float2)
    tex.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(reader.ConnectableAPI(), "result")
    surface.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).ConnectToSource(tex.ConnectableAPI(), "rgb")
    surface.CreateInput("emissiveColor", Sdf.ValueTypeNames.Color3f).ConnectToSource(tex.ConnectableAPI(), "rgb")
    surface.CreateOutput("surface", Sdf.ValueTypeNames.Token)
    material.CreateSurfaceOutput().ConnectToSource(surface.ConnectableAPI(), "surface")
    UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(material)
    # RTX can render back faces despite doubleSided=False. An opaque backing
    # prevents mirrored rear-side codes from becoming visual observations.
    back = UsdGeom.Cube.Define(stage, path + "/Backing")
    back.CreateSizeAttr(1.0)
    bx = UsdGeom.Xformable(back)
    bx.AddTranslateOp().Set(Gf.Vec3d(-.015, 0, 0))
    bx.AddScaleOp().Set(Gf.Vec3f(.02, w, h))
    backing = UsdShade.Material.Define(stage, path + "/BackMaterial")
    bs = UsdShade.Shader.Define(stage, path + "/BackMaterial/Surface")
    bs.CreateIdAttr("UsdPreviewSurface")
    bs.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(.3, .3, .3))
    bs.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(1.)
    bs.CreateOutput("surface", Sdf.ValueTypeNames.Token)
    backing.CreateSurfaceOutput().ConnectToSource(bs.ConnectableAPI(), "surface")
    UsdShade.MaterialBindingAPI.Apply(back.GetPrim()).Bind(backing)


def build(root, bridge=False):
    root = root.resolve()
    spec = write_spec(root, bridge)
    stage = Usd.Stage.CreateNew(str(root / "private" / "scene.usda"))
    stage.SetDefaultPrim(UsdGeom.Xform.Define(stage, "/World").GetPrim())
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    physics = UsdPhysics.Scene.Define(stage, "/World/PhysicsScene")
    physics.CreateGravityDirectionAttr(Gf.Vec3f(0, 0, -1))
    physics.CreateGravityMagnitudeAttr(9.81)
    dome = UsdLux.DomeLight.Define(stage, "/World/DomeLight")
    dome.CreateIntensityAttr(650)
    sun = UsdLux.DistantLight.Define(stage, "/World/Sun")
    sun.CreateIntensityAttr(1800)
    UsdGeom.Xformable(sun).AddRotateXYZOp().Set(Gf.Vec3f(-30, 20, 20))
    for box in spec["boxes"]:
        add_box(stage, "/World/Static/" + box["name"], box["pos"], box["size"], box["color"])
    for door in spec["doors"]:
        pos = [x + (o if door["initial_open"] else 0) for x, o in zip(door["pos"], door["open_offset"])]
        add_box(stage, "/World/Doors/" + door["id"], pos, door["size"], (0.38, 0.45, 0.50), True)
    for lift in spec["lifts"]:
        z = (lift["initial_floor"] - 1) * 3.0
        # One kinematic rigid body with compound collision shapes carries the passenger.
        cabin = UsdGeom.Xform.Define(stage, "/World/Cabins/" + lift["id"])
        UsdGeom.Xformable(cabin).AddTranslateOp().Set(Gf.Vec3d(lift["x"], lift["y"], z))
        UsdPhysics.RigidBodyAPI.Apply(cabin.GetPrim()).CreateKinematicEnabledAttr(True)
        for name, pos, size in (("Floor", (0, 0, -.1), (3.0, 3.2, .2)),
                                ("Back", (0, -1.5, 1.2), (3, .1, 2.4)),
                                ("Left", (-1.5, 0, 1.2), (.1, 3.2, 2.4)),
                                ("Right", (1.5, 0, 1.2), (.1, 3.2, 2.4))):
            add_box(stage, str(cabin.GetPath()) + "/" + name, pos, size, (.42, .48, .52))
        spec["signs"].append(dict(name=lift["id"] + "_CabinSign", pos=[lift["x"], -11.03, z + 1.0],
                                  yaw=90, payload={"kind": "cabin", "id": lift["id"]},
                                  label="CABIN " + lift["id"], size=1.4, attach_lift=lift["id"]))
    for sign in spec["signs"]:
        add_sign(stage, root / "private", sign)
    (root / "private" / "oracle.json").write_text(json.dumps(spec, indent=2))
    stage.GetRootLayer().Save()
    return spec


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bridge", action="store_true")
    args = parser.parse_args()
    spec = build(args.output, args.bridge)
    print(json.dumps({"boxes": len(spec["boxes"]), "signs": len(spec["signs"]), "doors": len(spec["doors"])}))
