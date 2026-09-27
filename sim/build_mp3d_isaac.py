"""Convert an authorized MP3D GLB to a static, textured PhysX scene.

MP3D's source GLB has Z-up vertex coordinates despite glTF's usual Y-up.
Habitat loads this dataset with x,y,z -> x,z,-y. Isaac preserves source Z-up.
No navigation mesh, endpoint, or synthetic floor is added to the environment.
"""
import argparse
import asyncio
import hashlib
import json
from pathlib import Path
import struct

ap=argparse.ArgumentParser()
ap.add_argument('--scene',default='x8F5xyUWy9e')
args=ap.parse_args()
root=Path(__file__).resolve().parents[2]
source=root/'assets/benchmarks/scene_datasets/mp3d'/args.scene/(args.scene+'.glb')
out=root/'assets/mp3d-isaac'/args.scene
out.mkdir(parents=True,exist_ok=True)
with source.open('rb') as f:
    f.read(12);length,_=struct.unpack('<II',f.read(8));gltf=json.loads(f.read(length))
if any(any(k in n for k in ('matrix','rotation','scale','translation')) for n in gltf['nodes']):
    raise RuntimeError('Source node transforms require an explicit coordinate audit')
accessors=[gltf['accessors'][p['attributes']['POSITION']] for m in gltf['meshes'] for p in m['primitives']]
expected_min=[min(a['min'][i] for a in accessors) for i in range(3)]
expected_max=[max(a['max'][i] for a in accessors) for i in range(3)]
from mp3d_textures import prepare_textured_gltf
conversion_source,texture_provenance=prepare_textured_gltf(source,out,root/'deps/basis_universal/bin/basisu')

from isaacsim import SimulationApp
app=SimulationApp({'headless':True,'create_new_stage':True,'multi_gpu':False,'active_gpu':0,
                   'physics_gpu':0,'fast_shutdown':True})
try:
    from isaacsim.core.utils.extensions import enable_extension
    enable_extension('omni.kit.asset_converter')
    import omni.kit.asset_converter as converter
    from pxr import Gf,Sdf,Usd,UsdGeom,UsdLux,UsdPhysics,UsdShade
    async def convert():
        context=converter.AssetConverterContext()
        context.export_preview_surface=True
        context.ignore_camera=True;context.ignore_light=True;context.ignore_animations=True
        context.use_meter_as_world_unit=True
        task=converter.get_instance().create_converter_task(str(conversion_source),str(out/'visual.usdc'),None,context)
        if not await task.wait_until_finished():
            raise RuntimeError(task.get_error_message())
    task=asyncio.ensure_future(convert())
    while not task.done():app.update()
    task.result()
    stage=Usd.Stage.CreateNew(str(out/'scene.usda'))
    world=UsdGeom.Xform.Define(stage,'/World');stage.SetDefaultPrim(world.GetPrim())
    UsdGeom.SetStageMetersPerUnit(stage,1.)
    UsdGeom.SetStageUpAxis(stage,UsdGeom.Tokens.z)
    environment=UsdGeom.Xform.Define(stage,'/World/MP3D')
    environment.GetPrim().GetReferences().AddReference('./visual.usdc')
    bb=UsdGeom.BBoxCache(Usd.TimeCode.Default(),['default','render'],useExtentsHint=False)
    bounds=bb.ComputeWorldBound(environment.GetPrim()).ComputeAlignedRange()
    actual_min,actual_max=list(bounds.GetMin()),list(bounds.GetMax())
    delta=max(abs(a-b) for a,b in zip(actual_min+actual_max,expected_min+expected_max))
    if delta>1e-3:raise RuntimeError(f'Converter changed source coordinates: max bound delta {delta}; audit before running')
    meshes,triangles,materials=0,0,0
    for prim in stage.Traverse():
        if prim.IsA(UsdGeom.Mesh):
            mesh=UsdGeom.Mesh(prim)
            mesh.CreateDoubleSidedAttr(True)
            UsdPhysics.CollisionAPI.Apply(prim).CreateCollisionEnabledAttr(True)
            UsdPhysics.MeshCollisionAPI.Apply(prim).CreateApproximationAttr('none')
            meshes+=1;triangles+=sum(n-2 for n in mesh.GetFaceVertexCountsAttr().Get())
        if prim.IsA(UsdShade.Shader):
            shader=UsdShade.Shader(prim)
            if shader.GetIdAttr().Get()=='UsdPreviewSurface':
                diffuse=shader.GetInput('diffuseColor')
                emission=shader.CreateInput('emissiveColor',Sdf.ValueTypeNames.Color3f)
                connected=diffuse.GetConnectedSource()
                if connected:emission.ConnectToSource(*connected)
                else:raise RuntimeError('Textured MP3D material lost its diffuse texture: '+str(prim.GetPath()))
                diffuse.DisconnectSource();diffuse.Set(Gf.Vec3f(0.))
                materials+=1
    # Scan textures already contain illumination. Emission preserves their
    # appearance; dome is only for the articulated robot, not extra floor support.
    UsdLux.DomeLight.Define(stage,'/World/RobotLight').CreateIntensityAttr(500.)
    physics=UsdPhysics.Scene.Define(stage,'/World/physicsScene')
    physics.CreateGravityDirectionAttr(Gf.Vec3f(0,0,-1));physics.CreateGravityMagnitudeAttr(9.81)
    stage.GetRootLayer().Save()
    manifest=dict(scene=args.scene,source=str(source),source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        converter='omni.kit.asset_converter',meters_per_unit=1,up_axis='Z',
        habitat_to_isaac=[[1,0,0],[0,0,-1],[0,1,0]],bounds_min=actual_min,bounds_max=actual_max,
        coordinate_bound_delta_m=delta,meshes=meshes,triangles=triangles,materials=materials,
        texture_provenance=texture_provenance,
        baked_texture_emission=True,static_triangle_collision=True,synthetic_support_floor=False,
        doors_and_furniture='static scan geometry, not articulated interactive objects')
    (out/'build.json').write_text(json.dumps(manifest,indent=2))
    print('MP3D_ISAAC_READY',json.dumps(manifest),flush=True)
finally:app.close()
