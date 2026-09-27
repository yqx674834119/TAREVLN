"""Decode MP3D's GOOGLE_texture_basis payloads for the USD converter.

Geometry buffers are copied byte-for-byte. Only texture representation changes.
The source GLB remains untouched. Basis Universal v2_0_2 is Apache-2.0.
"""
import hashlib
import json
from pathlib import Path
import struct
import subprocess
import re
from PIL import Image

BASIS_REVISION='433888a9736a8802fe9d17ae7f489b4cf7a45556'


def prepare_textured_gltf(source,out,basisu):
    with source.open('rb') as f:
        magic,version,total=struct.unpack('<III',f.read(12))
        if magic!=0x46546c67 or version!=2:raise ValueError('Expected GLB v2')
        size,kind=struct.unpack('<II',f.read(8));gltf=json.loads(f.read(size))
        size,kind=struct.unpack('<II',f.read(8))
        if kind!=0x004e4942:raise ValueError('Expected binary geometry chunk')
        binary=f.read(size)
    if not any(i.get('mimeType')=='image/x-basis' for i in gltf.get('images',[])):
        return source,{}
    if not basisu.is_file():raise RuntimeError('Basis Universal decoder required for MP3D embedded textures')
    prepared=out/'decoded';prepared.mkdir(exist_ok=True)
    records=[]
    for i,item in enumerate(gltf['images']):
        if item.get('mimeType')!='image/x-basis':raise ValueError('Mixed texture formats require explicit handling')
        view=gltf['bufferViews'][item['bufferView']]
        if view.get('buffer',0)!=0:raise ValueError('Unexpected external buffer')
        blob=binary[view.get('byteOffset',0):view.get('byteOffset',0)+view['byteLength']]
        basis=prepared/f'image-{i:03d}.basis';basis.write_bytes(blob)
        png=prepared/f'image-{i:03d}_unpacked_rgb_ETC1_RGB_0_0000.png'
        with (prepared/f'image-{i:03d}.log').open('w') as log:
            subprocess.run([str(basisu),'-unpack','-no_ktx','-format_only','0',str(basis)],
                           cwd=prepared,stdout=log,stderr=subprocess.STDOUT,check=True)
        if not png.exists():raise RuntimeError('Decoder did not produce base-level PNG: '+str(png))
        flag=re.search(r'Y Flipped: ([01])', (prepared/f'image-{i:03d}.log').read_text())
        if flag is None:raise RuntimeError('Decoder did not report texture orientation')
        flipped=flag[1]=='1'
        canonical=prepared/f'image-{i:03d}-gltf.png'
        with Image.open(png) as image:
            (image.transpose(Image.Transpose.FLIP_TOP_BOTTOM) if flipped else image).save(canonical)
        gltf['images'][i]={'name':item.get('name',f'image{i}'),'uri':canonical.name,'mimeType':'image/png'}
        records.append(dict(index=i,basis_sha256=hashlib.sha256(blob).hexdigest(),y_flip_restored=flipped,
                            png_sha256=hashlib.sha256(canonical.read_bytes()).hexdigest()))
    for texture in gltf['textures']:
        extension=texture.pop('extensions')
        if set(extension)!={'GOOGLE_texture_basis'}:raise ValueError('Unexpected texture extension')
        texture['source']=extension['GOOGLE_texture_basis']['source']
    for key in ('extensionsUsed','extensionsRequired'):
        if key in gltf:gltf[key]=[x for x in gltf[key] if x!='GOOGLE_texture_basis']
    (prepared/'source.bin').write_bytes(binary)
    gltf['buffers'][0]['uri']='source.bin'
    path=prepared/'scene.gltf';path.write_text(json.dumps(gltf))
    info=dict(basis_revision=BASIS_REVISION,decode_format='ETC1_RGB base mip to PNG',basis_binary_sha256=hashlib.sha256(basisu.read_bytes()).hexdigest(),
              textures=records,geometry_binary_sha256=hashlib.sha256(binary).hexdigest())
    (prepared/'provenance.json').write_text(json.dumps(info,indent=2))
    return path,info
