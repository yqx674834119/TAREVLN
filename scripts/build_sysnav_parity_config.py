"""Expand the pinned official simulation launch into an auditable ROS config.

No hand-copied algorithm defaults: YAML files and XML launch overrides are read
from the fixed upstream Git revision. Embodiment/runtime exceptions are explicit.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import re
import subprocess
import xml.etree.ElementTree as ET

import yaml

ROOT = Path(__file__).resolve().parents[1]
BASE = '478727e5a559be22822d5d093b57431449336287'


def build(source):
    inputs = {}
    def read(path):
        text = subprocess.check_output(['git','-C',str(source),'show',BASE+':'+path],text=True)
        inputs[path] = hashlib.sha256(text.encode()).hexdigest()
        return text
    def merge(dst, data, only=None):
        for node, body in data.items():
            if only is not None and node != only:
                continue
            dst.setdefault(node,dict(ros__parameters={} ))['ros__parameters'].update(body['ros__parameters'])

    # Record the reviewed topology and Python launch sources as well as YAML/XML.
    for path in (
        'src/base_autonomy/vehicle_simulator/launch/system_simulation_with_exploration_planner.launch',
        'src/exploration_planner/tare_planner/launch/explore_world_sim.launch',
        'src/exploration_planner/tare_planner/launch/room_segmentation.launch',
        'src/semantic_mapping/launch/semantic_mapping_sim.launch',
        'src/semantic_mapping/launch/detection_node.launch',
        'src/vlm_node/launch/vlm_node.launch'):
        read(path)

    config = yaml.safe_load(read('src/exploration_planner/tare_planner/config/matterport_sim.yaml'))
    for name in ('local_planner','terrain_analysis','terrain_analysis_ext'):
        file='src/base_autonomy/'+name+'/launch/'+name+'.launch'
        tree = ET.fromstring(read(file))
        arguments = {a.attrib['name']:a.attrib.get('default') for a in tree.findall('arg')}
        # Official top-level system_simulation_with_exploration_planner launch.
        arguments.update(checkTerrainConn='true',cameraOffsetZ='0.1',goalX='0.0',goalY='0.0')
        def substitute(s):
            return re.sub(r'\$\(var ([^)]+)\)', lambda m:arguments[m[1]], s)
        for node in tree.findall('node'):
            if node.attrib['pkg'] in ('tf2_ros',):
                continue
            node_name = node.attrib['name']
            for parameter in node.findall('param'):
                if 'from' in parameter.attrib:
                    path=substitute(parameter.attrib['from']).replace('$(find-pkg-share local_planner)','src/base_autonomy/local_planner')
                    merge(config,yaml.safe_load(read(path)),only=node_name)
                elif parameter.attrib['name']=='pathFolder':
                    # Expanded at runtime to the SAME pinned native path bank.
                    continue
                else:
                    config.setdefault(node_name,dict(ros__parameters={}))['ros__parameters'][parameter.attrib['name']] = yaml.safe_load(substitute(parameter.attrib['value']))
    merge(config,yaml.safe_load(read('src/semantic_mapping/config/mapping_mecanum_sim.yaml')))
    merge(config,yaml.safe_load(read('src/vlm_node/config/vlm_config.yaml')))
    for node in ('detection_node','semantic_mapping_node'):
        config[node]['ros__parameters'].update(device='auto',captioner_batch_size=16)
    # Included official non-policy nodes; boundary file is a public operating envelope.
    config['navigationBoundary'] = dict(ros__parameters=dict(sendBoundary=True,sendBoundaryInterval=2))
    config['sensor_scan'] = dict(ros__parameters={})
    return config, inputs


def generate(source, overrides):
    reference, inputs = build(source)
    result=copy.deepcopy(reference)
    differences=[]
    for node, parameters in overrides.items():
        for key, entry in parameters.items():
            if not entry.get('reason'):
                raise ValueError('Every override requires a reason')
            previous=reference.get(node,{}).get('ros__parameters',{}).get(key)
            result.setdefault(node,dict(ros__parameters={}))['ros__parameters'][key]=entry['value']
            differences.append(dict(node=node,key=key,upstream=previous,value=entry['value'],reason=entry['reason']))
    for body in result.values():
        body['ros__parameters']['use_sim_time']=True
    return result, dict(upstream_revision=BASE,profile='official-simulation-launch + explicit Isaac Go2 embodiment',
                        sources_sha256=inputs,overrides=differences,clock='all nodes use the same Isaac clock',
                        unresolved_parameter_overrides=[])


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,default=ROOT.parent/'deps/sysnav-full')
    p.add_argument('--check',action='store_true')
    a=p.parse_args()
    overrides=json.loads((ROOT/'config/sysnav-isaac-overrides.json').read_text())
    config,manifest=generate(a.source,overrides)
    path=ROOT/'config/sysnav-full-ros.yaml'
    text='# Generated by scripts/build_sysnav_parity_config.py; edit the documented overrides.\n'+yaml.safe_dump(config,sort_keys=False)
    if a.check:
        if yaml.safe_load(path.read_text())!=config:
            raise SystemExit('Config drift: regenerate the official SysNav profile')
    else:
        path.write_text(text)
        (ROOT/'config/sysnav-parity-manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps(dict(profile=manifest['profile'],nodes=len(config),explicit_overrides=len(manifest['overrides']),checked=a.check)))


if __name__=='__main__':main()
