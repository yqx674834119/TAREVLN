"""Read-only provenance of live nodes, resolved libraries, topics and parameters."""
import hashlib
import json
from pathlib import Path
import time


def capture_runtime(bridge, output, source, specs, children):
    from rcl_interfaces.srv import ListParameters, GetParameters
    from rclpy.parameter import parameter_value_to_python
    names=bridge.get_node_names()
    clients={name:(bridge.create_client(ListParameters,'/'+name+'/list_parameters'),
                   bridge.create_client(GetParameters,'/'+name+'/get_parameters'))
             for name in names if name!='isaac_sysnav_bridge'}
    lists={name:client[0].call_async(ListParameters.Request(prefixes=[],depth=10)) for name,client in clients.items()}
    deadline=time.monotonic()+8
    while not all(f.done() for f in lists.values()) and time.monotonic()<deadline:
        time.sleep(.02)
    requests={}
    result={'nodes':sorted(names),'parameters':{},'parameter_errors':[],'topics':{},'processes':[]}
    result['sensor_contract']={'input_pose':'GICP base_link','native_pose':'physical lidar origin',
        'lidar_in_base_m':[0.,0.,.45],'camera_in_sensor_m':[0.,0.,0.],
        'vehicle_frame':'yaw-only at sensor origin, matching native localPlanner',
        'base_link_in_sensor_m':[0.,0.,-.45],'registered_points':'map; unchanged by pose rebasing',
        'counts':dict(bridge.counts)}
    for name,future in lists.items():
        if future.done() and future.exception() is None:
            parameter_names=future.result().result.names
            requests[name]=(parameter_names,clients[name][1].call_async(GetParameters.Request(names=parameter_names)))
        else:
            result['parameter_errors'].append(name+': list timeout/error')
    deadline=time.monotonic()+8
    while not all(f.done() for _,f in requests.values()) and time.monotonic()<deadline:
        time.sleep(.02)
    for name,(parameter_names,future) in requests.items():
        if future.done() and future.exception() is None:
            result['parameters'][name]=dict(zip(parameter_names,map(parameter_value_to_python,future.result().values)))
        else:
            result['parameter_errors'].append(name+': get timeout/error')
    for topic in ['/clock','/state_estimation','/state_estimation_at_scan','/registered_scan','/sensor_scan',
                  '/camera/image','/tf','/tf_static','/navigation_boundary','/room_nodes_list','/room_mask',
                  '/door_cloud','/object_nodes_list','/way_point','/path','/cmd_vel']:
        result['topics'][topic]={key:[dict(node=x.node_name,type=x.topic_type) for x in call(topic)]
            for key,call in [('publishers',bridge.get_publishers_info_by_topic),('subscribers',bridge.get_subscriptions_info_by_topic)]}
    hashes={}
    for (package,executable),child in zip(specs,children):
        mapped=[]
        file=Path('/proc')/str(child.pid)/'maps'
        for line in file.read_text().splitlines() if file.exists() else []:
            path=line.split()[-1]
            if str(source) in path and path.startswith('/') and path not in mapped:
                mapped.append(path)
                p=Path(path)
                if p.is_file():hashes[path]=hashlib.sha256(p.read_bytes()).hexdigest()
        result['processes'].append(dict(package=package,executable=executable,pid=child.pid,
                                       running=child.poll() is None,mapped_project_files=mapped))
    result['loaded_file_sha256']=hashes
    (output/'runtime-contract.json').write_text(json.dumps(result,indent=2))
