"""Integration-contract checks against the pinned upstream, not copied defaults."""
import importlib.util
import json
from pathlib import Path
import sys
import unittest
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
sys.path.insert(0,str(ROOT/'sim'))
from build_sysnav_parity_config import generate, build
from audit_sysnav_parity import check_runtime
from vln_core.sysnav_sensor_contract import SensorObservationAdapter, LIDAR_IN_BASE
from vln_core.planning_geometry import rotation_wxyz, map_points
SOURCE=ROOT.parent/'deps/sysnav-full'


@unittest.skipUnless((SOURCE/'.git').exists(),'Pinned SysNav source required')
class SysNavParityTest(unittest.TestCase):
    def test_all_algorithm_parameters_inherit_official_launch(self):
        config,manifest=generate(SOURCE,json.loads((ROOT/'config/sysnav-isaac-overrides.json').read_text()))
        upstream,_=build(SOURCE)
        exceptions={(x['node'],x['key']) for x in manifest['overrides']}
        for node,body in upstream.items():
            for key,value in body['ros__parameters'].items():
                if (node,key) not in exceptions:self.assertEqual(config[node]['ros__parameters'][key],value)
        self.assertEqual(config['tare_planner_node']['ros__parameters']['kSensorRange'],3.)
        self.assertEqual(config['tare_planner_node']['ros__parameters']['kLookAheadDistance'],4.)
        self.assertEqual(config['terrainAnalysisExt']['ros__parameters']['checkTerrainConn'],True)

    def test_runtime_gate_rejects_duplicate_scan_publisher(self):
        runtime=dict(parameters={},topics={},processes=[],parameter_errors=[])
        for topic,node in {'/state_estimation_at_scan':'sensor_scan','/sensor_scan':'sensor_scan',
              '/navigation_boundary':'navigationBoundary','/way_point':'tare_planner_node',
                '/path':'localPlanner','/cmd_vel':'pathFollower','/room_nodes_list':'room_segmentation'}.items():
            runtime['topics'][topic]=dict(publishers=[dict(node=node)])
        runtime['topics']['/navigation_boundary']['publishers'].append(dict(node='room_segmentation'))
        self.assertEqual(check_runtime(runtime,{}),[])
        runtime['topics']['/state_estimation_at_scan']['publishers'].append(dict(node='isaac_sysnav_bridge'))
        self.assertTrue(check_runtime(runtime,{}))

    def test_runtime_gate_rejects_ignored_new_algorithm_parameter(self):
        runtime=dict(parameters={'tare_planner_node':{}},topics={},processes=[],parameter_errors=[])
        errors=check_runtime(runtime,{'tare_planner_node':{'ros__parameters':{'kSensorRange':3.}}})
        self.assertIn('Missing effective parameter: tare_planner_node.kSensorRange',errors)

    def test_actual_mapper_projection_matches_isaac_panorama(self):
        spec=importlib.util.spec_from_file_location('sysnav_projection',SOURCE/'src/semantic_mapping/semantic_mapping/cloud_image_fusion.py')
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        for azimuth,elevation in [(0.,0.),(.8,.3),(-1.4,-.4),(2.2,.55)]:
            ray=np.array([np.cos(elevation)*np.cos(azimuth),np.cos(elevation)*np.sin(azimuth),np.sin(elevation)])
            points=np.array([3*ray])
            pixel=module.scan2pixels_mecanum_sim(points)[0,:2]
            expected=[961-azimuth*1920/(2*np.pi),321-elevation*1920/(2*np.pi)]
            np.testing.assert_allclose(pixel,expected,atol=1.1)

    def test_ray_origin_under_roll_pitch_and_scan_endpoints_unchanged(self):
        q=np.array([.9,.2,-.3,.1]);q/=np.linalg.norm(q)
        rotation=rotation_wxyz(q)
        obs=dict(time=1.,position=[4.,-3.,.33],orientation=q.tolist(),velocity=[0.,0.,0.],
                 yaw=float(np.arctan2(rotation[1,0],rotation[0,0])),points=[[6.,2.,1.,0.]])
        converted=SensorObservationAdapter().convert(obs)
        np.testing.assert_allclose(converted['position'],obs['position']+rotation@LIDAR_IN_BASE)
        self.assertIs(converted['points'],obs['points'])
        self.assertEqual(obs['position'],[4.,-3.,.33])
        with self.assertRaises(ValueError):SensorObservationAdapter().convert(converted)
        point=map_points([[2.,0.,0.]],'vehicle',converted)[0]
        self.assertAlmostEqual(point[2],converted['position'][2])
        self.assertAlmostEqual(point[0],converted['position'][0]+2*np.cos(obs['yaw']))

    def test_sensor_velocity_uses_sensor_origin_motion(self):
        adapter=SensorObservationAdapter()
        obs=dict(time=1.,position=[0.,0.,.33],orientation=[1.,0.,0.,0.],velocity=[0.,0.,0.])
        a=adapter.convert(obs)
        obs.update(time=1.2,orientation=[np.cos(.1),0.,np.sin(.1),0.])
        b=adapter.convert(obs)
        expected=rotation_wxyz(obs['orientation']).T@((np.array(b['position'])-a['position'])/.2)
        np.testing.assert_allclose(b['velocity'],expected)


if __name__=='__main__':unittest.main()
