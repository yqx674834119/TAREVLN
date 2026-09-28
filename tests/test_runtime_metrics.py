"""Timing counters use wall time; appended/partial API records are consumed once."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

spec=importlib.util.spec_from_file_location('runtime_metrics',Path(__file__).resolve().parents[1]/'sim/runtime_metrics.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)


class RuntimeMetricTest(unittest.TestCase):
    def test_wall_clock_ratio_and_incremental_partial_api_records(self):
        with tempfile.TemporaryDirectory() as directory:
            output=Path(directory);(output/'agent').mkdir()
            log=output/'agent/api.jsonl'
            log.write_text('{"seconds":0.2,"status":"ok"}\n{"seconds":0.3')
            with patch.object(m.time,'perf_counter',side_effect=[100.,102.,104.]):
                metrics=m.RuntimeMetrics(output,10.,gui=True)
                metrics.renders=8;metrics.viewport_frames=2
                first=metrics.sample(11.,200)
                self.assertEqual(first['realtime_factor'],.5)
                self.assertEqual(first['render_fps'],4.)
                self.assertEqual(first['viewport_fps'],1.)
                self.assertEqual(first['vlm_api_calls'],1)
                with log.open('a') as stream:stream.write(',"status":"error"}\n')
                second=metrics.sample(12.,400)
                self.assertEqual(second['vlm_api_calls'],2)
                self.assertEqual(second['vlm_api_last_status'],'error')
                self.assertEqual(second['vlm_api_avg_ms'],250.)
                metrics.close()

    def test_absent_models_are_not_reported_as_zero_latency(self):
        with tempfile.TemporaryDirectory() as directory:
            metrics=m.RuntimeMetrics(Path(directory),0.)
            row=metrics.sample(0.,0)
            self.assertIsNone(row['viewport_fps'])
            self.assertIsNone(row['vlm_api_avg_ms'])
            self.assertNotIn('yolo_last_ms',row)
            metrics.close()
