"""Boundary and path-fidelity checks for the standalone R2R-CE evaluator."""
import importlib.util
from pathlib import Path
import unittest

spec = importlib.util.spec_from_file_location('benchmark_metrics', Path(__file__).resolve().parents[1]/'scripts/benchmarks/metrics.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


class BenchmarkMetricsTest(unittest.TestCase):
    def test_stop_is_required_and_three_metres_is_excluded(self):
        self.assertEqual(m.evaluate([[0,0,0]], [2.9], False, 4)['sr'], 0)
        self.assertEqual(m.evaluate([[0,0,0]], [3.0], True, 4)['sr'], 0)
        self.assertEqual(m.evaluate([[0,0,0]], [2.9], True, 4)['sr'], 1)

    def test_passing_goal_does_not_make_final_stop_successful(self):
        r = m.evaluate([[0,0,0], [4,0,0], [8,0,0]], [4,0,4], True, 4)
        self.assertEqual((r['sr'], r['osr'], r['spl']), (0, 1, 0))

    def test_detour_penalizes_spl_and_path_fidelity(self):
        gt = [[0,0,0], [1,0,0], [2,0,0]]
        good = m.evaluate(gt, [2,1,0], True, 2, gt)
        detour = m.evaluate([[0,0,0], [0,2,0], [2,2,0], [2,0,0]], [2,3,2,0], True, 2, gt)
        self.assertEqual(good['ndtw_exact'], 1)
        self.assertAlmostEqual(detour['spl'], 1/3)
        self.assertLess(detour['ndtw_exact'], good['ndtw_exact'])

    def test_stationary_rotations_do_not_duplicate_dtw_locations(self):
        self.assertEqual(m.deduplicate([[0,0,0], [0,0,0], [1,0,0]]), [[0,0,0], [1,0,0]])


if __name__ == '__main__':
    unittest.main()
