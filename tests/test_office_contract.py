import math
import unittest
from vln_policy.office_contract import validate_decision, confirmed_from_two_positions, unvisited_candidates


class OfficeContractTest(unittest.TestCase):
    def test_search_does_not_keep_selecting_visited_floor(self):
        candidates = [{"id": 0, "xyz": [1.,0.,.3]}, {"id": 1, "xyz": [4.,0.,.3]}]
        self.assertEqual(unvisited_candidates(candidates, [[0.,0.,.3],[1.,0.,.3]]), [candidates[1]])
        self.assertEqual(unvisited_candidates(candidates, []), candidates)
        self.assertEqual(unvisited_candidates(candidates, [[0.,0.,.3]]), candidates)

    def setUp(self):
        self.candidates = [{"id": 3, "xyz": [1., 2., .3]}]
        self.decision = {"action": "inspect", "candidate_id": 3, "target_visible": True,
                         "inside_target_area": False, "confidence": .9}

    def test_only_an_existing_candidate_can_be_selected(self):
        self.assertEqual(validate_decision(self.decision, self.candidates), self.candidates[0])
        for invalid in (4, True, "3", None):
            with self.assertRaises(ValueError):
                validate_decision(dict(self.decision, candidate_id=invalid), self.candidates)

    def test_truthy_strings_and_nonfinite_values_are_rejected(self):
        for changes in ({"inside_target_area": "false"}, {"confidence": math.nan}, {"confidence": 1.1}, {"action": "cmd_vel"}):
            with self.assertRaises(ValueError):
                validate_decision(dict(self.decision, **changes), self.candidates)

    def test_candidate_must_belong_to_the_selected_camera_view(self):
        views = {"front": [3], "left": [8]}
        self.assertEqual(validate_decision(dict(self.decision, selected_view="front"), self.candidates, views), self.candidates[0])
        with self.assertRaises(ValueError):
            validate_decision(dict(self.decision, selected_view="left"), self.candidates, views)

    def test_duplicate_stale_and_distant_views_cannot_complete(self):
        a = {"time": 5., "position": [0.,0.,.3]}
        for b in ({"time": 8., "position": [0.,0.,.3]}, {"time": 75., "position": [1.,0.,.3]}, {"time": 8., "position": [8.,0.,.3]}):
            self.assertFalse(confirmed_from_two_positions(a,b))
        self.assertTrue(confirmed_from_two_positions(a, {"time": 8., "position": [1.,0.,.3]}))
