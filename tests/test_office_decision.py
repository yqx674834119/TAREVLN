import unittest
from unittest.mock import Mock
from vln_policy.office_decision import visual_decision


class VisualDecisionTests(unittest.TestCase):
    def setUp(self):
        self.context = {"candidates": [{"id": 3, "xyz": [1.,2.,.3]}],
                        "candidate_view_ids": {"front": [3], "left": []}}
        self.valid = {"action": "inspect", "candidate_id": 3, "selected_view": "front",
                      "target_visible": False, "inside_target_area": False, "confidence": .5}

    def test_repeated_invalid_model_id_does_not_crash_or_command_motion(self):
        api = Mock()
        api.ask.return_value = dict(self.valid, candidate_id=99)
        result = visual_decision(api, 'task', [], self.context)
        self.assertEqual(api.ask.call_count, 2)
        self.assertEqual(result['action'], 'explore')
        self.assertIsNone(result['candidate_id'])
        self.assertFalse(result['inside_target_area'])

    def test_repaired_selection_is_validated(self):
        api = Mock()
        api.ask.side_effect = [dict(self.valid, selected_view='left'), self.valid]
        self.assertEqual(visual_decision(api, 'task', [], self.context), self.valid)

    def test_independent_visual_rejection_prevents_positive_arrival(self):
        api = Mock()
        claimed = dict(self.valid, action='confirm', target_visible=True, inside_target_area=True, confidence=.95)
        verifier = {'shared_meeting_table': True, 'multiple_chairs': True, 'wall_display': False,
                    'inside_area': True, 'confidence': .95, 'evidence': 'Only ceiling lights are visible'}
        api.ask.side_effect = [claimed, verifier]
        result = visual_decision(api, 'task with earlier claims', [], self.context)
        self.assertFalse(result['inside_target_area'])
        self.assertEqual(result['action'], 'explore')
        self.assertNotIn('earlier claims', api.ask.call_args.args[0])

    def test_arrival_is_checked_even_when_navigation_keeps_requesting_motion(self):
        api = Mock()
        navigation = dict(self.valid, target_visible=True, inside_target_area=False, confidence=.7)
        verifier = {'armchair_group': True, 'small_table': True, 'wall_display': True,
                    'inside_area': True, 'confidence': .9, 'evidence': 'Camera beside seating group'}
        api.ask.side_effect = [navigation, verifier]
        result = visual_decision(api, 'task', [], dict(self.context, task='lounge'))
        self.assertTrue(result['inside_target_area'])
        self.assertEqual(result['confidence'], .9)
        self.assertEqual(api.ask.call_args.args[2]['purpose'], 'arrival_verification')
