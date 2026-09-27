import unittest
from sim.contact_audit import accumulate_contact, contact_quality


class ContactAuditTests(unittest.TestCase):
    def test_proximity_with_no_force_is_not_physical_contact(self):
        pairs={}
        accumulate_contact(pairs,'/World/Go2/FL_foot/collisions/mesh','/ground',1.,[{'impulse':[0,0,.2],'separation':0.}])
        accumulate_contact(pairs,'/World/Go2/FL_calf/collisions/mesh','/ground',1.,[{'impulse':[0,0,0],'separation':.03}])
        result=contact_quality(pairs)
        self.assertTrue(result['contact_quality_passed'])
        self.assertEqual(len(result['nonfoot_contact_pairs']),1)
        self.assertEqual(result['nonfoot_physical_contact_pairs'],[])

    def test_force_or_geometric_touch_each_fail_nonfoot_check(self):
        for point in ({'impulse':[0,0,.001],'separation':.01}, {'impulse':[0,0,0],'separation':-.0001}):
            pairs={}
            accumulate_contact(pairs,'/World/Go2/Head_lower/collisions/mesh','/ground',1.,[point])
            self.assertEqual(len(contact_quality(pairs)['nonfoot_physical_contact_pairs']),1)

    def test_empty_reports_do_not_establish_contact_quality(self):
        pairs={}
        accumulate_contact(pairs,'/World/Go2/FL_foot/collisions/mesh','/ground',1.,[])
        self.assertFalse(contact_quality(pairs)['contact_quality_passed'])
