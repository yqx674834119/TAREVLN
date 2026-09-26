import unittest

from vln_maps.providers import AMapClient, MissingAMapKey, _bounded_bbox, coarse_osm_prior


class MapContractTests(unittest.TestCase):
    def test_hidden_indoor_and_entrance_tags_do_not_reach_policy(self):
        raw = {"elements": [
            {"type": "way", "id": 1, "geometry": [], "tags": {"building": "yes", "name": "A", "building:levels": "5", "entrance": "main"}},
            {"type": "node", "id": 2, "lat": 1, "lon": 2, "tags": {"entrance": "yes"}},
            {"type": "way", "id": 3, "geometry": [], "tags": {"highway": "footway", "indoor": "yes"}},
            {"type": "way", "id": 4, "geometry": [], "tags": {"highway": "footway"}},
        ]}
        prior = coarse_osm_prior(raw)
        self.assertEqual(len(prior["elements"]), 2)
        self.assertEqual(prior["elements"][0]["tags"], {"building": "yes", "name": "A"})
        self.assertEqual(prior["elements"][1]["tags"], {"highway": "footway"})

    def test_public_query_radius_is_bounded(self):
        with self.assertRaises(ValueError):
            _bounded_bbox(23, 113, 5000)

    def test_amap_key_is_required(self):
        import os
        old = os.environ.pop("AMAP_WEB_KEY", None)
        try:
            with self.assertRaises(MissingAMapKey):
                AMapClient()
        finally:
            if old is not None:
                os.environ["AMAP_WEB_KEY"] = old


if __name__ == "__main__":
    unittest.main()
