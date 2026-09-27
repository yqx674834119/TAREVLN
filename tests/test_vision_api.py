import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import urllib.error
from vln_core.vision_api import VisionAPI

class VisionTransportTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.key = self.root/'key'
        self.key.write_text('test-only-credential')
        self.key.chmod(0o600)
        self.api = VisionAPI(self.root/'evidence', self.key)

    def tearDown(self):
        self.temp.cleanup()

    def test_transient_timeout_retries_without_logging_credentials(self):
        payload = {'choices':[{'message':{'content':'{"action":"explore"}'}}]}
        with patch('urllib.request.urlopen', side_effect=[TimeoutError(), io.BytesIO(json.dumps(payload).encode())]) as call, patch('time.sleep'):
            self.assertEqual(self.api.ask('test', [b'test image'])['action'], 'explore')
        self.assertEqual(call.call_count, 2)
        report=json.loads((self.root/'evidence/call-0001/response.json').read_text())
        self.assertEqual(len(report['attempts']), 2)
        for path in (self.root/'evidence').rglob('*.json'):
            self.assertNotIn(self.key.read_text(), path.read_text())
            self.assertNotIn('Authorization', path.read_text())

    def test_auth_failure_is_not_retried(self):
        error=urllib.error.HTTPError('https://api.deepseek.com',403,'denied',{},None)
        with patch('urllib.request.urlopen', side_effect=error) as call, self.assertRaises(RuntimeError):
            self.api.ask('test', [])
        self.assertEqual(call.call_count,1)

    def test_network_retry_limit_stops_instead_of_faking_a_decision(self):
        with patch('urllib.request.urlopen', side_effect=TimeoutError()) as call, patch('time.sleep'), self.assertRaises(TimeoutError):
            self.api.ask('test', [])
        self.assertEqual(call.call_count,3)
        report=json.loads((self.root/'evidence/call-0001/response.json').read_text())
        self.assertEqual(report['status'],'error')
