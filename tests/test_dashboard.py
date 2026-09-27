import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock

spec = importlib.util.spec_from_file_location('dashboard_server',Path(__file__).resolve().parents[1]/'scripts/dashboard/server.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


class DashboardTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.project = self.root/'cross-building-floor-nav'
        self.project.mkdir()
        self.console = m.Console(self.root,self.project)

    def tearDown(self):
        self.temp.cleanup()

    def test_model_internal_reasoning_is_not_published(self):
        x = m.public_decision(dict(reason='A visible doorway supports moving left',action='inspect',
            reasoning_content='private chain',thinking='private',api_key='secret'))
        self.assertEqual(set(x),{'reason','action'})

    def test_task_and_command_arguments_are_allowlisted(self):
        self.console.episodes['7'] = {}
        cmd,_,_,_ = self.console.command('habitat:7',120,'test')
        self.assertEqual(cmd[cmd.index('--episode-id')+1],'7')
        with self.assertRaises(ValueError):
            self.console.command('habitat:7;echo injected',120,'test')
        with self.assertRaises(ValueError):
            self.console.command('office:lounge',-1,'test')

    def test_cannot_stop_unowned_job(self):
        with self.assertRaises(ValueError):
            self.console.stop('another-session-job')

    def test_duplicate_simulation_launch_is_rejected(self):
        proc = Mock();proc.poll.return_value=None
        self.console.jobs['own'] = dict(process=proc)
        with self.assertRaises(ValueError):
            self.console.start('habitat:7',120)

    def test_media_rejects_traversal_and_symlinks(self):
        run = self.project/'data/sim/office/runs/test';run.mkdir(parents=True)
        (run/'report.json').write_text('{"task":"lounge"}')
        (run/'frame.jpg').write_bytes(b'jpg')
        outside = self.root/'private.jpg';outside.write_bytes(b'secret')
        (run/'link.jpg').symlink_to(outside)
        self.assertEqual(self.console.media('office:test','frame.jpg'),run/'frame.jpg')
        for name in ['../../../../../../private.jpg','link.jpg','report.json']:
            with self.assertRaises(ValueError):
                self.console.media('office:test',name)


if __name__=='__main__':
    unittest.main()
