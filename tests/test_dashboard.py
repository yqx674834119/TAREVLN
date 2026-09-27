import importlib.util
import json
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

    def test_sysnav_launch_uses_full_native_stack(self):
        checkpoint=self.project/'data/readiness/training/go2-control-v5-rails/model_3595.pt'
        checkpoint.parent.mkdir(parents=True);checkpoint.touch()
        manifest=self.root/'assets/sysnav/provenance.json'
        manifest.parent.mkdir(parents=True);manifest.write_text('{}')
        command,_,record,_=self.console.command('sysnav:lounge',120,'native')
        self.assertIn('sim/run_sysnav_full.py',command)
        self.assertNotIn('sim/run_office.py',command)
        self.assertEqual(command[command.index('--odometry')+1],'gicp')
        self.assertEqual(record,'office:native')

    def test_native_decisions_are_reviewable_without_private_steps(self):
        run=self.project/'data/sim/office/runs/native'
        (run/'agent').mkdir(parents=True)
        (run/'report.json').write_text(json.dumps(dict(task='lounge',mode='sysnav_full',status='completed',success=True)))
        (run/'agent/api.jsonl').write_text(json.dumps(dict(id=1,time=1.,purpose='process_room_navigation_query',
            status='ok',result=dict(steps=['private deliberation'],final_answer=2,reason='Doorway is visible')))+'\n')
        detail=self.console.detail('office:native')
        self.assertEqual(detail['task_id'],'sysnav:lounge')
        self.assertEqual(detail['decisions'][0]['reason'],'Doorway is visible')
        self.assertNotIn('private deliberation',json.dumps(detail))

    def test_duplicate_simulation_launch_is_rejected(self):
        proc = Mock();proc.poll.return_value=None
        self.console.jobs['own'] = dict(process=proc)
        with self.assertRaises(ValueError):
            self.console.start('habitat:7',120)

    def test_planning_replay_never_uses_future_frames(self):
        run=self.project/'data/sim/office/runs/planning'
        p=run/'agent/planning';p.mkdir(parents=True)
        (run/'report.json').write_text('{"task":"lounge"}')
        entries=[]
        for t in (1.,2.):
            name=f'plan-{int(t*1000):010d}.json'
            row=dict(time=t,robot=dict(position=[t,0,0]),layers={})
            (p/name).write_text(json.dumps(row))
            entries.append(dict(**row,file=name))
        (p/'index.jsonl').write_text('\n'.join(json.dumps(x) for x in entries))
        self.assertFalse(self.console.planning('office:planning',.5)['available'])
        result=self.console.planning('office:planning',1.5)
        self.assertEqual(result['time'],1.)
        self.assertEqual(result['trail'],[[1.,0,0]])
        with self.assertRaises(ValueError):self.console.planning('office:planning',float('nan'))

    def test_media_rejects_traversal_and_symlinks(self):
        run = self.project/'data/sim/office/runs/test';run.mkdir(parents=True)
        (run/'report.json').write_text('{"task":"lounge"}')
        (run/'frame.jpg').write_bytes(b'jpg')
        outside = self.root/'private.jpg';outside.write_bytes(b'secret')
        (run/'link.jpg').symlink_to(outside)
        self.assertEqual(self.console.media('office:test','frame.jpg'),(run/'frame.jpg').resolve())
        for name in ['../../../../../../private.jpg','link.jpg','report.json']:
            with self.assertRaises(ValueError):
                self.console.media('office:test',name)


if __name__=='__main__':
    unittest.main()
