"""Read-only, network-isolated estimator process shared by replay and simulation."""
import json
from pathlib import Path
import select
import subprocess


class OdometryProcess:
    def __init__(self, root, out, denied_directory, method='gicp', voxel_size=.1,
                 initial_threshold=.5, fixed_threshold=None, prediction='constant_velocity', gicp_correspondence=.5):
        root, out = Path(root).resolve(), Path(out).resolve()
        out.mkdir(parents=True, exist_ok=True)
        ros, runtime = root.parent/'envs/ros', root.parent/'envs/odometry'
        command = ['bwrap', '--unshare-all', '--die-with-parent', '--new-session', '--clearenv',
                   '--ro-bind', '/usr', '/usr', '--ro-bind', '/lib', '/lib', '--ro-bind', '/lib64', '/lib64',
                   '--proc', '/proc', '--dev', '/dev', '--tmpfs', '/tmp', '--dir', '/tmp/home',
                   '--ro-bind', str(ros), str(ros), '--ro-bind', str(runtime), str(runtime),
                   '--ro-bind', str(root/'src/vln_localization'), '/app/vln_localization',
                   '--ro-bind', str(root/'scripts/odometry_worker.py'), '/worker.py',
                   '--setenv', 'PYTHONPATH', '/app', '--setenv', 'HOME', '/tmp/home',
                   '--setenv', 'LD_LIBRARY_PATH', str(ros/'lib'),
                   '--setenv', 'OMP_NUM_THREADS', '2', '--setenv', 'OPENBLAS_NUM_THREADS', '2',
                   str(runtime/'bin/python'), '-u', '/worker.py', '--deny-probe', str(denied_directory),
                   '--method', method, '--voxel-size', str(voxel_size),
                   '--initial-threshold', str(initial_threshold), '--prediction', prediction,
                   '--gicp-correspondence',str(gicp_correspondence)]
        if fixed_threshold is not None:
            command += ['--fixed-threshold', str(fixed_threshold)]
        self.log = (out/'worker.log').open('w')
        self.child = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                      stderr=self.log, text=True, bufsize=1)
        try:
            self.isolation = self._read()
            if self.isolation.get('ready') is not True or self.isolation.get('denied_directory_visible') is not False:
                raise RuntimeError('Estimator isolation not established')
            (out/'isolation.json').write_text(json.dumps(self.isolation, indent=2))
        except BaseException:
            self.close()
            raise

    def _read(self):
        if not select.select([self.child.stdout], [], [], 30)[0]:
            raise TimeoutError('Odometry worker timeout')
        line = self.child.stdout.readline()
        if not line:
            raise RuntimeError('Odometry worker exited')
        return json.loads(line)

    def ask(self, stamp, points_base_xyz):
        # No caller-supplied pose, map, labels or arbitrary fields cross the pipe.
        self.child.stdin.write(json.dumps({'time': stamp, 'points_base_xyz': points_base_xyz}, allow_nan=False)+'\n')
        self.child.stdin.flush()
        return self._read()

    def close(self):
        if not self.child.stdin.closed:
            self.child.stdin.close()
        try:
            self.child.wait(timeout=5)
        except subprocess.TimeoutExpired:
            self.child.terminate()
            self.child.wait(timeout=5)
        self.child.stdout.close()
        self.log.close()
