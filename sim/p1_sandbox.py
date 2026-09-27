"""Separate PID, mount and network namespaces for the policy, with JSON pipe RPC."""
import json
import os
from pathlib import Path
import select
import subprocess


class PolicyProcess:
    def __init__(self, project: Path, prior: Path, stderr):
        args = ["bwrap", "--unshare-all", "--die-with-parent", "--new-session", "--clearenv",
                "--ro-bind", "/usr", "/usr", "--ro-bind", "/lib", "/lib", "--ro-bind", "/lib64", "/lib64",
                "--proc", "/proc", "--dev", "/dev", "--tmpfs", "/tmp", "--dir", "/app", "--dir", "/input",
                "--ro-bind", str(project / "src/vln_policy"), "/app/vln_policy",
                "--ro-bind", str(project / "src/vln_core"), "/app/vln_core",
                "--ro-bind", str(prior), "/input/prior.json", "--setenv", "PYTHONPATH", "/app",
                "--setenv", "PYTHONNOUSERSITE", "1", "--chdir", "/app",
                "/usr/bin/python3", "-u", "-m", "vln_policy.p1_agent"]
        self.process = subprocess.Popen(args, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=stderr,
                                        text=True, bufsize=1, env={"PATH": "/usr/bin:/bin"})

    def ask(self, observation):
        self.process.stdin.write(json.dumps(observation, allow_nan=False) + "\n")
        self.process.stdin.flush()
        if not select.select([self.process.stdout], [], [], 15)[0]:
            raise TimeoutError("policy did not respond within 15 seconds")
        response = self.process.stdout.readline(262144)
        if not response:
            raise RuntimeError(f"policy exited: {self.process.poll()}")
        return json.loads(response)

    def close(self):
        if self.process.poll() is None:
            self.process.stdin.close()
            try:
                self.process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                self.process.terminate()
                self.process.wait(timeout=3)
