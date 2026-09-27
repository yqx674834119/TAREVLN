"""Isolate the navigation stack from evaluator assets while permitting its API."""
import glob
import json
import os
from pathlib import Path
import select
import subprocess


class OfficeProcess:
    def __init__(self, project: Path, output: Path, stderr, geometry_only=False, task="meeting"):
        root = project.parent
        ros = root / "envs/ros"
        source = root / "deps/sysnav"
        prefixes = sorted((source / "install").glob("*/share/ament_index"))
        installs = [p.parent.parent for p in prefixes]
        python = next((ros / "lib").glob("python3.*"))
        pythonpaths = [str(p) for prefix in installs for p in prefix.glob("lib/python3.*/site-packages")]
        libs = [str(p / "lib") for p in installs] + [str(ros / "lib"), str(source / "src/exploration_planner/tare_planner/or-tools/lib")]
        output.mkdir(parents=True, exist_ok=True)
        args = ["bwrap", "--unshare-all", "--share-net", "--die-with-parent", "--new-session", "--clearenv",
                "--ro-bind", "/usr", "/usr", "--ro-bind", "/lib", "/lib", "--ro-bind", "/lib64", "/lib64",
                "--proc", "/proc", "--dev", "/dev", "--tmpfs", "/tmp", "--dir", "/tmp/home",
                "--ro-bind", str(ros), str(ros), "--ro-bind", str(source), str(source),
                "--ro-bind", str(project / "src"), "/app",
                "--ro-bind", str(project / "sim/office_ros_worker.py"), "/worker.py",
                "--ro-bind", str(project / "config/office-ros.yaml"), "/config.yaml",
                "--bind", str(output), "/output",
                "--ro-bind", str(root / "secrets/deepseek-api-key"), "/run/secrets/deepseek-api-key"]
        for path in ("/etc/resolv.conf", "/etc/hosts", "/etc/nsswitch.conf", "/etc/ssl", "/etc/localtime"):
            if Path(path).exists():
                args += ["--ro-bind", path, path]
        env = {"PATH": str(ros / "bin") + ":/usr/bin:/bin", "HOME": "/tmp/home", "LANG": "C.UTF-8",
               "PYTHONNOUSERSITE": "1", "PYTHONPATH": ":".join(["/app", *pythonpaths]),
               "LD_LIBRARY_PATH": ":".join(libs), "AMENT_PREFIX_PATH": ":".join(map(str, [*installs, ros])),
               "ROS_DISTRO": "humble", "ROS_VERSION": "2", "ROS_DOMAIN_ID": "93", "ROS_LOCALHOST_ONLY": "1",
               "RMW_IMPLEMENTATION": "rmw_cyclonedds_cpp", "RCUTILS_LOGGING_USE_STDOUT": "0", "ROS_LOG_DIR": "/output/ros-logs"}
        env["ASAN_OPTIONS"] = "detect_leaks=0:abort_on_error=1"
        for name, value in env.items():
            args += ["--setenv", name, value]
        args += ["--chdir", "/tmp", str(ros / "bin/python"), "-u", "/worker.py", "--source", str(source), "--config", "/config.yaml", "--output", "/output"]
        args += ["--task", task]
        if geometry_only:
            args.append("--geometry-only")
        self.process = subprocess.Popen(args, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=stderr,
                                        text=True, bufsize=1, env={"PATH": "/usr/bin:/bin"})

    def ask(self, observation):
        self.process.stdin.write(json.dumps(observation, allow_nan=False) + "\n")
        self.process.stdin.flush()
        if not select.select([self.process.stdout], [], [], 90)[0]:
            raise TimeoutError("ROS/VLM worker timed out")
        line = self.process.stdout.readline(1048576)
        if not line:
            raise RuntimeError(f"ROS/VLM worker exited ({self.process.poll()})")
        return json.loads(line)

    def close(self):
        if self.process.poll() is None:
            self.process.stdin.close()
            try:
                self.process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                self.process.terminate()
                self.process.wait(timeout=5)
