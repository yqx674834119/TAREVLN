"""Isolate the navigation stack from evaluator assets while permitting its API."""
import glob
import json
import os
from pathlib import Path
import select
import subprocess
from urllib.parse import urlsplit


class FullSysNavProcess:
    def __init__(self, project: Path, output: Path, stderr, task="lounge", api_proxy=None,instruction=None,api_budget=150,enable_room_early_stop=False,vision_gpu=1,sam_gpu=2,vlm_gpu=3):
        root = project.parent
        ros = root / "envs/ros"
        source = root / "deps/sysnav-full"
        envroot = root / "envs/sysnav-full"
        gpulibs = root / "toolchains/nvidia-580.173.02/lib"
        prefixes = sorted((source / "install").glob("*/share/ament_index"))
        installs = [p.parent.parent for p in prefixes]
        python = next((ros / "lib").glob("python3.*"))
        pythonpaths = [str(p) for prefix in installs for p in prefix.glob("lib/python3.*/site-packages")]
        libs = [str(p / "lib") for p in installs] + [str(gpulibs), str(ros / "lib"), str(source / "src/exploration_planner/tare_planner/or-tools/lib")]
        output.mkdir(parents=True, exist_ok=True)
        for directory in ('workspace-output','debug','logs'):
            (output/directory).mkdir(exist_ok=True)
        args = ["bwrap", "--unshare-all", "--share-net", "--die-with-parent", "--new-session", "--clearenv",
                "--ro-bind", "/usr", "/usr", "--ro-bind", "/lib", "/lib", "--ro-bind", "/lib64", "/lib64",
                "--proc", "/proc", "--dev", "/dev", "--tmpfs", "/tmp", "--dir", "/tmp/home",
                "--ro-bind", str(ros), str(ros), "--ro-bind", str(source), str(source),
                "--ro-bind", str(project / "src"), "/app",
                "--ro-bind", str(project / "sim/sysnav_full_worker.py"), "/sim/sysnav_full_worker.py",
                "--ro-bind", str(project / "sim/sysnav_sensor_bridge.py"), "/sim/sysnav_sensor_bridge.py",
                "--ro-bind", str(project / "sim/sysnav_runtime_audit.py"), "/sim/sysnav_runtime_audit.py",
                "--ro-bind", str(project / "sim/planning_recorder.py"), "/sim/planning_recorder.py",
                "--ro-bind", str(envroot), str(envroot),
                "--ro-bind", str(root / "deps/sam2"), str(root / "deps/sam2"),
                "--ro-bind", str(gpulibs), str(gpulibs),
                "--ro-bind", str(root / "assets/sam"), "/models/sam",
                "--ro-bind", str(root / "assets/sysnav"), "/models/sysnav",
                "--ro-bind", str(project / "config/sysnav-full-ros.yaml"), "/config.yaml",
                "--ro-bind", str(project / "config/sysnav-navigation-boundary.ply"), "/boundary.ply",
                "--bind", str(output), "/output",
                "--ro-bind", str(root / "secrets/deepseek-api-key"), "/run/secrets/deepseek-api-key"]
        for path in ("/etc/resolv.conf", "/etc/hosts", "/etc/nsswitch.conf", "/etc/ssl", "/etc/localtime"):
            if Path(path).exists():
                args += ["--ro-bind", path, path]
        # GPU access only: no robot device or private scene/evaluator mounts.
        for device in glob.glob('/dev/nvidia*'):
            args += ['--dev-bind', device, device]
        args += ['--dir','/workspace', '--ro-bind',str(source/'src'),'/workspace/src',
                 '--bind',str(output/'workspace-output'),'/workspace/output',
                 '--bind',str(output/'debug'),'/workspace/debug',
                 '--bind',str(output/'logs'),'/workspace/logs']
        env = {"PATH": str(envroot / "bin") + ":" + str(ros / "bin") + ":/usr/bin:/bin", "HOME": "/tmp/home", "LANG": "C.UTF-8",
               "PYTHONNOUSERSITE": "1", "PYTHONPATH": ":".join(["/app", "/sim", str(source/"src/vlm_node"), str(source/"src/semantic_mapping"), str(root/"deps/sam2"), *pythonpaths]),
               "LD_LIBRARY_PATH": ":".join(libs), "AMENT_PREFIX_PATH": ":".join(map(str, [*installs, ros])),
               "ROS_DISTRO": "humble", "ROS_VERSION": "2", "ROS_DOMAIN_ID": "94", "ROS_LOCALHOST_ONLY": "1",
               "RMW_IMPLEMENTATION": "rmw_cyclonedds_cpp", "RCUTILS_LOGGING_USE_STDOUT": "0", "ROS_LOG_DIR": "/output/ros-logs"}
        env.update(VLM_PROVIDER='deepseek', DEEPSEEK_MODEL='deepseek-flash',
                   DEEPSEEK_API_KEY_FILE='/run/secrets/deepseek-api-key',
                   SYSNAV_API_LOG='/output/api.jsonl', SYSNAV_TIMING_DIR='/output', SYSNAV_MAX_API_CALLS=str(api_budget),
                   SYSNAV_YOLO_MODEL='/models/sysnav/yoloe-26x-seg.pt',
                   SYSNAV_TEXT_EMBEDDINGS='/models/sysnav/text-embeddings.pt',
                   SYSNAV_SAM_CHECKPOINT='/models/sam/sam2.1_hiera_base_plus.pt',
                   CUDA_VISIBLE_DEVICES=str(vlm_gpu), CUDA_DEVICE_ORDER='PCI_BUS_ID', SYSNAV_DETECTION_GPU=str(vision_gpu),
                   SYSNAV_SEMANTIC_GPU=str(sam_gpu), SYSNAV_VLM_GPU=str(vlm_gpu),
                   OMP_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4',
                   YOLO_CONFIG_DIR='/tmp/yolo', MPLCONFIGDIR='/tmp/matplotlib')
        # Ten ROS processes exceed the small localhost discovery range used by
        # this Humble build. This config is limited to this process tree.
        env['CYCLONEDDS_URI'] = '<CycloneDDS><Domain><Discovery><ParticipantIndex>auto</ParticipantIndex><MaxAutoParticipantIndex>64</MaxAutoParticipantIndex><Peers><Peer Address="127.0.0.1"/></Peers></Discovery></Domain></CycloneDDS>'
        env["ASAN_OPTIONS"] = "detect_leaks=0:abort_on_error=1"
        if api_proxy:
            endpoint=urlsplit(api_proxy)
            if (endpoint.scheme!='http' or endpoint.hostname!='127.0.0.1' or not endpoint.port
                or endpoint.username or endpoint.password or endpoint.path or endpoint.query or endpoint.fragment):
                raise ValueError('API relay must be an explicit loopback HTTP CONNECT endpoint')
            env['https_proxy']=api_proxy
        for name, value in env.items():
            args += ["--setenv", name, value]
        args += ["--chdir", "/workspace", str(envroot / "bin/python"), "-u", "/sim/sysnav_full_worker.py", "--source", str(source), "--config", "/config.yaml", "--output", "/output"]
        args += ["--task", task,"--boundary","/boundary.ply"]
        if enable_room_early_stop:args+=['--enable-room-early-stop']
        if instruction is not None:args+=['--instruction',instruction]
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
