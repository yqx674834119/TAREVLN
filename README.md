# Cross-Building and Cross-Floor Goal Navigation

This is the independent project workspace for A 5F → B 3F navigation. The project is stored under `~/VLN/cross-building-floor-nav` on the Mac and Host 5090. The TARE_WheelTECH production checkout and its separate `sim2real` Git repository are not runtime dependencies of this bootstrap.

## 项目概览

本项目实现一个在 Isaac Sim 中运行的室内语言目标导航系统，并保留向“粗地图先验下的跨楼宇、跨楼层导航”扩展的接口。当前可运行主线是单场景 Office / SysNav：Go2 通过 RGB 全景、虚拟 LiDAR、位姿和语言指令，在 TARE 生成的局部可达视点上探索房间并寻找目标物体。

当前主线包括：

- Isaac Sim 5.1 中的关节式 Unitree Go2 与 PhysX 碰撞。
- 原生 SysNav 的 room segmentation、semantic mapping、TARE、localPlanner、pathFollower。
- YOLO/SAM 视觉观测、GICP 可选里程计、DeepSeek Flash VLM 语义决策。
- TARE 可达候选、waypoint、路径、速度和 RGB/SAM/YOLO 结果的前端回放。
- OSM 与高德地图适配器，以及 MP3D 到 Isaac 的场景转换实验。

仿真运行不等同于实机验收；本仓库不会自动打开机器狗底盘或串口。

## 文件结构

```text
cross-building-floor-nav/
├── README.md                         # 项目说明、启动方法和架构
├── config/                           # ROS、SysNav、传感器、模型和来源清单
│   ├── sysnav-full-ros.yaml          # 原生 SysNav ROS 参数
│   ├── sysnav-full-provenance.json   # 版本、GPU、模型和隔离边界
│   └── sysnav-parity-manifest.json   # SysNav 对齐清单
├── src/
│   ├── vln_core/                     # 传感器、地图和几何接口
│   ├── vln_policy/                   # 任务描述、语义契约和策略适配
│   ├── vln_maps/                     # OSM / 高德地图 API
│   ├── vln_perception/               # SAM、YOLO 和观测处理
│   └── vln_localization/             # GICP 与定位进程
├── sim/                              # Isaac、MP3D 和 Go2 仿真入口
│   ├── run_sysnav_full.py            # 完整 SysNav + VLM + Go2 运行入口
│   ├── sysnav_full_worker.py         # ROS/VLM 子进程及信息隔离
│   ├── sysnav_go2_policy.py          # Go2 仿真控制接口
│   ├── gpu_lidar.py                  # CUDA LiDAR 加速与 PhysX 对照
│   ├── camera_rig.py                 # 同步多相机采集
│   ├── build_mp3d_isaac.py           # MP3D 场景转换
│   └── sysnav_evaluation.py          # 目标、可见性和停止条件评测
├── scripts/                          # 安装、编译、启动、诊断和报告脚本
│   ├── build_sysnav_full.sh          # 编译原生 ROS 包
│   ├── start_dashboard.sh            # 启动可视化前端
│   └── with_gpu_compat.sh            # 进程级 NVIDIA 动态库兼容
├── frontend/                         # 无构建依赖的前端页面
├── docs/                             # runbook、实验报告、基准和研究计划
├── patches/                          # SysNav 上游适配补丁
├── tests/                            # 纯 Python、接口、评测和前端测试
├── data/                             # 场景、缓存和运行结果（大文件不提交）
└── deps/                             # 上游 SysNav、SAM2、Isaac 等私有依赖
```

运行产生的记录位于 `data/sim/office/runs/<run-name>/`，包括 `report.json`、`trajectory.jsonl`、`agent/events.jsonl`、`agent/api.jsonl`、RGB/SAM/YOLO 图片、TARE 规划记录和源代码快照。`data/sim/office/private/oracle.json` 只供评测器使用，不挂载给导航策略。

## 整体算法架构

```mermaid
flowchart LR
    U[语言任务] --> T[TaskSpec / 任务条件]
    S[Isaac RGB-D\nLiDAR / 位姿 / IMU] --> A[Sensor Adapter]
    A --> P[YOLO + SAM\n房间分割与物体观测]
    P --> M[Room/Object Memory\n证据、覆盖率、负证据]
    T --> V[DeepSeek VLM\n语义分类与候选选择]
    M --> V
    V --> Q[语义子目标\nroom / object / inspect]
    Q --> R[TARE\n局部探索与可达视点]
    R --> W[/way_point]
    W --> L[localPlanner\n局部碰撞与路径]
    L --> F[pathFollower\n速度控制]
    F --> G[Go2 关节力矩策略]
    G --> S
    R --> D[/path、free_paths、viewpoints]
    A --> D
    D --> H[Dashboard\nRGB、轨迹、TARE、Waypoint、决策]
    E[Oracle Evaluator\n目标区域、停止条件、接触审计] -. 只评测 .-> H
```

系统职责边界如下：VLM 选择语义房间或观察候选，不生成速度和任意坐标；TARE 负责局部探索和可达视点；localPlanner 负责碰撞约束下的路径；pathFollower 和 Go2 策略负责执行。目标房间、目标包围盒和评测真值只进入独立 evaluator。

## 快速启动

以下命令在 Host 5090 执行，项目目录为 `~/VLN/cross-building-floor-nav`。

### 1. 检查 GPU 兼容库

```bash
cd ~/VLN/cross-building-floor-nav
bash scripts/setup_gpu_compat.sh       # 仅首次或驱动用户态库变化时执行
scripts/with_gpu_compat.sh nvidia-smi
```

`with_gpu_compat.sh` 只为子进程设置 `LD_LIBRARY_PATH` 和 Vulkan ICD，不替换系统驱动，也不需要重启。

### 2. 编译原生 SysNav

```bash
cd ~/VLN/cross-building-floor-nav
bash scripts/build_sysnav_full.sh
```

该脚本编译 `terrain_analysis`、`terrain_analysis_ext`、`local_planner`、`tare_planner`、`semantic_mapping` 和 `vlm_node` 等 ROS 包。完整依赖、版本和补丁见 `config/` 与 `patches/`。

### 3. 启动可视化前端

```bash
cd ~/VLN/cross-building-floor-nav
nohup bash scripts/start_dashboard.sh --lan-ip 10.4.15.106 \
  > ~/VLN/logs/dashboard-server.log 2>&1 < /dev/null &
```

浏览器访问 `http://10.4.15.106:8878/`；只在服务器本机访问时使用 `http://127.0.0.1:8878/`。Mac 也可以建立 SSH 转发：

```bash
ssh -N -L 127.0.0.1:8878:127.0.0.1:8878 5090
```

### 4. 启动完整 SysNav Office 仿真

```bash
cd ~/VLN/cross-building-floor-nav
OMNI_KIT_ACCEPT_EULA=YES bash scripts/with_gpu_compat.sh \
  ~/VLN/envs/isaac51/bin/python -u sim/run_sysnav_full.py \
  --name office-sysnav-demo \
  --task meeting_crossroom \
  --max-seconds 300 \
  --api-budget 150 \
  --odometry gicp \
  --lidar-profile uniform64 \
  --enable-room-early-stop \
  --lidar-backend cuda \
  --lidar-gpu 3 \
  --panorama-backend atlas \
  --isaac-gpu 0 \
  --vision-gpu 1 \
  --sam-gpu 2 \
  --vlm-gpu 3
```

常用任务是 `meeting`、`lounge` 和 `meeting_crossroom`。`--max-seconds -1` 表示不设置仿真时间上限，`--api-budget -1` 表示不设置 VLM 调用上限。模型密钥只从 `~/VLN/secrets/deepseek-api-key` 读取，不能写入命令行或 Git。

前端启动的任务会自动写入 `data/sim/office/runs/`。优先通过前端停止任务；命令行调试时可在对应运行目录创建 `STOP` 文件。

### 5. 查看结果和运行测试

```bash
cd ~/VLN/cross-building-floor-nav
python3 -m pytest -q tests
python3 scripts/summarize_sysnav_parity_run.py \
  data/sim/office/runs/office-sysnav-demo
```

GPU LiDAR 和同步相机的独立验证入口分别是 `sim/validate_gpu_lidar.py` 和 `sim/validate_camera_rig.py`。MP3D 场景转换和 R2R-CE 评测流程见 `docs/sysnav-crossroom-mp3d.md`，不要把 Isaac 转换结果直接当作官方 Habitat 分数。

## 运行配置和安全边界

- Isaac、ROS、SysNav、SAM2 和模型缓存都位于 `~/VLN`；不修改生产 `TARE_WheelTECH` 配置。
- 默认使用 ROS domain 94，导航进程通过独立文件系统命名空间运行。
- 评测器拥有 oracle 场景真值，策略只接收传感器和允许的语义候选。
- 当前结果是仿真证据；没有通过本 README 的命令自动启动实体机器狗。
- 前端局域网入口没有用户认证，只应在可信网络内使用。

## 进一步阅读

- [Office SysNav 运行说明](docs/office-runbook.md)
- [SysNav 单场景结果](docs/sysnav-full-single-scene.md)
- [SysNav 系统对齐记录](docs/sysnav-system-parity.md)
- [MP3D / Isaac 迁移与评测](docs/sysnav-crossroom-mp3d.md)
- [前端与历史运行说明](docs/benchmarks/dashboard.md)
- [研究计划与 CVPR 实验设计](docs/research-plan.md)

## Current stages

- **Harder SysNav tests and native planning replay:** the new cross-room display task ran for 300 s and timed out after 68.1 m; the original planner's room-switching behavior remains a limitation. MP3D meshes now load in Isaac with decoded textures, static collision, original episode poses, a Go2 spawn check, and isolated scoring. See [results and reproduction](docs/sysnav-crossroom-mp3d.md). The dashboard replays actual TARE samples, obstacles, paths and waypoints.

- **Physical-dog readiness goal:** active iteration with explicit simulation, perception, sensor, interface and field gates. See [current status](docs/readiness/STATUS.md). The project is not yet ready for physical motion.
- **Estimated localization:** isolated GICP can now drive TARE from rigid-mounted virtual scans. The 120 s dense-scan development run retained accurate simulated localization but timed out on semantic confirmation; see [localization evidence and limitations](docs/readiness/localization.md).

- **Single-scene Office integration:** native SysNav TARE / local planning / path following now connects Isaac RGB and point clouds to DeepSeek Flash visual decisions and articulated Go2 control. The policy receives reachable candidate IDs and has no access to the evaluator scene or goal regions. See [Office runbook](docs/office-runbook.md); successful task completion is a separate runtime gate from installation and compilation.

- **Map component:** OSM Nominatim resolves a named place, and Overpass downloads a small source snapshot. A strict exporter gives the policy only building identity/outlines and outdoor walkways. Raw OSM stays evaluator-side. AMap Web Service adapter supports geocoding, WGS84→GCJ02 conversion, and walking routes once `AMAP_WEB_KEY` is supplied. AMap coordinates remain tagged `GCJ02`; they are never silently merged with OSM WGS84 geometry.
- **Task component:** `vln_core.connections` is the initial directed connection-memory contract. A sighting or sign may guide investigation, but only a traversal plus distinct arrival-identity evidence becomes a reusable route. Capability restrictions, wrong signs, blocked edges and direction are represented; the P1 diagnostic now connects visible-sign perception and motor execution, while the proposed research planner remains to be integrated.
- **Simulation component:** Isaac Sim 5.1 is installed and the articulated Go2 has completed the synthetic P1 A5F → A1 → outdoor → B1 → B3F route on Host 5090. The baseline uses rendered QR signs, depth, ideal LiDAR/odometry and an API-assisted physical elevator. This is an integration baseline; TARE/VLM, real-map alignment and real-robot validation remain pending. Non-foot thigh contacts were recorded, so reaching the goal is not a clean locomotion-quality pass. See [P1 runbook](docs/p1-runbook.md) and [audit results](docs/p1-results.md).
- **GPU compatibility:** `scripts/setup_gpu_compat.sh` copies the already cached 580.173.02 userspace libraries into this project. `scripts/with_gpu_compat.sh` changes library paths for a child process only; it does not replace system libraries or reboot the host.

## OSM snapshots

```bash
PYTHONPATH=src python3 -m vln_maps.cli osm-place hkust-gz "Hong Kong University of Science and Technology (Guangzhou)" --radius-m 400
PYTHONPATH=src python3 -m vln_maps.cli osm-place harbour-city "Harbour City, Hong Kong" --radius-m 400
```

The output is `data/osm/<slug>/raw.json`, `prior.json`, and `metadata.json`. Review the resolved centroid and building identities before defining A/B; a search match alone is not a verified campus boundary. Data attribution: © OpenStreetMap contributors, ODbL. Public Nominatim is used at most once per place; cached results are reused. The Overpass request is spatially bounded and cached. Bulk data acquisition should use a regional extract instead of public services.

The current 400 m HKUST(GZ) snapshot has 29 building features and only 4 outdoor walkway features; Harbour City has 231 buildings and 448 walkway features. These are small map-acquisition checks, not validated navigable site graphs. The synthetic A/B simulator is intentionally independent of both real-world snapshots until building identities, floor plans, legal access and connection ground truth are supplied.

## AMap adapter

Apply for a **Web 服务 API** key. Put it in a `0600` file under `~/VLN/secrets/amap-web-key` on Host 5090, or export `AMAP_WEB_KEY` in a private shell. Do not commit or print it. The CLI takes the file via `--key-file` and returns a structured `MISSING_AMAP_KEY` status if absent. The adapter and keyless error path are implemented; live AMap calls cannot be validated until a key is available. Examples after provisioning:

```bash
PYTHONPATH=src python3 -m vln_maps.cli amap-geocode '香港科技大学（广州）' --key-file ~/VLN/secrets/amap-web-key
PYTHONPATH=src python3 -m vln_maps.cli amap-convert 113.4784431 22.8875346 --key-file ~/VLN/secrets/amap-web-key
# Pass GCJ02 coordinates returned by AMap, not raw OSM WGS84:
PYTHONPATH=src python3 -m vln_maps.cli amap-walk ORIGIN_LON ORIGIN_LAT DEST_LON DEST_LAT --key-file ~/VLN/secrets/amap-web-key
```

`walking_route` consumes **GCJ02** endpoints. Convert WGS84 points with the AMap conversion endpoint before calling it, and keep each provider's coordinates labeled. Hong Kong coverage and the account's regional entitlement require a real API response before either is claimed.

## Host 5090 installation

The bootstrap is launched in the background, with a PID and a full log under `~/VLN/logs/`. It installs no system packages and does not start TARE, a real robot, or a network-facing service.

```bash
nohup bash ~/VLN/cross-building-floor-nav/scripts/install_isaac51.sh \
  > ~/VLN/logs/isaac51-install.log 2>&1 < /dev/null &
echo $! > ~/VLN/logs/isaac51-install.pid
```

The user explicitly accepted NVIDIA's EULA for this task. Installation, GPU startup, articulated dynamics, camera/depth/IMU and physical elevator travel have now run. Native simulation, ROS 2 integration, and real-robot validation remain separate acceptance gates.

## Earlier MuJoCo smoke test

On Host 5090, with no reboot or system driver changes:

```bash
cd ~/VLN/cross-building-floor-nav
python3 sim/generate_demo_scene.py
MENAGERIE_CACHE_DIR=~/VLN/cache/mujoco_menagerie MUJOCO_GL=egl \
  scripts/with_gpu_compat.sh ~/VLN/envs/mujoco-go2/bin/python sim/run_mujoco_smoke.py
```

The report and frames are `data/sim/demo/mujoco_smoke.json`, `mujoco_smoke.png` (scene overview), and `mujoco_go2_a5f.png` (robot closeup). This smoke test checks only that the Go2 asset compiles in the campus scene, remains supported at A/F5 for 400 physics steps, and can be rendered through the project-local NVIDIA library path. The static `oracle.json` declares elevator and portal transitions for a future simulator skill API; they are **not** working robot actions. The companion `prior.json` deliberately omits those connections.

The preceding MuJoCo files belong to the earlier static bootstrap. The implemented doors/elevators and closed loop are in `sim/build_p1_usd.py` and `sim/run_p1.py`; use the P1 runbook for current commands. The simulator environment is `~/VLN/envs/isaac51`.

After you read and accept the [NVIDIA Omniverse license](https://docs.omniverse.nvidia.com/platform/latest/common/NVIDIA_Omniverse_License_Agreement.html), the prepared stage-only check is:

```bash
OMNI_KIT_ACCEPT_EULA=YES scripts/with_gpu_compat.sh \
  ~/VLN/envs/isaac51/bin/python sim/run_isaac_stage_smoke.py
```

This lightweight stage diagnostic is separate from the P1 route test. Downloads can be prepared in the background with `scripts/install_p1_components.sh`; it does not launch Isaac or accept a license automatically.

单场景 Office 的实际运行结果、失败案例和复核证据见 [实测报告](docs/office-results.md)，启动方式见 [运行说明](docs/office-runbook.md)。
