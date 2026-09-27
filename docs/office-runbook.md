# SysNav / TARE / DeepSeek 单场景实验

项目分支为 `codex/sysnav-isaac-office`。所有服务、源代码、环境、下载缓存、模型和日志均位于 5090 的 `/home/yqx/VLN`；不需要重启或替换系统 GPU 驱动。原 TARE_WheelTECH 和 P1 分支保留。

## 任务和信息边界

机器人在 NVIDIA Office 中接受单场景语言任务，默认 `--task meeting`：寻找有共享会议桌、多把椅子及大壁挂展示屏的会议区域。基础任务 `--task lounge`：寻找有成组扶手椅、小桌和壁挂屏的休息区。两个任务分别统计，不用休息区成功代替会议室成功。个人工位上的 PC 显示器、顶灯和窗户不算展示屏。机器人需要进入空闲地面区域，从两个不同位置观察确认，并静止至少一秒。

VLM 输入是四个方向的当前 RGB 图像、TARE 给出的可达观察候选及最近十条动作/执行记录。输出仅允许 `explore`、`inspect(candidate_id)`、`confirm`。候选位置来自当前点云形成的 TARE 图；模型不能给任意坐标或速度。到达复核还需给出当前图像中桌面像素；RGB-D 将其投影到三维，要求水平距离不超过 3 米、深度局部连续，且高度在 Office 地面以上 0.2–1.2 米。误指地面、墙面或缺失深度会拒绝并允许一次像素重标。这些是几何一致性门限，不能单独证明像素属于桌面；仍需结合视觉判定，误差见实测报告。两次观察的三维锚点还须相距不超过 1.5 米。至少两个近期正例观察位置相隔 0.6–4 米才能请求停止；评测器另外检查私有目标区域。

API 推理异步执行，等待期间给步态控制器零速度，物理和传感器继续运行。返回结果若已超过 60 秒或机器人相对拍摄位置漂移超过 0.35 米，则丢弃；原生 TARE 再检查候选的当前可达性。双视点确认的证据窗口为 60 秒，计算的是拍摄时间和拍摄位置。

TARE 候选按 12 个方位分别提供约 1.2 米和 2.8 米的近/远视点；确认阶段优先使用 1.8 米内的局部候选。短暂遮挡不清空先前已验证观察，证据在 60 秒后过期。

几何子目标保持到距目标小于 0.6 米、25 秒期限到达，或连续 10 秒没有至少 0.1 米的距离进展；这些反馈进入下一次 VLM 决策。`mission_paths` 计数记录 TARE 为语义子目标实际发布的有效路径。

搜索时保留实际行走位置，过滤距已到访位置小于 0.6 米的候选；确认阶段允许回访。候选的图像投影同时形成 `selected_view → candidate_id` 对应表，不匹配的选择会要求模型重选一次；第二次仍无效则记录拒绝原因，回退到原生几何探索，不将非法编号用于运动。历史提示只保存动作、位置和执行结果，不把旧视觉描述作为新画面的事实。导航模型报告目标线索且置信度至少 0.6 后，调用不含先前结论的独立 VLM 图像复核，分别判断任务设施、当前是否在目标旁以及置信度；只有复核满足全部设施、在区域内且置信度至少 0.8 才产生一个到达正例。导航选点与到达判断分离，最终空间真值始终只在评测器中使用。

默认关闭模型推理模式，单次网络尝试超时为 20 秒，临时网络错误和部分服务端错误最多尝试三次，认证失败不重试。等待期间保持零速度；所有尝试写入证据日志，失败不会返回模拟决策。API 逻辑调用上限为每轮 60 次，重试次数单独记录。

真实执行链为：Isaac 点云和位姿 → SysNav 原生 terrainAnalysis / TARE → `/way_point` → localPlanner → `/path` → pathFollower → `/cmd_vel` → Go2 关节力矩策略。导航进程被放入独立的文件系统命名空间，看不到 USD、目标区域或模拟器进程的场景文件。网络用于 DeepSeek API 和 localhost ROS，ROS domain 为 93。

本轮借鉴 SysNav 的分层职责，并使用其固定源码中的 TARE 和运动链。当前语义适配器基于观察候选；没有启动 SysNav 的整套 Room 分割、YOLO / SAM 物体图或实机 WebRTC。这一点与完整复现 SysNav 必须区分。

## 环境和代码

- Isaac：`~/VLN/envs/isaac51`，Python 3.11，Isaac Sim 5.1。
- ROS：`~/VLN/envs/ros`，RoboStack Humble，Python 3.12，与 Isaac 通过传感器 JSON 管道通信，避免 Python ABI 混用。
- USD 工具：`~/VLN/envs/usdtools`，不向 Isaac 环境安装另一个 usd-core。
- SysNav：`~/VLN/deps/sysnav`，来自 `478727e5a559be22822d5d093b57431449336287` 的仿真相关包，保留源代码许可证和基线 Git 快照。
- 修改记录：`patches/sysnav-isaac-mission.patch`；依赖锁：`config/ros-humble-linux-64.lock`。
- API：官方 `https://api.deepseek.com`，模型 `deepseek-flash`，密钥文件 `~/VLN/secrets/deepseek-api-key`，0600 权限。

后台下载：

```bash
cd ~/VLN/cross-building-floor-nav
nohup ~/VLN/envs/usdtools/bin/python scripts/fetch_usd_scene.py --workers 32 \
  > ~/VLN/logs/office-download.log 2>&1 < /dev/null &
```

下载器递归解析 USD 引用、材质属性、MDL 的相对模块导入及贴图；支持中断续传和逐文件 SHA-256。只有所有依赖完成后 `~/VLN/assets/usd-cache/office-download.json` 的 `complete` 才为 true。SHA-256 对应缓存中的文件；USD 中的路径已经本地化。

编译与启动：

```bash
bash scripts/build_sysnav.sh
~/VLN/envs/usdtools/bin/python sim/build_office.py
nohup env OMNI_KIT_ACCEPT_EULA=YES bash scripts/with_gpu_compat.sh \
  ~/VLN/envs/isaac51/bin/python -u sim/run_office.py \
  --name office-vlm-001 --max-seconds 600 \
  > ~/VLN/logs/office-vlm-001.log 2>&1 < /dev/null &
```

EULA 已由用户明确授权。仅使用 GPU 0；兼容库通过本项目脚本动态链接。`realRobot=false`，不连接机器狗或底盘串口。

构建器从 `sim/fixtures/office-evaluation.json` 生成私有 `data/sim/office/private/oracle.json`（会议室）及 `oracle_lounge.json`（休息区），其中定义合法起点及人工核对过的目标区域。评测夹具属于场景作者文件，不挂载给导航进程。该文件属于场景作者与评测器，不是策略输入；不要把它复制进 `config` 或语义提示词。

原生故障定位可使用 `scripts/build_sysnav_asan.sh`，并用 `scripts/replay_office_sensors.py` 回放保存的传感器。ASan 编译保持 Eigen 分配约定和 OR-Tools 9.8 所依赖的 Abseil 类布局一致；否则调试本身会产生 ABI 错误。回放只能验证对应输入的处理，不能当作自主任务成功。完成诊断后运行普通构建脚本恢复 Release。

已修复上游 `LocalCoveragePlanner::SelectViewPoint` 在队列缩小后仍按原始队列采样的问题；`scripts/test_sysnav_queue.py` 提取实际原生方法，使用可控视点覆盖数据运行 ASan 回归。原始方法在 `office-vlm-002` 中触发 ASan 越界报告，独立回归还触发了重复选点断言；修复后 10,000 次普通/前沿选择通过。办公室配置还将后方目标冻结角设为 180 度，以允许经过碰撞检查的双向短路径；没有关闭障碍或旋转碰撞检查。

## 证据与限制

每轮在 `data/sim/office/runs/<name>/` 保存：报告、原始 RGB / 深度 / 点云、位姿和命令轨迹、ROS 节点日志、TARE 输出计数、VLM 输入图片与决策、用量和延迟、文件访问隔离探针，以及运行代码和配置的副本。

`--preview` 是场景检查；`--geometry-only` 是原生几何链路测试，均不算 VLM 任务完成。只有普通任务运行中的 VLM 双视点确认、独立位置检查和停止条件同时满足，才可报告语义任务完成。任务完成与无碰撞运动分别报告；所有非脚部接触保留原始碰撞体名称，不能用任务成功掩盖运动问题。

传感器为理想 RGB-D 和虚拟多线扫描雷达，位姿为模拟器理想里程计。尚不代表真实 MID360 扫描、SLAM 漂移、实机传感器标定或 Go2 实机闭环。室内灯光和静态碰撞由场景构建器明确设置。

参考：[SysNav](https://github.com/zwandering/SysNav)、[DeepSeek Vision](https://api-docs.deepseek.com/guides/vision/)、[Isaac 5.1 环境资产](https://docs.isaacsim.omniverse.nvidia.com/5.1.0/assets/usd_assets_environments.html)。

可在对应运行目录创建 `STOP` 文件结束仿真，程序会保留报告与接触记录；优先使用它而非 Kit 的 SIGINT。复核视频可用 ROS 环境运行 `scripts/summarize_office.py <run> --oracle <private-oracle> --video` 生成，第三人称画面只供复核，不进入策略。
