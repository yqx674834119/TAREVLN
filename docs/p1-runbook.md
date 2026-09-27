# P1：A5F → B3F 仿真联调

本轮分支为 `codex/p1-closed-loop`。5090 的项目根目录为 `/home/yqx/VLN/cross-building-floor-nav`；环境、模型、日志均保存在 `/home/yqx/VLN`。未重启服务器，未修改系统驱动或 TARE_WheelTECH 实车部署。

## 本轮实现的范围

这是一套用于验证接口、物理转移和信息隔离的 **Isaac Sim 原生联调基线**。机器人是有 12 个受控关节的 Go2；策略给出局部速度目标，NVIDIA 已发布的平地行走网络生成关节目标，PD 控制器输出力矩。运行期间不通过修改机器人位姿实现行走或乘梯。

高层使用可见二维码标牌和 LiDAR 栅格 A*，还不是 TARE + VLM 研究方法。二维码编码的是现场标牌内容，由渲染 RGB 解码，再用相机深度反投影。没有调用物体真值位置接口。默认里程计来自仿真真值，必须作为理想定位假设披露。

第一版的摄像头、平面 LiDAR 和 IMU 是明确记录的虚拟配置，不能称为真实 Go2/MID360 标定。电梯呼叫和选层使用 `ELEVATOR_API_ASSISTED`，不能称为自主按键。

行走适配器还使用仿真轿厢的竖直速度，对低层策略的竖直本体速度作平台补偿。这是额外的仿真假设，不能声称已验证未知运动平台上的真实 Go2 控制；它不向高层提供目的楼层身份。

## 启动

用户已于本轮明确同意 NVIDIA Omniverse EULA。以下启动命令只应在接受该协议的环境运行。

```bash
cd ~/VLN/cross-building-floor-nav
# 小环境负责离线 USD 编写；Isaac 环境负责实际物理和渲染。
~/VLN/envs/usdtools/bin/python sim/build_p1_usd.py --output data/sim/p1
nohup env OMNI_KIT_ACCEPT_EULA=YES scripts/with_gpu_compat.sh \
  ~/VLN/envs/isaac51/bin/python -u sim/run_p1.py \
  --name baseline --max-seconds 500 \
  >~/VLN/logs/p1-baseline.log 2>&1 </dev/null &
echo $! >~/VLN/logs/p1-run.pid
```

不要在运行期间重建相同的 USD。每次试验使用新的 `--name`，保存失败证据。`--sensor-only` 只运行传感器检查；它不代表任务成功。`--start-yaw` 改变初始朝向。独立目录中用 `--bridge` 生成 2F 连廊变体，再通过 `--scene` 指向该目录；连廊是否成功必须单独验收。

资产下载脚本应在后台运行：

已有 Isaac 环境的新安装可以后台执行 `bash scripts/install_p1_components.sh`，建立独立 USD 编写环境并安装二维码/视频依赖；它要求已有 `bwrap`，不会在缺少隔离工具时退化为不隔离的策略进程，也不会启动模拟器。

```bash
nohup python3 scripts/fetch_p1_assets.py \
  >~/VLN/logs/p1-assets.log 2>&1 </dev/null &
```

它保存下载来源、大小和 SHA-256。Office/Hospital/Simple Room 在这里仅下载 USD 入口文件以检查依赖，不是完整离线场景包。

## 几何、观测和连接

- A 楼 5 层，B 楼 3 层，层高 3 m；每层独立楼板、墙体、门洞和碰撞体。
- 电梯井位于楼板外侧，楼板不贯穿井道。轿厢由运动学刚体及组合碰撞体构成，机器人靠接触随轿厢连续升降。
- 入口有独立门体；默认开启。电梯层门默认关闭，调用技能后滑开，关门后才能运行。
- `call` 要求在相应楼层并接近门口；`select` 要求已安全进入开门轿厢且目标层可服务。返回值不提供到达楼层身份。
- 相机记录 RGB、深度、内参、安装变换及时间戳；LiDAR 使用 PhysX 碰撞射线；IMU 使用 Isaac 物理传感器。
- 看到设施、看到去向、完成穿越和验证身份是不同状态。楼层牌验证到达身份后，连接记忆才进入 verified。

路线流程为：发现并接近 EA → 呼梯 → 视觉找到轿厢 → 走入 → 请求下降 → 读取到达楼层牌 → 走出 → 利用粗室外路网和局部碰撞观测离开 A → 观察 B 楼身份 → 进入并验证楼层 → 发现 EB → 乘梯到 3F → 验证 B3F → 离开轿厢并到公共区域停止。

这条高层流程是 P1 诊断策略，尚不具备通用层级路线搜索、错误连接回溯或 VLM 语义推理。可选连廊变体用于后续检查路线改变，不能用它替代这些模块。

## 信息隔离和成功条件

`public/prior.json` 仅含合成 OSM 风格的建筑位置/轮廓、室外步道、给定起点和语言目标结构。真实 OSM/高德数据没有伪装成这个合成场地的地图。

策略通过 `bubblewrap` 在独立网络、挂载、PID 命名空间运行，仅挂载策略源码、连接记忆库和公共先验。私有场景、oracle、机器人资产和用户 home 均不挂载。JSON 接口只提供观测、里程计和技能反馈。每次运行检查绝对路径、相对路径和 `/proc` 路径读取，结果保存在 `isolation.json`。这是本地研究进程隔离，不是针对内核漏洞的安全证明。

评测器独立检查目标区域、正确高度、已离开电梯、B3F 视觉身份，以及低速停止持续时间。发出 stop、门打开或选层 API 返回成功都不是成功条件。身体倾倒和异常接触另外记录。

产物位于 `data/sim/p1/runs/<name>/`：

| 文件 | 用途 |
|---|---|
| `report.json` | 任务状态、独立成功判定、传感器与接触统计、技能事件 |
| `trajectory.jsonl` | 位姿、速度、决策、轿厢连续高度 |
| `observations.jsonl` | 策略实际收到的观测与证据散列 |
| `rgb_*.png`, `depth_*.npz` | 渲染图像、深度与相机位姿 |
| `calibration.json` | 传感器假设与标定 |
| `isolation.json`, `policy.stderr` | 隔离检查和策略诊断 |

## 已定位的工程问题

1. 5090 系统库与已加载的 NVIDIA 内核模块版本不一致。项目局部库路径还必须提供无版本名的 `libcuda.so` 和 `libnvidia-ml.so`，因为 Kit 使用这些名字加载库。未修改系统驱动。
2. `SimulationApp(open_usd=...)` 必须同时禁止默认的新建 stage，否则可能检查的是空场景。
3. 物理步进和渲染频率不同。若每个控制循环直接使用 `world.step(render=True)`，会出现多步物理期间没有更新关节控制的问题。现在每 5 ms 更新控制并步进，独立渲染。
4. 运行开始后才增加接触报告 API 会使 articulation view 失效；已改为 reset 前配置。
5. 官方 Go2 平地策略在本配置的低平移、纯转向命令下会停滞。基线采用有 LiDAR 净空检查的小弧线转向，仍需在真实控制器上重新评估狭窄空间机动。
6. 标牌背面可能被 RTX 渲染为镜像，且悬挂位置可能互相遮挡。增加不透光背板，调整楼层牌位置，并使用 ZXing 解码。以后应改用真实指示牌材质和 OCR/VLM。
7. 导入的 Go2 视觉网格存在 UV primvar 警告。不能把警告当作物理失败，也不能据此声称外观资产已经完全修复。

## 迁移到真实感场景

先保持同一策略接口和评测器，把 P1 的单层几何换成 Office 或 Hospital 的局部区域。为导入场景检查单位、轴向、collision、门净宽、可达区域和传感器视线；视觉资产本身不保证物理导航可用。然后为各层建立独立子地图，用实际井道、层门和轿厢替换简化电梯。最后加入室外道路、入口搜索和多建筑关系。

| 可复用资源 | 适合复用的部分 | 仍需自行完成 |
|---|---|---|
| Isaac Office | 办公室、走廊、家具、电梯门视觉/动画资源 | 可载客轿厢、跨层物理、楼层身份、评测协议 |
| Isaac Hospital | 长走廊、房间、门口、复杂室内布置 | 楼层连接、门交互、可达性和机器人尺寸检查 |
| Simple Room / Warehouse | 快速验证传感器、障碍和局部规划 | 真实跨楼宇跨楼层结构 |
| SceneBlox | 房间/走廊布局与材质变化 | 垂直连接、任务采样和私有评测真值 |

入口 USD 实测直接外部引用数：Hospital 216、Office 539、Simple Room 55。必须下载完整依赖后才能宣称离线可用；这些数不是场景大小或完整依赖闭包。

后续研究验收仍缺：TARE 接口适配、冻结 VLM/OCR 与 3D 连接候选、连续探索/回访/重规划、真实传感器标定、带噪 SLAM、ROS 2 时间/TF 同步、地图与场景地理配准，以及跨场景统计。香港科技大学（广州）与海港城需要合法可用的楼层/连接资料和现场采集；OSM 建筑轮廓不包含这些室内信息。高德适配器已保留无密钥错误路径，实时请求仍等待 Web 服务 Key。

## 官方资料

- [Isaac 5.1 环境资产](https://docs.isaacsim.omniverse.nvidia.com/5.1.0/assets/usd_assets_environments.html)
- [SceneBlox](https://docs.isaacsim.omniverse.nvidia.com/5.1.0/replicator_tutorials/tutorial_replicator_sceneblox.html)
- [Office/Hospital ROS 2 导航示例](https://docs.isaacsim.omniverse.nvidia.com/5.1.0/ros2_tutorials/tutorial_ros2_multi_navigation.html)
- [相机](https://docs.isaacsim.omniverse.nvidia.com/5.1.0/sensors/isaacsim_sensors_camera.html)、[ROS 2 接入](https://docs.isaacsim.omniverse.nvidia.com/5.1.0/installation/install_ros.html)
- [Go2 平地策略配置](https://github.com/isaac-sim/IsaacLab/blob/v2.3.0/source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config/go2/flat_env_cfg.py)
