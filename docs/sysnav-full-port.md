# SysNav 原生仿真迁移

> 2026-09-28 系统复核：下文记录早期迁移历史。其“完整”指核心模块接入，
> 不是启动图/参数/传感器契约已等价。最新修复与限制见 [系统对齐审查](sysnav-system-parity.md)。

分支：`codex/sysnav-full-port`。官方源码固定为
`zwandering/SysNav@478727e5a559be22822d5d093b57431449336287`，独立依赖目录为
`~/VLN/deps/sysnav-full`，旧 `deps/sysnav` 与旧任务适配器保留作比较。

## 范围与模块

本次迁移以官方完整**仿真导航链**为范围。官方仿真本身跳过 ARISE/Livox
硬件包；此处用 Isaac 观测、GICP 里程计和 Go2 关节控制策略替换 Unity/实体接口。
这不构成实机验证，也不能称为 ARISE SLAM 验证。

| 模块 | 执行来源 | 证据 |
|---|---|---|
| 房间分割与拓扑 | 官方 `room_segmentation` | `/room_nodes_list` |
| 开放词汇检测、BoT-SORT | 官方 `detection_node`，YOLOE-26x | `/detection_result` |
| SAM2 与三维物体关联 | 官方 `semantic_mapping_node` | `/object_nodes_list` |
| 指令分解、房间分类与选择 | 官方 `vlm_reasoning_node`，DeepSeek 传输适配 | 原生查询/回答、`api.jsonl` |
| 物体类别、属性/空间条件确认 | 同一原生 VLM 节点 | target/object/anchor 查询与回答 |
| 探索、接近物体与停止条件 | 官方 `tare_planner_node` | 原生 `/way_point`、只读 `/sysnav/status` |
| 局部避障与路径跟随 | 官方 terrain/localPlanner/pathFollower | `/path`、`/cmd_vel` |
| 机器狗动力学 | Isaac PhysX、现有 Go2 v5 策略 | 关节、轨迹、接触记录与第三人称图像 |

桥接器不生成导航 waypoint 或速度决策，仅转发传感器、发布语言指令、监督进程，
并在命令失效/原生停止条件满足时输出零速度。原生 TARE 的 room/object 算法未被
旧 `/mission/subgoal` 自定义策略替代。原生可选 captioner 缺失时使用其既有
fallback；无用的 PyTorch3D/spaCy 导入移除，实际几何 IoU 算法仍为 Shapely。

## 平台适配

- ROS Humble/Python 3.12 兼容头文件；独立 `envs/sysnav-full` Python overlay。
- 官方 YOLOE-26x `.pt` 与原始词表替代机器专属 TensorRT `.engine`。
  使用官方 SAM2.1 Hiera base-plus 权重；运行前固定文字嵌入，禁止沙箱运行时下载。
- DeepSeek `deepseek-flash`，JSON 模式与本地 Pydantic 校验；保留原生八类查询流程。
  私钥只读挂载，日志不写私钥。界面提供观测、选择和简短决策依据。
- 六个同步 90° 虚拟相机转为 1920×640、360°×120° 全景；光心为 base+[0,0,0.45]。
  C++ 房间投影、Python 点云融合、VLM 包围盒投影共享这组外参。
- 继承已有缩短队列越界修复、Eigen ABI 编译设置；早期初始 waypoint 曾改为静止；系统对齐分支已恢复上游 12 m 前探，仅保留时钟就绪守卫。
- 修复固定源码版本的 Python/C++ 物体消息契约不一致：Python 发布批量
  `ObjectNodeList`，新增/更新为 `true`、删除为 `false`，匹配现有 C++ 消费者。
- 只读状态话题暴露原来的目标发现/接近条件，不修改原生判定阈值。
- ROS domain 94，localhost；与旧 domain 93 仿真实验隔离。
- GPU 通过项目局部动态库启动，无重启、驱动安装或全局配置修改。
- 仿真 Go2 到达后用一秒平滑插值切换至额定关节站姿，采用限力矩 PD 控制。
  该适配由原生 STOP 触发；不重置位姿/速度、不固定刚体、不关闭碰撞。
  修复原有步态策略零速度时仍踏步的问题，保持原来的三维速度停稳验收阈值。

## 单场景测试

任务：寻找接待区域中黑色、有软垫座面/靠背和扶手的椅子，靠近后停止。
调试任务曾误写成 student lounge；实际场景是接待等候区，原任务被模型拒绝，
保留该失败记录，修正场景描述后单独重跑，不将两种指令视为同一评测。
原生类别为 `chair`，属性与房间条件由 DeepSeek 从语言指令分解。
验收版本 `sysnav-office-object-v1` 在最终重跑前固定。评测器使用场景资产中
两个接待扶手椅的精确世界包围盒，要求：机器人离同一目标水平边界 ≤1 m、
预测三维物体位置距该包围盒 ≤0.4 m、相机到目标有首个碰撞命中为目标的视线、
机器人处于目标楼面高度，并以 <0.06 m/s 停稳 ≥1 s。接触审计另报，
`acceptance_passed` 同时要求任务成功和接触合格。标注与可见性射线均只在
评测器中运行，策略沙箱不可读。旧区域判定保留为诊断字段，不用于本次物体任务验收。
这是一项自定义单场景接近物体验收，不是标准 ObjectNav 或 VLN 的 SR/SPL。

`ui-20260928-014610-2e945` 仍保留旧区域判据下的失败状态，不用新规则回填。
会议室屏幕任务只提供入口，尚未验收，仍使用旧区域检查，界面明确标注。

```bash
cd ~/VLN/cross-building-floor-nav
bash scripts/with_gpu_compat.sh ~/VLN/envs/isaac51/bin/python -u sim/run_sysnav_full.py \
  --name sysnav-full-lounge --task lounge --max-seconds 120 \
  --checkpoint data/readiness/training/go2-control-v5-rails/model_3595.pt \
  --odometry gicp --lidar-profile uniform64 --api-proxy http://127.0.0.1:18119
```

运行记录在 `data/sim/office/runs/<name>`，各节点独立日志在 `agent/`，
`events.jsonl` 记录真实 ROS 消息，`api.jsonl` 记录 VLM 调用类型、结果、耗时和 token。
`isolation.json` 记录策略进程不可读 oracle/scene 的检查。全景图、机器狗图像、
LiDAR、关节与接触数据保留于评测侧。实际结果见运行报告，不以编译成功代替验收。
可用 `envs/sysnav-full/bin/python scripts/summarize_sysnav.py <run目录>` 重建
`review/summary.json` 和带物体真值、实际轨迹、三维速度与停止时刻的评审图。

## 在现有 5090 基础环境复现

现有前提：`envs/ros`、`envs/isaac51`、项目局部 GPU 动态库、Office 完整资产、
Go2 v5 checkpoint、`deps/sam2` 及私钥文件已经就绪。以下脚本不安装全局驱动。
本次环境完整版本记录在 `~/VLN/logs/sysnav-full-python.lock`，每次运行复制到
`source/python.lock`；模型哈希来自 `~/VLN/assets/sysnav/provenance.json`。

```bash
# 新机器或全新目录才执行；不要覆盖已有修改。
git clone https://github.com/zwandering/SysNav.git ~/VLN/deps/sysnav-full
git -C ~/VLN/deps/sysnav-full checkout -b codex/isaac-deepseek 478727e5a559be22822d5d093b57431449336287
git -C ~/VLN/deps/sysnav-full am ~/VLN/cross-building-floor-nav/patches/sysnav-full-isaac.patch
cd ~/VLN/cross-building-floor-nav
bash scripts/install_sysnav_full.sh
bash scripts/build_sysnav_full.sh
bash scripts/prepare_sysnav_models.sh
~/VLN/envs/sysnav-full/bin/python -m pip freeze > ~/VLN/logs/sysnav-full-python.lock
```

前端 `http://127.0.0.1:8878/` 选择「SysNav · 寻找接待区扶手椅」，启动后可查看
机器狗、全景 RGB、轨迹、原生房间/三维物体数量，以及 API 的结构化结果与简短依据。
只显示显式返回的决策摘要，不展示模型内部推理。

## 仍需扩展验证

- 此场景仅覆盖指令解析、房间分类/选择、物体类别/属性验证、导航接近和停止。
  带参照物与空间关系的指令分支虽已接入，不能以本次运行证明有效。
- 原生物体关联会反复触发类别/目标查询；每次运行有 150 次 API 预算。
  长距离探索需做查询去重、缓存和消融，不能直接外推当前运行性能。
- 六相机拼接、理想瞬时 LiDAR、无噪声的 GICP 测试不代表真实相机/LiDAR/IMU。
  静态地面、低速 Go2 控制策略也不代表真实楼梯或机器狗部署。
- 未验证实体机器人、ARISE SLAM、跨楼层、电梯、MP3D 标准任务或统计成功率。

参考：[SysNav 源码](https://github.com/zwandering/SysNav)、
[YOLOE 官方模型与预计算提示嵌入](https://docs.ultralytics.com/models/yoloe/)、
[SAM2 官方权重](https://github.com/facebookresearch/sam2)。
