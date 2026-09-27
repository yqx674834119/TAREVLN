# VLN 调研与第一轮 MP3D 评测

日期：2026-09-28。分支：`codex/vln-benchmark-audit`。

本轮选择 R2R、VLN-CE、HAMT、DUET、ETPNav 五篇作为研究与复现入口：它们分别覆盖任务奠基、连续环境、长历史建模、全局拓扑规划和连续空间拓扑导航。选择依据是方法影响、后续工作的采用以及官方实现可用性；没有声称它们是实时引用量排名前五。

**已完成的是一个真实 MP3D 场景的 Habitat 评测闭环，而不是五篇论文的训练模型复现，也不是 Isaac/Go2/TARE 的物理闭环验收。** 本轮保留既有机器狗分支，另开评测分支；没有重启服务器或改全局 GPU 驱动。

## 1. 五篇工作：任务、测评与对本项目的意义

| 工作与原文 | 核心设计 | 论文使用的主要测评 | 需要的输入和数据 | 本项目应如何使用 |
|---|---|---|---|---|
| [R2R，CVPR 2018](https://openaccess.thecvf.com/content_cvpr_2018/papers/Anderson_Vision-and-Language_Navigation_Interpreting_CVPR_2018_paper.pdf) | 将自然语言路线和真实室内全景观察结合，建立 VLN 基准 | R2R；NE、SR、Oracle SR、路径长度 | MP3D 全景图、离散连通图、路线及英文指令；Matterport3DSimulator | 统一指令、终点和未知建筑泛化概念；图节点跳转不能直接代表机器人行走 |
| [VLN-CE，ECCV 2020](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123730103.pdf) | 将路线移植到三维重建场景，执行小步前进/转向并面对碰撞 | R2R-CE；NE、SR、OSR、SPL、nDTW、路径长度 | MP3D GLB、navmesh、连续坐标任务、RGB-D；Habitat | 最合适的第一条标准测评链；需要 STOP、碰撞、步数上限和密集参考轨迹 |
| [HAMT，NeurIPS 2021](https://proceedings.neurips.cc/paper_files/paper/2021/file/2e5c2cb8d13e8fba78d95211440ba326-Paper.pdf) | 层级编码历史全景与动作，结合指令选择下一步 | R2R、RxR、R4R、R2R-Back、R2R-Last、REVERIE、CVDN；不同任务用 SR/SPL、nDTW/SDTW/CLS、目标定位或 Goal Progress | 对应注释、图连通关系、视觉特征和模型权重 | 参考长程记忆与回访；不能把 CVDN 的 GP 或 REVERIE 目标定位成功率混进 R2R 成功率 |
| [DUET，CVPR 2022 Oral](https://openaccess.thecvf.com/content/CVPR2022/papers/Chen_Think_Global_Act_Local_Dual-Scale_Graph_Transformer_for_Vision-and-Language_Navigation_CVPR_2022_paper.pdf) | 全局拓扑图与局部细粒度观察联合决策 | R2R、REVERIE、SOON；导航指标及各自目标 grounding 指标 | 全景特征、连通图；物体定位任务另需目标标注 | 是连接记忆与全局路线选择的重要方法参考；其图上动作权限需要单独标注 |
| [ETPNav，TPAMI，2024 在线发表／2025 卷期](https://arxiv.org/abs/2304.03047) | 视觉 waypoint 预测、在线拓扑图、指令规划与障碍恢复 | R2R-CE、RxR-CE；NE、SR、OSR、SPL、nDTW、SDTW | RGB-D 全景、waypoint 预测器、指令、预训练和任务权重 | 与 VLM＋TARE 最接近的连续导航对照；须统一相机覆盖、动作、碰撞和训练数据预算 |

官方代码已固定当前下载提交，见 `assets/benchmarks/public-acquisition.json`：[R2R](https://github.com/peteanderson80/Matterport3DSimulator)、[VLN-CE](https://github.com/jacobkrantz/VLN-CE)、[HAMT](https://github.com/cshizhe/VLN-HAMT)、[DUET](https://github.com/cshizhe/VLN-DUET)、[ETPNav](https://github.com/MarSaKi/ETPNav)。代码存放在 `~/VLN/deps/benchmarks/`，原始仓库未修改。

### 原论文数值，只作量级参考

| 论文表格／方法 | 协议与 split | SR (%) | SPL (%) |
|---|---|---:|---:|
| R2R Table 1，Student-forcing | 离散 R2R val_unseen | 21.8 | 原表未报 |
| VLN-CE Table 3 row 10，CMA 完整组合 | 连续 VLN-CE val_unseen，论文版本 | 32 | 30 |
| HAMT Table 5，HAMT | 离散 R2R val_unseen | 66 | 61 |
| DUET Table 6，DUET | 离散 R2R val_unseen | 72 | 60 |
| ETPNav Table 2，ETPNav | 连续 R2R-CE val_unseen | 57 | 49 |

这些数值来自上列原文表格，**不能合并排名，也不能与本轮单场景结果作直接胜负比较**。原始图导航、连续导航、版本、全景权限、训练/预训练和数据增强均不相同。最终应在相同 episode 集合、仿真版本和具身配置下重跑可比方法；尚未运行的论文模型必须标为 “未复现”。

## 2. 数据是什么，已经下载了什么

HM3D 与 MP3D 是不同场景集，不能用 HM3D 文件替换既有 R2R/RxR 的场景 ID。用户已提供 MP3D 获取来源，本轮使用 [Mrakas/mp3d 镜像](https://www.modelscope.cn/datasets/Mrakas/mp3d)，并保留 [Matterport3D 原始来源](https://niessner.github.io/Matterport/) 与使用条款归属。镜像 SHA256 校验说明下载与镜像一致，不独立证明镜像与原作者发布文件逐字节相同。

| 数据 | 路径（相对 `~/VLN/`） | 状态与用途 |
|---|---|---|
| 5 篇论文 PDF／官方 Git 仓库 | `assets/benchmarks/papers/`、`deps/benchmarks/` | 已下载；论文与代码来源分开记录 |
| 原始 R2R train/val_seen/val_unseen/test | `assets/benchmarks/tasks/R2R_*.json` | 已下载并检查 JSON；test 不拿来调参 |
| R2R-CE v1-3 原始任务 | `assets/benchmarks/tasks/R2R_VLNCE_v1-3/` | 已下载，ZIP CRC/SHA 记录 |
| R2R-CE v1-3 预处理及密集参考轨迹 | `assets/benchmarks/tasks/R2R_VLNCE_v1-3_preprocessed/` | 已下载；GT 只用于评测 nDTW |
| MP3D 90 个场景 | `assets/benchmarks/scene_datasets/mp3d/` | 首场景已完成，其余后台下载；看状态 JSON |
| ETPNav 作者托管的 R2R-CE v1-2、RxR-CE、预测器及权重 | `assets/benchmarks/etpnav/` | 后台下载；仅保存数据，不执行下载的脚本或加载 pickle/模型 |

ETPNav 资源来自其 [官方 README 指向的 ModelScope 仓库](https://modelscope.cn/models/admagic/ETPNav/files)。v1-2 与本轮 v1-3 分目录保存，不静默混用。未下载 REVERIE/SOON 等全部旁支任务或 HAMT/DUET 的全部预提取特征；第一轮聚焦五篇可关联的 R2R 系列任务。

本轮 R2R-CE v1-3 文件实际统计：train 10,819 条／61 场景；val_seen 778 条／53 场景；val_unseen 1,839 条／11 场景；test 3,408 条／18 场景。这是 instruction episode 数，不是独立路线数。

首场景 `x8F5xyUWy9e` 是 val_unseen 中 GLB 最小的场景，按下载体积选定，未依据模型结果挑选。它有 **102 条指令、34 条独立路线**。第一条为 episode 7 / trajectory 42：经过泳池，从吧台与椅子间走过，停在吧台拐角。VLM pilot 预定按 episode ID 取前 10 条：7、8、9、31、32、33、100、101、102、187，共 4 条路线；不能把这 10 条当成 10 个独立场景或独立路线。

## 3. 已搭建的评测链与公平性边界

`官方任务 → Habitat RGB-D/动作 → 独立策略 worker → 合法离散动作 → evaluator 计分`。

| 设置 | 本轮值 |
|---|---|
| 仿真 | Habitat-Sim 0.3.3，独立 `~/VLN/envs/habitat-bench`；复用已有基础运行时的只读包 |
| 行动 | 前进 0.25m，左右转 15°，STOP；上限 500 个原子动作；允许 sliding |
| 具身与相机 | radius 0.1m、height 1.5m、sensor height 1.25m；RGB 224×224，depth 256×256，HFOV 90° |
| 成功 | 策略发出 STOP，且最终 geodesic distance **严格小于 3m** |
| 路程 | 实际位置的连续欧氏位移累加，转向不增加路程 |
| 指标 | NE、SR、OSR、SPL、路径长度、步数、碰撞；官方密集 GT 上的 fastdtw nDTW/SDTW，并另存 exact DTW 交叉检查 |
| 固定随机性 | seed 20260928；随机基线为每 episode 派生种子，与论文原进程级随机流不保证逐条一致 |
| VLM | 已有 `deepseek-flash` 图像 API；单前视图、三个局部深度统计、指令和自身动作历史；最多 130 次请求；每次输出 1–4 个同类原子动作 |

设置来自 [VLN-CE task config](https://github.com/jacobkrantz/VLN-CE/blob/master/habitat_extensions/config/vlnce_task.yaml) 和 [Habitat-Lab 0.1.7 默认配置](https://github.com/facebookresearch/habitat-lab/blob/v0.1.7/habitat/config/default.py)。成功与 SPL 公式核对了 [原版实现](https://github.com/facebookresearch/habitat-lab/blob/v0.1.7/habitat/tasks/nav/nav.py)。原版 nDTW 默认使用 fastdtw，重复静止位置去重；没有用稀疏 `reference_path` 冒充密集 GT。

策略请求白名单只有 instruction、RGB、局部 depth、last_action、collision、step；不传目标坐标、目标距离、reference path、scene ID、GT 语义或导航网格。worker 与 evaluator 分进程，但当前同用户文件系统权限仍共享，因此是**接口隔离，不是对任意恶意策略的安全沙箱**。正式投稿评测应改为容器/只读挂载和评测服务，以隔离 oracle 文件。

这套环境比作者的 Habitat 0.1.7 更新，属于 **0.3.3 移植评测器**。现阶段只移植了 Random、Handcrafted 的逻辑，另设 STOP、特权 oracle 检查和我们自己的 VLM adapter。ETPNav/HAMT/DUET 的学习模型未运行；VLM adapter 也没有接入 TARE。四足动力学、足接触、SLAM、RGB-LiDAR 同步、速度控制属于 Isaac/Go2 后续验收，不能由此结果推断。

## 4. 如何与别的方法对比

第一组是严格共享协议的工程基线：STOP、Random、Handcrafted、VLM-only、几何探索＋VLM、完整 VLM＋TARE／连接记忆。在同一机器人抽象、相机权限、动作时间预算和数据 split 下运行。oracle 只用于检查 navmesh 和任务有效性，单独显示，禁止列为自主方法成绩。

第二组是连续空间学习方法：先复现 VLN-CE CMA 和 ETPNav。固定作者 checkpoint、配置、数据版本、Habitat 版本，先对齐小样本轨迹，再跑完整 val_unseen。ETPNav 常用全景 waypoint 预测；若我们的 VLM 只有单前视图，应分别报告原生配置，以及匹配视觉覆盖与动作权限的对照。

第三组是离散图方法：HAMT／DUET 保留其原始 R2R 设置作文献结果；移植到连续空间后必须明确叫 HAMT/DUET-based planner 加某种 controller，且报告碰撞、路径执行失败及候选权限。不可把模型输出图节点瞬移过去并称为机器人闭环。

正式实验以完整 val_unseen 为主，至少 3 个随机种子；置信区间按场景或路线聚类，避免三条同路线指令被当成完全独立样本。除成功率外记录 nDTW、SPL、碰撞、路径失败、VLM 次数/时延、端到端实时速度和停止错误。调参使用训练/开发划分；本次查看过的 val_unseen 场景只能继续当工程开发样本，不用于独立泛化结论。

优先消融：无图像／打乱指令（检验语义依赖）；无历史记忆；无拓扑回访；VLM-only 对比 TARE 提供可达候选；单次 STOP 对比独立视觉确认；前视与全景；无恢复策略；相同总请求和行动预算下各模块贡献。跨建筑 OSM 和跨楼层能力不属于本轮 R2R 验证，另建受控 P1 实验。

## 5. 复现与后台管理

所有路径在 5090 的 `~/VLN`。新环境不覆盖旧机器狗环境；GPU 使用现有 `scripts/with_gpu_compat.sh` 动态库包裹，不重启机器。

```bash
cd ~/VLN/cross-building-floor-nav
../envs/habitat-bench/bin/python -m unittest discover -s tests -p test_benchmark_metrics.py
HABITAT_SIM_LOG=quiet MAGNUM_LOG=quiet bash scripts/with_gpu_compat.sh \
  ../envs/habitat-bench/bin/python scripts/benchmarks/run_habitat.py \
  --limit 0 --output ../outputs/benchmarks/new-scene-run
```

输出目录必须是新目录，避免覆盖原始结果。VLM 单条执行用 `--policies vlm --limit 1`。需要原有 API 中继时，仅给该命令设置 `HTTPS_PROXY=http://127.0.0.1:18119`；中继必须仍在线。Token 均从 `~/VLN/secrets/` 的受限文件读取。

后台下载入口：首次用 `scripts/benchmarks/prepare_mp3d.py` 认证并生成清单，再运行 `fetch_mp3d.py`；附加资源用 `fetch_supplement.py`。进程独立于 SSH/聊天，最终状态分别为 `assets/benchmarks/mp3d-download-complete.json` 与 `etpnav-download-complete.json`。中间状态在对应 `*-download-status.json`，凡 `status` 不是 `verified` 的条目都要检查，不能只看计数就宣称完成。原始下载日志在 `~/VLN/logs/benchmark-*-download*.log`。

镜像额外夹带约 12GB 的代码和 conda 环境归档，已从任务范围排除；实际只下载 90 个场景的 360 个 GLB/navmesh/house/semantic 文件。调整后沿用已下载缓存并重新校验，因此进度计数曾重新累计，不表示数据丢失。排除清单在 `excluded-non-scene-files.json`。ETPNav 权重记录本地 SHA256 和服务给出的大小；未提供作者 SHA 的条目不能宣称经过作者独立哈希认证。

本次 pilot 保留了完整输入/动作/评分及协议 manifest，但未在运行开始时归档全部 runner 源码；报告中的源码哈希明确标为报告生成时版本。此后新增的前端运行会在仿真前归档 runner、metric、worker 与 API 源码。界面使用与验收见 [dashboard.md](dashboard.md)。

## 6. 当前发现与下一阶段

实际运行数据、可视化和逐条结果见同目录 `first-scene-results.md`。第一条 VLM 运行出现提前 STOP；同一环境的 oracle 能到达，说明主要问题在路线语义、局部恢复与停止验证，不能归咎于数据缺失。参考轨迹、碰撞和动作日志已保存，后续修复需保留本轮基线，不改成功阈值。

当前应先完成连续空间的可达候选和停止验证，再把同一个策略接口接回 TARE。Isaac 迁移时，MP3D 是扫描网格，必须处理尺度/坐标系、碰撞网格、材质、地面孔洞、动态传感器与 Go2 具身差异；Humanoid/圆柱体可走的 navmesh 不能直接证明 Go2 可达。先固定一条路线，在 Isaac 独立报告机体碰撞、足接触、定位漂移与控制时延，再扩场景。真正上机器狗前仍需完成既有安全链路和现场授权，当前没有实体测试结论。
