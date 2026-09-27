# 估计定位接入与证据边界

`--odometry gicp` 通过独立进程接入 LiDAR 里程计。估计器仅接收扫描时间和机体坐标下的 XYZ；bubblewrap 隔离进程无法读取实验目录、场景、oracle、真值轨迹，也不能联网。导航接收估计位姿配准后的点云和相机外参。重复发布不会把旧的估计位姿重新标记为新测量。

坐标系以首帧机体为相对原点，公开初始化条件为起步时水平、名义机体高度 0.4 m。没有输入起点世界 XY 或朝向。评测器单独用首帧真值做一次坐标对齐，计算后续误差；不做整条轨迹拟合。该初始化条件仍需在实机用重力与地面估计替代。

## 安装和运行

所有依赖装到 `~/VLN/envs/odometry`，版本见 `config/odometry-lock.txt`，不修改 ROS 或 Isaac Python 环境。

```bash
cd ~/VLN/cross-building-floor-nav
nohup bash scripts/install_odometry_components.sh > ../logs/odometry-install.log 2>&1 < /dev/null &

# 安装完成且没有其他 Isaac 进程后执行。名称必须尚不存在。
OMNI_KIT_ACCEPT_EULA=YES bash scripts/with_gpu_compat.sh \
  ../envs/isaac51/bin/python sim/run_office.py \
  --name office-gicp-example --task lounge --max-seconds 120 \
  --odometry gicp --lidar-profile uniform64 \
  --checkpoint data/readiness/training/go2-control-v4-r2/model_2796.pt
```

这是开发实验命令；v4 权重尚未通过 G1，不能据此作为默认模型或实机控制器。

## 已有回放

使用 [small_gicp](https://github.com/koide3/small_gicp) 1.0.1，GICP 对最近 10 个观测关键帧配准；每帧最多 50 次迭代、0.1 m 下采样、0.5 m 对应搜索范围。无回环和 IMU 融合。对照为 [KISS-ICP](https://github.com/PRBonn/kiss-icp) 1.3.0。失败记录全部保留。

| 回放 | 算法 | 位置 RMSE / 最大 / 末端，m | 最大角度误差 | 处理 p95 | 筛查 |
|---|---|---|---|---|---|
| lounge-006，开发 | KISS | 2.381 / 3.551 / 3.292 | 33.70° | 140.7 ms | 失败 |
| lounge-006，开发 | GICP | 0.0585 / 0.0790 / 0.0789 | 0.729° | 19.5 ms | 数值满足 |
| lounge-005，验证 | GICP | 0.0773 / 0.0917 / 0.0824 | 0.830° | 17.3 ms | 通过 |
| door-007，验证 | GICP | 0.0563 / 0.0867 / 0.0652 | 0.585° | 20.5 ms | 通过 |

验证前写入 `config/readiness/odometry-replay-gates.json`：最大位置误差 0.3 m、末端 0.2 m、最大角度 5°、处理 p95 200 ms、全部帧有效。这些记录来自同一 Office，不是未知场地验证。旧文件保存的是真值配准点云，因此导出器先撤销其配准；估计器只得到恢复的机体点云。这仍是理想虚拟传感器回放。

## 实时接入后的新失败

`office-lounge-008-gicp-v4` 改为传感器随机身转动的刚性安装模型，5 Hz、360×11 条瞬时射线。导航读取在线 GICP。运行约 11.4 s 后配准不收敛并停止；此前最大位置误差 0.703 m、最大角度误差 12.68°。这是失败的闭环开发实验，不能被旧回放的成功掩盖。

同段原始数据的静态运动先验和 0.05 m 下采样对照也失败。后续 `uniform64` 模式只增加竖直采样到 64 线，用来检查稀疏扫描对表面估计的影响。该射线模型不代表已知实机雷达，没有噪声、扫描畸变或真实丢包。所有新运行保存原始机体点云，并保留导致失效的帧。

`office-lounge-009-gicp-dense` 完成 119.915 s、30.936 m 的在线导航，600 帧配准全部有效。评测器首帧对齐后，位置 RMSE 0.444 mm、最大 1.013 mm、末端 0.759 mm，最大角度误差 0.0113°；估计器处理 p95 72.2 ms。这是精确静态几何、无噪声的虚拟扫描结果，不能外推真实传感器精度。完整仿真耗时 608.1 s，估计器处理时间不等于系统实时性能。

这一运行使用了 TARE 路径和真实 VLM API（29 次调用），没有非脚部物理接触；但两视角语义确认没有完成，任务以超时失败结束。它证明了估计定位能接入该开发闭环，未通过任务验收或硬件定位门槛。

## 尚不能用于实机的原因

注册收敛、Hessian 条件数和数值有限均不等于定位可靠；最初 KISS 回放在持续输出有效矩阵的同时漂移超过 3 m。当前所有输出明确为 `control_ready: false`。还需要匹配实际传感器、标定和同步，加入噪声与时间畸变测试，验证长程漂移/退化、IMU 融合或适配已有定位，并建立可验证的健康状态后才能接入硬件运动门控。
