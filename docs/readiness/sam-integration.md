# SAM2 图像分割接入

参考 SysNav 固定提交 `478727e5a559be22822d5d093b57431449336287` 的 `semantic_mapping_node.py:419–435`：先对整张图像计算 embedding，再批量传入检测框，用 `multimask_output=False` 得到掩膜，随后关联观测点云。完整参考源码在本地 `TARE_WheelTECH/vendor/sysnav`，只读使用。

本项目采用同一 SAM2 预测接口，但输入框来自在线 RGB-D 水平表面连通分量，并未接入 SysNav 的 YOLO/YOLOE TensorRT 检测器。数据盘已有 SAM2.1 large 权重，因此使用 large 配置；不能称为原版 SysNav base_plus 配置的复现。SAM3 仅发现文件，未使用。

## 数据流与边界

1. 当前深度、相机标定和估计位姿产生候选框；不存在真实深度的像素不能产生三维候选。
2. 独立 SAM2 服务只接收原始 RGB JPEG 和像素框，没有位姿、标签、USD 或 oracle。模型及权重只读，网络隔离；Unix socket 仅挂入策略进程。
3. SAM 掩膜与有效水平深度、候选高度切片相交，保留与原候选重叠的连通分量。从其内部选取像素，经原有多尺度平面检验后获得三维锚点。
4. VLM 从 S 编号中选择属于目标家具组的桌面。编号不带物体类别，模型生成的像素坐标不用于定位。
5. 独立 VLM 再检查锚点的物体身份；桌面关联的三维体素也限制在掩膜支持区域内。仍须两处观测确认、处于任务区域并停车，由独立评测器判定最终成功。

SAM 的预测 IoU 仅是模型自身的掩膜质量分数，不是物体身份概率。SAM 可能分出整把椅子、一个部件或错误区域，因此不能替代深度、物体归属和任务验证。SAM 开启时，服务失败或没有合格掩膜均拒绝到达确认，不能悄悄回退到无分割的成功判定。

## 5090 环境

所有新增文件位于 `~/VLN`。`assets/sam/sam2.1_hiera_large.pt` 是到数据盘现有文件的只读使用链接；固定 SHA 和版本见 `config/sam2-provenance.json`。`deps/sam2` 是既有干净源码的独立 clone，未修改原研究项目。原始权重不写入 Git。

模型使用 `envs/isaac51` 的 Python 3.11、`envs/torch-cu128` 的 CUDA 12.8 Torch，以及 `envs/sam2-deps` 中的 iopath、portalocker、tqdm。与 ROS Python 3.12 分进程运行。关闭可选 CUDA 掩膜后处理扩展，主编码器和解码器不变。GPU 2 用于分割，GPU 0 用于 Isaac；运行前检查占用，不能接管其他任务的 GPU。

`bash scripts/install_sam2_components.sh /path/to/sam2.1_hiera_large.pt` 可准备缺失源码和轻量依赖，并核查固定版本及权重 SHA。已在当前环境验证复用路径；缺失依赖时应将该安装命令放到后台，网络下载分支未在本轮重新执行。前置条件仍为项目已有 Isaac Python 和 Torch CUDA 12.8 环境。

```bash
cd ~/VLN/cross-building-floor-nav
OMNI_KIT_ACCEPT_EULA=YES bash scripts/with_gpu_compat.sh ../envs/isaac51/bin/python sim/run_office.py \
  --name UNIQUE_RUN_NAME --task lounge --max-seconds 120 \
  --odometry gicp --lidar-profile uniform64 --sam --sam-gpu 2 \
  --checkpoint data/readiness/training/go2-control-v4-r2/model_2796.pt
```

若现场使用了本文下述 API 中继，额外传 `--api-proxy http://127.0.0.1:18119`。不指定 `--sam` 可复现几何候选对照。每次运行保存 SAM 隔离结果、权重 SHA、图像/框请求、压缩掩膜和处理耗时；这些记录不包含 API 密钥。

## 当前 API 网络中继

5090 的 DeepSeek 直连曾因现有 VPN 假地址路由出现 TLS EOF。临时方案在 Mac 启动 `scripts/api_connect_relay.py`，通过 SSH 反向转发 `127.0.0.1:18119`。该中继只允许 `CONNECT api.deepseek.com:443`，不终止 TLS、不记录请求内容和密钥；其他目的地返回 403。没有修改系统 VPN、DNS 或路由。

中继当前依赖 Mac 与 SSH 在线，不能作为独立上机部署完成的证据。正式部署需要确认机器狗计算平台能直接或通过获准的网络出口访问模型服务。API 密钥保留在远端 `~/VLN/secrets/deepseek-api-key`，权限 0600。关闭临时中继时仅终止记录的本项目进程和 SSH 转发，不使用宽泛的进程清理命令。

## 证据范围

离线开发帧的掩膜已检查，隔离进程推理已通过。它们仅证明模型和接口可运行；不能据此声称分割精度提升、跨场景泛化或实机闭环通过。在线任务与对照结果在 `iteration-04.md` 记录。正式评测仍需独立人工标注的桌面/椅面掩膜、三维锚点和相机标定，报告掩膜 IoU、锚点误差、错误接受率和端到端任务成功率。
