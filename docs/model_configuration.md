# ReWAM 独立版模型配置

本页用于 release 整理期间的开发沟通，不作为最终发布文档。描述当前配置。完整可运行配置见 [rewam.yaml](../configs/model/rewam.yaml)。

## 网络组件

视觉分支明确区分编码器、特征校准和时序瓶颈，另有可选的重建诊断：

```yaml
vision_encoder:
  model_name: dinov3_vitl16
  output_dim: 1024

feature_calibration:
  layers: [11, 13, 15, 17, 19, 21, 23]

temporal_bottleneck:
  output_dim: 128
  qk_dim: 512
  gradient_mode: action_only
  causal_normalization:
    eps: 1.0e-5
    momentum: 0.1

reconstruction_probe:
  loss_weight: 1.0
  decoder_path: ${oc.env:RAEv2_DECODER_PATH,null}
```

上面仅展示关键字段，其余注意力参数、DINO 层选择等见完整配置。
`vision_encoder` 只负责冻结 DINO 的加载、输入归一化与中间层特征提取，返回未经校准的特征网格。
`feature_calibration.layers` 决定提取哪些层；校准模块执行无仿射 LayerNorm、跨层平均，
再加上最后一个选定层的空间均值。固定算法不提供额外开关。
DINO 调用使用 `norm=False`，不再替换其原始输出归一化层。

`vision_encoder.output_dim` 用于在延迟加载 DINO 之前构建 TRB 和重建探针；加载后会核对 DINO 的实际通道数。
patch size 从 DINO 的模型元信息取得，不再重复配置。
旧 `backend`、未使用的 Hugging Face `model_id`、`resize_input`、`norm_affine` 和相机数量字段已删除。
相机拆分移到输入处理边界，模型的 `camera_layout` 引用 `data.train.concat_multi_camera`，
仍保持已有三相机 Robotwin 拼接与拆分，不再把相机布局放入 DINO 配置。

`temporal_bottleneck` 是 ReWAM 必需组件，没有 `codec` 抽象、类型选择或 `enabled` 开关。
TRB 后的因果运行归一化也是架构组成部分，仅配置数值稳定项和统计更新率。
TRB 不再单独配置 `seed`。V 使用全局随机状态生成单位范数高斯初值，注意力输出投影仍初始化为零，
因此初始输出仍等于 V 投影。独立版在模型构建前应用训练全局 seed，训练阶段继续按 rank 设置随机状态。

`reconstruction_probe` 是梯度分离的可选诊断模块；设为 `null` 可省略，
探针重建的是教师表征，输入停止梯度，存在时正常训练其投影，不再使用 `enabled`、`trainable` 两个开关。
探针统一管理可训练的 3D deconv 和外部冻结的 RAEv2 RGB 解码器。
`reconstruction_probe.decoder_path` 指定 RAEv2 权重，默认读取 `RAEv2_DECODER_PATH`，未设置为 `null`。
`probe(states)` 返回重建的 DINO 特征，`probe.decode_rgb(features)` 按需加载并复用 RGB 解码器。
RAEv2 始终保持 `requires_grad=False` 和 eval，RGB 解码使用 no_grad，不进入优化器或 ReWAM state-dict；
调用模型的 `train()`/`requires_grad_()` 或迁移 dtype/device 都不会解冻它。`probe.release_decoder()` 仅释放外部解码器。
训练 loss 与普通动作推理不经过 RGB 解码器。路径为空时继续训练特征探针、计算验证 loss，
训练器仅跳过 RGB 验证；显式请求未来 RGB 可视化时仍须配置路径。

## 文本编码资源

`text_encoder` 直接描述实际使用的 UMT5 权重与 tokenizer：

```yaml
text_encoder:
  weights:
    model_id: DiffSynth-Studio/Wan-Series-Converted-Safetensors
    origin_file_pattern: models_t5_umt5-xxl-enc-bf16.safetensors
  tokenizer:
    model_id: Wan-AI/Wan2.1-T2V-1.3B
    origin_file_pattern: google/umt5-xxl/
  max_length: 128
```

这里两个仓库仅提供指定的文本资源，不加载其中的视频 DiT 或 VAE 权重。
也可以将 `weights` 或 `tokenizer` 整个替换为 `path: ./本地路径`，分别指向权重文件与 tokenizer 目录。
不再设置顶层 `model_id`、`tokenizer_model_id` 或 `redirect_common_files`，资源来源不会被隐式覆盖。

数据的 `context_len` 引用 `model.text_encoder.max_length`。
默认训练设置 `load_text_encoder: false`，直接使用 `text_embed_cache_dir` 下的文本缓存；
在线编码需设置 `load_text_encoder: true`。评测入口会加载文本编码器。
预计算脚本始终加载配置中的文本资源，与在线编码共用加载入口。

缓存文件继续使用已有的 `{prompt_sha256}.t5_len{长度}.wan22ti2v5b.pt` 格式。
其中 `wan22ti2v5b` 是保留的缓存格式标识，不再根据模型仓库名称生成，已有缓存可以继续读取。
本次仅调整资源配置及加载入口，不改变编码器、tokenizer、网络权重键或张量形状。

## 时序与状态语义

TRB 使用 `temporal_bottleneck.patch_size: [2,2,2]`，顺序严格为 **T/H/W**。
该参数同时决定非重叠分块大小与步幅，重建 probe 直接使用同一组三维值。
世界 DiT 的 `world_dit_config.patch_size` 是独立参数，仍保持 [1,1,1]。

数据采样只保留三个含义明确的入口：

```yaml
action_horizon: 64
observation_stride: 8
num_context_frames: ${model.temporal_bottleneck.patch_size.0}
```

action_horizon 是动作数量，单位为原始控制步；observation_stride 是图像采样间隔。
num_context_frames 包含所有历史帧和当前帧，历史数量自动为该值减一：

- patch_size=[2,2,2]：图像偏移 [-8,0,8,...,64]，10 帧、5 个状态。
- patch_size=[4,2,2]：图像偏移 [-24,-16,-8,0,8,...,64]，12 帧、3 个状态。

动作偏移始终为 [0,...,63]，共 64 个；底层 state 读取 [0,...,64]，共 65 个。
返回 proprio 时去掉最后一个 state，与动作长度对齐。这里保留原始 FastWAM 的时间对齐，
不把标签整体后移为 [1,...,64]。

数据集缺历史时重复 episode 首帧并标记 padding；历史重复增强把全部历史一起换成当前图像。
未来图像数必须能被 num_context_frames 整除；模型按实际输入帧数和 TRB 的时间分块推导组数。
不提供时序插值或特征左侧补零。

模型不再接收采样跨度或采样间隔，`representation` 配置整体删除。
TRB 从实际输入的 T/H/W 分块，padding mask 按相同的时间分块合并。
`latent_spatial_size` 同时删除，空间网格直接取 TRB 输出；每个相机特征网格仍须能被 H/W 分块整除。

世界调度器的自动 shift 在使用时根据实际 latent `[B,C,T,H,W]` 计算：
`future_tokens` 使用 `sqrt(C*(T-1)*H*W/shift_base_dim)`，`all_tokens` 使用全部 T 组。
批次大小不进入计算；训练采样 shift 改变时，相应训练权重归一化一起更新。
固定数值 shift 与推理调用中的 `sigma_shift` 覆盖继续生效。
联合推理长度由调用方提供 `num_video_frames` 和 `action_horizon`，图像数量包括历史与当前，
不要求与训练长度一致；动作单独推理只需要调用方提供动作长度。
默认 latent `[B,128,5,12,10]` 的自动 shift 仍为 `sqrt(128*4*12*10/4096)`。
世界模型输入、输出通道直接引用 `temporal_bottleneck.output_dim`，无需重复设置表征目标维度。

旧 `state_space: absolute` 表示预测完整视觉隐状态，区别于预测相对当前隐状态的差值。
现在固定使用完整状态，删除该开关和差值监控分支；这与机器人动作的绝对/增量坐标设置不同。
`normalize_target: none` 原本表示不对压缩前的视觉特征追加实验归一化。
该字段与旧特征统计路径、缓冲区和归一化函数已删除，TRB 后的因果归一化保留。
每个未来状态独立采样噪声时间步是最终方案，旧共享时间步配置分支也已移除。

## 梯度模式

按本次确认保留 `temporal_bottleneck.gradient_mode`，最终配置继续为 `action_only`。
发布实现仍要求该模式，与修改前的约束一致；没有在此次重构中扩展其他模式的训练行为。
动作损失的梯度传回 TRB，世界建模和重建诊断损失不更新 TRB。
TRB 固定可训练，不再通过 `trb_trainable` 标志决定是否加入优化器。

## 兼容与验证

配置与权重键迁移见[本轮开发记录](../../DEVELOPMENT_REFACTOR.md)。
模块命名改变了部分权重键；张量形状、参数枚举顺序和数值算法保持原样，尚未转换已训练权重。

小规模 CPU 对比覆盖有、无重建模块：全部权重、参数顺序、损失、所有参数梯度，
以及相同输入和随机种子的缓存动作推理结果均与修改前完全相等。
本次还修正 RoboTwin 入口读取旧 codec 属性名的问题，使真实历史帧和历史刷新观测生效。
这一输入处理修正需在真实评测环境继续验证；没有运行仿真器或已训练权重。

## 世界专家

`world_dit_config` 配置 `world_expert`，`world_scheduler` 配置世界建模的流匹配调度器，
`loss.lambda_world` 对应日志中的 `loss_world`。世界分支固定采用逐帧因果注意力，
不额外接收动作交叉注意力条件；动作与视觉 token 仍通过 MoT 联合计算。
已删除图像/VAE/control 条件标志及旧配置字段拦截。


## 图像尺寸配置（2026-10-19）

独立版 `shape_meta.images` 只保留相机 key，不再声明图像 `raw_shape` 或 `shape`。
目标尺寸只由图像 transform 指定，例如 `Resize.size: [240, 320]`，修改 resize 不需要
同步更新其他尺寸配置。processor 检查处理后张量为 `[T,3,H,W]` 且帧数正确，
空间尺寸直接来自 transform 的输出；多相机张量仍按已有流程堆叠和拼接。
动作和状态各自只在 shape 中声明向量维度，数据读取器、processor 和模型都引用这一来源。
LeRobot 数据集自身的 meta/info.json 仍遵守原有格式，不随本配置简化而改写。


## 动作和状态维度（2026-10-19）

`data.train.shape_meta.action[0].shape` 与 `data.train.shape_meta.state[0].shape`
分别是动作和状态维度的唯一配置来源。删除 `raw_shape` 及 processor 的
`action_output_dim`、`proprio_output_dim` 参数；processor 自动取得维度，继续校验原生张量。
模型配置直接引用这两个 shape，验证配置沿用训练配置的 metadata。
动作、状态可具有不同维度，不再要求人为同步多个字段。
默认维度仍各为 14，原有数据通道、归一化统计与模型权重形状保持一致。


## 数据位置与归一化统计文件

`data.train.dataset_root` 是数据位置，默认是单个路径；多个数据目录仍可使用路径列表。
验证集直接引用训练集的该配置，文本缓存预计算脚本也读取相同入口。
旧 `dataset_dirs` 配置及 Dataset 参数已改名，不提供旧名别名；底层通过官方 LeRobot 0.4.4 与本地只读格式适配类读取。

`data.train.dataset_norm_stats` 是已有归一化统计 JSON 的路径，默认 null 时训练集计算并保存
`dataset_stats.json`，验证集自动复用训练统计。它由 `pretrained_norm_stats` 改名，
不改变统计文件字段、归一化公式或训练权重。

action/state 固定读取每个维度跨预测位置共享的 `global_` 统计，不再提供按预测步归一化开关，
新生成的统计文件也不再写出 `stepwise_` 字段；已有文件中的多余字段不影响使用。
配置只保留统一的 `norm_default_mode`，默认 z-score。

配置中的 `norm_exception_mode` 已删除。按字段选择归一化方式属于通用 processor 能力，
实现中保留默认 None 的可选覆盖参数；它不是 LeRobot Dataset 的必需接口。
当前 ReWAM 配置不启用任何覆盖。

`num_obs_steps`、`num_image_obs_steps` 及 Dataset 向 processor 注入计数的逻辑均已移除。
图像序列长度直接与本次采样的 `image_is_pad` 长度对齐，网络再验证 TRB 分组所需长度，
因此不会把 65 个控制时刻误当成图像数量，也不需要重复维护 2× 的 10 帧或 4× 的 12 帧。


## 官方 LeRobot 数据读取

独立版现通过官方 LeRobot 0.4.4 读取数据，本地 LeRobotDataset 继承官方类，
在内存中适配 v2.0/v2.1 存储格式，同时支持 v3.0。原始 JSONL、Parquet、MP4 不改写。
ReWAM 的采样、相机布局、动作/状态归一化和文本缓存仍由现有包装层负责。
详细的依赖变更、验证与 NumPy/评测环境冲突见[本轮开发记录](../../LEROBOT_DEPENDENCY_REFACTOR.md)。


## ReWAM 数据包装层收敛（2026-10-20）

BaseLerobotDataset 已合并到 RobotVideoDataset，删除原有中间包装类。
RobotVideoDataset 统一处理采样、训练/验证划分、归一化、历史增强、相机拼接与文本缓存；
self.reader 连接底层 LeRobot 读取器，self.processor 直接提供模型验证所需的反归一化接口。
数据集采用扁平目录，配置 target 为 `rewam.datasets.robot_video_dataset.RobotVideoDataset`，processor 为 `rewam.datasets.rewam_processor.ReWAMProcessor`。图像转换使用 torchvision 官方 `ConvertImageDtype`，相机布局和统计归一化工具集中在 `rewam.datasets.utils`。
完整模型输入和统计与合并前逐元素一致；网络权重没有变化。
