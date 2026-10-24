# ReWAM 独立版权重兼容与迁移记录

现已提供显式转换工具，完整使用方式见[历史权重迁移指南](../../CHECKPOINT_MIGRATION.md)。
选择 `--target fastwam`，迁移历史权重的顶层组件名称，并将 `model_class` 从 `RA` 更新为 `ReWAM`。
原 checkpoint 的 1662 个张量已通过当前组合配置的键与形状只读校验，尚未另存实际训练权重。

2026-10-20 本轮不新增权重键或张量形状变更。设备和 dtype 改为从实际注册参数读取，
避免 `.to()` 或 `.bfloat16()` 后使用过期属性。网络计算与参数枚举顺序保持原样。

下文保留此前逐轮重构记录；工具使用当前 `reconstruction_probe` 名称，
历史中间名称 `feature_reconstructor` 也会显式迁移至该名称。

## 影响保存内容的更名

| 原名称 | 发布版名称 | 影响 |
| --- | --- | --- |
| Python 包 `fastwam` | `rewam` | 导入路径和 Hydra target 改变；通过 pickle 保存的完整模型对象需要迁移类路径。仅保存张量的数据不包含 Python 模块路径。 |
| `RA` / `create_ra` | `ReWAM` / `create_rewam` | 工厂名称及 `model_class` 元数据改变。 |
| `representation_codec` | `trb` | 权重文件顶层键及完整模型、Accelerate state-dict 前缀改变。 |
| `codec_latent_norm` | `causal_norm` | 权重文件键与完整 state-dict 前缀改变；须保留运行均值、方差及批次计数。 |
| `codec_decoder` | `feature_reconstructor` | 可选的梯度分离重建诊断模块，其权重文件键和完整 state-dict 前缀改变。 |
| `representation_encoder` | `feature_calibration` | 模块更名；冻结的特征教师仍不注册到 checkpoint。 |
| `FAEAttentionCausalCodec` | `TemporalRepresentationBottleneck` | 仅类名改变，内部张量名称和形状保持原样。 |
| `CausalRunningChannelNorm` | `CausalRunningNormalization` | 仅类名改变，缓冲区名称保持原样。 |
| `CausalCodecFeatureDecoder` | `DetachedFeatureReconstructor` | 仅类名改变，投影权重保持原样。 |

`mot` 和 `proprio_encoder` 的权重键保持原样。MoT 的 `video`、`action` 专家标识及内部 DiT 张量名称保持原样。
TRB 的 `q_projection`、`k_projection`、`value_projection`、`output_projection` 和相机偏置名称保持原样。

所有张量形状保持原样：DINO 维度 1024、patch 特征维度 8192、TRB 输出维度 128、
TRB Q/K 维度 512、8 个注意力头、世界模型维度 2048、动作模型维度 1024，专家仍为原有 30 层。
旧最终配置中关于输出 512 通道的注释已经过时，实际组合配置的输出维度为 128。

## 配置与行为

最终模型与任务配置为 `model=rewam`、`task=robotwin_rewam`，已展开旧最终配置的继承链。
`codec.type` 更名为 `trb`，配置位置 `representation.codec` 保留。
移除实验 codec、特征教师选项和预训练世界模型初始化入口，数据、模型与缓存路径改为可配置的发布路径。

权重加载器现在要求所有启用组件均存在，并严格加载张量。
历史文件缺少 TRB 或归一化统计时，不再静默使用随机瓶颈或恒等归一化。
需先明确检查并迁移相关权重键和配置，再加载历史文件；该转换尚未实现或执行。

Accelerate/DeepSpeed 的完整恢复文件还包含优化器和调度器状态。仅转换模型权重的文件不能用于完整训练恢复。
最终配置的参数枚举顺序保持原样，但恢复时仍须单独核对优化器参数组、参数名、统计量、训练步数和保存的配置。

## 数值验证

固定随机种子后，小规模 CPU 模型与源版本 `79a03ea` 的初始张量、训练损失和 TRB 梯度完全相等。
发布版测试还覆盖严格权重恢复、时序采样配置及 AGRS 梯度隔离。
生产 checkpoint 与 GPU 数值一致性需在权重转换获确认后继续验证。


## 1014 配置去重补充

- 删除数据集的 `video_size` 和拼接后的缩放、中心裁剪；Robotwin 布局直接确定输出为高 384、宽 320。
  保留拼接前的相机缩放及插值方式，随机输入回归检查表明与旧最终流程逐像素完全相等。
- 删除未使用的 `camera_key`，移除上层数据集的 `global_sample_stride` 配置和参数。
  底层原生采样仍固定为默认步长 1，`action_video_freq_ratio` 保留。
- 删除最终配置中未生效的编码器 `input_size`：`resize_input: false` 时不读取它。
  删除 TRB 配置中只能取 `[2,2,2]` 的 `kernel_size` 和 `stride`，继续使用原有构造默认值。
- 验证集引用训练集的公共配置，保留 `is_training_set: false` 和 `history_repeat_probability: 0.0`。
  数据路径、时序长度、相机形状和处理器等覆盖参数会同步到验证集，避免两份配置漂移。
  验证归一化统计仍由运行入口优先使用明确指定的统计文件，再使用训练集统计。
- 缓存文本的 `context_len` 引用 `model.text_encoder.max_length`，验证图像变换引用训练变换。

以上不改变权重键、参数形状、当前实验的采样时间线或像素值。旧配置中删除的字段需要移除；
若改过全局步长、拼接输出尺寸或使用其他图像布局，不能据此认为新旧预处理等价。
尚未转换任何已训练权重。


### 1014 processor 接口收敛

- 删除 `num_output_cameras` 配置、构造参数及相机补零、截断逻辑。
  处理器按 `shape_meta.images` 的顺序输出实际配置的相机；Robotwin 布局仍要求三路相机。
- `train_transforms`、`val_transforms` 合并为单个 `transforms`，训练与验证使用同一处理流程。
  处理器的 train/eval 模式继续用于验证时保存未归一化的 `gt_action`，不再用于选择图像变换。
- 同步 `ReWAMProcessor`、原有基类接口、数据配置和测试。旧保存配置加载前需删除
  `num_output_cameras`，把两套相同的图像变换改为 `transforms`。
  若旧训练和验证变换不同，不能直接将其视为同一流程。

原有 `default` 动作、状态字典及归一化统计键保留，不改动动作顺序和统计格式。
三路相机的随机输入在训练与验证模式下，与本次修改前处理器的全部输出逐项完全相等。
网络权重键和形状保持原样，尚未转换已训练权重。


### 1014 动作与状态改为原生张量

数据加载器现在直接返回 `sample["action"]` 和 `sample["state"]` 张量。
已移除 `{"default": tensor}` 包装、`ConcatLeftAlign` 合并器以及验证、部署中的拆分和重新合并。
处理器只接受单个原生动作向量和单个原生状态向量，原始维度、处理后维度和输出维度须一致。

配置移除 `processor.action_state_merger`、`processor.action_state_transforms`。
同时删除重复的 `BaseProcessor`、未使用的字典式后处理入口、字段式增量动作掩码入口及旧统计缓存查询入口。
验证输出的 `gt_action` 由单字段字典改为原始动作张量；其数值保持原样。
使用旧 processor 配置时，必须先移除上述旧字段，不能直接传给新构造函数。
若完整 pickle 对象含旧 processor 或合并器，还需在旧代码环境读取后迁移；这不影响纯模型张量文件。

**统计文件格式保留**：加载和新计算的统计仍使用 `action.default`、`state.default`，无需重算已有统计。
归一化仍采用相同的均值、标准差、数值稳定项与 `[-5,5]` 截断。
按时间步的动作归一化与其逆变换也保留。统计键只在统计读取、保存边界使用，不再包装训练样本。

CPU 验证显示，旧、新数据加载与处理器的图像、动作、状态和掩码数值完全相等，
两段人工构造 episode 的全部统计量及保存键完全相等，部署归一化和动作 L1/L2 指标也完全相等。
网络张量键、形状和采样时间线保持原样，尚未转换已训练权重。


### 1014 TRB 顶层配置与最终网络路径

当前模型配置与架构说明见[模型配置](model_configuration.md)。本轮路径迁移：

| 旧路径 | 当前路径或处理 |
| --- | --- |
| `model.representation.codec` | `model.temporal_bottleneck`，不再使用 codec 抽象 |
| `model.representation.encoder` | `model.feature_calibration`，删除单一教师的 `name` 选择 |
| `representation.codec.latent_norm` | `temporal_bottleneck.causal_normalization`，删除 `enabled/type` |
| `representation.codec.decoder` | `model.feature_reconstruction`，仅保留损失权重；`null` 表示省略诊断模块 |
| `representation.codec.gradient_mode` | `temporal_bottleneck.gradient_mode`，保留，当前仍为 `action_only` |
| `representation.target_dim` | 删除，使用 `temporal_bottleneck.output_dim` |
| `normalize_target`、旧 latent stats 路径与缓冲区 | 删除，固定不对压缩前特征追加实验归一化 |
| `state_space` | 删除，固定预测完整视觉隐状态 |
| `temporal_align_mode`、`temporal_indices` | 删除，仅支持严格对应的双帧组 |
| `temporal_padding`、kernel/stride 参数 | 删除，TRB 固定处理完整 2×2×2 特征块 |
| `data.*.include_previous_video_frame` | 删除，始终包含真实历史帧；初始历史重复策略保留 |

`trb`、`causal_norm`、`feature_reconstructor` 的保存键及内部张量名称保持原样。
删除的 latent mean/variance 缓冲区原本是非持久、未启用的缓冲区，不属于 checkpoint。
TRB 输出仍为 128，Q/K 仍为 512，优化器参数顺序保持原样。
内部 helper 和诊断指标由 `codec` 更名为 `trb`，日志消费端若依赖旧指标名需同步。
旧配置中的废弃字段会明确报错；不提供静默兼容别名，也没有转换训练产物。

RoboTwin 部署入口原本仍读取旧的 `representation_codec_uses_history` 属性，导致未启用真实历史刷新。
现在按最终 ReWAM 架构始终使用历史/当前帧，并在历史刷新步请求观测。
这修正了当前整理版本的输入行为，不改变权重形状；真实已训练权重的评测需检查该变化。
相同输入、固定种子的核心缓存动作推理与修改前完全相等。

小规模 CPU 对比验证有、无重建模块两种情形：全部模型张量、参数顺序、训练损失、
所有参数梯度均完全相等。尚未转换任何已训练权重，也未运行 GPU 或仿真器。

### 1014 删除固定的图像条件标志

删除 `model.representation_dit_config.has_image_input`、`WanVideoDiT` 的同名构造参数
及仅允许 `False` 的断言。该参数没有实际计算分支，删除后网络结构、张量键和形状均保持原样。
旧保存配置在传给新版模型之前，需要删除此字段；纯权重张量无需因本次修改转换。


### 1014 本轮世界专家与重建探针命名

本文件保留整理过程的历史记录，供 release 开发使用，不作为最终发布文档。
当前名称、权重键映射和验证结果以[本轮开发记录](../../DEVELOPMENT_REFACTOR.md)为准。
删除旧配置拦截和兼容别名，世界专家统一为 `world_expert`，重建探针为 `reconstruction_probe`。
网络张量形状和数值计算保持原样，尚未转换任何已训练权重。

### 1014 FastWAM 上游更新整合

本轮的 RoPE 设备驻留、请求内动作文本条件缓存和可选动作编译未新增或更改序列化权重键，
张量形状与参数顺序保持原样。LeRobot 3.0 与在线文本是可选的数据输入能力，
原有默认采样与统计文件格式保持原样。尚未转换任何已训练权重。
与本轮修改前快照比较，两种重建探针设置的全部权重、损失、梯度和固定种子 CPU 动作结果完全相等。
GPU 编译数值与真实性能需后续验证，详见[上游整合记录](upstream_fastwam_20261014.md)。


## 2026-10-16：编码器与特征校准分离

DINO 资源从 `feature_calibration` 移到同级 `encoder`，`feature_calibration` 仅保留 `layers`。
删除视觉分支旧 backend/Hugging Face 分支、resize/norm 开关和手填 patch size；
patch size 从实际 DINO 元信息读取，输出通道规格保留在 `encoder.output_dim` 供延迟构建 TRB 使用。
独立版相机布局改由 `camera_layout` 引用数据侧设置；插件继续接收独立视角。

原始特征由 `DINOEncoder` 提取，再由 `FeatureCalibration` 完成无仿射归一化和融合。
DINO 使用 `norm=False` 提取特征，不再替换原模型的输出 norm。
校准的 epsilon、运算顺序和融合公式保持原样，两个模块都没有需要保存的可训练参数。
DINO 仍冻结、延迟加载且不注册进 ReWAM 的 state-dict；用于 dtype 跟随的空 buffer 不持久化。

本轮不改变现有 ReWAM state-dict 键、权重张量形状或参数顺序，不需要额外的张量键转换；
此前重构记录中的迁移仍然需要执行。旧保存配置需要按上述边界拆分，
旧完整模型 pickle 包含的类路径/对象结构已改变，应通过新配置重建模型后加载张量，不能直接视为兼容。
没有转换或写入任何已训练权重。


## 2026-10-16：TRB 初始化统一使用全局随机状态

删除 temporal_bottleneck.seed、TRB 构造参数和独立 CPU Generator。
V 保留单位范数高斯初始化，注意力输出投影保留零初始化与残差相加。
独立版在构建模型前设置 cfg.seed，各 rank 从相同模型初值开始；训练器继续使用原有的 rank 随机偏移。
HWM 原有训练入口已在构建框架前设置全局 seed，无需修改共享入口。

这会改变从零训练的初值，并额外消耗全局 RNG，因此 V、之后初始化的偏置/模块初值和后续随机序列都可能变化。
网络公式、state-dict 键、张量形状、参数顺序及优化器结构不变。
旧保存配置需删除 TRB 局部 seed；已有权重张量无需因本轮再做转换。
完整加载相同权重并固定推理噪声/采样 seed 时，结果保持一致。
若未固定采样随机源，构建阶段 RNG 消耗变化可能改变后续采样值。

验证仅加载此前人工初始化的小模型张量：12 组 FP32/BF16、Base/Joint、探针开关用例，
严格加载后特征、损失、全部梯度、运行统计和固定种子动作均与修改前精确相等。
没有转换、加载或改写已训练权重。


## 2026-10-16：reconstruction_probe 统一管理 RGB 解码器

配置路径由 representation.decoder_path 移到 reconstruction_probe.decoder_path；
环境变量由 REWAM_DECODER_PATH 改为 RAEv2_DECODER_PATH。插件/HWM 使用 framework 下的同名字段。
probe 管理注册的可训练 projection 和未注册的外部冻结 decoder，原模型级 representation_decoder 字段与方法删除。
新类位于 reconstruction_probe.py；temporal_bottleneck.py 不再导出 ReconstructionProbe。

训练头的权重键 reconstruction_probe.projection.weight/bias、形状、参数顺序和初始化保持原样。
冻结的 RAEv2 权重只在 decode_rgb 时按路径加载，始终 eval/no_grad，不进入优化器与 ReWAM state-dict。
旧完整 state-dict 若是在模型级 RAEv2 仍加载时保存，可能包含 representation_decoder.*；
这些外部资源键不属于新版 ReWAM checkpoint，后续迁移时应单独确认并移除，继续从路径加载 RAEv2。
完整 Python 模型 pickle 的类路径与对象结构也已变化，应通过新配置重建并加载张量。

decoder_path 为 null 时继续训练特征解压头、计算验证 loss，独立版验证不再隐式要求 RGB 解码器。
提供路径时，独立版保留原来的 RGB 验证与可视化流程，复用一份缓存，并在验证结束释放。
未转换、加载或改写任何已训练权重；数值对比只使用人工初始化的小模型和临时的微型 RAEv2 权重。


## 2026-10-16：backbone 命名与独立版目录整理

DINOEncoder 的 `teacher` / `_teacher` 改为 `backbone` / `_backbone`；内部冻结 DINO
仍不注册成子模块。state_dict 不受影响；如果历史文件使用完整 Python 对象 pickle，
后续转换时需处理实例缓存字段，不能用属性兼容别名替代显式迁移。

独立版 UMT5 及其辅助类从 `rewam.models.wan22.wan_video_text_encoder`、
`rewam.models.helpers` 移到 `rewam.models.text_encoder`；配置工具移到 `rewam.utils.config`。
动作/世界 DiT、TRB、特征探针的权重键、张量形状及参数顺序保持一致。删除的 `ActionHead`
没有在实际 ActionDiT 中实例化。类路径变化只涉及完整对象 pickle；标准 state_dict 无需因此转换。
本次仅记录，未转换任何实际训练权重。


## 2026-10-16：移除动作骨干预训练初始化

ReWAM 从配置直接构造 ActionDiT；删除单独加载 WAN 衍生动作骨干初始化权重的入口。
该入口仅用于训练前的初始化，不负责恢复 ReWAM 权重或训练状态。
标准 state_dict 的名称、形状和参数顺序均保持不变；初始化随机数消耗和前向计算不变。
旧配置中的动作骨干预训练路径和跳过初始化标记需删除，不保留旧字段兼容分支。
恢复训练和推理继续使用完整 ReWAM checkpoint；尚未转换实际训练权重。

独立版删除 `ActionDiT.from_pretrained()`、`backbone_key_set()`、
`action_dit_pretrained_path` 和 `skip_dit_load_from_pretrain`。`model_paths` 中的
初始化来源字段改为 `action_dit: RANDOM_INIT`；此字段不是模型权重键。
`ReWAM.load_checkpoint()`、训练器的完整状态恢复及部署权重加载保持原样。


## 2026-10-19：移除独立版图像尺寸 metadata

`data.*.shape_meta.images` 中不再声明图像 `raw_shape` / `shape`，只保留相机 key。
图像尺寸只通过 transform 配置，processor 保留帧数、RGB 通道和维数检查。
复用旧运行配置时可删除这些图像尺寸字段；动作和状态的维度字段、
数据归一化统计及训练权重保持原样，不涉及权重转换。


## 2026-10-19：动作与状态维度配置统一

删除动作/状态的 `raw_shape`，各自仅保留 `shape`；删除 processor 构造参数
`action_output_dim`、`proprio_output_dim`。模型的 action_dim / proprio_dim 直接从
shape_meta 引用，processor 自动取得相同的维度。
旧运行配置应删除重复字段，并更新模型中的对应引用；不提供旧参数兼容别名。
默认 14 维、动作/状态的通道顺序、归一化统计格式及训练权重键/形状不变，
因此不需要因本次配置整理转换实际训练权重。


## 2026-10-19：时间压缩倍数与多帧历史

TRB 配置新增 `temporal_compression`，默认 2。历史帧数自动为倍数减一，
训练采样、时间分组、推理历史窗口、padding mask 与重建 probe 共用该设定。
Standalone 的旧 `representation.temporal_groups` 改为 `num_frames` / `frame_stride` 引用数据配置，
推理历史输入由单张 `previous_image` 改为 `history_images [B,C,T_history,H,W]`。
HWM / plugin 保留主框架已有的单历史帧输入；多历史帧使用 `history_image [3,T_history,H,W,3]`。

默认 2× 的 state dict 键和张量形状不变，FP32/BF16 的 TRB 输出和输入/参数梯度逐元素一致。
选择 4× 属于新的网络规格：TRB Q/K/V 投影的输入列数翻倍，
probe 的 ConvTranspose3d 时间核从 2 变为 4；其余参数键不变。
旧 2× 权重不能直接严格加载到 4×，也不存在不改变模型语义的自动等价转换。
这里只记录兼容性影响，未转换任何已训练权重。


## 2026-10-19：数据配置名称整理

Standalone 的 `pretrained_norm_stats` 改为 `dataset_norm_stats`，`dataset_dirs` 改为 `dataset_root`。
后者默认采用单路径字符串，也支持路径列表以保持多数据集能力；训练/验证引用、Dataset 参数和文本缓存脚本同步调整。
旧运行配置需修改这两个名称；不保留旧名别名。底层 LeRobot reader 的内部接口不变。
统计文件、归一化算法、模型参数名和张量形状不变，已有 `dataset_stats.json` 可继续使用。
HWM 与 plugin 本轮不修改；没有转换已训练权重。


## 2026-10-19：归一化分支与 processor 计数精简

- 删除 use_stepwise_action_norm 参数和分支，action/state 固定读取 global_ 统计。
- 新统计文件只生成 global_ 字段；旧统计文件保留的 stepwise_ 字段不会被使用。
- 删除 num_obs_steps 与 num_image_obs_steps 参数、属性，以及 Dataset 的计数注入。
  图像序列长度以采样 mask 为准，保持 2×/4× 数据流程。
- norm_exception_mode 从配置删除；按字段覆盖作为通用 processor 能力，仍以默认 None 的可选参数保留。
- 旧运行配置需移除被删除的参数；模型权重键、形状和默认归一化计算不变，未转换真实权重。

固定同一批 episode 的归约顺序，对比修改前后：所有 global_ 统计以及 z-score、min/max、q01/q99
三种模式的归一化和反归一化结果均逐元素一致（rtol=atol=0）。


## 2026-10-19：采样语义与三维 TRB 分块配置

- data.num_frames: 65 → action_horizon: 64；底层 obs_size 仍为 65，action_size 仍为 64。
- action_video_freq_ratio → observation_stride。
- data.temporal_compression → num_context_frames，引用 model.temporal_bottleneck.patch_size.0。
- TRB 的 temporal_compression → patch_size: [T,H,W]；默认 [2,2,2]。
- representation 也使用 action_horizon / observation_stride，重建 probe 的三维核和步幅引用 TRB。
- 评测入口与历史缓存同步新名称；没有旧名别名或 retired 检查。

默认图像、action、state 采样偏移和模型参数键/形状不变。
TRB 和 probe 的 FP32/BF16 参数、输出与梯度与修改前逐元素一致。

改变任意 patch_size 轴会改变 patch 内元素的组织和对应 probe 几何；
Q/K/V 的输入维度为 encoder_dim * T * H * W，probe 权重时间/空间核也随之改变。
即使不同 patch 的乘积相同、线性层形状相同，其元素含义也不相同，不能据此认为旧权重可等价使用。
这里只记录迁移影响，未转换任何真实权重。


## 2026-10-19：采样规划与模型解耦

删除整个 representation 配置，包括 action_horizon、observation_stride、latent_spatial_size。
同时删除无参数的 RepresentationConfig、预设 temporal_groups 以及不再被使用的 as_hw 工具。
视频时间分组与 padding mask 按实际输入长度推导，世界调度器自动 shift 按实际 latent C/T/H/W 计算。
自动 shift 改变时同步重算训练权重归一化；固定数值 shift 和推理 sigma_shift 覆盖保留。
推理调用负责提供长度，允许与训练长度不同。action/state metadata 及数据采样偏移保持原状。

纯张量 checkpoint 的键、形状、参数枚举顺序不变；旧运行配置需移除整个 representation 段。
原有默认网格和长度的 shift 数值不变。对于原先由配置与实际输入不一致而算错 shift 的场景，
新行为会使用实际形状，因此不能认为与错误旧配置等价。
当前发布默认 `[B,128,5,12,10]`、future_tokens 的 shift 仍为 sqrt(128*4*12*10/4096)。

FP32/BF16、有/无 probe、固定/自动 shift 共 8 组小模型验证：权重、训练损失、全部参数梯度、
相同种子的动作与联合推理结果逐元素一致。联合推理使用占位 RGB 帧，不涉及真实 RGB 解码。
没有加载、转换或改写任何已训练权重。HWM 与 plugin 不在本轮范围内。


## 2026-10-19：改用官方 LeRobot 依赖

删除内置 LeRobot 复制实现，本地 LeRobotDataset 继承官方 LeRobot 0.4.4，适配 v2.x 文件布局。
v3.0 使用同一读取流程；本地数据文件、已有统计 JSON 和文本 embedding 缓存不迁移。
网络代码、权重键、张量形状和默认采样时间线均未改变，没有转换任何已训练权重。
真实 Robotwin v2.1 子集在 PyAV 与 TorchCodec 后端的样本、全部归一化统计与旧读取器逐元素一致。
环境依赖调整及尚待确认的 RoboTwin NumPy 冲突见[开发记录](../../LEROBOT_DEPENDENCY_REFACTOR.md)。


## 2026-10-20：视觉编码器命名

视觉配置与模型属性 encoder 改名为 vision_encoder。Standalone 使用 model.vision_encoder，
HWM / StarVLA plugin 使用 framework.vision_encoder，旧 checkpoint 附带配置需同步改名。
DINO backbone 仍不注册进 state-dict，没有持久化参数或 buffer 因本次更名而变化；
可训练张量键、形状和参数顺序保持一致，没有转换已训练权重。不保留旧参数别名。


## 2026-10-20：合并数据包装层

删除 BaseLerobotDataset 类及其文件，职责并入 RobotVideoDataset；没有旧类导入别名。
现有 Hydra Dataset target、配置、采样时间线和统计 JSON 格式保持原样。
训练器从 val_dataset.processor 取得反归一化接口。
本次不修改模型权重、张量形状或网络参数顺序，不需要因此转换权重。
若额外用 pickle 保存了完整旧 Dataset 对象，应从配置重新创建 Dataset。


## 2026-10-20：专家唯一持有与训练分支精简

独立版由 `mot.mixtures.video/action` 唯一注册两组专家，`world_expert` 和 `action_expert` 为只读访问属性。
删除根部重复注册路径，避免完整 `state_dict()` 重复列出同一组张量。
标准 `save_checkpoint()` 的分组件权重格式保持原样，`mot` 内的键、张量形状、数值和参数枚举次序保持一致。
完整模块状态不再包含重复的 `world_expert.*`、`action_expert.*`；规范路径为
`mot.mixtures.video.*`、`mot.mixtures.action.*`，优化器参数名称也使用该路径。
因此旧完整 Accelerate 状态不能直接假定兼容，需另行处理旧别名与优化器记录。
本轮工具支持历史标准模型权重，不迁移完整训练恢复目录。

训练固定使用当前方案的逐帧 timestep，删除未使用的单 timestep 损失分支、恒等包装和目标转发函数。
原有逐帧损失计算次序、采样随机数、AGRS 和重建探针保持原样。
