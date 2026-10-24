<div align="center">

# ReWAM: Rethinking Representations for World-Action Modeling

Haoyi Jiang<sup>1</sup>, Liu Liu<sup>3</sup>, Xinjiang Wang<sup>3</sup>, Zhihao Sun<sup>4</sup>,
Zequn Chen<sup>3</sup>, Sen Wang<sup>5</sup>, Xinjie Wang<sup>3</sup>, Xia Chen<sup>3</sup>,<br>
Jingfeng Yao<sup>1</sup>, Weiheng Zhao<sup>1</sup>, Shanglin Yuan<sup>1</sup>, Zhizhong Su<sup>3</sup>,
Wei Sui<sup>2</sup>, Wenyu Liu<sup>1</sup>, Xinggang Wang<sup>1</sup><br>

<sup>1</sup>Huazhong University of Science & Technology,
<sup>2</sup>D-Robotics,<br>
<sup>3</sup>Horizon Robotics,
<sup>4</sup>Fudan University,
<sup>5</sup>Xi'an Jiaotong University

[![Project page](https://img.shields.io/badge/project%20page-hustvl.github.io%2FReWAM-blue)](https://hustvl.github.io/ReWAM/)
[![arXiv](https://img.shields.io/badge/arXiv-2609.38163-red?logo=arXiv&logoColor=red)](https://arxiv.org/abs/2609.38163)
[![Hugging Face Model](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-TODO-yellow)]()
[![License: MIT](https://img.shields.io/github/license/hustvl/ReWAM)](LICENSE)

</div>

ReWAM learns robot actions and future world states in a shared representation space. The `main` branch of this repository contains the complete framework used for our RoboTwin training and evaluation.

We also provide a [starVLA_plugin](https://github.com/hustvl/GaussTR/tree/starVLA_plugin) implementation of ReWAM. The RoboDojo results reported in our paper were obtained using our internal data and a training pipeline built on StarVLA. We have ported the core code to this standalone plugin, but have not evaluated its performance on RoboDojo.

## Setup

### Installation

```bash
git clone --recursive --single-branch https://github.com/hustvl/ReWAM.git
cd ReWAM
pip install -e .
```

For simulation evaluation, initialize the pinned RoboTwin submodule and apply the ReWAM compatibility patch:

```bash
bash scripts/setup_robotwin.sh
```

This script prepares only the RoboTwin source code. Follow the [RoboTwin installation guide](https://robotwin-platform.github.io/doc/usage/robotwin-install.html) to install the simulator dependencies and download the assets.

### Data Preparation

Download the dataset from [yuanty/robotwin2.0-fastwam](https://huggingface.co/datasets/yuanty/robotwin2.0-fastwam). Set `ROBOTWIN_DATA_ROOT` to the extracted LeRobot dataset directory containing `meta/`, `data/`, and `videos/`.

```bash
export ROBOTWIN_DATA_ROOT=PATH_TO_ROBOTWIN
export ROBOTWIN_TEXT_CACHE="${ROBOTWIN_DATA_ROOT}/text_embeds"
python scripts/precompute_text_embeds.py task=robotwin_rewam
```

## Usage

### Training

Launch training on a single node with eight GPUs:

```bash
bash scripts/train.sh 8 task=robotwin_rewam
```

For multi-node training, run the launcher on each node using the same `NNODES`, `MASTER_ADDR`, `MASTER_PORT`, and `RUN_ID`, with a unique `NODE_RANK` for each node.

### Evaluation

After completing the RoboTwin simulator setup above, evaluate a model checkpoint using the normalization statistics from the corresponding training run:

```bash
python evaluation/robotwin/run_robotwin_manager.py \
  task=robotwin_rewam \
  ckpt=PATH_TO_OUTPUTS/checkpoints/weights/step_XXXXXX.pt \
  EVALUATION.dataset_stats_path=PATH_TO_OUTPUTS/dataset_stats.json \
  MULTIRUN.num_gpus=8 MULTIRUN.max_tasks_per_gpu=2
```

For multi-node evaluation, add the following settings to the command above:

```
+MULTIRUN.num_nodes=${NNODES} \
+MULTIRUN.node_rank=${NODE_RANK} \
+MULTIRUN.master_addr=${MASTER_ADDR} \
+MULTIRUN.master_port=${EVAL_MASTER_PORT} \
```

## Acknowledgements

This implementation builds on [Fast-WAM](https://github.com/yuantianyuan01/FastWAM). We thank the authors for sharing their code and data.

## Citation

If you use ReWAM in your research, please cite:

```bibtex
@misc{jiang2026rewam,
  title={Rethinking Representations for World-Action Modeling},
  author={Haoyi Jiang and Liu Liu and Xinjiang Wang and Zhihao Sun and Zequn Chen and Sen Wang and Xinjie Wang and Xia Chen and Jingfeng Yao and Weiheng Zhao and Shanglin Yuan and Zhizhong Su and Wei Sui and Wenyu Liu and Xinggang Wang},
  journal={arXiv preprint arXiv:2609.38163},
  year={2026},
  url={https://arxiv.org/abs/2609.38163}
}
```
