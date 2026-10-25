# ReWAM StarVLA Plugin

This branch provides a standalone StarVLA plugin ported from our internal ReWAM framework. The complete training and deployment pipeline has not yet been validated with this plugin.

## Installation

Install StarVLA first, then run the following command from this repository's root directory:

```bash
pip install -e .
```

To enable online text encoding with UMT5, install the optional text dependencies:

```bash
pip install -e '.[text]'
```

## Usage

Use `rewam_starvla.run` to register the ReWAM plugin before launching a StarVLA module. StarVLA's `starVLA` and `deployment` packages must be importable in the active Python environment.

```bash
python -m rewam_starvla.run starVLA.training.train_starvla \
  --config_yaml examples/rewam_base.yaml.yaml
```
