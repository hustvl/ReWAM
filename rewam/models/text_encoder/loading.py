from dataclasses import dataclass
from typing import Any

import torch

from .umt5 import HuggingfaceTokenizer, WanTextEncoder
from .weights import ModelConfig, hash_model_file, load_state_dict


@dataclass
class TextComponents:
    text_encoder: WanTextEncoder | None
    tokenizer: HuggingfaceTokenizer | None
    text_encoder_path: str | None
    tokenizer_path: str | None


WAN22_MODEL_REGISTRY = [
    {
        "model_hash": "9c8818c2cbea55eca56c7b447df170da",
        "model_name": "wan_video_text_encoder",
        "model_class": WanTextEncoder,
    }
]


def _load_registered_model(
    path,
    model_name: str,
    torch_dtype: torch.dtype,
    device: str,
    model_kwargs_override: dict[str, Any] | None = None,
):
    model_hash = hash_model_file(path)

    matched_config = None
    for config in WAN22_MODEL_REGISTRY:
        if config["model_hash"] == model_hash and config["model_name"] == model_name:
            matched_config = config
            break
    if matched_config is None:
        raise ValueError(
            f"Cannot detect model type for {model_name}. File: {path}. "
            f"Model hash: {model_hash}. This standalone package follows DiffSynth hash-based loading."
        )

    model_class = matched_config["model_class"]
    model_kwargs = dict(matched_config.get("extra_kwargs", {}))
    if model_kwargs_override is not None:
        model_kwargs.update(model_kwargs_override)
    state_dict_converter = matched_config.get("state_dict_converter")

    model = model_class(**model_kwargs)
    state_dict = load_state_dict(path, torch_dtype=torch_dtype, device="cpu")
    if state_dict_converter is not None:
        state_dict = state_dict_converter(state_dict)

    model.load_state_dict(state_dict, strict=False)
    model = model.to(device=device, dtype=torch_dtype)
    return model


def load_text_components(
    *,
    weights: dict[str, Any],
    tokenizer: dict[str, Any],
    max_length: int,
    device: str = "cuda",
    torch_dtype: torch.dtype = torch.bfloat16,
    load_text_encoder: bool = True,
) -> TextComponents:
    if not load_text_encoder:
        return TextComponents(
            text_encoder=None,
            tokenizer=None,
            text_encoder_path=None,
            tokenizer_path=None,
        )

    text_config = ModelConfig(**weights)
    tokenizer_config = ModelConfig(**tokenizer)
    text_config.download_if_necessary()
    tokenizer_config.download_if_necessary()
    text_encoder = _load_registered_model(
        text_config.path,
        "wan_video_text_encoder",
        torch_dtype=torch_dtype,
        device=device,
    )
    tokenizer = HuggingfaceTokenizer(
        name=tokenizer_config.path,
        seq_len=int(max_length),
        clean="whitespace",
    )
    return TextComponents(
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        text_encoder_path=str(text_config.path),
        tokenizer_path=str(tokenizer_config.path),
    )
