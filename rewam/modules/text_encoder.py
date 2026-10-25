from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn


class TransformersT5Encoder(nn.Module):
    """Standard T5 / UMT5 encoder from ``transformers``.

    Parameters
    ----------
    pretrained_model_name_or_path:
        HuggingFace model id or local path, e.g. ``"google/umt5-xxl"``.
    torch_dtype:
        Data type for the encoder weights.
    device:
        Target device.
    kwargs:
        Forwarded to the selected encoder class's ``from_pretrained``.
    """

    def __init__(
        self,
        pretrained_model_name_or_path: str,
        torch_dtype: torch.dtype = torch.float32,
        device: str | torch.device = "cpu",
        init_from_config_only: bool = False,
        **kwargs: Any,
    ):
        super().__init__()
        from transformers import AutoConfig, T5EncoderModel, UMT5EncoderModel

        config_keys = (
            "cache_dir",
            "force_download",
            "local_files_only",
            "revision",
            "subfolder",
            "token",
            "trust_remote_code",
        )
        config_kwargs = {key: kwargs[key] for key in config_keys if key in kwargs}
        config = AutoConfig.from_pretrained(
            pretrained_model_name_or_path,
            **config_kwargs,
        )
        encoder_types = {
            "t5": T5EncoderModel,
            "umt5": UMT5EncoderModel,
        }
        try:
            encoder_type = encoder_types[config.model_type]
        except KeyError as exc:
            raise ValueError(
                f"Unsupported text encoder model_type={config.model_type!r}; "
                f"expected one of {sorted(encoder_types)}."
            ) from exc

        if init_from_config_only:
            with torch.device("meta"):
                self._encoder = encoder_type(config)
            self._encoder = self._encoder.to(dtype=torch_dtype)
        else:
            self._encoder = encoder_type.from_pretrained(
                pretrained_model_name_or_path,
                config=config,
                torch_dtype=torch_dtype,
                **kwargs,
            ).to(device=device)
        meta_parameters = [name for name, parameter in self._encoder.named_parameters() if parameter.is_meta]
        if meta_parameters and not init_from_config_only:
            raise RuntimeError(
                f"Text encoder loaded from {pretrained_model_name_or_path!r} still has "
                f"{len(meta_parameters)} meta parameters; the checkpoint is incomplete."
            )
        self._encoder.eval()
        for p in self._encoder.parameters():
            p.requires_grad_(False)

    @property
    def text_dim(self) -> int:
        """Output dimension (``d_model``)."""
        return int(self._encoder.config.d_model)

    def forward(self, ids: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        """Encode token ids → ``[B, L, text_dim]``."""
        out = self._encoder(input_ids=ids, attention_mask=mask)
        return out.last_hidden_state  # [B, L, d_model]


class TransformersTokenizer:
    """Standard HuggingFace tokenizer.

    Parameters
    ----------
    pretrained_model_name_or_path:
        HuggingFace model id or local tokenizer path.
    seq_len:
        Max sequence length (padding + truncation).  If *None*, no padding.
    kwargs:
        Forwarded to ``AutoTokenizer.from_pretrained``.
    """

    def __init__(
        self,
        pretrained_model_name_or_path: str,
        seq_len: int | None = None,
        **kwargs: Any,
    ):
        from transformers import AutoTokenizer

        self._tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path, **kwargs)
        self._seq_len = seq_len

    def __call__(
        self,
        sequence: str | list[str],
        *,
        return_mask: bool = False,
        add_special_tokens: bool = True,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, torch.Tensor] | torch.Tensor:
        if isinstance(sequence, str):
            sequence = [sequence]

        tok_kwargs: dict[str, Any] = {"return_tensors": "pt"}
        if self._seq_len is not None:
            tok_kwargs.update(padding="max_length", truncation=True, max_length=self._seq_len)
        tok_kwargs.update(kwargs)

        ids = self._tokenizer(sequence, **tok_kwargs)

        if return_mask:
            return ids.input_ids, ids.attention_mask
        return ids.input_ids


__all__ = ["TransformersT5Encoder", "TransformersTokenizer"]
