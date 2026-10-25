from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import torch
import torch.utils.checkpoint as torch_checkpoint

from .dit import scaled_dot_product_attention

# ---------------------------------------------------------------------------
# Token-layout / state / binding dataclasses
# ---------------------------------------------------------------------------


@dataclass
class MoTTokenLayout:
    """Per-branch token metadata used to compile attention masks / caches.

    Local groups must be mutually exclusive and together cover every token with
    ``valid_mask == True``.  The MoT / mask compiler only reads this; it does
    not interpret the *model* semantics of any group.

    Attributes:
        groups: local group name -> bool ``[B, S]``.  ``"video"`` for Wan,
            ``{"video", "context"}`` for LingBot, ``"action"`` for the action
            branch.
        frame_ids: int64 ``[B, S]``.  Time tokens carry ``0..T-1``; tokens
            without video-time semantics carry ``-1``.  ``None`` when the branch
            has no time semantics (e.g. plain action).
        valid_mask: bool ``[B, S]`` excluding padding tokens.
    """

    groups: dict[str, torch.Tensor]
    frame_ids: torch.Tensor | None
    valid_mask: torch.Tensor


@dataclass
class MoTExpertState:
    """Mutable per-branch state threaded through the MoT layer loop.

    ``tokens`` is the only field that may evolve layer-to-layer; ``layout`` and
    ``payload`` must be treated as read-only within a single forward so that
    gradient-checkpoint closures can safely capture them.

    Attributes:
        tokens: ``[B, S, D_model]``, the branch's full attention sequence.
        layout: :class:`MoTTokenLayout` (read-only).
        payload: opaque, expert-private conditioning (RoPE freqs, t_mod,
            cross-attention context/mask, head params, grid meta...).  Created
            and consumed only by the expert; the MoT never interprets it.
    """

    tokens: torch.Tensor
    layout: MoTTokenLayout
    payload: Any


@dataclass
class MoTBranchBinding:
    """A bound (branch-name, expert, state) triple passed into a MoT forward.

    The ``expert`` is a runtime reference, NOT registered as a MoT submodule
    (the framework owns it).  The same expert may bind multiple independent
    states (e.g. future IDM ``noisy_video`` / ``cond_video``).
    """

    name: str
    expert: Any
    state: MoTExpertState


@dataclass
class MoTLayerIO:
    """Per-(layer, branch) Q/K/V + residual produced by ``mot_layer_qkv``.

    ``q``/``k``/``v`` are already head-reshaped, q/k-normed and RoPE-applied
    (all done inside the expert) and laid out as ``[B, H, S, Dh]``.  ``residual``
    is the pre-attention ``x`` for the residual path.  ``payload`` carries this
    layer's expert-private post-attention inputs (gates / shifts / scales /
    cross-attention context).
    """

    q: torch.Tensor
    k: torch.Tensor
    v: torch.Tensor
    residual: torch.Tensor
    payload: Any


# ---------------------------------------------------------------------------
# Interaction rules are supplied by the model; MoT only interprets token layouts.
# ---------------------------------------------------------------------------


_ALLOWED_MODES = ("full", "key_first_frame", "first_frame_causal", "per_frame_causal")


@dataclass(frozen=True)
class InteractionRule:
    """A single directed allow-rule ``query <- key`` under a given ``mode``.

    Group refs are global ``"<branch>.<local_group>"`` strings, e.g.
    ``"video.video"``, ``"action.action"``.  Modes:

      * ``full``: query group reads all valid key-group tokens.
      * ``key_first_frame``: query group reads only key tokens with
        ``frame_ids == 0``.
      * ``first_frame_causal``: first-frame query tokens (``frame_ids == 0``)
        read only first-frame keys; all other query tokens read all valid keys.
      * ``per_frame_causal``: a query reads key tokens from its own or earlier
        frames. Both groups must provide frame ids.
    """

    query: str
    key: str
    mode: str

    def __post_init__(self):
        if self.mode not in _ALLOWED_MODES:
            raise ValueError(f"InteractionRule mode must be one of {_ALLOWED_MODES}, got {self.mode!r}")
        for ref, label in ((self.query, "query"), (self.key, "key")):
            if not isinstance(ref, str) or "." not in ref:
                raise ValueError(f"InteractionRule {label} must be a '<branch>.<group>' ref, got {ref!r}")


@dataclass
class LayerKVCache:
    """Per-layer cached K/V for a single branch's exported token groups.

    K/V are ``[B, H, S_cache, Dh]`` and are post q/k-norm AND post-RoPE, matching
    the reference ``kv_cache.append({"k": k, "v": v})`` (caching pre-norm /
    pre-RoPE projections would change attention).
    """

    k: torch.Tensor
    v: torch.Tensor
    token_indices: torch.Tensor  # int64 [B, S_cache], original positions in branch seq


@dataclass
class BranchKVCache:
    """A cached key-branch usable as a read-only key side in ``forward_with_cache``.

    The policy compiler treats a cache as a key branch whose only group is
    ``exported_global_groups`` (e.g. ``("video.video",)``), so the same mask
    compilation code path handles both live branches and caches.
    """

    source_branch: str
    exported_groups: tuple[str, ...]
    exported_global_groups: tuple[str, ...]
    valid_mask: torch.Tensor  # bool [B, S_cache]
    frame_ids: torch.Tensor | None  # int64 [B, S_cache]
    layers: list[LayerKVCache] = field(default_factory=list)

    @property
    def num_layers(self) -> int:
        return len(self.layers)


# ---------------------------------------------------------------------------


class MoT(torch.nn.Module):
    """Shared backbone-agnostic MoT layer.

    Holds no experts and no trainable weights.  Experts are passed as runtime
    references inside :class:`MoTBranchBinding`.  Use :meth:`validate_expert` at
    framework construction time for early capability checks; per-call consistency
    (layer / head / head_dim / batch / dtype / device across bindings) is
    enforced in :meth:`forward`.
    """

    def __init__(self, *, gradient_checkpoint: bool = False):
        super().__init__()
        self.gradient_checkpoint = bool(gradient_checkpoint)

    # -- construction-time capability check --------------------------------

    @staticmethod
    def validate_expert(expert: Any) -> list[str]:
        """Return a list of missing-capability error strings (empty == OK).

        Called by the framework once experts exist to validate the
        "runtime capability validation at construction" requirement without the
        MoT having to own the experts.
        """
        problems: list[str] = []
        for prop in ("mot_num_layers", "mot_num_heads", "mot_head_dim"):
            if not hasattr(expert, prop):
                problems.append(f"missing property '{prop}'")
            else:
                try:
                    val = getattr(expert, prop)
                    int(val)
                except (TypeError, ValueError) as e:
                    problems.append(f"property '{prop}' not int-convertible: {e}")
        for meth in ("mot_layer_qkv", "mot_layer_post_attention", "mot_finalize"):
            if not callable(getattr(expert, meth, None)):
                problems.append(f"missing method '{meth}'")
        return problems

    # -- per-call consistency validation ----------------------------------

    @staticmethod
    def _validate_bindings(
        branches: Sequence[MoTBranchBinding],
    ) -> tuple[int, int, int, int, torch.device, torch.dtype]:
        if len(branches) == 0:
            raise ValueError("`branches` must contain at least one MoTBranchBinding.")
        names = [b.name for b in branches]
        if len(set(names)) != len(names):
            raise ValueError(f"Branch names must be unique within a forward, got {names}")

        first = branches[0]
        num_layers = int(first.expert.mot_num_layers)
        num_heads = int(first.expert.mot_num_heads)
        head_dim = int(first.expert.mot_head_dim)
        batch = int(first.state.tokens.shape[0])
        device = first.state.tokens.device
        dtype = first.state.tokens.dtype

        for b in branches[1:]:
            if int(b.expert.mot_num_layers) != num_layers:
                raise ValueError(
                    "All branches must share num_layers; "
                    f"got {num_layers} and {int(b.expert.mot_num_layers)}"
                )
            if int(b.expert.mot_num_heads) != num_heads:
                raise ValueError(
                    "All branches must share num_heads; " f"got {num_heads} and {int(b.expert.mot_num_heads)}"
                )
            if int(b.expert.mot_head_dim) != head_dim:
                raise ValueError(
                    "All branches must share head_dim; " f"got {head_dim} and {int(b.expert.mot_head_dim)}"
                )
            if int(b.state.tokens.shape[0]) != batch:
                raise ValueError(
                    "All branches must share batch size; " f"got {batch} and {int(b.state.tokens.shape[0])}"
                )
            if b.state.tokens.device != device or b.state.tokens.dtype != dtype:
                raise ValueError("All branch tokens must share device and dtype.")
        return num_layers, num_heads, head_dim, batch, device, dtype

    # -- SDPA wrapper -----------------------------------------------------

    def _mixed_attention(
        self,
        q_cat: torch.Tensor,
        k_cat: torch.Tensor,
        v_cat: torch.Tensor,
        attention_mask: torch.Tensor,
        num_heads: int,
    ) -> torch.Tensor:
        attn_mask = attention_mask.to(device=q_cat.device)

        def _forward(q, k, v):
            return scaled_dot_product_attention(q=q, k=k, v=v, num_heads=num_heads, ctx_mask=attn_mask)

        return _forward(q_cat, k_cat, v_cat)

    # -- public entry 1: full coupled forward -----------------------------

    def forward(
        self,
        branches: Sequence[MoTBranchBinding],
        attention_mask: torch.Tensor,
    ) -> dict[str, MoTExpertState]:
        """Run all branches through every layer with a shared mixed SDPA.

        Returns a dict ``{branch_name: updated MoTExpertState}``.  The state's
        ``tokens`` are the final (pre-finalize) hidden states and its
        ``payload`` / ``layout`` are carried through unchanged, so the framework
        can call ``expert.mot_finalize(out[name])`` directly.  This mirrors the
        reference returning ``tokens_all`` plus the per-expert pre-state used by
        ``post_dit`` / ``mot_finalize``.
        """
        num_layers, num_heads, _, _, _, _ = self._validate_bindings(branches)
        _check_attention_mask(attention_mask, branches, cache_branches=())

        # Mutable per-branch state; rebind the binding's state each layer.
        states: dict[str, MoTExpertState] = {b.name: b.state for b in branches}
        seq_lens = [int(b.state.tokens.shape[1]) for b in branches]
        names = [b.name for b in branches]
        experts = [b.expert for b in branches]

        for layer_idx in range(num_layers):
            payloads_before = [id(states[n].payload) for n in names]
            layer_states = tuple(states[name] for name in names)

            def coupled_layer(
                *tokens: torch.Tensor,
                _layer_idx: int = layer_idx,
                _layer_states: tuple[MoTExpertState, ...] = layer_states,
            ) -> tuple[torch.Tensor, ...]:
                current_states = tuple(
                    MoTExpertState(tokens=token, layout=state.layout, payload=state.payload)
                    for token, state in zip(tokens, _layer_states, strict=True)
                )
                ios = tuple(
                    expert.mot_layer_qkv(_layer_idx, state)
                    for expert, state in zip(experts, current_states, strict=True)
                )
                mixed = self._mixed_attention(
                    torch.cat([io.q for io in ios], dim=2),
                    torch.cat([io.k for io in ios], dim=2),
                    torch.cat([io.v for io in ios], dim=2),
                    attention_mask,
                    num_heads,
                )

                next_tokens = []
                start = 0
                for seq_len, io, expert, state in zip(seq_lens, ios, experts, current_states, strict=True):
                    end = start + seq_len
                    next_state = expert.mot_layer_post_attention(
                        _layer_idx,
                        mixed[:, :, start:end, :],
                        io,
                        state,
                    )
                    if id(next_state.payload) != id(state.payload):
                        raise RuntimeError(
                            "Expert rebound payload within a coupled MoT layer; "
                            "payload must remain read-only for gradient checkpointing."
                        )
                    next_tokens.append(next_state.tokens)
                    start = end
                return tuple(next_tokens)

            input_tokens = tuple(state.tokens for state in layer_states)
            if self.gradient_checkpoint and torch.is_grad_enabled():
                output_tokens = torch_checkpoint.checkpoint(
                    coupled_layer,
                    *input_tokens,
                    use_reentrant=False,
                )
            else:
                output_tokens = coupled_layer(*input_tokens)
            states = {
                name: MoTExpertState(tokens=token, layout=state.layout, payload=state.payload)
                for name, token, state in zip(names, output_tokens, layer_states, strict=True)
            }

            # Checkpoint closures require payload identity to remain stable.
            for n, before in zip(names, payloads_before, strict=True):
                if id(states[n].payload) != before:
                    raise RuntimeError(
                        f"Branch '{n}' rebound payload during layer {layer_idx}; "
                        "payload must be read-only within a forward (gradient-"
                        "checkpoint closure invariant)."
                    )

        return states

    # -- public entry 2: single-branch prefill + K/V cache export --------

    def prefill_branch_cache(
        self,
        branch: MoTBranchBinding,
        self_attention_mask: torch.Tensor,
        export_groups: tuple[str, ...],
    ) -> BranchKVCache:
        """Run a single branch once and cache per-layer K/V for ``export_groups``.

        Mirrors the reference ``prefill_video_cache``: each layer first advances
        the branch state using full-sequence self-attention, then K/V for the
        exported token groups (post q/k-norm, post-RoPE) are written to the
        cache.  For v1 Base ``export_groups=("video",)`` with a batch-uniform
        Wan layout, ``S_cache == S_branch`` and a fast index-select path is used.
        """
        expert = branch.expert
        num_layers = int(expert.mot_num_layers)
        num_heads = int(expert.mot_num_heads)
        head_dim = int(expert.mot_head_dim)
        if len(export_groups) == 0:
            raise ValueError("`export_groups` must be non-empty.")

        state = branch.state
        layout = state.layout
        global_groups = [f"{branch.name}.{g}" for g in export_groups]
        # exported token mask: union of the requested local groups within valid.
        export_mask = torch.zeros_like(layout.valid_mask)
        for g in export_groups:
            if g not in layout.groups:
                raise ValueError(
                    f"Branch '{branch.name}' has no local group '{g}'; "
                    f"available: {sorted(layout.groups.keys())}"
                )
            export_mask = export_mask | (layout.groups[g] & layout.valid_mask)

        # Batch-uniform fast path: every sample exports the same token positions.
        batch_uniform = bool(export_mask.all(dim=0).all().item())
        if batch_uniform:
            token_indices = torch.where(export_mask[0])[0]  # [S_cache]
            s_cache = int(token_indices.shape[0])
            cache_indices = token_indices.view(1, s_cache).expand(state.tokens.shape[0], -1)
        else:
            # Dense cache export requires the same token positions in every batch item.
            raise NotImplementedError("Cache export requires identical token positions across the batch.")

        _check_attention_mask(self_attention_mask, [branch], cache_branches=())

        layers: list[LayerKVCache] = []
        for layer_idx in range(num_layers):
            io = expert.mot_layer_qkv(layer_idx, state)
            mixed = self._mixed_attention(io.q, io.k, io.v, self_attention_mask, num_heads)
            state = expert.mot_layer_post_attention(layer_idx, mixed, io, state)
            # Cache post-norm, post-RoPE K/V for the exported tokens.
            # io.k/v are [B, H, S, Dh]; gather exported positions along dim=2.
            idx = cache_indices.to(io.k.device)
            gather_index = idx[:, None, :, None].expand(-1, num_heads, -1, head_dim)
            k_cached = torch.gather(io.k, 2, gather_index)
            v_cached = torch.gather(io.v, 2, gather_index)
            layers.append(LayerKVCache(k=k_cached, v=v_cached, token_indices=cache_indices))

        exported_valid = torch.gather(layout.valid_mask, 1, cache_indices.to(layout.valid_mask.device))
        exported_frame = None
        if layout.frame_ids is not None:
            exported_frame = torch.gather(layout.frame_ids, 1, cache_indices.to(layout.frame_ids.device))

        return BranchKVCache(
            source_branch=branch.name,
            exported_groups=tuple(export_groups),
            exported_global_groups=tuple(global_groups),
            valid_mask=exported_valid,
            frame_ids=exported_frame,
            layers=layers,
        )

    # -- public entry 3: active branches + cached key branches -----------

    def forward_with_cache(
        self,
        active_branches: Sequence[MoTBranchBinding],
        caches: Sequence[BranchKVCache],
        attention_mask: torch.Tensor,
    ) -> dict[str, MoTExpertState]:
        """Run ``active_branches`` with cached K/V prepended as read-only keys.

        Column order of the concatenated keys is ``[cache_0, ..., cache_{C-1},
        active_0, ..., active_{A-1}]``.  The mask passed in MUST have key columns
        in this exact order (see :func:`compile_attention_mask`).  Returns
        ``{active_name: updated MoTExpertState}`` for ``expert.mot_finalize``.
        """
        if len(caches) == 0:
            return self.forward(active_branches, attention_mask)

        num_layers, num_heads, _head_dim, _batch, _device, _dtype = self._validate_bindings(active_branches)
        for c in caches:
            if c.num_layers != num_layers:
                raise ValueError(
                    f"Cache '{c.source_branch}' has {c.num_layers} layers; " f"expected {num_layers}."
                )
        _check_attention_mask(attention_mask, list(active_branches), cache_branches=list(caches))

        states: dict[str, MoTExpertState] = {b.name: b.state for b in active_branches}
        seq_lens = [int(b.state.tokens.shape[1]) for b in active_branches]
        names = [b.name for b in active_branches]

        for layer_idx in range(num_layers):
            q_chunks: list[torch.Tensor] = []
            k_active_chunks: list[torch.Tensor] = []
            v_active_chunks: list[torch.Tensor] = []
            ios: list[MoTLayerIO] = []
            experts: list[Any] = []

            for b in active_branches:
                io = b.expert.mot_layer_qkv(layer_idx, states[b.name])
                ios.append(io)
                experts.append(b.expert)
                q_chunks.append(io.q)
                k_active_chunks.append(io.k)
                v_active_chunks.append(io.v)

            # Key order: caches first, then active.
            k_cat = torch.cat([c.layers[layer_idx].k for c in caches] + k_active_chunks, dim=2)
            v_cat = torch.cat([c.layers[layer_idx].v for c in caches] + v_active_chunks, dim=2)
            q_cat = torch.cat(q_chunks, dim=2)
            mixed = self._mixed_attention(q_cat, k_cat, v_cat, attention_mask, num_heads)

            start = 0
            for name, seq_len, io, expert in zip(names, seq_lens, ios, experts, strict=True):
                end = start + seq_len
                mixed_slice = mixed[:, :, start:end, :]
                states[name] = expert.mot_layer_post_attention(layer_idx, mixed_slice, io, states[name])
                start = end

        return states


# ---------------------------------------------------------------------------
# Mask shape validation
# ---------------------------------------------------------------------------


def _branch_seq_len(branch: MoTBranchBinding) -> int:
    return int(branch.state.tokens.shape[1])


def _cache_seq_len(cache: BranchKVCache) -> int:
    return int(cache.layers[0].k.shape[2])


def _check_attention_mask(
    attention_mask: torch.Tensor,
    branches: Sequence[MoTBranchBinding],
    *,
    cache_branches: Sequence[BranchKVCache],
) -> None:
    if attention_mask.ndim not in (2, 4):
        raise ValueError(
            f"`attention_mask` must be 2D [Sq, Sk] or 4D [B, 1, Sq, Sk], got {tuple(attention_mask.shape)}"
        )
    sq = sum(_branch_seq_len(b) for b in branches)
    sk = sum(_cache_seq_len(c) for c in cache_branches) + sum(_branch_seq_len(b) for b in branches)
    actual = tuple(attention_mask.shape)
    if attention_mask.ndim == 2:
        if actual != (sq, sk):
            raise ValueError(f"`attention_mask` shape {actual} != expected [Sq, Sk] = [{sq}, {sk}]")
    else:
        if actual[1:] != (1, sq, sk):
            raise ValueError(
                f"`attention_mask` shape {actual} != expected [B, 1, Sq, Sk] = "
                f"[{actual[0]}, 1, {sq}, {sk}]"
            )


# ---------------------------------------------------------------------------
# Mask compiler: resolved policy + layouts -> bool attention mask
# ---------------------------------------------------------------------------


def _branch_group_masks(branch: MoTBranchBinding) -> dict[str, torch.Tensor]:
    """Global-group -> bool [B, S] masks for a live branch."""
    device = branch.state.tokens.device
    return {f"{branch.name}.{gname}": gmask.to(device) for gname, gmask in branch.state.layout.groups.items()}


def _cache_group_masks(cache: BranchKVCache) -> dict[str, torch.Tensor]:
    """Global-group -> bool [B, S_cache] masks for a cached key branch.

    A cache exports one or more groups; every cached token belongs to the
    *first* exported group (this is all the mask compiler needs to match the
    ``key_first_frame`` / ``full`` rules for a cached video branch).
    """
    primary = cache.exported_global_groups[0]
    valid = cache.valid_mask.to(cache.valid_mask.device)
    return {primary: valid}


def _gather_global_groups(
    bindings: Sequence[Any], group_mask_getter
) -> tuple[dict[str, torch.Tensor], torch.Tensor, torch.Tensor, torch.Tensor]:
    """Concatenate per-binding group masks / valid / frame along the sequence dim.

    Returns ``(global_masks, valid[B, S_total], frame[B, S_total], S_total)``.
    ``global_masks`` maps each global group name to a bool ``[B, S_total]``
    tensor (only True over the binding(s) that define that group).
    """
    B = int(_first_batch(bindings))
    device = _first_device(bindings)
    # First pass: total sequence length across bindings.
    sizes = []
    for b in bindings:
        gmasks = group_mask_getter(b)
        sizes.append(int(next(iter(gmasks.values())).shape[1]))
    s_total = sum(sizes)

    global_masks: dict[str, torch.Tensor] = {}
    valid_parts: list[torch.Tensor] = []
    frame_parts: list[torch.Tensor] = []
    offset = 0
    for b, s in zip(bindings, sizes, strict=True):
        gmasks = group_mask_getter(b)
        # Each group mask is [B, s]; zero-pad it into the full [B, s_total]
        # sequence so global_masks[gname] is True only over this binding.
        for gname, gmask in gmasks.items():
            if gmask.shape != (B, s):
                raise ValueError(f"Group '{gname}' mask shape {tuple(gmask.shape)} != ({B}, {s})")
            padded = torch.zeros((B, s_total), dtype=torch.bool, device=device)
            padded[:, offset : offset + s] = gmask
            if gname in global_masks:
                global_masks[gname] = global_masks[gname] | padded
            else:
                global_masks[gname] = padded
        bvalid = torch.zeros((B, s), dtype=torch.bool, device=device)
        for gmask in gmasks.values():
            bvalid = bvalid | gmask
        valid_parts.append(bvalid)
        frame_parts.append(_binding_frame(b))
        offset += s
    valid = (
        torch.cat(valid_parts, dim=1) if valid_parts else torch.empty((B, 0), dtype=torch.bool, device=device)
    )
    frame = (
        torch.cat(frame_parts, dim=1) if frame_parts else torch.empty((B, 0), dtype=torch.long, device=device)
    )
    return global_masks, valid, frame, s_total


def _first_batch(bindings: Sequence[Any]) -> int:
    b = bindings[0]
    if isinstance(b, BranchKVCache):
        return int(b.valid_mask.shape[0])
    return int(b.state.tokens.shape[0])


def _first_device(bindings: Sequence[Any]) -> torch.device:
    b = bindings[0]
    if isinstance(b, BranchKVCache):
        return b.valid_mask.device
    return b.state.tokens.device


def _binding_frame(b: Any) -> torch.Tensor:
    if isinstance(b, BranchKVCache):
        B, S = b.valid_mask.shape
        dev = b.valid_mask.device
        if b.frame_ids is not None:
            return b.frame_ids.to(dev)
        return torch.full((B, S), -1, dtype=torch.long, device=dev)
    layout = b.state.layout
    B, S = b.state.tokens.shape[0], b.state.tokens.shape[1]
    dev = b.state.tokens.device
    if layout.frame_ids is not None:
        f = layout.frame_ids.to(dev)
        if f.shape != (B, S):
            raise ValueError(f"Branch '{b.name}' frame_ids shape {tuple(f.shape)} != ({B}, {S})")
        return f
    return torch.full((B, S), -1, dtype=torch.long, device=dev)


def compile_attention_mask(
    *,
    query_bindings: Sequence[MoTBranchBinding],
    key_bindings: Sequence[Any],
    rules: Sequence[InteractionRule],
    default_deny: bool = True,
) -> torch.Tensor:
    """Build a bool attention mask ``[Sq, Sk]`` (batch-uniform) or ``[B,1,Sq,Sk]``.

    ``key_bindings`` may contain :class:`MoTBranchBinding` (live key branches)
    and :class:`BranchKVCache` (read-only cached key branches).  The key column
    order is exactly ``list(key_bindings)``; for ``forward_with_cache`` callers
    must pass ``key_bindings=[*caches, *active_branches]`` to match the MoT's
    concatenation order.

    Structural safety:
      * every valid query token must have >=1 visible valid key;
      * the returned 2D mask is only used when every batch row is identical
        (no cross-sample leakage), otherwise a 4D per-batch mask is returned.
    """
    q_groups, q_valid, q_frame, s_q = _gather_global_groups(query_bindings, _branch_group_masks)

    def _key_group_masks(kb):
        if isinstance(kb, BranchKVCache):
            return _cache_group_masks(kb)
        return _branch_group_masks(kb)

    k_groups, k_valid, k_frame, s_k = _gather_global_groups(key_bindings, _key_group_masks)

    B = int(q_valid.shape[0])
    device = q_valid.device
    mask = torch.zeros((B, s_q, s_k), dtype=torch.bool, device=device)
    for rule in rules:
        if rule.query not in q_groups or rule.key not in k_groups:
            # Rules for absent optional token groups contribute no attention.
            continue
        q_sel = q_groups[rule.query] & q_valid  # [B, Sq]
        k_sel = k_groups[rule.key] & k_valid  # [B, Sk]
        if rule.mode == "full":
            allow = k_sel  # [B, Sk]
        elif rule.mode == "key_first_frame":
            allow = k_sel & (k_frame == 0)  # [B, Sk]
        elif rule.mode == "first_frame_causal":
            # First-frame query rows (frame==0) read only first-frame keys;
            # all other query rows read all valid keys of this group.
            k_ff = k_sel & (k_frame == 0)  # [B, Sk]
            q_ff = q_frame == 0  # [B, Sq]
            allow = torch.where(
                q_ff.unsqueeze(-1),
                k_ff.unsqueeze(1).expand(-1, s_q, -1),
                k_sel.unsqueeze(1).expand(-1, s_q, -1),
            )  # [B, Sq, Sk]
            mask = mask | (q_sel.unsqueeze(-1) & allow)
            continue
        elif rule.mode == "per_frame_causal":
            if (q_sel & (q_frame < 0)).any() or (k_sel & (k_frame < 0)).any():
                raise ValueError("per_frame_causal requires frame_ids on both query and key groups.")
            allow = k_sel.unsqueeze(1) & (k_frame.unsqueeze(1) <= q_frame.unsqueeze(-1))
            mask = mask | (q_sel.unsqueeze(-1) & allow)
            continue
        else:  # pragma: no cover - guarded by InteractionRule.__post_init__
            raise ValueError(f"Unsupported mode: {rule.mode}")
        mask = mask | (q_sel.unsqueeze(-1) & allow.unsqueeze(1))

    # --- structural safety ----------------------------------------------
    visible = mask.any(dim=-1)  # [B, Sq]
    bad = q_valid & ~visible
    if bad.any():
        n = int(bad.sum().item())
        raise ValueError(
            f"Structural mask safety: {n} valid query token(s) have no visible key; "
            "every valid query group needs at least one allow-rule."
        )

    # --- collapse to 2D when batch-uniform --------------------------------
    batch_uniform = bool((mask == mask[0:1]).all().item())
    if batch_uniform:
        return mask[0].contiguous()  # [Sq, Sk]
    return mask.unsqueeze(1).contiguous()  # [B, 1, Sq, Sk]
