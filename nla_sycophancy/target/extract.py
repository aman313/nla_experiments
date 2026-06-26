"""M0: residual-stream activation extraction from the target model.

Extracts the layer-``l`` residual stream (the *output* of transformer block
``l``, which HF exposes as ``hidden_states[l + 1]``) at chosen token positions.

We provide two equivalent paths:

- :func:`extract_residual_hidden_states` — uses ``output_hidden_states=True``;
  simplest and the reference for correctness.
- :class:`ResidualExtractor` — registers a forward hook on a single block so we
  don't pay to materialize every layer's hidden state when scanning many
  rollouts. A unit test asserts the two agree.

Position resolution (:func:`resolve_positions`) maps the pre-registered
:class:`~nla_sycophancy.io.schema.Position` set onto concrete token indices given
the prompt/answer token boundaries.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from nla_sycophancy.io.schema import Position


def extract_residual_hidden_states(model, input_ids, layer: int) -> np.ndarray:
    """Return the block-``layer`` residual stream for every token: ``[T, d]``.

    ``hidden_states[layer + 1]`` is the output of block ``layer`` (``[0]`` is the
    embedding layer). Operates on the first batch element.
    """
    import torch

    if not torch.is_tensor(input_ids):
        input_ids = torch.tensor(input_ids)
    if input_ids.ndim == 1:
        input_ids = input_ids.unsqueeze(0)
    with torch.no_grad():
        out = model(input_ids=input_ids.to(model.device), output_hidden_states=True)
    hs = out.hidden_states[layer + 1]  # [B, T, d]
    return hs[0].float().cpu().numpy()


@dataclass
class _HookState:
    captured: Optional["object"] = None


class ResidualExtractor:
    """Forward-hook based extractor for one transformer block.

    Usage::

        ex = ResidualExtractor(model, layer=20)
        with ex:
            model(input_ids=ids)
        vecs = ex.last()        # [T, d] numpy
    """

    def __init__(self, model, layer: int):
        self.model = model
        self.layer = layer
        self._state = _HookState()
        self._handle = None
        self._block = self._locate_block(model, layer)

    @staticmethod
    def _locate_block(model, layer: int):
        # Cover the common HF causal-LM layouts: ``model.model.layers`` (Llama/
        # Qwen/Mistral), ``model.model.language_model.layers`` (Gemma-3 wrapper),
        # and ``model.transformer.h`` (GPT-2-style).
        candidates = []
        inner = getattr(model, "model", model)
        if hasattr(inner, "layers"):
            candidates = inner.layers
        elif hasattr(inner, "language_model") and hasattr(inner.language_model, "layers"):
            candidates = inner.language_model.layers
        elif hasattr(inner, "h"):
            candidates = inner.h
        elif hasattr(model, "transformer") and hasattr(model.transformer, "h"):
            candidates = model.transformer.h
        assert candidates is not None and len(candidates) > layer, (
            f"could not locate block {layer} on {type(model).__name__}"
        )
        return candidates[layer]

    def _hook(self, _module, _inputs, output):
        # Decoder blocks return a tuple ``(hidden_state, ...)``; some return a
        # bare tensor. Capture the residual-stream hidden state either way.
        self._state.captured = output[0] if isinstance(output, tuple) else output

    def __enter__(self) -> "ResidualExtractor":
        self._handle = self._block.register_forward_hook(self._hook)
        return self

    def __exit__(self, *exc) -> None:
        if self._handle is not None:
            self._handle.remove()
            self._handle = None

    def last(self) -> np.ndarray:
        """Return the most recently captured residual stream: ``[T, d]``."""
        assert self._state.captured is not None, "no forward pass captured yet"
        return self._state.captured[0].float().cpu().numpy()


@dataclass(frozen=True)
class TokenBoundaries:
    """Token-index boundaries within a tokenized (prompt + answer) sequence.

    All indices are into the full token sequence fed to the target model.

    - ``user_end``: index of the last user-prompt token.
    - ``assist_start``: index of the first assistant/control token.
    - ``answer``: index of the emitted answer-option token.
    """

    user_end: int
    assist_start: int
    answer: int
    seq_len: int

    def __post_init__(self) -> None:
        assert 0 <= self.user_end < self.seq_len, "user_end out of range"
        assert 0 <= self.assist_start < self.seq_len, "assist_start out of range"
        assert 0 <= self.answer < self.seq_len, "answer out of range"


def resolve_positions(b: TokenBoundaries) -> dict[Position, int]:
    """Map pre-registered positions to concrete token indices.

    ``preans`` is the token immediately *before* the answer option — the
    decision point in a no-CoT/answer-only format (the primary position).
    """
    preans = max(b.answer - 1, 0)
    return {
        Position.USEREND: b.user_end,
        Position.ASSIST0: b.assist_start,
        Position.PREANS: preans,
        Position.ANS: b.answer,
    }
