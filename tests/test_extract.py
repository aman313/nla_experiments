"""Tier-0 extraction tests on a tiny randomly-initialized HF model (CPU)."""

import numpy as np
import pytest

from nla_sycophancy.io.schema import Position
from nla_sycophancy.target.extract import (
    ResidualExtractor,
    TokenBoundaries,
    extract_residual_hidden_states,
    resolve_positions,
)

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")


def _tiny_model():
    from transformers import LlamaConfig, LlamaForCausalLM

    cfg = LlamaConfig(
        vocab_size=64, hidden_size=32, intermediate_size=64,
        num_hidden_layers=4, num_attention_heads=4, num_key_value_heads=4,
        max_position_embeddings=64,
    )
    torch.manual_seed(0)
    model = LlamaForCausalLM(cfg).eval()
    return model


def test_hook_matches_output_hidden_states():
    model = _tiny_model()
    ids = torch.randint(0, 64, (1, 12))
    layer = 2

    ref = extract_residual_hidden_states(model, ids, layer)

    ex = ResidualExtractor(model, layer=layer)
    with ex:
        with torch.no_grad():
            model(input_ids=ids)
    hook = ex.last()

    assert ref.shape == (12, 32)
    assert hook.shape == ref.shape
    np.testing.assert_allclose(hook, ref, rtol=1e-4, atol=1e-4)


def test_extractor_handle_removed_after_context():
    model = _tiny_model()
    ex = ResidualExtractor(model, layer=1)
    with ex:
        pass
    assert ex._handle is None


def test_resolve_positions():
    b = TokenBoundaries(user_end=10, assist_start=12, answer=20, seq_len=21)
    pos = resolve_positions(b)
    assert pos[Position.USEREND] == 10
    assert pos[Position.ASSIST0] == 12
    assert pos[Position.PREANS] == 19  # immediately before the answer
    assert pos[Position.ANS] == 20


def test_token_boundaries_validation():
    with pytest.raises(AssertionError):
        TokenBoundaries(user_end=30, assist_start=2, answer=5, seq_len=10)
