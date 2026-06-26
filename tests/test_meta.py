from pathlib import Path

import pytest
import yaml

from nla_sycophancy.nla import meta

GOLDEN = (
    Path(__file__).resolve().parents[1]
    / "nla_sycophancy" / "vendor" / "examples" / "qwen7b_layer20_step4200.txt"
)

INJ_ID, LEFT_ID, RIGHT_ID, INJ_CHAR = 149705, 29, 522, "\u320e"


class FakeTokenizer:
    """Minimal tokenizer satisfying the load_nla_config contract."""

    unk_token_id = 0

    def encode(self, text, add_special_tokens=False):
        if text == INJ_CHAR:
            return [INJ_ID]
        return [ord(c) for c in text]

    def apply_chat_template(self, messages, tokenize=True, add_generation_prompt=True):
        # ... <concept>㈎</concept> ... -> [.., LEFT, INJ, RIGHT, ..]
        return [101, LEFT_ID, INJ_ID, RIGHT_ID, 102]


def _write_meta(tmp_path: Path, **overrides) -> Path:
    m = {
        "kind": "nla_model",
        "d_model": 3584,
        "extraction": {"injection_scale": 150.0},
        "tokens": {
            "injection_char": INJ_CHAR,
            "injection_token_id": INJ_ID,
            "injection_left_neighbor_id": LEFT_ID,
            "injection_right_neighbor_id": RIGHT_ID,
        },
        "prompt_templates": {"av": "<concept>{injection_char}</concept>"},
    }
    m.update(overrides)
    p = tmp_path / "nla_meta.yaml"
    p.write_text(yaml.safe_dump(m))
    return tmp_path


def test_load_and_validate_ok(tmp_path):
    ckpt = _write_meta(tmp_path)
    cfg = meta.load_and_validate(ckpt, FakeTokenizer())
    assert cfg.d_model == 3584
    assert cfg.injection_token_id == INJ_ID
    assert cfg.injection_scale == 150.0


def test_load_and_validate_detects_token_drift(tmp_path):
    ckpt = _write_meta(tmp_path,
                       tokens={"injection_char": INJ_CHAR,
                               "injection_token_id": 999999,  # wrong
                               "injection_left_neighbor_id": LEFT_ID,
                               "injection_right_neighbor_id": RIGHT_ID})
    with pytest.raises(AssertionError):
        meta.load_and_validate(ckpt, FakeTokenizer())


def test_load_and_validate_detects_neighbor_drift(tmp_path):
    ckpt = _write_meta(tmp_path,
                       tokens={"injection_char": INJ_CHAR,
                               "injection_token_id": INJ_ID,
                               "injection_left_neighbor_id": 7,  # wrong
                               "injection_right_neighbor_id": RIGHT_ID})
    with pytest.raises(AssertionError):
        meta.load_and_validate(ckpt, FakeTokenizer())


def test_parse_golden_example_structure():
    g = meta.parse_golden_example(GOLDEN)
    assert g.extraction_model == "Qwen/Qwen2.5-7B-Instruct"
    assert g.layer == 20
    assert g.d_model == 3584
    assert abs(g.fve_denominator - 0.7335) < 1e-9
    assert g.user_message == "What are you hiding?"
    assert len(g.tokens) > 50
    assert any(t.section == "REPLY" for t in g.tokens)


def test_golden_fve_relation_holds():
    g = meta.parse_golden_example(GOLDEN)
    # fve_nrm == 1 - mse_nrm / denominator for every parsed token
    for t in g.tokens[:40]:
        expected = 1.0 - t.mse_nrm / g.fve_denominator
        assert abs(expected - t.fve_nrm) < 5e-3


def test_golden_indistribution_filter_skips_system_prompt():
    g = meta.parse_golden_example(GOLDEN)
    ind = g.in_distribution_tokens(min_index=24)
    assert all(t.index >= 24 for t in ind)
    assert len(ind) < len(g.tokens)
