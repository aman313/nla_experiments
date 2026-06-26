from nla_sycophancy.judge.rubric import (
    DIM_KEYS,
    build_judge_messages,
    parse_judge_response,
)


def test_messages_contain_all_dims():
    msgs = build_judge_messages("some explanation")
    blob = " ".join(m["content"] for m in msgs)
    for k in DIM_KEYS:
        assert k in blob
    assert "some explanation" in blob


def test_parse_clean_json():
    text = ('{"D_beliefaware": 0.9, "D_factaware": 0.2, "D_agreement": 0.8, '
            '"D_resist": 0.0, "D_commit": 0.5, "D_eval": 0.1}')
    d = parse_judge_response(text)
    assert d is not None
    assert d["D_agreement"] == 0.8 and d["D_beliefaware"] == 0.9
    assert set(d) == set(DIM_KEYS)


def test_parse_with_surrounding_prose():
    text = ('Sure! Here are my scores:\n{"D_beliefaware": 1.0, "D_factaware": 0.3, '
            '"D_agreement": 0.7, "D_resist": 0.1, "D_commit": 0.4, "D_eval": 0.0}\n'
            'Let me know if you need more.')
    d = parse_judge_response(text)
    assert d is not None and d["D_agreement"] == 0.7


def test_parse_clamps_and_defaults_missing():
    text = '{"D_agreement": 1.7, "D_beliefaware": -0.5}'
    d = parse_judge_response(text)
    assert d["D_agreement"] == 1.0  # clamped
    assert d["D_beliefaware"] == 0.0  # clamped
    assert d["D_eval"] == 0.0  # missing -> default


def test_parse_garbage_returns_none():
    assert parse_judge_response("I refuse to answer.") is None
