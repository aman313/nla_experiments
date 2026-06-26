import math

import pytest

from nla_sycophancy.io.schema import (
    AttritionReport,
    Item,
    Rollout,
)


def test_item_validation():
    it = Item(id="x", question="q?", options=("a", "b", "c", "d"), correct_idx=2)
    assert it.n_options == 4
    with pytest.raises(AssertionError):
        Item(id="x", question="q?", options=("a",), correct_idx=0)
    with pytest.raises(AssertionError):
        Item(id="x", question="q?", options=("a", "b"), correct_idx=5)


def test_rollout_option_probs_from_logprobs():
    r = Rollout(variant_id="v", sample_idx=0, answer_idx=0,
                option_logprobs=(0.0, math.log(3.0)))
    probs = r.option_probs
    assert math.isclose(sum(probs), 1.0)
    # second option has 3x the unnormalized weight
    assert math.isclose(probs[1] / probs[0], 3.0, rel_tol=1e-6)


def test_attrition_report_balances():
    rep = AttritionReport()
    rep.add("raw", 100)
    rep.add("kept", 60)
    d = rep.to_dict()
    assert d["raw"] == 100 and d["kept"] == 60
    assert "−40" in rep.render()
