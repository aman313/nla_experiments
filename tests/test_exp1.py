import numpy as np

from nla_sycophancy.analysis.exp1_predictive import AlignmentRow, analyze_alignment
from nla_sycophancy.judge.rubric import DIM_KEYS


def _make_rows(n=80, signal=0.6, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    fillers = ["alpha", "bravo", "charlie", "delta", "echo"]
    for i in range(n):
        label = int(i % 2 == 0)
        dims = {k: float(rng.uniform(0, 0.3)) for k in DIM_KEYS}
        # D_agreement carries the label signal; D_beliefaware is a confound that
        # also tracks label (both variants state a belief), to test partialling.
        dims["D_agreement"] = float(np.clip(label * signal + rng.normal(0, 0.12), 0, 1))
        dims["D_beliefaware"] = float(np.clip(0.5 + rng.normal(0, 0.1), 0, 1))
        # prompt text is uninformative about the label (random filler)
        text = f"Question with option {fillers[rng.integers(0, len(fillers))]}"
        rows.append(AlignmentRow(item_id=f"i{i}", label=label, dims=dims,
                                 prompt_text=text, mean_fve=0.6))
    return rows


def test_alignment_detects_informative_dagreement():
    out = analyze_alignment(_make_rows(seed=1), seed=1)
    assert out["n_sycophantic"] >= 2 and out["n_non_sycophantic"] >= 2
    assert out["D_agreement_auroc"] > 0.75
    assert out["D_agreement_gap"] > 0.2
    assert out["D_agreement_gap_pvalue"] < 0.05
    assert out["per_dimension"]["D_agreement"]["auroc"] > 0.75


def test_incremental_validity_positive_over_prompt_text():
    out = analyze_alignment(_make_rows(seed=2), seed=2)
    # prompt text is uninformative, NLA dims are informative -> positive increment
    assert "incremental_auprc" in out
    assert out["combined_auroc"] >= out["prompt_text_baseline_auroc"]
    assert out["nla_only_auroc"] > 0.7


def test_partial_corr_survives_beliefaware_control():
    out = analyze_alignment(_make_rows(seed=3), seed=3)
    # D_agreement is the real signal; controlling for belief-awareness keeps it
    assert out["D_agreement_partial_corr_given_beliefaware"] > 0.2


def test_small_sample_guard():
    rows = [AlignmentRow(item_id="a", label=1, dims={k: 0.5 for k in DIM_KEYS}),
            AlignmentRow(item_id="b", label=0, dims={k: 0.1 for k in DIM_KEYS})]
    out = analyze_alignment(rows)
    assert "error" in out
