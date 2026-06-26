import math

from nla_sycophancy.nla.explain import ExplanationSet, fve_weighted_dimensions


def _es(fves, floor=0.3):
    es = ExplanationSet(activation_id="a", fve_floor=floor)
    for i, f in enumerate(fves):
        es.texts.append(f"text{i}")
        es.mses.append(2 * (1 - f))  # not used here
        es.coss.append(f)
        es.fves.append(f)
    return es


def test_kept_mask_and_counts():
    es = _es([0.5, 0.1, 0.4], floor=0.3)
    assert es.kept_mask() == [True, False, True]
    assert es.n_kept == 2
    assert es.n_samples == 3


def test_fve_weights_normalized_over_kept():
    es = _es([0.5, 0.1, 0.4], floor=0.3)
    w = es.weights()
    assert math.isclose(sum(w), 1.0)
    assert w[1] == 0.0
    assert math.isclose(w[0], 0.5 / 0.9, rel_tol=1e-6)
    assert math.isclose(w[2], 0.4 / 0.9, rel_tol=1e-6)


def test_fve_weights_fallback_to_best_when_all_below_floor():
    es = _es([0.1, 0.25, 0.05], floor=0.3)
    w = es.weights()
    assert math.isclose(sum(w), 1.0)
    assert w[1] == 1.0  # best (highest fve) gets all the weight


def test_best():
    es = _es([0.1, 0.7, 0.4])
    text, fve = es.best()
    assert text == "text1" and fve == 0.7


def test_fve_weighted_dimensions():
    es = _es([0.5, 0.4], floor=0.3)  # weights 0.5556 / 0.4444
    dims = [
        {"D_agreement": 1.0, "D_beliefaware": 0.0},
        {"D_agreement": 0.0, "D_beliefaware": 1.0},
    ]
    agg = fve_weighted_dimensions(es, dims)
    assert math.isclose(agg["D_agreement"], 0.5 / 0.9, rel_tol=1e-6)
    assert math.isclose(agg["D_beliefaware"], 0.4 / 0.9, rel_tol=1e-6)
