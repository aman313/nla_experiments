import numpy as np

from nla_sycophancy.analysis import stats
from nla_sycophancy.nla.ar_client import direction_mse, fve_nrm, mse_from_cos


def test_mse_cos_relation():
    assert abs(mse_from_cos(1.0) - 0.0) < 1e-9
    assert abs(mse_from_cos(0.0) - 2.0) < 1e-9
    assert abs(mse_from_cos(0.9) - 0.2) < 1e-9


def test_direction_mse_matches_cos():
    rng = np.random.default_rng(0)
    a = rng.standard_normal(64)
    b = a + 0.1 * rng.standard_normal(64)
    mse, cos = direction_mse(a, b)
    assert abs(mse - mse_from_cos(cos)) < 1e-6


def test_fve_nrm():
    # mse equal to baseline variance => fve 0; perfect => fve 1
    assert abs(fve_nrm(0.7335, 0.7335) - 0.0) < 1e-9
    assert abs(fve_nrm(0.0, 0.7335) - 1.0) < 1e-9


def test_bootstrap_ci_covers_mean():
    rng = np.random.default_rng(1)
    data = rng.normal(5.0, 1.0, size=500)
    point, lo, hi = stats.bootstrap_ci(data, n_boot=2000, seed=2)
    assert lo < 5.0 < hi
    assert abs(point - data.mean()) < 1e-9


def test_permutation_test_detects_difference():
    rng = np.random.default_rng(3)
    x = rng.normal(0.0, 1.0, 200)
    y = rng.normal(1.0, 1.0, 200)
    obs, p = stats.permutation_test(
        x, y, lambda a, b: a.mean() - b.mean(), n_perm=2000, seed=4
    )
    assert p < 0.05


def test_permutation_test_null_is_nonsignificant():
    rng = np.random.default_rng(5)
    x = rng.normal(0.0, 1.0, 200)
    y = rng.normal(0.0, 1.0, 200)
    _, p = stats.permutation_test(
        x, y, lambda a, b: a.mean() - b.mean(), n_perm=2000, seed=6
    )
    assert p > 0.05


def test_incremental_auprc_positive_when_extra_informative():
    rng = np.random.default_rng(7)
    n = 400
    y = rng.integers(0, 2, n)
    base = rng.normal(0, 1, n)               # uninformative
    extra = y + rng.normal(0, 0.5, n)        # informative about label
    res = stats.cv_incremental_auprc(base, extra, y, seed=8)
    assert res["auroc_combined"] > res["auroc_base"]
    assert res["auprc_increment"] > 0.05


def test_partial_correlation_removes_confound():
    rng = np.random.default_rng(9)
    z = rng.normal(0, 1, 500)
    x = z + rng.normal(0, 0.1, 500)
    y = z + rng.normal(0, 0.1, 500)
    # x,y strongly correlate via z; partialling z out should shrink toward 0
    full = np.corrcoef(x, y)[0, 1]
    partial = stats.partial_correlation(x, y, z)
    assert full > 0.8
    assert abs(partial) < 0.3


def test_multiplicity_corrections():
    pvals = [0.001, 0.04, 0.5, 0.9]
    holm = stats.holm_bonferroni(pvals, alpha=0.05)
    bh = stats.benjamini_hochberg(pvals, alpha=0.05)
    assert holm[0] is True and holm[2] is False
    assert bh[0] is True and bh[3] is False


def test_auroc_auprc_perfect():
    y = [0, 0, 1, 1]
    s = [0.1, 0.2, 0.8, 0.9]
    assert abs(stats.auroc(y, s) - 1.0) < 1e-9
    assert abs(stats.auprc(y, s) - 1.0) < 1e-9
