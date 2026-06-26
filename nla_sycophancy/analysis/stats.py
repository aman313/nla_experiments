"""Statistics utilities: bootstrap CIs, permutation tests, AUPRC/AUROC,
incremental validity, multiplicity correction, partial correlation.

Kept dependency-light (numpy + scikit-learn + scipy) and unit-tested on
synthetic data with known ground truth.
"""

from __future__ import annotations

from typing import Callable, Optional, Sequence

import numpy as np


def bootstrap_ci(
    values: Sequence[float],
    statistic: Callable[[np.ndarray], float] = np.mean,
    *,
    n_boot: int = 10000,
    alpha: float = 0.05,
    seed: int = 0,
) -> tuple[float, float, float]:
    """Percentile bootstrap CI. Returns ``(point, lo, hi)``."""
    rng = np.random.default_rng(seed)
    arr = np.asarray(values, dtype=np.float64)
    assert arr.ndim == 1 and arr.size > 0
    point = float(statistic(arr))
    idx = rng.integers(0, arr.size, size=(n_boot, arr.size))
    boot = np.array([statistic(arr[i]) for i in idx])
    lo = float(np.quantile(boot, alpha / 2))
    hi = float(np.quantile(boot, 1 - alpha / 2))
    return point, lo, hi


def permutation_test(
    x: Sequence[float],
    y: Sequence[float],
    statistic: Callable[[np.ndarray, np.ndarray], float],
    *,
    n_perm: int = 10000,
    seed: int = 0,
    alternative: str = "two-sided",
) -> tuple[float, float]:
    """Label-permutation test over the concatenation of ``x`` and ``y``.

    ``statistic(x, y)`` must be a scalar (e.g. difference in means). Returns
    ``(observed, p_value)``.
    """
    rng = np.random.default_rng(seed)
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    obs = statistic(x, y)
    pooled = np.concatenate([x, y])
    nx = x.size
    count = 0
    for _ in range(n_perm):
        perm = rng.permutation(pooled)
        stat = statistic(perm[:nx], perm[nx:])
        if alternative == "two-sided":
            count += abs(stat) >= abs(obs)
        elif alternative == "greater":
            count += stat >= obs
        else:
            count += stat <= obs
    p = (count + 1) / (n_perm + 1)
    return float(obs), float(p)


def auroc(y_true: Sequence[int], scores: Sequence[float]) -> float:
    from sklearn.metrics import roc_auc_score

    return float(roc_auc_score(y_true, scores))


def auprc(y_true: Sequence[int], scores: Sequence[float]) -> float:
    from sklearn.metrics import average_precision_score

    return float(average_precision_score(y_true, scores))


def recall_at_fpr(y_true: Sequence[int], scores: Sequence[float], fpr: float = 0.01) -> float:
    """Recall (TPR) at a fixed false-positive rate."""
    from sklearn.metrics import roc_curve

    fprs, tprs, _ = roc_curve(y_true, scores)
    ok = fprs <= fpr
    return float(tprs[ok].max()) if ok.any() else 0.0


def cv_incremental_auprc(
    X_base: np.ndarray,
    X_extra: np.ndarray,
    y: Sequence[int],
    *,
    n_splits: int = 5,
    seed: int = 0,
) -> dict[str, float]:
    """Cross-validated incremental AUPRC/AUROC of an augmented model over base.

    Trains logistic regression on (a) base features and (b) base+extra features
    with stratified K-fold CV, returns out-of-fold AUPRC/AUROC for each and the
    increment (combined - base). This is the Exp-1 primary-metric machinery
    (M0 = base prompt-text features; M2 = base + NLA features).
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold
    from sklearn.preprocessing import StandardScaler

    X_base = np.asarray(X_base, dtype=np.float64)
    X_extra = np.asarray(X_extra, dtype=np.float64)
    y = np.asarray(y, dtype=int)
    if X_base.ndim == 1:
        X_base = X_base[:, None]
    if X_extra.ndim == 1:
        X_extra = X_extra[:, None]
    X_comb = np.hstack([X_base, X_extra])

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    oof_base = np.zeros(len(y))
    oof_comb = np.zeros(len(y))
    for tr, te in skf.split(X_base, y):
        for X, oof in ((X_base, oof_base), (X_comb, oof_comb)):
            sc = StandardScaler().fit(X[tr])
            clf = LogisticRegression(max_iter=1000)
            clf.fit(sc.transform(X[tr]), y[tr])
            oof[te] = clf.predict_proba(sc.transform(X[te]))[:, 1]

    base_ap, comb_ap = auprc(y, oof_base), auprc(y, oof_comb)
    base_roc, comb_roc = auroc(y, oof_base), auroc(y, oof_comb)
    return {
        "auprc_base": base_ap,
        "auprc_combined": comb_ap,
        "auprc_increment": comb_ap - base_ap,
        "auroc_base": base_roc,
        "auroc_combined": comb_roc,
        "auroc_increment": comb_roc - base_roc,
    }


def partial_correlation(
    x: Sequence[float],
    y: Sequence[float],
    covars: np.ndarray,
) -> float:
    """Partial correlation of ``x`` and ``y`` controlling for ``covars``.

    Residualize both x and y on covars (with intercept) via least squares, then
    correlate the residuals. Used for D_agreement's unique variance over the
    prompt-reading dimension(s).
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    covars = np.asarray(covars, dtype=np.float64)
    if covars.ndim == 1:
        covars = covars[:, None]
    A = np.hstack([np.ones((covars.shape[0], 1)), covars])

    def _resid(v: np.ndarray) -> np.ndarray:
        beta, *_ = np.linalg.lstsq(A, v, rcond=None)
        return v - A @ beta

    rx, ry = _resid(x), _resid(y)
    denom = np.linalg.norm(rx) * np.linalg.norm(ry)
    return float(rx @ ry / denom) if denom > 0 else 0.0


def holm_bonferroni(pvalues: Sequence[float], alpha: float = 0.05) -> list[bool]:
    """Holm-Bonferroni step-down. Returns a reject/accept mask in input order."""
    p = np.asarray(pvalues, dtype=np.float64)
    order = np.argsort(p)
    m = len(p)
    reject = np.zeros(m, dtype=bool)
    for rank, idx in enumerate(order):
        thresh = alpha / (m - rank)
        if p[idx] <= thresh:
            reject[idx] = True
        else:
            break
    return reject.tolist()


def benjamini_hochberg(pvalues: Sequence[float], alpha: float = 0.05) -> list[bool]:
    """Benjamini-Hochberg FDR control. Returns a reject mask in input order."""
    p = np.asarray(pvalues, dtype=np.float64)
    m = len(p)
    order = np.argsort(p)
    reject = np.zeros(m, dtype=bool)
    max_k = -1
    for rank, idx in enumerate(order, start=1):
        if p[idx] <= alpha * rank / m:
            max_k = rank
    if max_k > 0:
        for rank, idx in enumerate(order, start=1):
            if rank <= max_k:
                reject[idx] = True
    return reject.tolist()
