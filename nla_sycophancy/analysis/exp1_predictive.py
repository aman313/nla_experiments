"""M2/Exp-1: how aligned are NLA explanations with the counterfactual label?

Within the incorrect-belief slice (prompt structurally constant), we ask whether
the FVE-weighted NLA judge dimensions at ``t_preans`` distinguish *sycophantic*
from *non-sycophantic* items, and whether they do so **beyond a prompt-text-only
baseline** (incremental validity / non-leakage).

Primary read-outs:
- AUROC/AUPRC of ``D_agreement`` alone (the target construct).
- AUROC of an NLA-only logistic model (all dimensions).
- Incremental AUPRC/AUROC of (prompt-text + NLA) over prompt-text alone.
- Partial correlation of ``D_agreement`` with the label, controlling for
  ``D_beliefaware`` (does the agreement signal carry unique variance beyond
  "the model read the user's stated belief"?).
- Mean ``D_agreement`` gap (sycophantic − non) with a permutation test.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from nla_sycophancy.analysis import stats
from nla_sycophancy.judge.rubric import DIM_KEYS


@dataclass
class AlignmentRow:
    item_id: str
    label: int                 # 1 = sycophantic, 0 = non-sycophantic
    dims: dict[str, float]     # FVE-weighted NLA judge dimensions
    prompt_text: str = ""      # incorrect-belief prompt (M0 baseline features)
    mean_fve: float = 0.0


def _dim_matrix(rows: Sequence[AlignmentRow]) -> np.ndarray:
    return np.array([[r.dims.get(k, 0.0) for k in DIM_KEYS] for r in rows], float)


def analyze_alignment(rows: Sequence[AlignmentRow], *, seed: int = 0) -> dict:
    """Compute the NLA-vs-counterfactual alignment metrics."""
    y = np.array([r.label for r in rows], int)
    n_pos, n_neg = int(y.sum()), int((1 - y).sum())
    out: dict = {
        "n_items": len(rows),
        "n_sycophantic": n_pos,
        "n_non_sycophantic": n_neg,
        "base_rate": round(float(y.mean()), 4) if len(y) else 0.0,
        "mean_fve": round(float(np.mean([r.mean_fve for r in rows])), 4) if rows else 0.0,
    }
    if n_pos < 2 or n_neg < 2:
        out["error"] = "need >=2 items per class for alignment metrics"
        return out

    X = _dim_matrix(rows)

    # 1) per-dimension univariate alignment (AUROC + mean gap).
    per_dim = {}
    for j, k in enumerate(DIM_KEYS):
        col = X[:, j]
        roc = stats.auroc(y, col) if len(np.unique(col)) > 1 else 0.5
        gap = float(col[y == 1].mean() - col[y == 0].mean())
        per_dim[k] = {"auroc": round(roc, 4), "mean_syco": round(float(col[y == 1].mean()), 4),
                      "mean_non": round(float(col[y == 0].mean()), 4), "gap": round(gap, 4)}
    out["per_dimension"] = per_dim

    # 2) D_agreement headline (AUROC/AUPRC + permutation test on the gap).
    da = X[:, DIM_KEYS.index("D_agreement")]
    out["D_agreement_auroc"] = round(stats.auroc(y, da), 4)
    out["D_agreement_auprc"] = round(stats.auprc(y, da), 4)
    obs, p = stats.permutation_test(
        da[y == 1], da[y == 0], lambda a, b: a.mean() - b.mean(),
        n_perm=5000, seed=seed, alternative="greater",
    )
    out["D_agreement_gap"] = round(float(obs), 4)
    out["D_agreement_gap_pvalue"] = round(float(p), 4)

    # 3) partial correlation: D_agreement vs label controlling for D_beliefaware.
    db = X[:, DIM_KEYS.index("D_beliefaware")]
    out["D_agreement_corr_raw"] = round(
        float(np.corrcoef(da, y)[0, 1]) if len(np.unique(da)) > 1 else 0.0, 4
    )
    out["D_agreement_partial_corr_given_beliefaware"] = round(
        stats.partial_correlation(da, y.astype(float), db), 4
    )

    # 4) multivariate models (cross-validated). NLA-only and prompt-text baseline,
    #    plus the incremental validity of prompt-text + NLA over prompt-text alone.
    n_splits = min(5, n_pos, n_neg)
    if n_splits >= 2:
        nla_only = stats.cv_incremental_auprc(
            np.zeros((len(y), 1)), X, y, n_splits=n_splits, seed=seed
        )
        out["nla_only_auroc"] = round(nla_only["auroc_combined"], 4)
        out["nla_only_auprc"] = round(nla_only["auprc_combined"], 4)

        texts = [r.prompt_text for r in rows]
        if any(texts):
            from sklearn.feature_extraction.text import TfidfVectorizer

            tfidf = TfidfVectorizer(max_features=500, ngram_range=(1, 2)).fit_transform(
                texts
            ).toarray()
            inc = stats.cv_incremental_auprc(tfidf, X, y, n_splits=n_splits, seed=seed)
            out["prompt_text_baseline_auroc"] = round(inc["auroc_base"], 4)
            out["prompt_text_baseline_auprc"] = round(inc["auprc_base"], 4)
            out["combined_auroc"] = round(inc["auroc_combined"], 4)
            out["combined_auprc"] = round(inc["auprc_combined"], 4)
            out["incremental_auroc"] = round(inc["auroc_increment"], 4)
            out["incremental_auprc"] = round(inc["auprc_increment"], 4)

    out["interpretation"] = _interpret(out)
    return out


def _interpret(out: dict) -> str:
    da = out.get("D_agreement_auroc", 0.5)
    inc = out.get("incremental_auprc")
    pc = out.get("D_agreement_partial_corr_given_beliefaware", 0.0)
    p = out.get("D_agreement_gap_pvalue", 1.0)
    parts = [f"D_agreement AUROC={da:.3f} (p={p:.3f} for syco>non gap)."]
    if inc is not None:
        if inc > 0.02:
            parts.append(f"NLA adds incremental AUPRC={inc:+.3f} over prompt text "
                         f"(evidence beyond leakage).")
        else:
            parts.append(f"NLA incremental AUPRC={inc:+.3f} over prompt text "
                         f"(little/no signal beyond prompt — consistent with leakage).")
    parts.append(f"D_agreement partial corr | belief-aware = {pc:+.3f}.")
    return " ".join(parts)
