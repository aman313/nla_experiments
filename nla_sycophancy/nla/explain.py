"""M2: NLA explanation generation + reconstruction-fidelity gating.

For each extracted activation we sample ``n_av`` AV explanations (T=1), score
each with the AR critic (FVE / MSE), and gate/weight by reconstruction fidelity.
The kept explanations + their FVE weights are handed to the judge (see
:mod:`judge.grade`), whose per-dimension scores are then FVE-weighted into a
single feature vector per (item, position).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np

from nla_sycophancy.nla.ar_client import fve_nrm


@dataclass
class ExplanationSet:
    """All AV samples for one activation, with AR fidelity per sample."""

    activation_id: str
    texts: list[str] = field(default_factory=list)
    mses: list[float] = field(default_factory=list)
    coss: list[float] = field(default_factory=list)
    fves: list[float] = field(default_factory=list)
    fve_floor: float = 0.0

    @property
    def n_samples(self) -> int:
        return len(self.texts)

    def kept_mask(self) -> list[bool]:
        return [f >= self.fve_floor for f in self.fves]

    @property
    def n_kept(self) -> int:
        return sum(self.kept_mask())

    def best(self) -> Optional[tuple[str, float]]:
        """Best-of-N explanation by FVE."""
        if not self.fves:
            return None
        i = int(np.argmax(self.fves))
        return self.texts[i], self.fves[i]

    def weights(self) -> list[float]:
        """FVE weights over kept samples (clamped at 0), normalized to sum 1.

        If nothing passes the floor, fall back to the single best sample so the
        item is never silently dropped (its low FVE is recorded for filtering).
        """
        mask = self.kept_mask()
        w = [max(f, 0.0) if m else 0.0 for f, m in zip(self.fves, mask)]
        s = sum(w)
        if s <= 0:
            b = self.best()
            if b is None:
                return [0.0] * self.n_samples
            out = [0.0] * self.n_samples
            out[int(np.argmax(self.fves))] = 1.0
            return out
        return [x / s for x in w]


def explain_activation(
    av,
    ar,
    activation: np.ndarray,
    *,
    activation_id: str = "",
    n_av: int = 8,
    temperature: float = 1.0,
    max_new_tokens: int = 160,
    fve_floor: float = 0.3,
    fve_denominator: float = 0.7335,
) -> ExplanationSet:
    """Sample ``n_av`` AV explanations and score each with the AR critic."""
    texts = av.verbalize_batch(
        [activation] * n_av, temperature=temperature, max_new_tokens=max_new_tokens
    )
    es = ExplanationSet(activation_id=activation_id, fve_floor=fve_floor)
    for t in texts:
        mse, cos = ar.score(t, activation)
        es.texts.append(t)
        es.mses.append(mse)
        es.coss.append(cos)
        es.fves.append(fve_nrm(mse, fve_denominator))
    return es


def fve_weighted_dimensions(
    es: ExplanationSet,
    per_sample_dims: Sequence[dict[str, float]],
) -> dict[str, float]:
    """FVE-weighted mean of judge dimensions over the AV samples.

    ``per_sample_dims[i]`` is the judge's dimension dict for ``es.texts[i]``.
    """
    assert len(per_sample_dims) == es.n_samples, "dim count != sample count"
    weights = es.weights()
    keys = sorted({k for d in per_sample_dims for k in d})
    out: dict[str, float] = {}
    for k in keys:
        out[k] = float(sum(w * d.get(k, 0.0) for w, d in zip(weights, per_sample_dims)))
    return out
