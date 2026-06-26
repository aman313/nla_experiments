"""Activation reconstructor (AR / critic) wrapper + reconstruction-fidelity math.

The pure-math helpers (:func:`direction_mse`, :func:`mse_from_cos`,
:func:`fve_nrm`) are unit-testable on CPU with no model. :class:`ARClient`
lazily wraps the vendored ``NLACritic`` for GPU reconstruction/scoring.

Convention (see kitft/nla-inference README): both predicted and gold vectors are
L2-normalized to ``mse_scale = sqrt(d_model)`` before the MSE, so

    MSE = 2 * (1 - cos),  range [0, 4],  orthogonal -> 2.

FVE (fraction of variance explained, normalized) compares the reconstruction MSE
to the predict-the-mean baseline variance of the training distribution:

    fve_nrm = 1 - mse_nrm / Var(v_nrm)_train.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Optional

import numpy as np


def mse_from_cos(cos: float) -> float:
    """Direction MSE from cosine similarity under L2-to-sqrt(d) normalization."""
    return 2.0 * (1.0 - cos)


def cos_from_mse(mse: float) -> float:
    return 1.0 - mse / 2.0


def fve_nrm(mse: float, denominator: float) -> float:
    """Normalized fraction-of-variance-explained.

    ``denominator`` is Var(v_nrm) of the training distribution (the
    predict-the-mean baseline MSE). fve=0 means "no better than the mean";
    fve=1 means perfect reconstruction.
    """
    assert denominator > 0, "fve denominator (training variance) must be > 0"
    return 1.0 - mse / denominator


def direction_mse(
    pred: np.ndarray, gold: np.ndarray, mse_scale: Optional[float] = None
) -> tuple[float, float]:
    """Return ``(mse, cos)`` for two vectors, both L2-normalized to ``mse_scale``.

    If ``mse_scale`` is None, uses sqrt(d_model). Equivalent to the vendored
    ``NLACritic.score`` math, factored out for testing without a model.
    """
    pred = np.asarray(pred, dtype=np.float64)
    gold = np.asarray(gold, dtype=np.float64)
    assert pred.shape == gold.shape, f"shape mismatch {pred.shape} vs {gold.shape}"
    d = pred.shape[-1]
    s = math.sqrt(d) if mse_scale is None else float(mse_scale)
    pn = pred / max(np.linalg.norm(pred), 1e-12) * s
    gn = gold / max(np.linalg.norm(gold), 1e-12) * s
    mse = float(((pn - gn) ** 2).mean())
    cos = float((pn @ gn) / (np.linalg.norm(pn) * np.linalg.norm(gn)))
    return mse, cos


class ARClient:
    """Lazy wrapper over the vendored ``NLACritic`` (GPU)."""

    def __init__(self, checkpoint_dir: str | Path, *, device: str = "cuda:0"):
        from nla_sycophancy.vendor.nla_inference import NLACritic

        self._critic = NLACritic(str(checkpoint_dir), device=device)
        self.mse_scale = self._critic.mse_scale

    def reconstruct(self, explanation: str) -> np.ndarray:
        return self._critic.reconstruct(explanation).numpy()

    def score(self, explanation: str, original: np.ndarray) -> tuple[float, float]:
        """Return ``(mse_nrm, cos)`` of reconstructing ``explanation`` vs original."""
        return self._critic.score(explanation, original)
