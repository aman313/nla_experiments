"""Activation verbalizer (AV / actor) wrapper.

Thin wrapper over the vendored ``NLAClient``, which injects an activation as a
single token embedding into the actor's fixed prompt and decodes a natural
language explanation via an SGLang ``input_embeds`` request. The heavy lifting
(template, scale, neighbor checks) lives in the vendored single-file client; we
only add config validation and an N-sample convenience method.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np


class AVClient:
    """Wraps the vendored ``NLAClient`` actor."""

    def __init__(
        self,
        checkpoint_dir: str | Path,
        *,
        sglang_url: str = "http://localhost:30000",
        injection_scale_override: Optional[float] = None,
        device: str = "cpu",
    ):
        from nla_sycophancy.vendor.nla_inference import NLAClient

        self._client = NLAClient(
            str(checkpoint_dir),
            sglang_url=sglang_url,
            injection_scale_override=injection_scale_override,
            device=device,
        )
        self.cfg = self._client.cfg

    def verbalize(
        self,
        activation: np.ndarray,
        *,
        temperature: float = 1.0,
        max_new_tokens: int = 200,
        extract_explanation: bool = True,
    ) -> str:
        return self._client.generate(
            activation,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
            extract_explanation=extract_explanation,
        )

    def verbalize_n(
        self,
        activation: np.ndarray,
        n_samples: int,
        *,
        temperature: float = 1.0,
        max_new_tokens: int = 200,
    ) -> list[str]:
        """Sample ``n_samples`` AV explanations at temperature ``T`` (default 1.0)."""
        return [
            self._client.generate(
                activation, temperature=temperature, max_new_tokens=max_new_tokens
            )
            for _ in range(n_samples)
        ]
