"""M0: load + validate the NLA sidecar contract, and parse golden examples.

Two responsibilities:

1. :func:`load_and_validate` — thin wrapper over the vendored
   ``nla_inference.load_nla_config`` that asserts the ``nla_meta.yaml`` sidecar
   against a *live* tokenizer (injection token id, neighbor ids, prompt-template
   round-trip). This is the #1 silent-failure guard (wrong injection position →
   garbage AV) and needs only the tokenizer, not the GPU model.

2. :func:`parse_golden_example` — read a ``kitft/nla-inference`` worked-example
   transcript (``examples/*.txt``) into structured per-token records so the M0
   golden-MSE gate can assert reproduction within tolerance.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from nla_sycophancy.vendor import nla_inference as _nv

# Re-export the sidecar dataclass + loader so callers import from one place.
NLAConfig = _nv.NLAConfig


def load_and_validate(
    checkpoint_dir: str | Path,
    tokenizer: Any,
    injection_scale_override: Optional[float] = None,
) -> NLAConfig:
    """Load ``nla_meta.yaml`` and assert it against the live tokenizer.

    Raises ``AssertionError`` on any drift (tokenizer version, template,
    neighbor tokens). Returns the validated :class:`NLAConfig`.
    """
    return _nv.load_nla_config(
        checkpoint_dir, tokenizer, injection_scale_override=injection_scale_override
    )


# ─── Golden example transcript parsing ───────────────────────────────────────

_HEADER_RE = {
    "av_checkpoint": re.compile(r"AV checkpoint:\s*(\S+)"),
    "ar_checkpoint": re.compile(r"AR checkpoint:\s*(\S+)"),
    "extraction_model": re.compile(r"Extraction model:\s*(\S+)"),
    "d_model": re.compile(r"d_model:\s*(\d+)"),
}
_LAYER_RE = re.compile(r"Extraction:\s*layer\s*(\d+)")
_FVE_DENOM_RE = re.compile(r"fve_nrm:\s*1\s*-\s*mse_nrm\s*/\s*(\d+\.\d+)")
_USER_MSG_RE = re.compile(r"User message:\s*'(.*)'")
_TOKEN_LINE_RE = re.compile(
    r"^\[\s*(\d+)\]\s+(PROMPT|REPLY)\s+token=(.*?)\s+"
    r"\|\|v\|\|=([0-9.]+)\s+mse_nrm=([\-0-9.]+)\s+"
    r"cos=([\-0-9.]+)\s+fve_nrm=([\-0-9.]+)\s*$"
)


@dataclass(frozen=True)
class GoldenToken:
    index: int
    section: str          # "PROMPT" or "REPLY"
    token_repr: str       # the printed token='...' value
    raw_norm: float
    mse_nrm: float
    cos: float
    fve_nrm: float
    decode_text: str = "" # the AV's (greedy) explanation for this token


@dataclass(frozen=True)
class GoldenExample:
    av_checkpoint: str
    ar_checkpoint: str
    extraction_model: str
    layer: int
    d_model: int
    fve_denominator: float
    user_message: str
    tokens: tuple[GoldenToken, ...]

    def reply_tokens(self) -> tuple[GoldenToken, ...]:
        return tuple(t for t in self.tokens if t.section == "REPLY")

    def in_distribution_tokens(self, min_index: int = 24) -> tuple[GoldenToken, ...]:
        """Tokens the example flags as in-distribution (skips the system prompt).

        Early-sequence + system-prompt positions are under-sampled in training
        and decode unreliably; the transcript's own summary excludes them.
        """
        return tuple(t for t in self.tokens if t.index >= min_index)


def parse_golden_example(path: str | Path) -> GoldenExample:
    """Parse a ``kitft/nla-inference`` worked-example transcript file."""
    text = Path(path).read_text()

    def _h(key: str) -> str:
        m = _HEADER_RE[key].search(text)
        assert m, f"golden example missing header field {key!r}: {path}"
        return m.group(1)

    layer_m = _LAYER_RE.search(text)
    assert layer_m, f"golden example missing extraction layer: {path}"
    denom_m = _FVE_DENOM_RE.search(text)
    assert denom_m, f"golden example missing fve_nrm denominator: {path}"
    user_m = _USER_MSG_RE.search(text)
    assert user_m, f"golden example missing user message: {path}"

    tokens: list[GoldenToken] = []
    lines = text.splitlines()
    pending: Optional[dict] = None
    decode_lines: list[str] = []

    def _flush() -> None:
        if pending is None:
            return
        tokens.append(
            GoldenToken(
                index=pending["index"],
                section=pending["section"],
                token_repr=pending["token_repr"],
                raw_norm=pending["raw_norm"],
                mse_nrm=pending["mse_nrm"],
                cos=pending["cos"],
                fve_nrm=pending["fve_nrm"],
                decode_text="\n".join(decode_lines).strip(),
            )
        )

    for line in lines:
        m = _TOKEN_LINE_RE.match(line)
        if m:
            _flush()
            idx, section, tok_repr, norm, mse, cos, fve = m.groups()
            pending = {
                "index": int(idx),
                "section": section,
                "token_repr": tok_repr,
                "raw_norm": float(norm),
                "mse_nrm": float(mse),
                "cos": float(cos),
                "fve_nrm": float(fve),
            }
            decode_lines = []
        elif pending is not None:
            # decode text is indented; stop collecting at a section rule line
            if line.startswith(("─", "═", "§")) or line.lstrip().startswith(
                ("4.", "5.", "SUMMARY")
            ):
                _flush()
                pending = None
                decode_lines = []
            elif line.strip():
                decode_lines.append(line.strip())
    _flush()
    assert tokens, f"golden example parsed 0 per-token rows: {path}"

    return GoldenExample(
        av_checkpoint=_h("av_checkpoint"),
        ar_checkpoint=_h("ar_checkpoint"),
        extraction_model=_h("extraction_model"),
        layer=int(layer_m.group(1)),
        d_model=int(_h("d_model")),
        fve_denominator=float(denom_m.group(1)),
        user_message=user_m.group(1),
        tokens=tuple(tokens),
    )
