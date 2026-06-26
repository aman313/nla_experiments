"""M1: target-model rollouts with per-option logprobs.

Two ways to estimate the per-option answer distribution for an answer-only MCQ:

- :func:`score_options` (preferred, low-variance): one forward pass; read the
  next-token logprob of each option letter at the assistant generation point.
- :func:`sample_answers`: sample N short completions and parse the letter
  (empirical distribution; also yields refusal/format-error rate).

:func:`parse_answer_letter` (the parser) is pure-Python and unit-tested on CPU.
Generation helpers require a HF model and run on GPU (Modal).
"""

from __future__ import annotations

import re
from typing import Optional, Sequence

from nla_sycophancy.data.variants import LETTERS
from nla_sycophancy.io.schema import Rollout

_LETTER_RE = re.compile(r"\b([A-H])\b")


def parse_answer_letter(text: str, n_options: int) -> Optional[int]:
    """Parse the chosen option index from a model completion.

    Accepts a bare letter, ``A)``, ``(A)``, ``Answer: A``, etc. Returns the
    0-based option index, or None if no valid single option is found (refusal /
    malformed / ambiguous multi-answer).
    """
    valid = set(LETTERS[:n_options])
    # Prefer the first standalone capital letter token.
    found: list[str] = []
    for m in _LETTER_RE.finditer(text.upper()):
        if m.group(1) in valid:
            found.append(m.group(1))
    if not found:
        return None
    # If multiple *distinct* options are asserted, treat as ambiguous.
    if len(set(found)) > 1:
        # allow a repeated single answer ("A. A is correct") but reject A vs B
        first = found[0]
        if any(f != first for f in found):
            return None
    return LETTERS.index(found[0])


def option_letter_token_ids(tokenizer, n_options: int) -> list[int]:
    """Token id emitted for each option letter at the answer position.

    Uses the no-leading-space encoding (assistant emits the bare letter after
    the generation prompt). Asserts each letter is a single token.
    """
    ids = []
    for i in range(n_options):
        enc = tokenizer.encode(LETTERS[i], add_special_tokens=False)
        assert len(enc) == 1, (
            f"option letter {LETTERS[i]!r} is not a single token: {enc}"
        )
        ids.append(enc[0])
    return ids


def score_options(model, tokenizer, prompt: str, n_options: int,
                  variant_id: str = "") -> Rollout:
    """Single forward pass -> per-option next-token logprob.

    Returns a :class:`Rollout` whose ``option_logprobs`` are the (unnormalized
    over the full vocab) log-softmax logprobs of each option letter, and whose
    ``answer_idx`` is the argmax option.
    """
    import torch

    ids = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=True, add_generation_prompt=True, return_tensors="pt",
    ).to(model.device)
    with torch.no_grad():
        logits = model(input_ids=ids).logits[0, -1]
    logprobs = torch.log_softmax(logits.float(), dim=-1)
    letter_ids = option_letter_token_ids(tokenizer, n_options)
    opt_lp = [float(logprobs[t]) for t in letter_ids]
    answer_idx = int(max(range(n_options), key=lambda i: opt_lp[i]))
    return Rollout(
        variant_id=variant_id,
        sample_idx=0,
        answer_idx=answer_idx,
        option_logprobs=tuple(opt_lp),
        raw_text="",
        format_ok=True,
    )


def sample_answers(model, tokenizer, prompt: str, n_options: int, *,
                   n_samples: int = 16, temperature: float = 1.0,
                   max_new_tokens: int = 8, variant_id: str = "",
                   seed: int = 0) -> list[Rollout]:
    """Sample N completions, parse each into a :class:`Rollout`."""
    import torch

    torch.manual_seed(seed)
    ids = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=True, add_generation_prompt=True, return_tensors="pt",
    ).to(model.device)
    rollouts: list[Rollout] = []
    for k in range(n_samples):
        with torch.no_grad():
            out = model.generate(
                ids, do_sample=temperature > 0, temperature=max(temperature, 1e-5),
                max_new_tokens=max_new_tokens, pad_token_id=tokenizer.eos_token_id,
            )
        text = tokenizer.decode(out[0, ids.shape[1]:], skip_special_tokens=True)
        idx = parse_answer_letter(text, n_options)
        rollouts.append(
            Rollout(
                variant_id=variant_id,
                sample_idx=k,
                answer_idx=idx,
                option_logprobs=tuple([0.0] * n_options),
                raw_text=text,
                format_ok=idx is not None,
            )
        )
    return rollouts
