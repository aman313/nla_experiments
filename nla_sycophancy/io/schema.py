"""Typed records for every pipeline stage.

These mirror the schemas in `docs/03_implementation_plan.md §3`. We use frozen
dataclasses (no pydantic dependency) so the Tier-0 CPU tests stay lightweight.
Each record is JSON/parquet-serializable via :func:`dataclasses.asdict`.

Stages: Item -> Variant -> Rollout -> Label -> Activation -> Explanation -> Grade.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import Optional


class VariantKind(str, enum.Enum):
    """Counterfactual variant (Kirch et al. 2026 / arXiv:2509.21344)."""

    CONTROL = "control"            # question only, no stated user belief
    CORRECT = "correct"            # user states the correct option as belief
    INCORRECT = "incorrect"        # user states a specific wrong option as belief


class Regime(str, enum.Enum):
    """Elicitation regime (separate dataset slices)."""

    NATURAL = "natural"                    # no agree/disagree instruction (primary)
    INSTRUCTION = "instruction"            # explicit "agree with the user"
    INCENTIVIZED = "incentivized"          # indirect social/reward pressure


class Position(str, enum.Enum):
    """Pre-registered extraction token positions (NLA layer residual stream)."""

    USEREND = "userend"        # last user-prompt token (leakage probe)
    ASSIST0 = "assist0"        # first assistant/control token
    PREANS = "preans"          # token immediately before the answer (PRIMARY)
    ANS = "ans"                # the answer-option token


class Bucket(str, enum.Enum):
    """Behavioral sycophancy bucket assigned by :mod:`target.label`."""

    SYCOPHANTIC = "sycophantic"            # switches to user-suggested wrong option
    NON_SYCOPHANTIC = "non_sycophantic"    # stays correct across all variants
    OTHER_WRONG = "other_wrong"            # switched to a *different* wrong option
    CONTROL_INCORRECT = "control_incorrect"  # not confidently correct in control
    REFUSAL = "refusal"                    # refusal / malformed / multi-answer
    UNRESOLVED = "unresolved"              # did not meet any clear criterion


@dataclass(frozen=True)
class Item:
    """A filtered factual multiple-choice question."""

    id: str
    question: str
    options: tuple[str, ...]
    correct_idx: int
    source: str = "opentriviaqa"
    category: Optional[str] = None

    def __post_init__(self) -> None:
        assert len(self.options) >= 2, f"item {self.id}: need >=2 options"
        assert 0 <= self.correct_idx < len(self.options), (
            f"item {self.id}: correct_idx {self.correct_idx} out of range "
            f"for {len(self.options)} options"
        )

    @property
    def n_options(self) -> int:
        return len(self.options)


@dataclass(frozen=True)
class Variant:
    """A prompt built for one (item, kind, regime, belief-strength)."""

    item_id: str
    kind: VariantKind
    regime: Regime
    prompt: str
    # index of the option asserted as the user's belief (None for control)
    belief_idx: Optional[int] = None
    # belief-strength level for the dose-response sweep (0 = no/again default)
    strength: int = 0

    @property
    def id(self) -> str:
        b = "x" if self.belief_idx is None else str(self.belief_idx)
        return f"{self.item_id}:{self.kind.value}:{self.regime.value}:s{self.strength}:b{b}"


@dataclass(frozen=True)
class Rollout:
    """One sampled (or scored) response to a variant."""

    variant_id: str
    sample_idx: int
    answer_idx: Optional[int]            # parsed chosen option, None if unparseable
    option_logprobs: tuple[float, ...]   # per-option logprob (len == n_options)
    raw_text: str = ""
    format_ok: bool = True

    @property
    def option_probs(self) -> tuple[float, ...]:
        """Softmax-normalized per-option probabilities."""
        import math

        m = max(self.option_logprobs)
        exps = [math.exp(lp - m) for lp in self.option_logprobs]
        z = sum(exps)
        return tuple(e / z for e in exps)


@dataclass(frozen=True)
class Label:
    """Per-item sycophancy label derived from control + incorrect rollouts."""

    item_id: str
    regime: Regime
    control_correct_p: float             # P(correct) in control variant
    switch_to_user_wrong_p: float        # P(user-suggested wrong) in incorrect variant
    switch_to_other_wrong_p: float       # P(any other wrong) in incorrect variant
    bucket: Bucket
    continuous_switch: float             # continuous outcome (= switch_to_user_wrong_p)
    belief_idx: Optional[int] = None     # the wrong option suggested in incorrect variant


@dataclass(frozen=True)
class Activation:
    """A residual-stream vector extracted at one token position."""

    rollout_id: str
    position: Position
    layer: int
    vec_path: str                        # content-addressed path in the artifact store
    raw_norm: float = 0.0


@dataclass(frozen=True)
class Explanation:
    """One AV explanation of an activation, with AR reconstruction fidelity."""

    activation_id: str
    av_sample_idx: int
    text: str
    fve: float                           # fraction of variance explained (nrm)
    mse: float                           # direction MSE = 2(1 - cos)
    cos: float = 0.0


@dataclass(frozen=True)
class Grade:
    """Judge scores for one explanation (each dimension in [0, 1])."""

    explanation_id: str
    D_beliefaware: float
    D_factaware: float
    D_agreement: float
    D_resist: float
    D_commit: float
    D_eval: float


@dataclass
class AttritionReport:
    """Reconciles counts at every filter stage (must always balance).

    Stages added with ``is_chain=True`` (the default) are sequential filters and
    render a drop delta vs the previous chain stage. Stages added with
    ``is_chain=False`` are categorical breakdowns (e.g. per-bucket counts) and
    render without a delta.
    """

    stages: list[tuple[str, int, bool]] = field(default_factory=list)

    def add(self, name: str, count: int, *, is_chain: bool = True) -> None:
        self.stages.append((name, count, is_chain))

    def to_dict(self) -> dict[str, int]:
        return {name: count for name, count, _ in self.stages}

    def render(self) -> str:
        lines = ["Attrition report:"]
        prev: Optional[int] = None
        for name, count, is_chain in self.stages:
            if is_chain:
                delta = "" if prev is None else f"  (−{prev - count})"
                prev = count
            else:
                delta = ""
            lines.append(f"  {name:<32} {count:>8}{delta}")
        return "\n".join(lines)
