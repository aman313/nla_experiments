"""M1: load + filter the OpenTriviaQA multiple-choice source.

This is the dataset used for sycophancy in the reference paper
(arXiv:2509.21344, "Linear probes rely on textual evidence"), which follows
Kirch et al. 2026: filtered OpenTriviaQA questions converted to a fixed-format
MCQ, then expanded into counterfactual variants (see :mod:`data.variants`).

OpenTriviaQA (https://github.com/uberspot/OpenTriviaQA, CC-BY-SA) stores one
file per category; each question block is::

    #Q <question text>
    ^ <correct answer text>
    A <option A>
    B <option B>
    C <option C>
    D <option D>

The ``^`` line repeats the *text* of the correct option; we resolve it to an
option index. Blocks are separated by blank lines.
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Iterable, Optional

from nla_sycophancy.io.schema import Item

_OPTION_RE = re.compile(r"^([A-H])\s+(.*)$")


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip().casefold()


def parse_opentriviaqa(text: str, *, source: str = "opentriviaqa",
                       category: Optional[str] = None) -> list[Item]:
    """Parse the categorized OpenTriviaQA text format into :class:`Item` records.

    Malformed blocks (no resolvable correct option, <2 options, empty question)
    are skipped silently; use :func:`load_opentriviaqa` for attrition counting.
    """
    items: list[Item] = []
    block: list[str] = []

    def flush(block_lines: list[str]) -> None:
        if not block_lines:
            return
        question_parts: list[str] = []
        correct_text: Optional[str] = None
        options: list[str] = []
        mode = None
        for raw in block_lines:
            line = raw.rstrip("\n")
            if line.startswith("#Q"):
                question_parts.append(line[2:].strip())
                mode = "q"
            elif line.startswith("^"):
                correct_text = line[1:].strip()
                mode = "c"
            else:
                m = _OPTION_RE.match(line)
                if m:
                    options.append(m.group(2).strip())
                    mode = "o"
                elif mode == "q" and line.strip():
                    question_parts.append(line.strip())
        question = " ".join(p for p in question_parts if p).strip()
        if not question or correct_text is None or len(options) < 2:
            return
        correct_idx = None
        for i, opt in enumerate(options):
            if _norm(opt) == _norm(correct_text):
                correct_idx = i
                break
        if correct_idx is None:
            return
        qid = hashlib.sha1(
            f"{source}|{category}|{question}|{'|'.join(options)}".encode()
        ).hexdigest()[:16]
        items.append(
            Item(
                id=qid,
                question=question,
                options=tuple(options),
                correct_idx=correct_idx,
                source=source,
                category=category,
            )
        )

    for line in text.splitlines():
        if line.startswith("#Q") and block:
            flush(block)
            block = [line]
        else:
            block.append(line)
    flush(block)
    return items


def load_opentriviaqa(categories_dir: str | Path,
                      categories: Optional[Iterable[str]] = None) -> list[Item]:
    """Load every (or selected) category file from a local OpenTriviaQA clone."""
    root = Path(categories_dir)
    assert root.exists(), f"categories dir not found: {root}"
    files = sorted(p for p in root.iterdir() if p.is_file())
    if categories is not None:
        wanted = set(categories)
        files = [p for p in files if p.name in wanted]
    items: list[Item] = []
    for p in files:
        text = p.read_text(encoding="utf-8", errors="replace")
        items.extend(parse_opentriviaqa(text, category=p.name))
    return items


def filter_items(
    items: list[Item],
    *,
    n_options: Optional[int] = 4,
    min_question_len: int = 8,
    dedupe: bool = True,
) -> list[Item]:
    """Apply the standard MCQ filters used by the reference protocol.

    - ``n_options``: keep only questions with exactly this many options (paper
      uses 4; pass None to keep any >=2).
    - ``min_question_len``: drop trivially short questions.
    - ``dedupe``: drop duplicate (question, options) blocks across categories.
    """
    out: list[Item] = []
    seen: set[str] = set()
    for it in items:
        if n_options is not None and it.n_options != n_options:
            continue
        if len(it.question) < min_question_len:
            continue
        # reject options that are duplicated within an item (ambiguous answer)
        if len({_norm(o) for o in it.options}) != it.n_options:
            continue
        if dedupe:
            key = _norm(it.question) + "||" + "|".join(_norm(o) for o in it.options)
            if key in seen:
                continue
            seen.add(key)
        out.append(it)
    return out
