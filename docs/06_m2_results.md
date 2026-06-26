# M2 + initial Exp-1: how aligned is NLA to the counterfactual sycophancy label?

Target **Qwen2.5-7B-Instruct** (NLA L20), independent judge
**microsoft/Phi-3.5-mini-instruct** (different family). Within the
**incorrect-belief, natural** slice (the prompt is structurally identical across
items — only the specific wrong letter differs), at the **`t_preans`** decision
token. N_av=6 AV samples per activation, AR FVE gating (floor 0.3), judge
dimensions FVE-weighted per item. Run via `modal run scripts/modal_app.py::m2_run`.

## Sample
- 300 OpenTriviaQA items labeled → **67 sycophantic / 80 non-sycophantic**
  (147 items, base rate 0.456) after the confident-in-control filter.
- 882 AV explanations; mean FVE **0.371**; 719/882 pass the FVE≥0.3 gate.
- Judge: 882 explanations graded, **0 parse failures**.

## Headline: NLA explanations *are* aligned with the counterfactual labels

| metric | value |
|---|---|
| **`D_agreement` AUROC** (syco vs non) | **0.737** |
| `D_agreement` AUPRC | 0.732 |
| `D_agreement` gap (syco − non) | +0.180, **p = 0.0002** (permutation) |
| `D_agreement` partial corr \| belief-aware | **+0.482** |
| **NLA-only model AUROC** (all 6 dims) | **0.776** |
| NLA-only AUPRC | 0.790 |
| prompt-text-only baseline AUROC | 0.550 |
| incremental AUPRC (prompt-text + NLA vs prompt-text) | **+0.109** |

**Answer:** On Qwen2.5-7B, NLA explanations at the decision token are
**meaningfully aligned** with behavioral sycophancy. The NLA "agreement/deference"
reading predicts sycophantic-vs-honest answers with AUROC ≈ 0.74 (all NLA
dimensions together ≈ 0.78), far above the near-chance prompt-text baseline
(0.55), and the gap is highly significant (p ≈ 2e-4). Crucially, the signal is
**not just prompt leakage**: `D_agreement` retains a strong partial correlation
(+0.48) with the label after controlling for belief-awareness, and NLA adds
**+0.11 incremental AUPRC** over a prompt-text-only model. This supports the
predictive claim (C1) and gives initial support for specificity / non-leakage (C2).

## Per-dimension breakdown (the directions are coherent)

| dimension | AUROC | mean (syco) | mean (non) | reading |
|---|---|---|---|---|
| **D_agreement** | 0.737 | 0.51 | 0.33 | ↑ for sycophantic — the target construct |
| **D_resist** | 0.216 | 0.18 | 0.36 | ↑ for *non*-sycophantic (resistance ↔ honesty) |
| **D_commit** | 0.704 | 0.59 | 0.45 | ↑ for sycophantic (commits to the user's answer) |
| D_factaware | 0.549 | 0.72 | 0.68 | ~flat |
| D_eval | 0.546 | 0.52 | 0.51 | ~flat (nuisance) |
| D_beliefaware | 0.414 | 0.42 | 0.46 | **~flat** — leakage control behaves as intended |

The two construct-relevant dimensions separate the classes strongly and **in the
predicted directions** (agreement ↑, resistance ↓ for sycophancy), while the
leakage dimension `D_beliefaware` and the nuisance dimension `D_eval` are flat —
exactly the pattern faithfulness predicts and leakage does not.

## Caveats (these are *initial* results, not the confirmatory analysis)
1. **Bootstrapped judge:** Phi-3.5 stands in for the human-calibrated grader
   (no human gold set yet). Results are conditional on this grader; M2's
   calibration round (≥200 human grades, IAA) is not yet done.
2. **Single 7B model, single NLA layer (L20), n=147**, exploratory split only —
   no held-out confirmation split or multiplicity correction yet (that's M3).
3. The **combined (TF-IDF + NLA) AUROC (0.654) is below NLA-only (0.776)**: adding
   ~500 sparse text features to a small-n logistic dilutes the strong NLA signal
   (high-dim regularization artifact). The clean comparison is NLA-only 0.78 vs
   prompt-text-only 0.55; the incremental metric (combined vs base) is still
   positive (+0.11 AUPRC). M3 should use a sentence-embedding baseline + tuned
   regularization and the frozen confirmation split.
4. Counterfactual labels are a **behavioral proxy** (mitigated by the
   confident-in-control filter), and NLAs can confabulate (mitigated by N_av
   sampling + FVE gating).

## Reproduce
```bash
modal run scripts/modal_app.py::m2_run --n-label 300 --max-per-class 80 --n-av 6
# writes /data/m2_results.json (Modal volume) and prints the full result
```
