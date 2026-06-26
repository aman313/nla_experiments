# Critique of the v1 Experimentation Plan

This document reviews the first-draft plan for measuring whether NLA (Natural
Language Autoencoder) explanations are *faithful* to **counterfactual sycophancy
labels**. The goal is to surface methodological problems before committing
implementation effort. Issues are grouped by severity.

Throughout, "target model" = the model whose activations are explained;
"AV" = activation verbalizer (`vector → text`); "AR" = activation reconstructor
(`text → vector`); the sycophancy labeling protocol follows the counterfactual
control/correct-belief/incorrect-belief construction from arXiv:2509.21344
(which itself follows Kirch et al.).

---

## A. Blocking / framing issues

### A1. Model availability mismatch (most important practical constraint)
The released, ready-to-use NLAs are trained for **open models only** — Qwen2.5-7B
(L20), Gemma-3-12B (L32), Gemma-3-27B (L41), Llama-3.3-70B (L53). The
counterfactual sycophancy methodology in the reference paper was demonstrated on
Llama-3.1-8B-Instruct. **There is no released NLA for Llama-3.1-8B and none for
Claude.** v1 implicitly assumed an NLA exists for whatever model we want to study.

Consequence: the *target model* must be one of the four with a public NLA, OR we
must train an NLA ourselves (multi-day GRPO on 2×8×H100 per the paper — out of
scope for a first pass). The plan must pick a target from the available NLA set
and generate the sycophancy dataset *on that model*.

### A2. "Faithful to counterfactual labels" needs a precise definition
v1 oscillated between three distinct claims without separating them:

1. **Predictiveness** — NLA-derived features predict the behavioral label.
2. **Specificity / non-leakage** — the signal is not just the AV re-reading the
   user's stated belief from the prompt.
3. **Causal faithfulness** — the explanation names a factor that is causally
   responsible for the switch.

These are increasingly strong and require different experiments. A faithfulness
claim that conflates them is not defensible. The final plan must state which
claim each experiment supports.

### A3. The counterfactual label is itself a proxy, not ground truth
A behavioral answer-switch under user pressure is *evidence of sycophantic
behavior*, not a readout of an internal "desire to please." Some switches are
**rational deference** (treating the user as a weak evidence source), which is
arguably not pejorative sycophancy. Validating NLA explanations against this
label measures agreement with a behavioral proxy, and the ceiling is the
proxy's own validity. This must be stated as a scope limitation, and we should
prefer items where the model is *confidently correct* in control (so a switch is
more clearly unjustified).

---

## B. Confounds that can manufacture a false positive result

### B1. Prompt leakage makes cross-variant comparison nearly uninterpretable
In the incorrect-belief variant the prompt *literally contains* the wrong answer.
An AV that simply transcribes "the user believes B" will look like it detects
sycophancy. v1's Experiment 1 (predict label from explanation) and Experiment 2
(difference-in-differences across variants) are both vulnerable: the variant
identity is trivially decodable from prompt text, so any across-variant signal
is confounded with prompt content.

**Fix:** the primary analysis must be **within the incorrect-belief variant**,
contrasting *sycophantic* vs *non-sycophantic* items. Here the prompt is
structurally identical (both contain a wrong user belief); only the model's
internal handling differs. This is the only contrast where prompt-reading alone
cannot separate the classes. The across-variant DiD should be demoted to a
secondary/sanity analysis, not the main result.

### B2. Correctness/agreement confound in the correct-belief variant
In the correct-belief variant, "agreeing with the user" and "being correct"
coincide. Any "agreement" NLA signal there is ambiguous. The clean manipulation
is the **incorrect-belief** variant only. v1 mentioned this but still built
metrics that averaged over both variants.

### B3. Reasoning leakage / CoT
If the target model verbalizes reasoning ("the user says B, but actually..."),
the activation — and therefore the AV — may reflect explicit reasoning text
rather than a latent disposition. v1's Experiment 6 addressed this but it was
not central. Faithfulness in a **no-CoT / answer-only** setting, with extraction
*before* the answer token, is much stronger evidence and should be a core
condition, not an afterthought.

### B4. Judge/target shared priors (circularity)
v1 relies heavily on an LLM judge to bucket explanations into "agreement
motivation," "deference," etc. If the judge shares a model family / priors with
the target or the AV, it may hallucinate sycophancy interpretations from neutral
text. Needs: an independent judge model, a frozen rubric calibrated against a
human gold set (the reference NLA paper hand-graded ~186 explanations and tuned
until the grader matched, then froze it — replicate that protocol), and
inter-annotator agreement reporting.

### B5. No specificity / placebo behavior
v1 has no negative control behavior. Without one we cannot show the NLA signal is
*specific to sycophancy* rather than firing on any "user asserts a belief"
context. Need (a) a non-sycophantic but belief-laden control task, and (b)
adversarial "looks sycophantic but isn't" items (cf. the reference paper's
adversarial probe validation).

---

## C. Statistical and measurement issues

### C1. Stochastic, single-rollout labels
Target models sampled at T>0 give noisy switch labels from a single rollout.
Need either greedy decoding or N rollouts with a **switching probability**, and a
defensible threshold (or treat probability as a continuous outcome). "Switched to
the user's wrong answer" must be distinguished from "switched to a *different*
wrong answer" (confusion, not sycophancy).

### C2. Class imbalance and shrinking N after filtering
Filtering to "correct in control" then "natural sycophancy is rare" can leave few
positives. AUROC alone is misleading under imbalance — report **AUPRC, base
rates, and recall at fixed low FPR**. Do a power/sample-size estimate before
running, not after.

### C3. Multiple comparisons
v1 sweeps many token positions × variants × metrics × layers-if-available with no
correction. This is a garden of forking paths. Need **pre-registration** of the
primary hypothesis/metric/position, a held-out confirmation split, and
multiplicity control (Holm/BH) for secondary analyses.

### C4. The "NLAAgree − PromptBelief" composite is not well-posed
Subtracting two judge scores measured on different rubric dimensions assumes a
shared scale they don't have. Better: enter `user-belief-awareness` and
`agreement/deference` as **separate regressors** predicting behavior and measure
the **unique/incremental** contribution of the agreement dimension (partial
correlation, or incremental AUROC over a prompt-text-only baseline).

### C5. Arbitrary token aggregation
v1 inherits the reference paper's "any-of-N tokens" aggregation, which the
authors themselves flag as arbitrary (it underestimated awareness on blackmail).
Pre-specify the aggregation (e.g., max over a defined pre-answer window) and
report sensitivity to the choice.

### C6. NLA inference is itself stochastic
The AV samples at T=1; a single explanation is a sample. Need N samples per
activation with a defined aggregation, and/or **best-of-N ranked by AR
reconstruction** — and the gating/weighting by reconstruction quality (FVE/MSE)
must be explicit, with the FVE distribution reported (untrustworthy
reconstructions should be down-weighted or excluded).

---

## D. Causal experiment (v1 Experiment 4) issues

### D1. Steering is weak and confounded
The reference paper reports NLA-derived steering succeeds ~50% of the time and
can produce incoherent outputs. An edit-then-reconstruct `Δ = AR(edited) −
AR(orig)` changes norm and possibly many directions at once. Needs:
- norm-matched **random-direction** and **semantics-preserving-reword** controls,
- a **steering-strength sweep** with coherence monitoring (KL / perplexity vs
  unsteered) to avoid "it worked because we broke the model,"
- application at the layer the NLA was trained on *and* a check that this is near
  where the decision is made (the decision may live at another layer/token).

### D2. Causal null result is ambiguous
If steering fails to reduce switching, it could be because (a) the explanation is
unfaithful, or (b) the AR reconstruction / single-layer single-token patch is
just too weak. So the causal experiment can *support* faithfulness on success but
cannot *refute* it on failure. State this asymmetry.

---

## E. Scope / practical issues

- **Single NLA layer.** We are restricted to the released layer. If sycophancy is
  represented elsewhere, the NLA will miss it — a scope limitation, not evidence
  of unfaithfulness. (The reference NLA paper explicitly found reward-model
  features only at a specific layer.)
- **Compute.** NLA inference generates ~500 tokens per activation and is
  impractical past ~10k tokens/transcript. With N samples × many tokens × many
  items this is the dominant cost and must be budgeted; MCQ trivia transcripts
  are short, which helps.
- **Reproducibility.** v1 never pinned model versions, decoding params, seeds,
  `nla_meta.yaml` scale factors, or the exact extraction token. These must be
  fixed and logged.
- **Verbalizable-only ceiling.** NLAs can only surface what is *verbalizable* from
  the activation; a null is not proof of absence.

---

## F. Summary of required changes for v2

1. Pick a target model **with a released NLA**; generate the sycophancy dataset on
   that model (default: Llama-3.3-70B or Gemma-3-12B).
2. Make the **within-incorrect-belief, sycophantic-vs-non** contrast the primary
   analysis; demote cross-variant DiD to secondary.
3. Separate the three faithfulness claims (predictive / specific / causal) and map
   each experiment to one.
4. Add a **prompt-text-only baseline** and require NLA signal to beat it
   (incremental validity), plus a **placebo behavior** for specificity.
5. Center a **no-CoT, pre-answer-token** condition to defeat reasoning leakage.
6. Replace the composite metric with incremental-validity regression.
7. Add probabilistic/greedy labels, AUPRC + recall@FPR, power analysis,
   pre-registration, held-out confirmation, multiplicity control.
8. Add a frozen, human-calibrated, independent judge with IAA.
9. Add proper causal controls (norm-matched random + reword), strength sweep,
   coherence checks; state the success/failure asymmetry.
10. Gate/weight by AR reconstruction quality; sample N AV explanations.
11. Document scope limits: single layer, verbalizability ceiling, proxy-label
    ceiling, compute.
