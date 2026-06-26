# Implementation Plan

Engineering plan to execute `02_final_experiment_plan.md`. Optimized so the
**primary result (Exp 1)** is reachable with the least infrastructure, then
layered outward to the causal and reliability experiments.

## 0. Guiding constraints

- Target model must have a released NLA. Default **Gemma-3-12B (L32)** for the
  build/iterate phase (fits more easily than Llama-3.3-70B), with **Llama-3.3-70B
  (L53)** as the headline target if GPU budget allows.
- Reuse, do not reinvent: drive AV/AR via the recipe in `kitft/nla-inference`
  (`nla_inference.py` actor client + `NLACritic`); load `nla_meta.yaml` and
  **assert against the live tokenizer at startup** (never hardcode token IDs,
  injection scale, or prompt template).
- Everything seedable, versioned, and resumable; activations and explanations are
  expensive — cache aggressively and key caches by config hash.

---

## 1. Repository layout

```
nla_experiments/
├── docs/                         # these planning docs
├── configs/
│   ├── model_gemma3_12b.yaml     # target + NLA checkpoint ids, layer, scales
│   ├── model_llama3_70b.yaml
│   └── experiment.yaml           # N rollouts, thresholds τ/θ, N_av, FVE floor, seeds
├── nla_sycophancy/
│   ├── data/
│   │   ├── source.py             # load + filter OpenTriviaQA-style MCQ
│   │   └── variants.py           # build control / correct / incorrect / strength sweep
│   ├── target/
│   │   ├── rollout.py            # batched generation, per-option logprobs
│   │   ├── label.py              # switch-prob, confident-in-control filter, buckets
│   │   └── extract.py            # hook layer l, grab t_userend/t_assist0/t_preans/t_ans
│   ├── nla/
│   │   ├── av_client.py          # wraps nla_inference actor (SGLang input_embeds)
│   │   ├── ar_client.py          # NLACritic: text -> vector, FVE/MSE
│   │   └── meta.py               # load + validate nla_meta.yaml vs tokenizer
│   ├── judge/
│   │   ├── rubric.py             # 6 dimensions, frozen prompt, blinded inputs
│   │   └── grade.py              # independent judge calls + caching
│   ├── steering/
│   │   └── ar_steer.py           # edit explanation -> AR delta -> patch target
│   ├── analysis/
│   │   ├── exp1_predictive.py    # M0/M1/M2, incremental AUPRC, partial corr
│   │   ├── exp2_placebo.py
│   │   ├── exp3_doseresponse.py
│   │   ├── exp5_causal.py
│   │   ├── exp6_reliability.py   # recurrence + claim ablation
│   │   └── stats.py              # bootstrap CIs, permutation, Holm/BH, power
│   └── io/
│       ├── cache.py              # content-addressed artifact store (parquet)
│       └── schema.py             # pydantic records for every stage
├── scripts/                      # thin CLIs calling the package
├── tests/                        # unit + a tiny end-to-end smoke test
├── requirements.txt
└── README.md
```

---

## 2. Dependencies

- `torch`, `transformers`, `safetensors`, `accelerate` (target model + hooks).
- **SGLang** (or vLLM) for AV inference with `input_embeds` injection, per the NLA
  inference recipe; `nla_inference.py` vendored or pip-installed.
- `huggingface_hub` (pull `kitft/nla-*` checkpoints + `nla_meta.yaml`).
- `datasets` (OpenTriviaQA / trivia source), `pandas`/`pyarrow` (parquet cache).
- `scikit-learn` (M0/M1/M2 classifiers, AUPRC/AUROC), `numpy`, `scipy`,
  `statsmodels` (partial corr, multiplicity).
- Judge: an API/local model **independent** of the target family.
- A venv check first (per repo rule): use existing `.venv`/`venv` if present,
  else create one only when implementation actually starts.

---

## 3. Data schemas (key records)

- `Item{id, question, options[], correct_idx, source}`
- `Variant{item_id, kind∈{control,correct,incorrect}, regime, strength, prompt}`
- `Rollout{variant_id, sample_idx, answer_idx, option_logprobs, raw_text, format_ok}`
- `Label{item_id, control_correct_p, switch_to_user_wrong_p, bucket, continuous_switch}`
- `Activation{rollout_id, position∈{userend,assist0,preans,ans}, layer, vec_path}`
- `Explanation{activation_id, av_sample_idx, text, fve, mse}`
- `Grade{explanation_id, D_beliefaware, D_factaware, D_agreement, D_resist, D_commit, D_eval}`

All persisted as parquet, keyed by a hash of `(model_rev, nla_rev, config)`.

---

## 4. Milestones

### M0 — Harness & validation (build trust before scale)
- Load target + NLA; `meta.py` asserts injection token/scale/template against the
  tokenizer; reproduce one worked example from `nla-inference/examples/` (match
  per-token MSE) as a **correctness gate**.
- `extract.py` returns the right activation for a known token (unit test).
- Tiny end-to-end smoke test on 5 items.

### M1 — Dataset & labels
- Implement source loading + filtering; build the three variants + strength sweep
  + three regimes.
- Rollouts with per-option logprobs; `label.py` produces confident-in-control
  filter, switch probabilities, and buckets; emit an **attrition report**.
- **Pilot (~100 items)** to estimate base rates/variance → finalize N, τ, θ and the
  power calc; **pre-register** primary slice/position/metric (freeze a config file
  + git tag).

### M2 — NLA + judge pipeline
- AV sampling (N_av), AR reconstruction + FVE; FVE gating/weighting; caching.
- Judge rubric; **calibration round** vs ≥200 human-graded explanations; freeze
  rubric; report grader–human + human–human agreement.

### M3 — Primary analysis (Exp 1) + leakage (Exp 7)
- M0 prompt-text baseline, M1 NLA, M2 combined; cross-validated **incremental
  AUPRC/AUROC**, partial correlation of `D_agreement`, calibration; on the frozen
  confirmation split.
- Position/regime comparison for leakage characterization.
- **Decision checkpoint:** if M2 ≤ M0 (leakage-only), report and stop before
  investing in causal infra.

### M4 — Specificity, dose–response, reliability (Exp 2, 3, 6)
- Placebo task pipeline reuse; dose–response monotonicity + within-item corr;
  recurrence + claim-ablation reliability weights.

### M5 — Causal steering (Exp 5)
- `ar_steer.py`: explanation edit → AR delta → patched generation; norm-matched
  random + reword + belief-only controls; strength sweep + coherence (KL/PPL)
  monitor; non-sycophantic transfer control.

### M6 — Replication & write-up
- Re-run headline (Exp 1 + Exp 5) on the second target model if budget allows;
  assemble figures, attrition, judge-agreement, and caveats from
  `02_final_experiment_plan.md §8`.

---

## 5. Compute budget (order-of-magnitude)

- AV generates ~500 tokens/activation. Cost ≈
  `items × variants × regimes × positions × N_av × 500 tokens`. Keep the **primary**
  run lean: incorrect-belief + control only, `t_preans` (+`t_userend` for leakage),
  natural regime, N_av=8. MCQ transcripts are short, so target-model rollouts are
  cheap relative to AV inference. Batch AV via SGLang; cache everything.
- Causal/strength experiments are run only on a high-confidence subset to bound
  cost.

---

## 6. Testing & QA

- Unit tests: meta/tokenizer assertions, extraction token correctness, label
  logic (switch-to-user-wrong vs switch-to-other-wrong), FVE math.
- Golden test: reproduce a `nla-inference` example MSE within tolerance.
- Determinism: fixed seeds; snapshot tests on a frozen 10-item fixture.
- Data integrity: attrition report must reconcile counts at every filter.

---

## 7. Key risks & mitigations

| Risk | Mitigation |
|---|---|
| Wrong injection scale/token → garbage AV | `meta.py` startup asserts + golden-MSE gate |
| Result is pure prompt leakage | Incremental-validity (M2 vs M0) gate at M3 before causal work |
| Few positives after filtering | Pilot + power calc; broaden source; continuous switch outcome |
| Judge bias/circularity | Independent model, frozen human-calibrated rubric, blinding, IAA |
| Steering breaks the model | Strength sweep + KL/PPL coherence monitor; norm-matched controls |
| Single-layer blindspot | State as scope limit; (stretch) train an NLA at another layer later |
| Compute blowup from AV | Lean primary slice, batching, content-addressed caching |

---

## 8. Stretch goals (out of v2 scope)

- Train an NLA at a second layer to test layer sensitivity of sycophancy signal.
- Compare NLA against linear-probe / SAE monitors on the same counterfactual
  labels (ties directly back to the arXiv:2509.21344 leakage findings).
- Extend to multi-turn / agentic sycophancy beyond single-turn MCQ.
