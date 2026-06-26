# M0 + M1: implementation & validation results

Target model: **Qwen2.5-7B-Instruct**, NLA layer **L20** (`kitft/nla-qwen2.5-7b-L20-{av,ar}`).
All GPU runs executed on Modal (single A100-40GB). The AV runs via the
transformers `inputs_embeds` path (the documented SGLang fallback), deterministic
under greedy decoding.

## M0 — Harness & validation

### Tier 0 (CPU, hermetic): 46 unit tests pass
meta/tokenizer contract, golden-example parsing, residual-extraction hook vs
`output_hidden_states`, variant construction, labeling buckets, answer parsing,
statistics on synthetic ground truth.

### Golden-MSE plumbing gate (`golden_mse_gate`)
Re-extracts the worked example's layer-20 activations and validates them against
the recorded transcript.

| metric | value |
|---|---|
| Gate A (extraction + AR) pass | **True** |
| AR MAE vs golden mse_nrm | **0.0012** |
| AR corr vs golden mse_nrm | **0.9997** |
| extraction raw-norm match rate | **1.00** |
| mean reproduced mse_nrm / golden | 0.1918 / 0.191 |
| mean reproduced fve_nrm | 0.739 |
| Gate B (AV coherence) pass | **True** (ASCII English, parseable `<explanation>`) |

Interpretation: extraction at L20, AR reconstruction, and AV injection are all
wired correctly (scales, token ids, embedding scaling, layer). This is the
plumbing gate the rest of the pipeline depends on.

### 5-item end-to-end smoke (`smoke_test`)
data → variants → rollout (per-option logprobs) → extract (`t_preans`) → AV → AR
ran for all 5 items; explanations coherent; reconstruction mse 0.40–0.49,
cos 0.75–0.80.

## M1 — Dataset & labels

Dataset: **OpenTriviaQA** (the reference paper's sycophancy source), 4-option
MCQ, filtered → **42,860** items available (≥ the 2,000 raw target).

### Pilot (`m1_pilot`, 100 items, τ=0.8, θ=0.5, per-option-logprob labeling)

```
labeled_items                         100
confident_in_control                   64   (−36)
bucket_sycophantic                     28
bucket_non_sycophantic                 33
bucket_other_wrong                      3
bucket_control_incorrect               36
bucket_refusal                          0
bucket_unresolved                       0
mean_control_correct_p              0.647
mean_switch_to_user_wrong_p         0.565
confident_correct_rate              0.64
```

Interpretation:
- **Base rate** among confident-in-control items: ~44% sycophantic / ~52%
  non-sycophantic — a healthy, near-balanced split for the Exp-1 primary
  analysis.
- To reach the pre-registered ≥300 / ≥300 class target, scale to ~1,400 labeled
  items (28% sycophantic, 33% non-sycophantic of labeled). The filtered pool
  (42,860) is far larger than required.
- Qwen2.5-7B is markedly sycophantic under a confident incorrect-belief prompt
  (mean switch ≈ 0.57), which is favorable signal for the downstream NLA study.

## Notes / decisions
- **Target = Qwen2.5-7B**: no `HF_TOKEN` available in this environment, so gated
  Gemma/Llama can't be pulled. Qwen + its `kitft` NLA checkpoints are ungated and
  match the Tier-1 golden-test target. Add an `HF_TOKEN` Modal secret to promote
  the headline run to Gemma-3-12B / Llama-3.3-70B.
- `transformers` pinned `<5` (the kitft recipe and vendored client assume the 4.x
  `apply_chat_template`/`generate` behavior).
- Next (M2): AV N-sampling + FVE gating + the independent judge rubric.
