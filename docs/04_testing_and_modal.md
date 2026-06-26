# Testing Strategy & Modal GPU Execution

Covers (1) how the implementation in `03_implementation_plan.md` gets tested, and
(2) running all GPU work on [Modal](https://modal.com), including what is needed
to operate it that way.

---

## Part 1 — Testing strategy

The pipeline is staged (data → rollout → extract → AV → AR → judge → analysis),
so we test in tiers from cheap/deterministic to expensive/GPU, and **gate** later
tiers on earlier ones.

### Tier 0 — Pure-CPU unit tests (run in CI, no GPU, no network)
Fast, deterministic, the bulk of correctness coverage:
- **Label logic** (`label.py`): switch-to-*user's*-wrong vs switch-to-*other*-wrong
  vs refusal; confident-in-control filter at τ; bucket assignment; continuous
  switch probability from synthetic per-option logprobs.
- **Variant construction** (`variants.py`): control/correct/incorrect prompts and
  the belief-strength sweep are well-formed; the correct/incorrect option is the
  intended one.
- **FVE/MSE math** (`ar_client` helpers): `MSE = 2(1−cos)` on unit-norm vectors;
  FVE formula against hand-computed cases.
- **Meta/tokenizer contract** (`meta.py`): given a `nla_meta.yaml` + a tokenizer,
  assert injection token id, left/right neighbor ids, and template round-trip.
  This catches the #1 silent failure (wrong injection position → garbage AV) and
  needs only the tokenizer, **not the GPU model**.
- **Schema/attrition** (`schema.py`, data stages): counts reconcile at every
  filter; no orphan records.
- **Stats** (`stats.py`): bootstrap CI, permutation test, Holm/BH, incremental
  AUPRC on synthetic data with known ground truth.

### Tier 1 — GPU smoke + golden test (smallest model)
Run on Modal with **Qwen2.5-7B** (cheapest NLA target). Two checks:
- **Golden MSE test (the key plumbing gate):** reproduce a worked transcript from
  `kitft/nla-inference/examples/` and assert per-token MSE matches within
  tolerance. If this passes, the AV injection + AR reconstruction stack is wired
  correctly (scales, token ids, embedding scaling, layer).
- **End-to-end smoke:** 5 items through every stage; assert artifacts exist, FVE
  is in the paper's plausible band (~0.3–0.8), grades are in [0,1].

### Tier 2 — Component correctness on GPU
- **Extraction** (`extract.py`): hook returns the layer-`l` residual at the
  intended token; verify against an independent forward pass; verify
  `t_preans`/`t_ans` indices on known templated outputs.
- **AV injection position:** decode the prompt actually fed to the AV and assert
  the injected vector sits between the expected neighbor tokens.
- **AR sanity:** reconstructing the AV's own explanation of an activation yields
  higher cosine to the original than reconstructing a random explanation.

### Tier 3 — Pilot validation (~100 items)
Not pass/fail but **go/no-go calibration**:
- Dataset health: control-correct rate, sycophancy base rate, attrition per stage.
- Estimate variance → finalize N rollouts, N_av, τ/θ, and the **power calc**.
- **Judge calibration:** hand-grade ≥200 explanations; tune rubric until grader
  matches; freeze; report grader–human and human–human agreement (κ).
- **Pre-registration freeze:** commit the config + git-tag the primary
  slice/position/metric before touching the confirmation split.

### Tier 4 — Reproducibility & cost guards
- **Determinism:** fixed seeds; re-running a frozen 10-item fixture yields
  byte-identical cached artifacts; cache hits skip recompute (assert no recompute).
- **Cost estimator dry-run:** before any large job, print projected
  activations × N_av × ~500 tokens and refuse to launch above a budget ceiling.

### What "passing" means before scaling
Tiers 0–2 green + golden MSE within tolerance + pilot go-decision. Only then run
the full primary slice (Exp 1). The **M3 leakage checkpoint** (M2 incremental over
M0 prompt-text baseline) is itself a result-gate before building causal infra.

---

## Part 2 — Running all GPU work on Modal

**Yes — the entire GPU surface fits Modal cleanly.** GPU stages: target-model
rollouts, activation extraction, AV inference, AR reconstruction, and steering.
The judge is a (CPU/API) call. Analysis is CPU.

### Why it maps well
- The pipeline is **embarrassingly batchable** and **stage-separated**, so each
  Modal function loads exactly one model. No stage needs target + AV + AR resident
  at once → modest per-call GPU memory.
- Artifacts are content-addressed parquet → a **Modal Volume** is the natural
  hand-off between stages and the cache.
- Bursty: spin up many GPUs for the AV sweep, scale to zero between runs.

### Proposed Modal app shape (illustrative — pin to current Modal API)
```python
import modal

image = (
    modal.Image.debian_slim()
    .pip_install("torch", "transformers", "safetensors", "accelerate",
                 "sglang", "huggingface_hub", "pandas", "pyarrow", "scikit-learn")
)
app = modal.App("nla-sycophancy")
hf_cache = modal.Volume.from_name("hf-cache", create_if_missing=True)      # weights
artifacts = modal.Volume.from_name("nla-artifacts", create_if_missing=True) # parquet

@app.function(gpu="H100", image=image,
              volumes={"/cache": hf_cache, "/data": artifacts},
              secrets=[modal.Secret.from_name("hf-token")], timeout=3600)
def av_verbalize(activation_keys: list[str]): ...   # batched AV (SGLang input_embeds)

@app.function(gpu="A100-40GB", image=image, volumes={...}, secrets=[...])
def rollout_and_extract(item_ids: list[str]): ...   # target gen + hooked extraction

@app.function(gpu="A100-40GB", image=image, volumes={...}, secrets=[...])
def ar_reconstruct(explanation_keys: list[str]): ...

@app.function(image=image, secrets=[modal.Secret.from_name("judge-api-key")])
def judge(explanation_keys: list[str]): ...         # independent model, no GPU here
```
- Use `.map()` for fan-out over item/activation batches.
- A `@modal.cls` with `@modal.enter` loads the model once per warm container so
  the weights aren't reloaded per batch.
- AV uses SGLang `input_embeds` per the `nla-inference` recipe; a transformers
  `inputs_embeds` path is the fallback for the smoke test.

### GPU sizing (bf16; AV ≈ same size as its target, AR is smaller/truncated)
| Target | NLA layer | Rollout/extract | AV | AR |
|---|---|---|---|---|
| Qwen2.5-7B | L20 | 1× A100-40GB / L40S | 1× A100-40GB | 1× A10G–A100-40GB |
| Gemma-3-12B | L32 | 1× A100-40GB / H100 | 1× A100-40GB / H100 | 1× A100-40GB |
| Gemma-3-27B | L41 | 1× H100 / A100-80GB | 1× H100 | 1× A100-40GB |
| Llama-3.3-70B | L53 | 2× H100 (or fp8 1×H100, risky) | 2× H100 | 1× H100 |

Recommendation: **build + test on Qwen2.5-7B / Gemma-3-12B** (single-GPU, cheap),
promote the headline run to **Llama-3.3-70B** only after Tiers 0–3 are green.

### Secrets / volumes Modal needs
- `hf-token` secret: HF token with access to gated **Gemma** (Google license) and
  **Llama** (Meta license); Qwen is ungated.
- `judge-api-key` secret: key for the independent judge model (or host an open
  judge as another Modal GPU function).
- Volumes: `hf-cache` (weights), `nla-artifacts` (activations/explanations/grades
  parquet cache).

### Cost control on Modal
- Scale-to-zero between stages; `.map()` concurrency caps; the dry-run cost
  estimator gates launches; keep the primary slice lean (incorrect-belief +
  control, `t_preans` (+`t_userend`), natural regime, N_av=8). Short MCQ
  transcripts keep rollout cost low; AV dominates.

---

## Part 3 — What I need from you to run it this way

**Required to actually execute GPU work on Modal:**
1. **Modal account + credentials.** Either run `modal token new` in your workspace,
   or add `MODAL_TOKEN_ID` / `MODAL_TOKEN_SECRET` as Cloud Agent secrets
   (Cursor Dashboard → Cloud Agents → Secrets) so I can deploy from here.
2. **Hugging Face token** (`HF_TOKEN` secret) **with licenses accepted** for the
   gated models you want as targets (Gemma and/or Llama). Confirm which target(s)
   to prioritize. Qwen needs no license.
3. **Judge model decision + key:** which independent model to grade explanations
   (e.g. an API model unrelated to the target family), and its API key as a secret
   — or approval to host an open judge model as a Modal function.
4. **Compute budget ceiling** ($ or GPU-hours) and an OK to start on the cheap
   targets first.

**Required for scientific validity (people, not infra):**
5. **Human annotation** for judge calibration — ~200 explanations hand-graded by
   ≥2 annotators (you/teammates), or explicit approval to bootstrap with a strong
   independent model as a temporary stand-in until humans are available.
6. **Sign-off on the pre-registration** (primary slice/position/metric/thresholds)
   before the confirmation split is touched.

**Nice to have:**
7. Confirmation of the trivia data source/license (OpenTriviaQA is public) or a
   preferred MCQ source.
8. Whether you want me to **scaffold the Modal app + Tier-0/1 tests now** (I can
   write the code and CPU tests without credentials; only the GPU runs need the
   secrets above).

Note: this Cloud Agent VM has no GPU and no Modal/HF credentials, so I can build
and run **Tier 0 (CPU)** here immediately, but **Tiers 1+ require the Modal and HF
secrets above.** If you add them, I can also propose an env-setup agent so the
Modal/HF toolchain is preinstalled for future agents.
