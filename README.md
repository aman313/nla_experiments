# nla_experiments

Are NLA (Natural Language Autoencoder) explanations faithful to **counterfactual
sycophancy** labels? This repo implements the experiment plan in
[`docs/`](docs/) — see `02_final_experiment_plan.md` for the science and
`03_implementation_plan.md` for the engineering milestones.

## Status

| Milestone | Scope | State |
|---|---|---|
| **M0** — Harness & validation | sidecar/tokenizer contract, residual extraction, golden-MSE gate, smoke test | implemented; CPU tests green, GPU golden gate runs on Modal |
| **M1** — Dataset & labels | OpenTriviaQA load+filter, counterfactual variants, per-option-logprob rollouts, sycophancy labeling + attrition, pilot | implemented; CPU tests green, pilot runs on Modal |

The reference paper (arXiv:2509.21344, *"Linear probes rely on textual
evidence"*, following Kirch et al. 2026) uses **OpenTriviaQA** for its sycophancy
scenario; M1 uses the same source and the same control / correct-belief /
incorrect-belief counterfactual construction.

## Layout

```
configs/                 model + experiment YAML (Qwen2.5-7B is the default target)
nla_sycophancy/
  data/      source.py (OpenTriviaQA)   variants.py (counterfactual variants)
  target/    rollout.py  label.py  extract.py
  nla/       meta.py  av_client.py  ar_client.py
  analysis/  stats.py
  io/        schema.py  cache.py
  vendor/    nla_inference.py (kitft/nla-inference, Apache-2.0) + examples/
scripts/     modal_app.py (GPU), run_m0_golden.py, run_m1_pilot.py
tests/       Tier-0 CPU tests
```

## Quick start (CPU / Tier-0)

```bash
python -m virtualenv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python -m pytest -q          # Tier-0 unit tests (no GPU, no network)
```

## GPU work (Modal)

All GPU stages (target rollouts, activation extraction, AV inference, AR
reconstruction) run on [Modal](https://modal.com). Set `MODAL_TOKEN_ID` /
`MODAL_TOKEN_SECRET` (and `HF_TOKEN` for gated Gemma/Llama targets; Qwen is
ungated), then:

```bash
modal run scripts/modal_app.py::golden_mse_gate     # M0 golden-MSE plumbing gate
modal run scripts/modal_app.py::smoke_test          # M0 5-item end-to-end smoke
modal run scripts/modal_app.py::m1_pilot            # M1 pilot rollouts + attrition
```

See `docs/04_testing_and_modal.md` for the full testing tiers and Modal recipe.
