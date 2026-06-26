# nla_experiments

Research design for testing whether **Natural Language Autoencoder (NLA)**
explanations of LLM activations are *faithful* to **counterfactual sycophancy
labels**.

NLAs (Fraser-Taliente, Kantamneni, Ong et al., *Transformer Circuits*, 2026;
code: [`kitft/natural_language_autoencoders`](https://github.com/kitft/natural_language_autoencoders),
inference: [`kitft/nla-inference`](https://github.com/kitft/nla-inference))
translate residual-stream activations into natural language via an activation
verbalizer (AV) and back via an activation reconstructor (AR). The counterfactual
sycophancy labeling methodology follows
[arXiv:2509.21344](https://arxiv.org/pdf/2509.21344) (and Kirch et al.):
control / user-suggests-correct / user-suggests-incorrect variants, labeling an
item sycophantic when the model abandons a correct control answer for the user's
suggested wrong option.

## Documents

1. [`docs/01_critique.md`](docs/01_critique.md) — issues found in the first-draft
   plan (model availability, prompt leakage, confounds, statistics, causal-test
   pitfalls).
2. [`docs/02_final_experiment_plan.md`](docs/02_final_experiment_plan.md) — the
   revised plan: within-incorrect-belief primary contrast, incremental validity
   over a prompt-text baseline, placebo specificity, dose–response, AR-steering
   causal test, reliability heuristics, pre-registration, and caveats.
3. [`docs/03_implementation_plan.md`](docs/03_implementation_plan.md) — repo
   layout, dependencies, schemas, milestones, compute budget, testing, risks.
4. [`docs/04_testing_and_modal.md`](docs/04_testing_and_modal.md) — tiered testing
   strategy, running all GPU stages on Modal (app shape, GPU sizing, secrets),
   and what's needed to operate it that way.

## Core question

Not merely *"can NLA explanations predict the sycophancy label?"* — which prompt
leakage can satisfy trivially — but *"do they surface the internal disposition
that causes the switch, beyond what is readable from the prompt text?"*
