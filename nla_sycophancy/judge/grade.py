"""M2: independent judge model that scores NLA explanations on the rubric.

The judge must be **independent** of the target/AV/AR family (here Qwen) to
avoid shared-prior circularity. Default is ``microsoft/Phi-3.5-mini-instruct``
(MIT, ungated, a different model family). It can be swapped for a stronger API
model once a key is available; the rubric and parsing are model-agnostic.

Decoding is greedy (deterministic) so the frozen grader is reproducible.
"""

from __future__ import annotations

from typing import Optional

from nla_sycophancy.judge.rubric import (
    DIM_KEYS,
    build_judge_messages,
    parse_judge_response,
)

DEFAULT_JUDGE = "microsoft/Phi-3.5-mini-instruct"


class JudgeModel:
    def __init__(
        self,
        model_id: str = DEFAULT_JUDGE,
        *,
        device: str = "cuda",
        dtype: str = "bfloat16",
    ):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.model_id = model_id
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        # Left-pad for batched decoder generation.
        self.tokenizer.padding_side = "left"
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id, torch_dtype=getattr(torch, dtype), trust_remote_code=True,
        ).to(device).eval()
        self.device = device

    def _generate(self, prompts: list[str], max_new_tokens: int) -> list[str]:
        import torch

        enc = self.tokenizer(
            prompts, return_tensors="pt", padding=True, truncation=True,
            max_length=2048,
        ).to(self.device)
        with torch.no_grad():
            out = self.model.generate(
                **enc, max_new_tokens=max_new_tokens, do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id,
            )
        gen = out[:, enc["input_ids"].shape[1]:]
        return self.tokenizer.batch_decode(gen, skip_special_tokens=True)

    def grade_batch(
        self,
        explanations: list[str],
        *,
        batch_size: int = 16,
        max_new_tokens: int = 96,
    ) -> list[Optional[dict[str, float]]]:
        prompts = [
            self.tokenizer.apply_chat_template(
                build_judge_messages(e), tokenize=False, add_generation_prompt=True
            )
            for e in explanations
        ]
        results: list[Optional[dict[str, float]]] = []
        for i in range(0, len(prompts), batch_size):
            chunk = prompts[i:i + batch_size]
            outs = self._generate(chunk, max_new_tokens)
            results.extend(parse_judge_response(o) for o in outs)
        return results

    def grade(self, explanation: str, **kw) -> Optional[dict[str, float]]:
        return self.grade_batch([explanation], **kw)[0]


def zero_dims() -> dict[str, float]:
    """A neutral all-zero dimension dict (used when grading fails)."""
    return {k: 0.0 for k in DIM_KEYS}
