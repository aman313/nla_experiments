"""Transformers-backed activation verbalizer (the documented SGLang fallback).

The vendored :class:`NLAClient` drives the AV through an SGLang ``input_embeds``
server. For the M0 golden gate / smoke test and small M1 runs we use the
``transformers`` ``inputs_embeds`` path instead (kitft/nla-inference README:
"a transformers ``inputs_embeds`` path is the fallback for the smoke test").

This avoids the heavy/fragile SGLang dependency, is deterministic under greedy
decoding, and reuses the *exact* injection math from the vendored single-file
client (template, scale, embedding scale, neighbor-checked injection) so it
stays faithful to the trained recipe.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np

from nla_sycophancy.vendor.nla_inference import (
    EXPLANATION_RE,
    inject_at_marked_positions,
    load_nla_config,
    normalize_activation,
    resolve_embed_scale,
)


class TransformersAV:
    """AV inference via ``model.generate(inputs_embeds=...)``."""

    def __init__(
        self,
        checkpoint_dir: str | Path,
        *,
        device: str = "cuda",
        dtype: str = "bfloat16",
        injection_scale_override: Optional[float] = None,
    ):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        checkpoint_dir = str(checkpoint_dir)
        self.tokenizer = AutoTokenizer.from_pretrained(
            checkpoint_dir, trust_remote_code=True
        )
        self.cfg = load_nla_config(
            checkpoint_dir, self.tokenizer,
            injection_scale_override=injection_scale_override,
        )
        self.embed_scale = resolve_embed_scale(checkpoint_dir)
        self.model = AutoModelForCausalLM.from_pretrained(
            checkpoint_dir, torch_dtype=getattr(torch, dtype),
            trust_remote_code=True,
        ).to(device).eval()
        self.device = device
        self._embed_layer = self.model.get_input_embeddings()

    def _build_inputs_embeds(self, v_raw: np.ndarray):
        import torch

        content = self.cfg.actor_prompt_template.format(
            injection_char=self.cfg.injection_char
        )
        input_ids = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": content}],
            tokenize=True, add_generation_prompt=True,
        )
        ids_t = torch.tensor(input_ids, dtype=torch.long).unsqueeze(0)
        with torch.no_grad():
            embeds = (self._embed_layer(ids_t.to(self.device)) * self.embed_scale).float()
        v = torch.as_tensor(np.asarray(v_raw, dtype=np.float32))
        assert v.numel() == self.cfg.d_model, (
            f"activation length {v.numel()} != d_model {self.cfg.d_model}"
        )
        v_scaled = normalize_activation(v.view(1, -1), self.cfg.injection_scale)
        injected = inject_at_marked_positions(
            ids_t.to(self.device), embeds, v_scaled.to(self.device),
            self.cfg.injection_token_id,
            self.cfg.injection_left_neighbor_id,
            self.cfg.injection_right_neighbor_id,
        )
        return injected.to(self._embed_layer.weight.dtype)

    def verbalize(
        self,
        activation: np.ndarray,
        *,
        temperature: float = 0.0,
        max_new_tokens: int = 200,
        extract_explanation: bool = True,
    ) -> str:
        import torch

        embeds = self._build_inputs_embeds(activation)
        attn = torch.ones(embeds.shape[:2], dtype=torch.long, device=self.device)
        do_sample = temperature > 0
        gen_kwargs = dict(
            inputs_embeds=embeds, attention_mask=attn,
            max_new_tokens=max_new_tokens,
            pad_token_id=self.tokenizer.eos_token_id,
            do_sample=do_sample,
        )
        if do_sample:
            gen_kwargs["temperature"] = temperature
        with torch.no_grad():
            out = self.model.generate(**gen_kwargs)
        # With inputs_embeds, generate() returns only the new tokens.
        text = self.tokenizer.decode(out[0], skip_special_tokens=True)
        if not extract_explanation:
            return text
        m = EXPLANATION_RE.search(text)
        return m.group(1).strip() if m else text
