"""Embed texts with a decoder LLM → one vector per text (pooled hidden states)."""

from __future__ import annotations

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from rich import print
from config import EmbedConfig
from pool import Pooler


class TextEmbedder:
    """Frozen decoder LLM wrapped to produce one pooled vector per input string."""

    def __init__(self, config: EmbedConfig | None = None) -> None:
        self.config = config or EmbedConfig()
        self.model_name = self.config.model_name
        self.device = "mps" if torch.backends.mps.is_available() else "cpu"
        self.pool_mode = self.config.pool_mode
        self.pooler = Pooler(self.pool_mode)
        self.dtype = self.config.dtype
        self.max_length = self.config.max_length

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(self.model_name).to(
            device=self.device, dtype=self.dtype
        )
        self.model.eval()


    @property
    def hidden_size(self) -> int:
        return int(self.model.config.hidden_size)

    def sync(self) -> None:
        if self.device == "mps":
            torch.mps.synchronize()

    def embed(
        self,
        texts: list[str],
        *,
        mode: str | None = None,
        dtype: torch.dtype | None = None,
        max_length: int | None = None,
    ) -> torch.Tensor:
        """Return one vector per text: [B, hidden_size].

        If max_length is set, truncate (and pad) to that many tokens — critical
        for long BEV states on Mac so batches stay in memory and finish fast.
        """
        # Checking and adjusting the model datatype to the configuration
        if dtype is not None and dtype != self.dtype:
            self.model.to(dtype=dtype)
            self.dtype = dtype

        effective_max_length = (
            max_length if max_length is not None else self.max_length
        )
        tok_kwargs: dict = {"return_tensors": "pt", "padding": True}
        if effective_max_length is not None:
            tok_kwargs["truncation"] = True
            tok_kwargs["max_length"] = effective_max_length

        batch = self.tokenizer(texts, **tok_kwargs)
        # We will get 2 arrays (token ids and also attention mask)
        batch = {k: v.to(self.device) for k, v in batch.items()}

        # We will get the hidden states of the last layer
        pooler = Pooler(mode) if mode is not None and mode != self.pool_mode else self.pooler
        with torch.no_grad():
            out = self.model(**batch, output_hidden_states=True)
            hidden = out.hidden_states[-1]  # [B, L, H]
            return pooler.pool(hidden, batch["attention_mask"])
