"""Batch container for variable-option decision examples with deduplication and [B, N] mask."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch


@dataclass
class OptionBatch:
    """B states, N unique options, [B, N] ownership mask, and [B] target column indices."""

    state_texts: list[str]
    unique_options: list[str]
    mask: torch.Tensor  # [B, N] bool: True if option j belongs to state i
    target_idx: torch.Tensor  # [B] long: column index in [0, N) of state i's correct option
    total_options: int  # sum of option counts before deduplication

    @classmethod
    def from_examples(cls, examples: list[dict[str, Any]]) -> OptionBatch:
        if not examples:
            raise ValueError("examples must be non-empty")

        state_texts: list[str] = []
        unique_options: list[str] = []
        opt_to_id: dict[str, int] = {}
        row_option_ids: list[list[int]] = []
        target_ids: list[int] = []
        total_options = 0

        for ex in examples:
            state_texts.append(cls._format_state(ex))
            ids: list[int] = []
            for choice in ex["choices"]:
                opt_text = cls._format_option(choice)
                oid = opt_to_id.get(opt_text)
                if oid is None:
                    oid = len(unique_options)
                    opt_to_id[opt_text] = oid
                    unique_options.append(opt_text)
                ids.append(oid)
            total_options += len(ids)
            row_option_ids.append(ids)
            target_ids.append(ids[int(ex["correct_idx"])])

        B = len(state_texts)
        N = len(unique_options)
        mask = torch.zeros((B, N), dtype=torch.bool)
        for i, ids in enumerate(row_option_ids):
            mask[i, ids] = True

        return cls(
            state_texts=state_texts,
            unique_options=unique_options,
            mask=mask,
            target_idx=torch.tensor(target_ids, dtype=torch.long),
            total_options=total_options,
        )

    @staticmethod
    def _format_state(ex: dict[str, Any]) -> str:
        return f"Situation: {ex['state']}\nQuestion: {ex['question']}"

    @staticmethod
    def _format_option(choice: str) -> str:
        return f"Option: {choice}"
