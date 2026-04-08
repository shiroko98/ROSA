from __future__ import annotations

from typing import Dict, List, Optional, Sequence

from rosa_addressing import AddressMeta, ExactMatchRosaState, SuffixAutomatonRosaState


def _state_for_sequence_mode(
    sequence_mode: str,
    *,
    min_match_len: int,
    special_ids: Optional[set],
    forbid_special_target: bool,
):
    if sequence_mode == "online_exact":
        state_cls = ExactMatchRosaState
    elif sequence_mode == "online_sam":
        state_cls = SuffixAutomatonRosaState
    else:
        raise ValueError(f"不支持为 sequence_mode={sequence_mode} 构建训练地址缓存。")
    return state_cls(
        min_match_len=min_match_len,
        special_ids=special_ids,
        forbid_special_target=forbid_special_target,
    )


def _meta_rows_to_precomputed(rows: Sequence[AddressMeta]) -> Dict[str, List[int]]:
    return {
        "rosa_ids": [row.addr_id for row in rows],
        "fired_match_lens": [row.fired_match_len for row in rows],
        "raw_best_lens": [row.raw_match_len for row in rows],
    }


def build_sequence_online_precomputed_rosa(
    docs_tokens: Sequence[Sequence[int]],
    *,
    sequence_mode: str,
    min_match_len: int,
    special_ids: Optional[set],
    forbid_special_target: bool,
    memory_prefixes: Optional[Sequence[Sequence[int]]] = None,
) -> List[Dict[str, List[int]]]:
    if memory_prefixes is not None and len(memory_prefixes) != len(docs_tokens):
        raise ValueError("memory_prefixes 长度必须与 docs_tokens 一致。")

    out: List[Dict[str, List[int]]] = []
    for doc_idx, ids in enumerate(docs_tokens):
        state = _state_for_sequence_mode(
            sequence_mode,
            min_match_len=min_match_len,
            special_ids=special_ids,
            forbid_special_target=forbid_special_target,
        )
        prefix = list(memory_prefixes[doc_idx]) if memory_prefixes is not None else []
        if prefix:
            state.prefill(prefix)
        rows = state.prefill(ids)
        out.append(_meta_rows_to_precomputed(rows))
    return out
