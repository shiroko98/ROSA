from __future__ import annotations

from typing import Dict, List, Optional, Sequence

from rosa_addressing import ExactMatchRosaState, RosaStateSnapshot, SuffixAutomatonRosaState


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
        raise ValueError(f"不支持为 sequence_mode={sequence_mode} 构建训练状态快照。")
    return state_cls(
        min_match_len=min_match_len,
        special_ids=special_ids,
        forbid_special_target=forbid_special_target,
    )


def build_sequence_online_state_snapshots(
    docs_tokens: Sequence[Sequence[int]],
    *,
    sequence_mode: str,
    snapshot_interval: int,
    min_match_len: int,
    special_ids: Optional[set],
    forbid_special_target: bool,
    memory_prefixes: Optional[Sequence[Sequence[int]]] = None,
) -> List[Dict[str, Sequence]]:
    if snapshot_interval <= 0:
        raise ValueError("snapshot_interval 必须为正整数。")
    if memory_prefixes is not None and len(memory_prefixes) != len(docs_tokens):
        raise ValueError("memory_prefixes 长度必须与 docs_tokens 一致。")

    out: List[Dict[str, Sequence]] = []
    for doc_idx, ids in enumerate(docs_tokens):
        prefix = list(memory_prefixes[doc_idx]) if memory_prefixes is not None else []
        state = _state_for_sequence_mode(
            sequence_mode,
            min_match_len=min_match_len,
            special_ids=special_ids,
            forbid_special_target=forbid_special_target,
        )
        if prefix:
            state.prefill(prefix)
        positions: List[int] = [0]
        snapshots: List[RosaStateSnapshot] = [state.snapshot()]
        next_snapshot_pos = snapshot_interval
        for pos, token_id in enumerate(ids, start=1):
            state.update_one(token_id)
            if pos == next_snapshot_pos:
                positions.append(pos)
                snapshots.append(state.snapshot())
                next_snapshot_pos += snapshot_interval
        out.append(
            {
                "positions": tuple(positions),
                "snapshots": tuple(snapshots),
            }
        )
    return out
