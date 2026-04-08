from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch

from rosa_runtime import RosaAddressBatch


@dataclass(frozen=True)
class AddressMeta:
    addr_id: int
    raw_match_len: int
    fired_match_len: int
    valid_mask: bool
    special_mask: bool = False
    source_type: str = "token_exact"

    @property
    def hit_flag(self) -> bool:
        return self.valid_mask


@dataclass(frozen=True)
class RosaStateSnapshot:
    token_ids: Tuple[int, ...]
    num_tokens: int
    last_address: Optional[AddressMeta] = None


def build_address_meta(
    addr_id: int,
    raw_match_len: int,
    *,
    min_match_len: int,
    special_mask: bool = False,
    source_type: str = "token_exact",
) -> AddressMeta:
    valid = addr_id >= 0 and raw_match_len >= min_match_len and not special_mask
    fired = raw_match_len if valid else 0
    return AddressMeta(
        addr_id=addr_id if valid else -1,
        raw_match_len=raw_match_len,
        fired_match_len=fired,
        valid_mask=valid,
        special_mask=special_mask,
        source_type=source_type,
    )


def exact_match_step_address(
    history: Sequence[int],
    token_id: int,
    *,
    min_match_len: int,
    special_ids: Optional[set] = None,
    forbid_special_target: bool = True,
    source_type: str = "token_exact",
) -> AddressMeta:
    special_ids = special_ids or set()
    gi = len(history)
    combined = list(history) + [token_id]

    best_len = 0
    best_j = -1
    for j in range(gi):
        if combined[j] != token_id:
            continue
        m = 1
        while j - m >= 0 and gi - m >= 0 and combined[j - m] == combined[gi - m]:
            m += 1
        if m > best_len or (m == best_len and j > best_j):
            best_len = m
            best_j = j

    addr_id = -1
    special_mask = False
    if best_j >= 0 and best_j + 1 < len(combined):
        candidate = combined[best_j + 1]
        if forbid_special_target and candidate in special_ids:
            special_mask = True
        else:
            addr_id = candidate

    return build_address_meta(
        addr_id,
        best_len,
        min_match_len=min_match_len,
        special_mask=special_mask,
        source_type=source_type,
    )


class ExactMatchRosaState:
    """
    参考级在线 ROSA 状态机。
    先保证 prefill / update_one 语义稳定，与现有整段 exact-match 定义逐 token 对齐；
    后续再在这个接口下替换为更高性能的状态结构。
    """

    def __init__(
        self,
        *,
        min_match_len: int = 1,
        special_ids: Optional[set] = None,
        forbid_special_target: bool = True,
        source_type: str = "token_exact",
    ):
        self.min_match_len = min_match_len
        self.special_ids = set(special_ids or set())
        self.forbid_special_target = forbid_special_target
        self.source_type = source_type
        self.reset()

    def reset(self) -> None:
        self._token_ids: List[int] = []
        self._last_address: Optional[AddressMeta] = None

    def prefill(self, token_ids: Sequence[int]) -> List[AddressMeta]:
        return [self.update_one(token_id) for token_id in token_ids]

    def update_one(self, token_id: int) -> AddressMeta:
        meta = exact_match_step_address(
            self._token_ids,
            token_id,
            min_match_len=self.min_match_len,
            special_ids=self.special_ids,
            forbid_special_target=self.forbid_special_target,
            source_type=self.source_type,
        )
        self._token_ids.append(token_id)
        self._last_address = meta
        return meta

    def snapshot(self) -> RosaStateSnapshot:
        return RosaStateSnapshot(
            token_ids=tuple(self._token_ids),
            num_tokens=len(self._token_ids),
            last_address=self._last_address,
        )

    def clone(self) -> "ExactMatchRosaState":
        cloned = ExactMatchRosaState(
            min_match_len=self.min_match_len,
            special_ids=self.special_ids,
            forbid_special_target=self.forbid_special_target,
            source_type=self.source_type,
        )
        cloned._token_ids = list(self._token_ids)
        cloned._last_address = self._last_address
        return cloned


def _coerce_batch_token_rows(token_ids: Any) -> List[List[int]]:
    if isinstance(token_ids, torch.Tensor):
        if token_ids.dim() == 1:
            return [token_ids.detach().cpu().tolist()]
        if token_ids.dim() == 2:
            return token_ids.detach().cpu().tolist()
        raise ValueError("token_ids tensor 只能是 1D 或 2D。")
    return [list(row) for row in token_ids]


class OnlineRosaBatchState:
    """批量在线 ROSA 状态，提供 prefill + decode/update 最小闭环。"""

    def __init__(self, states: Sequence[Any]):
        self.states = list(states)

    @classmethod
    def create(
        cls,
        batch_size: int,
        *,
        min_match_len: int = 1,
        special_ids: Optional[set] = None,
        forbid_special_target: bool = True,
        source_type: str = "token_exact",
        state_backend: str = "sam",
    ) -> "OnlineRosaBatchState":
        if state_backend == "exact_list":
            state_cls = ExactMatchRosaState
        elif state_backend == "sam":
            state_cls = OnlineRosaState
        else:
            raise ValueError(f"未知 state_backend: {state_backend}")
        return cls(
            [
                state_cls(
                    min_match_len=min_match_len,
                    special_ids=special_ids,
                    forbid_special_target=forbid_special_target,
                    source_type=source_type,
                )
                for _ in range(batch_size)
            ]
        )

    def reset(self) -> None:
        for state in self.states:
            state.reset()

    def prefill(self, token_ids: Any, *, pad_id: Optional[int] = None) -> None:
        rows = _coerce_batch_token_rows(token_ids)
        if len(rows) != len(self.states):
            raise ValueError("prefill 的 batch 大小必须与 OnlineRosaBatchState 中的 state 数量一致。")
        for state, row in zip(self.states, rows):
            filtered = [tok for tok in row if pad_id is None or tok != pad_id]
            if filtered:
                state.prefill(filtered)

    def address_tokens(
        self,
        token_ids: Any,
        *,
        pad_id: Optional[int] = None,
        device: Optional[torch.device] = None,
    ) -> Dict[str, torch.Tensor]:
        rows = _coerce_batch_token_rows(token_ids)
        if len(rows) != len(self.states):
            raise ValueError("address_tokens 的 batch 大小必须与 OnlineRosaBatchState 中的 state 数量一致。")

        batch_meta: List[List[AddressMeta]] = []
        for state, row in zip(self.states, rows):
            row_meta: List[AddressMeta] = []
            for tok in row:
                if pad_id is not None and tok == pad_id:
                    row_meta.append(
                        build_address_meta(
                            -1,
                            0,
                            min_match_len=state.min_match_len,
                            source_type=state.source_type,
                        )
                    )
                    continue
                row_meta.append(state.update_one(tok))
            batch_meta.append(row_meta)
        return pack_address_meta_batch(batch_meta, device=device)

    def snapshot(self) -> List[RosaStateSnapshot]:
        return [state.snapshot() for state in self.states]

    def clone(self) -> "OnlineRosaBatchState":
        return OnlineRosaBatchState([state.clone() for state in self.states])


def build_online_rosa_batch_state(
    batch_size: int,
    *,
    min_match_len: int = 1,
    special_ids: Optional[set] = None,
    forbid_special_target: bool = True,
    source_type: str = "token_exact",
    state_backend: str = "sam",
) -> OnlineRosaBatchState:
    return OnlineRosaBatchState.create(
        batch_size,
        min_match_len=min_match_len,
        special_ids=special_ids,
        forbid_special_target=forbid_special_target,
        source_type=source_type,
        state_backend=state_backend,
    )


def pack_address_meta_batch(
    batch_meta: Sequence[Sequence[AddressMeta]],
    *,
    device: Optional[torch.device] = None,
) -> Dict[str, torch.Tensor]:
    bsz = len(batch_meta)
    seqlen = max((len(row) for row in batch_meta), default=0)

    addr_ids = torch.full((bsz, seqlen), -1, dtype=torch.long, device=device)
    raw_match_lens = torch.zeros((bsz, seqlen), dtype=torch.long, device=device)
    fired_match_lens = torch.zeros((bsz, seqlen), dtype=torch.long, device=device)
    valid_mask = torch.zeros((bsz, seqlen), dtype=torch.bool, device=device)
    special_mask = torch.zeros((bsz, seqlen), dtype=torch.bool, device=device)

    for b, row in enumerate(batch_meta):
        for i, meta in enumerate(row):
            addr_ids[b, i] = meta.addr_id
            raw_match_lens[b, i] = meta.raw_match_len
            fired_match_lens[b, i] = meta.fired_match_len
            valid_mask[b, i] = meta.valid_mask
            special_mask[b, i] = meta.special_mask

    return {
        "addr_ids": addr_ids,
        "raw_match_lens": raw_match_lens,
        "fired_match_lens": fired_match_lens,
        "valid_mask": valid_mask,
        "special_mask": special_mask,
    }


def make_rosa_address_batch(addressed: Dict[str, torch.Tensor], *, source: str = "unknown") -> RosaAddressBatch:
    return RosaAddressBatch(
        addr_ids=addressed["addr_ids"],
        raw_match_lens=addressed["raw_match_lens"],
        fired_match_lens=addressed["fired_match_lens"],
        valid_mask=addressed["valid_mask"],
        special_mask=addressed["special_mask"],
        source=source,
    )


def _pad_filtered_mem(mem_row: List[int], pad_id: int) -> List[int]:
    return [x for x in mem_row if x != pad_id]


def online_rosa_address_meta_with_memory(
    input_ids: torch.Tensor,
    memory_ids: Optional[torch.Tensor],
    min_match_len: int,
    pad_id: int,
    special_ids: Optional[set] = None,
    forbid_special_target: bool = True,
) -> List[List[AddressMeta]]:
    bsz, _ = input_ids.shape
    special_ids = special_ids or set()
    seqs = input_ids.detach().cpu().tolist()
    mems = memory_ids.detach().cpu().tolist() if memory_ids is not None else [[] for _ in range(bsz)]
    out: List[List[AddressMeta]] = []

    for b in range(bsz):
        seq = seqs[b]
        mem = _pad_filtered_mem(mems[b], pad_id)
        row: List[AddressMeta] = []
        state = ExactMatchRosaState(
            min_match_len=min_match_len,
            special_ids=special_ids,
            forbid_special_target=forbid_special_target,
        )
        if mem:
            state.prefill(mem)
        row.extend(state.prefill(seq))
        out.append(row)
    return out


class SuffixAutomatonRosaState:
    """真正的在线 SAM 状态机，用于 prefill/decode 主线。"""

    def __init__(
        self,
        *,
        min_match_len: int = 1,
        special_ids: Optional[set] = None,
        forbid_special_target: bool = True,
        source_type: str = "token_exact",
    ):
        self.min_match_len = min_match_len
        self.special_ids = set(special_ids or set())
        self.forbid_special_target = forbid_special_target
        self.source_type = source_type
        self.reset()

    def reset(self) -> None:
        self._token_ids: List[int] = []
        self._last_address: Optional[AddressMeta] = None
        self._trans: List[Dict[int, int]] = [dict()]
        self._link: List[int] = [-1]
        self._length: List[int] = [0]
        self._endpos: List[int] = [-1]
        self._last = 0

    def _new_state(self) -> int:
        self._trans.append({})
        self._link.append(-1)
        self._length.append(0)
        self._endpos.append(-1)
        return len(self._trans) - 1

    def _lookup_token(self, idx: int, token_id: int) -> int:
        if idx < len(self._token_ids):
            return self._token_ids[idx]
        if idx == len(self._token_ids):
            return token_id
        return -1

    def prefill(self, token_ids: Sequence[int]) -> List[AddressMeta]:
        return [self.update_one(token_id) for token_id in token_ids]

    def update_one(self, token_id: int) -> AddressMeta:
        cur_idx = len(self._token_ids)
        r = self._new_state()
        self._length[r] = self._length[self._last] + 1
        p = self._last

        while p != -1 and token_id not in self._trans[p]:
            self._trans[p][token_id] = r
            p = self._link[p]

        if p == -1:
            self._link[r] = 0
        else:
            q = self._trans[p][token_id]
            if self._length[p] + 1 == self._length[q]:
                self._link[r] = q
            else:
                u = self._new_state()
                self._trans[u] = self._trans[q].copy()
                self._length[u] = self._length[p] + 1
                self._link[u] = self._link[q]
                self._endpos[u] = self._endpos[q]

                while p != -1 and self._trans[p].get(token_id) == q:
                    self._trans[p][token_id] = u
                    p = self._link[p]

                self._link[q] = u
                self._link[r] = u

        v = r
        best_m = 0
        addr_id = -1
        special_mask = False
        while v != -1:
            if self._length[v] > 0 and self._endpos[v] >= 0:
                best_m = self._length[v]
                if best_m >= self.min_match_len:
                    candidate_idx = self._endpos[v] + 1
                    candidate = self._lookup_token(candidate_idx, token_id)
                    if candidate >= 0:
                        if self.forbid_special_target and candidate in self.special_ids:
                            special_mask = True
                        else:
                            addr_id = candidate
                break
            v = self._link[v]

        self._last = r
        v = self._last
        while v != -1 and self._endpos[v] < cur_idx:
            self._endpos[v] = cur_idx
            v = self._link[v]

        self._token_ids.append(token_id)
        meta = build_address_meta(
            addr_id,
            best_m,
            min_match_len=self.min_match_len,
            special_mask=special_mask,
            source_type=self.source_type,
        )
        self._last_address = meta
        return meta

    def snapshot(self) -> RosaStateSnapshot:
        return RosaStateSnapshot(
            token_ids=tuple(self._token_ids),
            num_tokens=len(self._token_ids),
            last_address=self._last_address,
        )

    def clone(self) -> "SuffixAutomatonRosaState":
        cloned = SuffixAutomatonRosaState(
            min_match_len=self.min_match_len,
            special_ids=self.special_ids,
            forbid_special_target=self.forbid_special_target,
            source_type=self.source_type,
        )
        cloned._token_ids = list(self._token_ids)
        cloned._last_address = self._last_address
        cloned._trans = [row.copy() for row in self._trans]
        cloned._link = list(self._link)
        cloned._length = list(self._length)
        cloned._endpos = list(self._endpos)
        cloned._last = self._last
        return cloned


OnlineRosaState = SuffixAutomatonRosaState


def _stateful_online_sam_address_meta_with_memory(
    input_ids: torch.Tensor,
    memory_ids: Optional[torch.Tensor],
    min_match_len: int,
    pad_id: int,
    special_ids: Optional[set] = None,
    forbid_special_target: bool = True,
) -> List[List[AddressMeta]]:
    bsz, _ = input_ids.shape
    special_ids = special_ids or set()
    seqs = input_ids.detach().cpu().tolist()
    mems = memory_ids.detach().cpu().tolist() if memory_ids is not None else [[] for _ in range(bsz)]
    out: List[List[AddressMeta]] = []

    for b in range(bsz):
        seq = seqs[b]
        mem = _pad_filtered_mem(mems[b], pad_id)
        row: List[AddressMeta] = []
        state = SuffixAutomatonRosaState(
            min_match_len=min_match_len,
            special_ids=special_ids,
            forbid_special_target=forbid_special_target,
        )
        if mem:
            state.prefill(mem)
        row.extend(state.prefill(seq))
        out.append(row)
    return out


def _sam_sequence_address_meta_with_memory(
    input_ids: torch.Tensor,
    memory_ids: Optional[torch.Tensor],
    min_match_len: int,
    pad_id: int,
    special_ids: Optional[set] = None,
    forbid_special_target: bool = True,
) -> List[List[AddressMeta]]:
    bsz, seqlen = input_ids.shape
    special_ids = special_ids or set()
    seqs = input_ids.detach().cpu().tolist()
    mems = memory_ids.detach().cpu().tolist() if memory_ids is not None else [[] for _ in range(bsz)]
    out: List[List[AddressMeta]] = []

    for b in range(bsz):
        seq = seqs[b]
        mem = _pad_filtered_mem(mems[b], pad_id)
        combined = mem + seq
        offset = len(mem)

        preds, match_lens = sam_rosa_predict(combined, min_match_len=min_match_len)
        sliced_preds = preds[offset:offset + seqlen]
        sliced_match_lens = match_lens[offset:offset + seqlen]

        row: List[AddressMeta] = []
        for pred, raw_m in zip(sliced_preds, sliced_match_lens):
            special_mask = pred >= 0 and forbid_special_target and pred in special_ids
            row.append(
                build_address_meta(
                    pred,
                    raw_m,
                    min_match_len=min_match_len,
                    special_mask=special_mask,
                )
            )
        out.append(row)

    return out


def online_sam_address_meta_with_memory(
    input_ids: torch.Tensor,
    memory_ids: Optional[torch.Tensor],
    min_match_len: int,
    pad_id: int,
    special_ids: Optional[set] = None,
    forbid_special_target: bool = True,
    implementation: str = "fast",
) -> List[List[AddressMeta]]:
    if implementation == "stateful":
        return _stateful_online_sam_address_meta_with_memory(
            input_ids=input_ids,
            memory_ids=memory_ids,
            min_match_len=min_match_len,
            pad_id=pad_id,
            special_ids=special_ids,
            forbid_special_target=forbid_special_target,
        )
    if implementation != "fast":
        raise ValueError(f"未知 online_sam implementation: {implementation}")
    return _sam_sequence_address_meta_with_memory(
        input_ids=input_ids,
        memory_ids=memory_ids,
        min_match_len=min_match_len,
        pad_id=pad_id,
        special_ids=special_ids,
        forbid_special_target=forbid_special_target,
    )


def naive_rosa_address_meta_with_memory(
    input_ids: torch.Tensor,
    memory_ids: Optional[torch.Tensor],
    min_match_len: int,
    pad_id: int,
    special_ids: Optional[set] = None,
    forbid_special_target: bool = True,
) -> List[List[AddressMeta]]:
    bsz, seqlen = input_ids.shape
    special_ids = special_ids or set()
    seqs = input_ids.detach().cpu().tolist()
    mems = memory_ids.detach().cpu().tolist() if memory_ids is not None else [[] for _ in range(bsz)]
    out: List[List[AddressMeta]] = []

    for b in range(bsz):
        seq = seqs[b]
        mem = _pad_filtered_mem(mems[b], pad_id)
        combined = mem + seq
        offset = len(mem)
        row: List[AddressMeta] = []

        for i in range(seqlen):
            gi = offset + i
            cur = combined[gi]
            if cur < 0:
                row.append(build_address_meta(-1, 0, min_match_len=min_match_len))
                continue

            best_len = 0
            best_j = -1
            for j in range(gi):
                if combined[j] != cur:
                    continue
                m = 1
                while j - m >= 0 and gi - m >= 0 and combined[j - m] == combined[gi - m]:
                    m += 1
                if m > best_len or (m == best_len and j > best_j):
                    best_len = m
                    best_j = j

            addr_id = -1
            special_mask = False
            if best_j >= 0 and best_j + 1 < len(combined):
                candidate = combined[best_j + 1]
                if forbid_special_target and candidate in special_ids:
                    special_mask = True
                else:
                    addr_id = candidate
            row.append(
                build_address_meta(
                    addr_id,
                    best_len,
                    min_match_len=min_match_len,
                    special_mask=special_mask,
                )
            )
        out.append(row)

    return out


def sam_rosa_predict(seq: Sequence[int], min_match_len: int = 1) -> Tuple[List[int], List[int]]:
    n = len(seq)
    pred = [-1] * n
    match_len = [0] * n
    if n == 0:
        return pred, match_len

    s = 2 * n + 1
    trans: List[Optional[Dict[int, int]]] = [None] * s
    link = [-1] * s
    length = [0] * s
    endpos = [-1] * s

    trans[0] = {}
    last = 0
    z = 1

    for i, t in enumerate(seq):
        r = z
        z += 1
        trans[r] = {}
        length[r] = length[last] + 1
        p = last

        while p != -1 and t not in trans[p]:
            trans[p][t] = r
            p = link[p]

        if p == -1:
            link[r] = 0
        else:
            q = trans[p][t]
            if length[p] + 1 == length[q]:
                link[r] = q
            else:
                u = z
                z += 1
                trans[u] = trans[q].copy()
                length[u] = length[p] + 1
                link[u] = link[q]
                endpos[u] = endpos[q]

                while p != -1 and trans[p].get(t) == q:
                    trans[p][t] = u
                    p = link[p]

                link[q] = u
                link[r] = u

        v = r
        a = -1
        best_m = 0
        while v != -1:
            if length[v] > 0 and endpos[v] >= 0:
                best_m = length[v]
                if best_m >= min_match_len:
                    idx = endpos[v] + 1
                    if 0 <= idx < n:
                        a = seq[idx]
                break
            v = link[v]

        pred[i] = a
        match_len[i] = best_m
        last = r

        v = last
        while v != -1 and endpos[v] < i:
            endpos[v] = i
            v = link[v]

    return pred, match_len


def sam_rosa_address_meta_with_memory(
    input_ids: torch.Tensor,
    memory_ids: Optional[torch.Tensor],
    min_match_len: int,
    pad_id: int,
    special_ids: Optional[set] = None,
    forbid_special_target: bool = True,
) -> List[List[AddressMeta]]:
    return _sam_sequence_address_meta_with_memory(
        input_ids=input_ids,
        memory_ids=memory_ids,
        min_match_len=min_match_len,
        pad_id=pad_id,
        special_ids=special_ids,
        forbid_special_target=forbid_special_target,
    )


def rosa_addressing_with_memory(
    input_ids: torch.Tensor,
    memory_ids: Optional[torch.Tensor],
    min_match_len: int,
    pad_id: int,
    special_ids: Optional[set] = None,
    forbid_special_target: bool = True,
    backend: str = "sam",
) -> Dict[str, torch.Tensor]:
    if backend == "naive":
        batch_meta = naive_rosa_address_meta_with_memory(
            input_ids=input_ids,
            memory_ids=memory_ids,
            min_match_len=min_match_len,
            pad_id=pad_id,
            special_ids=special_ids,
            forbid_special_target=forbid_special_target,
        )
    elif backend == "sam":
        batch_meta = sam_rosa_address_meta_with_memory(
            input_ids=input_ids,
            memory_ids=memory_ids,
            min_match_len=min_match_len,
            pad_id=pad_id,
            special_ids=special_ids,
            forbid_special_target=forbid_special_target,
        )
    else:
        raise ValueError(f"未知 rosa backend: {backend}")
    return pack_address_meta_batch(batch_meta, device=input_ids.device)


class RosaAddressEngine:
    """统一的 ROSA 地址引擎接口，兼容 reference backend 与 sequence-online 扫描。"""

    def __init__(
        self,
        *,
        min_match_len: int,
        pad_id: int,
        backend: str = "sam",
        sequence_mode: str = "reference_backend",
        online_sam_impl: str = "fast",
        special_ids: Optional[set] = None,
        forbid_special_target: bool = True,
        source_type: str = "token_exact",
    ):
        self.min_match_len = min_match_len
        self.pad_id = pad_id
        self.backend = backend
        self.sequence_mode = sequence_mode
        self.online_sam_impl = online_sam_impl
        self.special_ids = set(special_ids or set())
        self.forbid_special_target = forbid_special_target
        self.source_type = source_type

    def _state_backend(self) -> str:
        if self.sequence_mode == "online_exact":
            return "exact_list"
        return "sam"

    def init_state(self, batch_size: int) -> OnlineRosaBatchState:
        return build_online_rosa_batch_state(
            batch_size,
            min_match_len=self.min_match_len,
            special_ids=self.special_ids,
            forbid_special_target=self.forbid_special_target,
            source_type=self.source_type,
            state_backend=self._state_backend(),
        )

    def forward_step(
        self,
        input_ids: torch.Tensor,
        state: OnlineRosaBatchState,
        *,
        device: Optional[torch.device] = None,
    ) -> RosaAddressBatch:
        batch = make_rosa_address_batch(
            state.address_tokens(
                input_ids,
                pad_id=self.pad_id,
                device=device or input_ids.device,
            ),
            source=f"step:{self.sequence_mode}",
        )
        return batch.to(device) if device is not None else batch

    def forward_seq(
        self,
        input_ids: torch.Tensor,
        memory_ids: Optional[torch.Tensor] = None,
        *,
        device: Optional[torch.device] = None,
    ) -> RosaAddressBatch:
        if self.sequence_mode == "online_exact":
            batch_meta = online_rosa_address_meta_with_memory(
                input_ids=input_ids,
                memory_ids=memory_ids,
                min_match_len=self.min_match_len,
                pad_id=self.pad_id,
                special_ids=self.special_ids,
                forbid_special_target=self.forbid_special_target,
            )
            batch = make_rosa_address_batch(
                pack_address_meta_batch(batch_meta, device=input_ids.device),
                source="seq:online_exact",
            )
        elif self.sequence_mode == "online_sam":
            batch_meta = online_sam_address_meta_with_memory(
                input_ids=input_ids,
                memory_ids=memory_ids,
                min_match_len=self.min_match_len,
                pad_id=self.pad_id,
                special_ids=self.special_ids,
                forbid_special_target=self.forbid_special_target,
                implementation=self.online_sam_impl,
            )
            batch = make_rosa_address_batch(
                pack_address_meta_batch(batch_meta, device=input_ids.device),
                source="seq:online_sam",
            )
        elif self.sequence_mode == "reference_backend":
            batch = make_rosa_address_batch(
                rosa_addressing_with_memory(
                    input_ids=input_ids,
                    memory_ids=memory_ids,
                    min_match_len=self.min_match_len,
                    pad_id=self.pad_id,
                    special_ids=self.special_ids,
                    forbid_special_target=self.forbid_special_target,
                    backend=self.backend,
                ),
                source=f"seq:reference:{self.backend}",
            )
        else:
            raise ValueError(f"未知 rosa sequence_mode: {self.sequence_mode}")
        return batch.to(device) if device is not None else batch
