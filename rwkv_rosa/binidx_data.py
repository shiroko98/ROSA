"""Read upstream packed SFT binidx without re-tokenization or loading .bin into RAM.

EOT-delimited examples are unpacked before exact-content deduplication and splitting.
The selected physical rows are seed-random, not a potentially source-sorted prefix.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path

import numpy as np

from third_party.rwkv_lm_v7.binidx import MMapIndexedDataset


@dataclass
class BinidxDocument:
    tokens: object
    masks: object
    row: int
    start: int
    end: int
    content_hash: str
    supervised_tokens: int

    @property
    def input_ids(self):
        return self.tokens.get(self.row, self.start, self.end - self.start)

    @property
    def loss_mask(self):
        return self.masks.get(self.row, self.start, self.end - self.start)


def resolve_binidx_prefix(path):
    path = Path(path)
    if path.is_dir():
        candidates = sorted(p for p in (path / 'results').glob('*.idx') if not p.name.endswith('.mask.idx'))
        if not candidates:
            candidates = sorted(p for p in path.glob('*.idx') if not p.name.endswith('.mask.idx'))
        if len(candidates) != 1:
            raise ValueError(f'Expected exactly one token binidx prefix in {path}, got {len(candidates)}')
        path = candidates[0].with_suffix('')
    elif path.suffix in ('.bin', '.idx'):
        path = path.with_suffix('')
    for suffix in ('.bin', '.idx', '.mask.bin', '.mask.idx'):
        if not Path(str(path) + suffix).is_file():
            raise FileNotFoundError(str(path) + suffix)
    return str(path.resolve())


def load_binidx_documents(path, *, seed=42, max_rows=2048, eot_id=65532, max_docs=0):
    prefix = resolve_binidx_prefix(path)
    tokens, masks = MMapIndexedDataset(prefix), MMapIndexedDataset(prefix + '.mask')
    if not np.array_equal(tokens.sizes, masks.sizes):
        raise ValueError('Token and loss-mask binidx lengths differ.')
    # This SFT format has one packed record per indexed item. Do not silently
    # interpret sentence-level indices as complete packed records.
    if not np.array_equal(tokens.doc_idx, np.arange(len(tokens) + 1)):
        raise ValueError('Expected one physical packed record per binidx document.')
    if not np.array_equal(tokens.doc_idx, masks.doc_idx):
        raise ValueError('Token and mask document indices differ.')
    row_ids = np.random.default_rng(seed).permutation(len(tokens))[:max_rows or None]
    docs, seen = [], set()
    duplicate = empty = unsupervised = unterminated = 0
    selected_hash = hashlib.sha256()
    min_id, max_id = None, None
    for i, row in enumerate(row_ids):
        row = int(row)
        ids, mask = tokens[row], masks[row]
        if not np.isin(mask, [0, 1]).all():
            raise ValueError(f'Non-binary loss mask in row {row}')
        lo, hi = int(ids.min()), int(ids.max())
        min_id = lo if min_id is None else min(min_id, lo)
        max_id = hi if max_id is None else max(max_id, hi)
        selected_hash.update(row.to_bytes(8, 'little'))
        selected_hash.update(ids.tobytes())
        selected_hash.update(mask.tobytes())
        boundaries = (np.flatnonzero(ids == eot_id) + 1).tolist()
        if not boundaries or boundaries[-1] != len(ids):
            boundaries.append(len(ids))
            unterminated += 1
        start = 0
        for end in boundaries:
            # Upstream inserts a newline between packed examples. Strip only
            # leading LF separators (token 11), not arbitrary prompt content.
            left = start
            while left < end and int(ids[left]) == 11:
                left += 1
            start = end
            if end - left < 2:
                empty += 1
                continue
            supervised = int(mask[left + 1:end].sum())
            if not supervised:
                unsupervised += 1
                continue
            # Explicit int64 little-endian representation makes hashes invariant
            # to source storage dtype. Duplicate targets remain in ONE split.
            digest = hashlib.sha256(np.asarray(ids[left:end], dtype='<i8').tobytes()).hexdigest()
            if digest in seen:
                duplicate += 1
                continue
            seen.add(digest)
            docs.append(BinidxDocument(tokens, masks, row, left, end, digest, supervised))
            if max_docs and len(docs) >= max_docs:
                break
        if (i + 1) % 256 == 0:
            print(f'binidx: read {i+1}/{len(row_ids)} packed rows, {len(docs)} unique supervised examples', flush=True)
        if max_docs and len(docs) >= max_docs:
            row_ids = row_ids[:i + 1]
            break
    if len(docs) < 3:
        raise ValueError('Need at least three distinct supervised examples after unpacking.')
    files = {}
    for suffix in ('.bin', '.idx', '.mask.bin', '.mask.idx'):
        p = Path(prefix + suffix)
        stat = p.stat()
        files[suffix] = {'path': str(p), 'size': stat.st_size, 'mtime_ns': stat.st_mtime_ns}
        if suffix.endswith('.idx'):
            files[suffix]['sha256'] = hashlib.sha256(p.read_bytes()).hexdigest()
    metadata = {
        'format': 'packed_sft_binidx', 'prefix': prefix, 'source_files': files,
        'source_packed_rows': len(tokens), 'source_stored_tokens': int(tokens.sizes.sum()),
        'selected_rows': row_ids.tolist(), 'selected_rows_sha256': selected_hash.hexdigest(),
        'selected_rows_count': len(row_ids), 'unique_supervised_examples': len(docs),
        'duplicates_removed': duplicate, 'padding_segments_removed': empty,
        'unsupervised_segments_removed': unsupervised, 'unterminated_rows': unterminated,
        'token_id_min': min_id, 'token_id_max': max_id, 'eot_id': eot_id,
        'split_unit': 'EOT-delimited exact-deduplicated example',
        'integrity_note': 'index files + selected row bytes hashed; complete .bin files NOT hashed',
        'leakage_limit': 'exact token duplicates removed; related conversations/near-duplicates not certified absent',
    }
    print(f'binidx ready: {len(docs)} examples from {len(row_ids)} random rows, {duplicate} duplicates removed', flush=True)
    return docs, metadata
