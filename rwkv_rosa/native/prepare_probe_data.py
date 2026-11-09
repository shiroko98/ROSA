"""Bounded real-data suffixes for native context/placement profiling only.

Does not alter upstream MyDataset: each derived row has exactly ctx_len+1 tokens,
with the original supervision mask.  Not a full-corpus training preparation tool.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from third_party.rwkv_lm_v7.binidx import MMapIndexedDataset
from rwkv_rosa.binidx_data import resolve_binidx_prefix


def prepare(source, output, context, rows=16):
    prefix = resolve_binidx_prefix(source)
    out = Path(output)
    out.mkdir(parents=True, exist_ok=False)
    data, masks = MMapIndexedDataset(prefix), MMapIndexedDataset(prefix + '.mask')
    ids, targets, selected = [], [], []
    for i in range(min(len(data), 4096)):
        if len(data[i]) < context + 1:
            continue
        row = np.array(data[i][-(context + 1):], dtype=np.uint16)
        mask = np.array(masks[i][-(context + 1):], dtype=np.uint8)
        if not mask[1:].sum():
            continue
        ids.append(row); targets.append(mask)
        selected.append({'source_row': i, 'source_start': len(data[i]) - context - 1,
                         'supervised_tokens': int(mask[1:].sum()),
                         'sha256': hashlib.sha256(row.tobytes() + mask.tobytes()).hexdigest()})
        if len(ids) == rows:
            break
    if len(ids) != rows:
        raise ValueError('Not enough real supervised suffixes; no synthetic replacement allowed.')
    destination = out / 'sft'
    for suffix, arrays in [('', ids), ('.mask', targets)]:
        np.concatenate(arrays).tofile(str(destination) + suffix + '.bin')
        with MMapIndexedDataset.Index.writer(str(destination) + suffix + '.idx', arrays[0].dtype.type) as writer:
            writer.write([context + 1] * rows, list(range(rows + 1)))
    metadata = {'source': prefix, 'ctx_len': context, 'rows': rows, 'selected': selected,
                'purpose': 'controlled native SFT benchmark, not full-corpus training or independent quality evaluation'}
    (out / 'manifest.json').write_text(json.dumps(metadata, indent=2) + '\n')
    print(str(destination.resolve()))
    return str(destination.resolve())


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', required=True); p.add_argument('--output', required=True)
    p.add_argument('--context', type=int, required=True); p.add_argument('--rows', type=int, default=16)
    a = p.parse_args()
    if a.context < 16 or a.context % 16 or a.rows < 2:
        p.error('Context must be a positive multiple of 16; at least two rows required.')
    prepare(a.source, a.output, a.context, a.rows)
