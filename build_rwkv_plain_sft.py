#!/usr/bin/env python3
"""Rebuild raw messages JSONL as original-G1 text, EOS=0, packed masked binidx."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import shutil
import time

import numpy as np

from rwkv_rosa.plain_sft import ROOT, FORMAT_VERSION, SOURCE_URL, load_tokenizer, encode_record, RowPacker
from third_party.rwkv_lm_v7.binidx import MMapIndexedDataset

TOKENIZER = None


def write_json(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')
    tmp.replace(path)


def build_file(job):
    source, prefix, context, signature, max_records = job
    source, prefix = Path(source), Path(prefix)
    meta = prefix.with_suffix('.json')
    if meta.exists():
        saved = json.loads(meta.read_text())
        if saved.get('signature') == signature and saved.get('status') == 'complete':
            expected = saved['rows'] * (context + 1)
            if Path(str(prefix) + '.bin').stat().st_size == expected * 2 and Path(str(prefix) + '.mask.bin').stat().st_size == expected:
                return saved
    stats = {'source': str(source), 'signature': signature, 'status': 'running', 'lines': 0,
             'records': 0, 'filtered_records': 0, 'invalid_records': 0, 'encoded_tokens': 0,
             'supervised_tokens_before_packing': 0, 'rows': 0}
    raw_hash = hashlib.sha256()
    started = time.monotonic(); last_progress = started
    token_path = Path(str(prefix) + '.bin.tmp'); mask_path = Path(str(prefix) + '.mask.bin.tmp')
    progress = Path(str(prefix) + '.progress.json')
    with source.open('rb') as stream, token_path.open('wb') as tokens, mask_path.open('wb') as masks, \
            Path(str(prefix) + '.errors.jsonl').open('w') as errors:
        packer = RowPacker(tokens, masks, context + 1)
        for number, line in enumerate(stream, 1):
            if max_records and number > max_records:
                break
            stats['lines'] += 1; raw_hash.update(line)
            if not line.strip():
                continue
            try:
                record = json.loads(line)
                if not isinstance(record, dict):
                    raise ValueError('JSONL record must be an object')
                encoded = encode_record(record, TOKENIZER)
                if encoded is None:
                    stats['filtered_records'] += 1
                    continue
                ids, mask = encoded
                stats['records'] += 1; stats['encoded_tokens'] += len(ids)
                stats['supervised_tokens_before_packing'] += int(mask.sum())
                packer.add(ids, mask)
            except (ValueError, TypeError, KeyError, UnicodeError) as exc:
                stats['invalid_records'] += 1
                errors.write(json.dumps({'source': str(source), 'line': number,
                                         'type': type(exc).__name__, 'error': str(exc)[:500]}, ensure_ascii=False) + '\n')
            now = time.monotonic()
            if now - last_progress >= 5:
                stats.update(rows=packer.rows, elapsed_seconds=now - started,
                             source_bytes_read=stream.tell(), source_total_bytes=source.stat().st_size)
                write_json(progress, stats); last_progress = now
        packer.flush()
        if stats['supervised_tokens_before_packing'] != packer.supervised:
            raise AssertionError('Packing dropped or duplicated supervised targets')
        stats.update(rows=packer.rows, supervised_tokens=packer.supervised,
                     padding_tokens=packer.padding, split_documents=packer.split_documents,
                     unsupervised_chunks_omitted=packer.unsupervised_rows,
                     source_sha256=raw_hash.hexdigest(), elapsed_seconds=time.monotonic() - started)
    token_path.replace(str(prefix) + '.bin'); mask_path.replace(str(prefix) + '.mask.bin')
    stats['status'] = 'complete'; write_json(meta, stats); write_json(progress, stats)
    return stats


def build(args):
    global TOKENIZER
    source = Path(args.source).resolve(strict=True)
    out = Path(args.output).resolve(); out.mkdir(parents=True, exist_ok=True)
    parts = out / 'parts'; parts.mkdir(exist_ok=True)
    results = out / 'results'; results.mkdir(exist_ok=True)
    sources = sorted(source.rglob('*.jsonl')) if source.is_dir() else [source]
    if args.max_files:
        sources = sources[:args.max_files]
    if not sources:
        raise ValueError('No source JSONL files')
    code_hash = hashlib.sha256()
    for path in (Path(__file__), ROOT / 'rwkv_rosa/plain_sft.py', ROOT / 'cpp_extensions/rwkv_plain_tokenizer.cpp', Path(args.vocab)):
        code_hash.update(path.read_bytes())
    config = {'format': FORMAT_VERSION, 'source': str(source), 'context': args.context,
              'stored_tokens_per_row': args.context + 1, 'eos_token_id': 0, 'pad_token_id': 0,
              'ordinary_max_token_id': 65529, 'supervision': 'last assistant including response separator and EOS',
              'template_source': SOURCE_URL, 'source_count': len(sources), 'source_bytes': sum(p.stat().st_size for p in sources),
              'code_and_vocabulary_sha256': code_hash.hexdigest(), 'workers': args.workers,
              'max_records_per_file': args.max_records_per_file, 'started_at': datetime.now().astimezone().isoformat()}
    config['signature'] = hashlib.sha256(json.dumps({**config, 'started_at': None, 'workers': None,
        'files': [(str(p), p.stat().st_size, p.stat().st_mtime_ns) for p in sources]}, sort_keys=True).encode()).hexdigest()
    manifest = out / 'build_manifest.json'
    if manifest.exists():
        old = json.loads(manifest.read_text())
        if old.get('signature') != config['signature']:
            raise ValueError('Existing output belongs to a different builder/config/source set; use a new output')
        if old.get('status') == 'complete':
            return old
    write_json(manifest, {**config, 'status': 'tokenizing'})
    TOKENIZER = load_tokenizer(args.vocab)
    jobs = [(str(path), str(parts / f'part{i:05d}'), args.context,
             hashlib.sha256((config['signature'] + str(path)).encode()).hexdigest(), args.max_records_per_file)
            for i, path in enumerate(sources)]
    completed = []
    # Large shards first, deterministic merge order later. Fork shares the
    # immutable compiled trie; workers never initialize CUDA or run torch ops.
    by_size = sorted(jobs, key=lambda job: Path(job[0]).stat().st_size, reverse=True)
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=multiprocessing.get_context('fork')) as pool:
        futures = [pool.submit(build_file, job) for job in by_size]
        for future in as_completed(futures):
            row = future.result(); completed.append(row)
            progress = {'status': 'tokenizing', 'completed_files': len(completed), 'total_files': len(jobs),
                        'records': sum(x['records'] for x in completed), 'rows': sum(x['rows'] for x in completed),
                        'last_completed_source': row['source']}
            write_json(out / 'build_progress.json', progress)
            print(json.dumps(progress), flush=True)
    stats = [json.loads(Path(job[1]).with_suffix('.json').read_text()) for job in jobs]
    total_rows = sum(row['rows'] for row in stats)
    if total_rows == 0:
        raise ValueError('No supervised rows were produced')
    write_json(out / 'build_progress.json', {'status': 'merging', 'total_rows': total_rows})
    prefix = results / 'SFT_RWKV7_13B'
    for suffix, dtype in (('', np.uint16), ('.mask', np.uint8)):
        destination = Path(str(prefix) + suffix + '.bin')
        temporary = Path(str(destination) + '.tmp')
        with temporary.open('wb') as stream:
            for job in jobs:
                part = Path(job[1] + suffix + '.bin')
                with part.open('rb') as src:
                    shutil.copyfileobj(src, stream, length=16 * 1024 * 1024)
        if temporary.stat().st_size != total_rows * (args.context + 1) * np.dtype(dtype).itemsize:
            raise AssertionError('Merged binidx size mismatch')
        temporary.replace(destination)
        index = Path(str(prefix) + suffix + '.idx')
        with MMapIndexedDataset.Index.writer(str(index) + '.tmp', dtype) as writer:
            writer.write([args.context + 1] * total_rows, list(range(total_rows + 1)))
        Path(str(index) + '.tmp').replace(index)
    summary = {key: sum(row.get(key, 0) for row in stats) for key in
               ('records', 'filtered_records', 'invalid_records', 'encoded_tokens', 'supervised_tokens',
                'padding_tokens', 'split_documents', 'unsupervised_chunks_omitted')}
    final = {**config, **summary, 'status': 'complete', 'rows': total_rows,
             'stored_tokens': total_rows * (args.context + 1), 'data_prefix': str(prefix), 'sources': stats,
             'completed_at': datetime.now().astimezone().isoformat()}
    write_json(manifest, final); write_json(out / 'build_progress.json', {k: v for k, v in final.items() if k != 'sources'})
    print(json.dumps({k: v for k, v in final.items() if k != 'sources'}), flush=True)
    return final


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', required=True); parser.add_argument('--output', required=True)
    parser.add_argument('--vocab', default=str(ROOT / 'rwkv_vocab_v20230424.txt'))
    parser.add_argument('--context', type=int, default=192512); parser.add_argument('--workers', type=int, default=64)
    parser.add_argument('--max-files', type=int, default=0); parser.add_argument('--max-records-per-file', type=int, default=0)
    args = parser.parse_args()
    if args.context < 2 or args.workers < 1:
        parser.error('Positive context and worker count required')
    import torch
    torch.set_num_threads(1)
    build(args)


if __name__ == '__main__':
    main()
