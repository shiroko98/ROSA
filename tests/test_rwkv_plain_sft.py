"""Original-G1 tokenization, masks, EOS, packing, and raw-JSONL rebuild."""
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace

import numpy as np
import pytest

from rwkv_rosa.plain_sft import ROOT, vocabulary, load_tokenizer, render_record, encode_record, RowPacker
from third_party.rwkv_lm_v7.tokenizer import TRIE
from third_party.rwkv_lm_v7.binidx import MMapIndexedDataset


@pytest.fixture(scope='module')
def tokenizer():
    return load_tokenizer()


def test_compiled_greedy_ids_and_byte_masks_match_filtered_reference(tokenizer):
    trie = TRIE()
    for word, index in vocabulary(ROOT / 'rwkv_vocab_v20230424.txt'):
        trie.add(word, val=(word, index))
    rng = np.random.default_rng(42)
    alphabet = list('abc 中文😀é\n<>|:_09')
    samples = ['User: 你好\n\nAssistant: <think></think>回答。\n\n',
               '<|im_start|><|im_end|><|endoftext|><think><tool_call>',
               *[''.join(rng.choice(alphabet, size=200).tolist()) for _ in range(20)]]
    for text in samples:
        raw = text.encode('utf-8'); start = len(raw) // 2; end = len(raw) - 1
        ids, mask = tokenizer.encode(raw, start, end)
        expected_ids = []; expected_mask = []; position = 0
        while position < len(raw):
            stop, _, values = trie.find_longest(raw, position)
            word, index = next(iter(values))
            expected_ids.append(index); expected_mask.append(int(stop > start and position < end)); position = stop
        assert np.frombuffer(ids, '<u2').tolist() == expected_ids
        assert np.frombuffer(mask, 'u1').tolist() == expected_mask
        assert max(expected_ids) < 65530


def test_format_final_assistant_mask_tools_reasoning_and_eos(tokenizer):
    record = {'tools': [{'function': {'name': 'weather'}}], 'messages': [
        {'role': 'system', 'content': '你是助手'}, {'role': 'user', 'content': '天气？'},
        {'role': 'assistant', 'content': '', 'tool_calls': [{'function': {'name': 'weather', 'arguments': '{}'}}]},
        {'role': 'tool', 'name': 'weather', 'content': '晴'},
        {'role': 'assistant', 'content': '晴天', 'reasoning_content': '检查结果'},
        {'role': 'tool', 'content': 'trailing tool is not an assistant target'}]}
    raw, start, end = render_record(record)
    text = raw.decode(); assert 'User: 天气？\n\nAssistant:' in text
    assert 'User: Function output (weather):\n晴\n\nAssistant:' in text
    assert '<think>\n检查结果\n</think>\n\n晴天' in text
    assert 'trailing tool' not in text
    assert raw[start:end].decode().startswith(' <think>')
    ids, mask = encode_record(record, tokenizer)
    assert ids[-1] == 0 and mask[-1] == 1
    assert ids.max() < 65530
    pieces = dict((i, b) for b, i in vocabulary(ROOT / 'rwkv_vocab_v20230424.txt'))
    supervised = b''.join(pieces[int(i)] for i, m in zip(ids[:-1], mask[:-1]) if m).decode()
    assert '检查结果' in supervised and '晴天' in supervised
    assert 'weather' not in supervised and '天气？' not in supervised
    assert encode_record({'messages': [{'role': 'user', 'content': 'no answer'}]}, tokenizer) is None


def test_packing_preserves_shifted_supervision_and_masks_padding():
    tokens, masks = io.BytesIO(), io.BytesIO()
    packer = RowPacker(tokens, masks, 9)
    source = [(np.arange(1, 6, dtype=np.uint16), np.array([0, 0, 1, 1, 1], dtype='u1')),
              (np.arange(20, 40, dtype=np.uint16), np.array([0] * 10 + [1] * 10, dtype='u1')),
              (np.array([7, 8, 0], dtype='u2'), np.array([0, 1, 1], dtype='u1'))]
    for ids, mask in source:
        packer.add(ids, mask)
    packer.flush()
    ids = np.frombuffer(tokens.getvalue(), '<u2').reshape(-1, 9)
    mask = np.frombuffer(masks.getvalue(), 'u1').reshape(-1, 9)
    expected = np.concatenate([a[b.astype(bool)] for a, b in source])
    np.testing.assert_array_equal(ids[mask.astype(bool)], expected)
    assert not mask[:, 0].any()
    assert packer.unsupervised_rows == 1
    assert packer.supervised == sum(int(b.sum()) for _, b in source)


def test_raw_rebuild_is_loadable_fixed_length_and_resumable(tmp_path):
    import build_rwkv_plain_sft
    source = tmp_path / 'source'; source.mkdir()
    records = [{'messages': [{'role': 'user', 'content': f'问{i}'},
                             {'role': 'assistant', 'content': '答案。' * 30}]} for i in range(5)]
    (source / 'a.jsonl').write_text('\n'.join(json.dumps(x, ensure_ascii=False) for x in records) + '\nBAD JSON\n')
    args = SimpleNamespace(source=str(source), output=str(tmp_path / 'out'), context=64,
                           workers=2, max_files=0, max_records_per_file=0,
                           vocab=str(ROOT / 'rwkv_vocab_v20230424.txt'))
    result = build_rwkv_plain_sft.build(args)
    assert result['status'] == 'complete' and result['records'] == 5 and result['invalid_records'] == 1
    data = MMapIndexedDataset(result['data_prefix']); masks = MMapIndexedDataset(result['data_prefix'] + '.mask')
    assert np.all(data.sizes == 65) and np.array_equal(data.sizes, masks.sizes)
    assert sum(int(masks[i][1:].sum()) for i in range(len(data))) == result['supervised_tokens']
    assert all(int(data[i].max()) < 65530 for i in range(len(data)))
    assert build_rwkv_plain_sft.build(args)['signature'] == result['signature']
