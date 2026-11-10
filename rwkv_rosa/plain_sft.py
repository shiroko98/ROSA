"""Original G1 text conversation format, ordinary RWKV bytes, and packed SFT."""
from __future__ import annotations

import ast
from pathlib import Path
import json
import os
import subprocess
import sys
import sysconfig

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
FORMAT_VERSION = 'rwkv_g1_plain_last_assistant_v1'
SOURCE_URL = 'https://huggingface.co/BlinkDL/rwkv7-g1#chat-prompt-note-better-replace-all-nn-in-user_prompt-to-n-as-i-am-using-nn-as-chat-round-separator-in-pretrain-data'


def vocabulary(path):
    result = []
    for line in Path(path).read_text().splitlines():
        left, right = line.index(' '), line.rindex(' ')
        token = ast.literal_eval(line[left:right])
        if isinstance(token, str):
            token = token.encode('utf-8')
        index = int(line[:left])
        if 0 < index < 65530:
            result.append((token, index))
    return result


def load_tokenizer(vocab_path=ROOT / 'rwkv_vocab_v20230424.txt'):
    build = ROOT / '.codex-build' / 'plain_tokenizer'
    build.mkdir(parents=True, exist_ok=True)
    binary = build / ('rwkv_plain_tokenizer_ext' + sysconfig.get_config_var('EXT_SUFFIX'))
    source = ROOT / 'cpp_extensions/rwkv_plain_tokenizer.cpp'
    if not binary.exists() or binary.stat().st_mtime_ns < source.stat().st_mtime_ns:
        import torch
        temporary = binary.with_suffix(binary.suffix + '.tmp')
        compiler = os.environ.get('CXX', 'g++')
        subprocess.run([compiler, '-O3', '-std=c++17', '-shared', '-fPIC',
                        '-I' + sysconfig.get_paths()['include'],
                        '-I' + str(Path(torch.__file__).parent / 'include'),
                        str(source), '-o', str(temporary)], check=True)
        temporary.replace(binary)
    if str(build) not in sys.path:
        sys.path.insert(0, str(build))
    from rwkv_plain_tokenizer_ext import Tokenizer
    return Tokenizer(vocabulary(vocab_path))


def text_content(value):
    if value is None:
        return ''
    if isinstance(value, str):
        return value.replace('\r\n', '\n').replace('\r', '\n')
    if isinstance(value, list):
        return ''.join(text_content(x if isinstance(x, str) else x.get('text', x.get('output', '')))
                       for x in value if isinstance(x, (str, dict)))
    return json.dumps(value, ensure_ascii=False)


def render_record(record):
    """Return UTF-8 text plus the final assistant's supervised byte interval.

    Role labels are ordinary text. Thinking/tool markup is also ordinary text;
    never bind it to private/untrained high vocabulary IDs. Existing thinking is
    preserved; no fabricated thinking prefix or extra default system is added.
    """
    messages = record.get('messages')
    if not isinstance(messages, list) or not messages:
        raise ValueError('Expected nonempty messages list')
    last = next((i for i in range(len(messages) - 1, -1, -1)
                 if isinstance(messages[i], dict) and messages[i].get('role') == 'assistant'), None)
    if last is None:
        return None
    # Match the existing SFT objective: final assistant only. Trailing tool
    # messages cannot be supervised as an assistant answer and are excluded.
    messages = messages[:last + 1]
    parts = []
    tools = record.get('tools')
    if tools:
        parts.append('System: Tools:\n' + json.dumps(tools, ensure_ascii=False) + '\n\n')
    train_start = train_end = None
    length = sum(len(x.encode('utf-8')) for x in parts)
    for i, message in enumerate(messages):
        if not isinstance(message, dict):
            raise ValueError('Each message must be an object')
        role = message.get('role')
        content = text_content(message.get('content')).strip()
        if role == 'assistant':
            reasoning = text_content(message.get('reasoning_content')).strip()
            if reasoning and '<think>' not in content:
                content = '<think>\n' + reasoning + '\n</think>\n\n' + content
            calls = message.get('tool_calls')
            if calls:
                chunks = []
                for call in calls:
                    payload = call.get('function', call)
                    chunks.append('<tool_call>\n' + json.dumps(payload, ensure_ascii=False) + '\n</tool_call>')
                content += ('\n' if content else '') + '\n'.join(chunks)
            prefix = 'Assistant:'
        elif role == 'user':
            prefix = 'User:'
        elif role in ('system', 'developer'):
            prefix = 'System:'
        elif role in ('tool', 'function'):
            prefix = 'User:'
            name = str(message.get('name', '')).strip()
            content = 'Function output' + (f' ({name})' if name else '') + ':\n' + content
        else:
            raise ValueError(f'Unsupported role {role!r}')
        if i == last:
            if not content:
                return None
            train_start = length + len(prefix.encode('utf-8'))
        piece = prefix + ' ' + content + '\n\n'
        parts.append(piece)
        length += len(piece.encode('utf-8'))
        if i == last:
            train_end = length
    encoded = ''.join(parts).encode('utf-8')
    return encoded, train_start, train_end


def encode_record(record, tokenizer):
    rendered = render_record(record)
    if rendered is None:
        return None
    text, start, end = rendered
    ids, mask = tokenizer.encode(text, start, end)
    # Official RWKV end-of-document token is ID 0. Padding also uses 0, with
    # mask=0; the real final EOS is supervised, unlike padding.
    return np.frombuffer(ids + b'\0\0', dtype='<u2'), np.frombuffer(mask + b'\1', dtype=np.uint8)


class RowPacker:
    """Stream whole conversations into fixed rows; overlap one token on splits."""
    def __init__(self, tokens_file, masks_file, stored_length):
        if stored_length < 2:
            raise ValueError('At least two stored tokens required')
        self.tokens_file, self.masks_file = tokens_file, masks_file
        self.length = stored_length
        self.ids = np.zeros(stored_length, dtype='<u2')
        self.mask = np.zeros(stored_length, dtype=np.uint8)
        self.used = self.rows = self.supervised = self.padding = self.unsupervised_rows = self.split_documents = 0

    def flush(self):
        if not self.used:
            return
        # The first stored token has no preceding input in this row. It is not
        # counted as a target; each split carries its preceding overlap token.
        self.mask[0] = 0
        count = int(self.mask.sum())
        if count:
            self.tokens_file.write(self.ids.tobytes())
            self.masks_file.write(self.mask.tobytes())
            self.rows += 1; self.supervised += count; self.padding += self.length - self.used
        else:
            self.unsupervised_rows += 1
        self.ids.fill(0); self.mask.fill(0); self.used = 0

    def add(self, ids, mask):
        if len(ids) != len(mask):
            raise ValueError('Token/mask lengths differ')
        if len(ids) > self.length:
            self.flush(); self.split_documents += 1
            for start in range(0, len(ids) - 1, self.length - 1):
                segment = ids[start:start + self.length]; targets = mask[start:start + self.length]
                self.ids[:len(segment)] = segment; self.mask[:len(segment)] = targets
                self.used = len(segment); self.flush()
            return
        if self.used + len(ids) > self.length:
            self.flush()
        end = self.used + len(ids)
        self.ids[self.used:end] = ids; self.mask[self.used:end] = mask
        self.used = end
        if self.used == self.length:
            self.flush()
