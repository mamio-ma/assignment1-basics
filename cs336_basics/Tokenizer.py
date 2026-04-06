from functools import lru_cache
from typing import List, Dict
import os
import json
from tests.common import unicode_str_to_bytes

class Tokenizer:
    _vocab: Dict[int, bytes]
    _merge: list[tuple[bytes, bytes]]
    _special_tokens: list[str] | None

    def __init__(self, vocab: Dict[int, bytes], merges: list[tuple[bytes, ...]], special_tokens: list[str] | None = None):
        self._vocab = vocab
        self._merges = merges
        self._special_tokens = special_tokens

    @classmethod
    def from_files(cls, vocab_filepath, merges_filepath, special_tokens=None):
        _vocab = {}
        with open(vocab_filepath) as f:
            vocab = json.load(f)
            for k, v in vocab.items():
                if k in special_tokens:
                    _vocab[k] = v
                    continue
                _vocab[unicode_str_to_bytes(k)] = v

        _merge = []

        with open(merges_filepath) as f:
            for line in f:
                line = line.strip()
                k, v = line.split(' ')
                _merge.append(tuple([unicode_str_to_bytes(k), unicode_str_to_bytes(v)]))

        return cls(_vocab, _merge)


    def encode(self, text: str) -> List[int]:
        raise NotImplementedError

