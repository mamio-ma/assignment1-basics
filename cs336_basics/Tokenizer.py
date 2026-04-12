from functools import lru_cache
from typing import List, Dict
import os
import json
from tests.common import unicode_str_to_bytes
import regex as re
import pathlib

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
                if special_tokens is not None and k in special_tokens:
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
        pre_token_words: List[str] = []
        encode_words: List[int] = []

        if self._special_tokens:
            pattern = "|".join(re.escape(t) for t in self._special_tokens)
            special_token_iter = re.finditer(pattern, text)
            text_without_special_token = re.split(pattern, text)
        else:
            text_without_special_token = [text]
            special_token_iter = None

        for corpus in text_without_special_token:
            PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
            print(corpus)
            for match in re.finditer(PAT, corpus):
                pre_token_words.append(match.group())

            if special_token_iter:
                special_token = next(special_token_iter).group()
                pre_token_words.append(special_token)

        vocab_reverse: Dict[bytes, int] = {}
        for k, v in self._vocab.items():
            vocab_reverse[v] = k

        for word in pre_token_words:

            if self._special_tokens is not None and word in self._special_tokens:
                encode_words.append(vocab_reverse[word.encode("utf-8")])
                continue

            word_in_bytes: tuple[bytes, ...] = tuple(bytes([b]) for b in word.encode("utf-8"))
            ## apply merge
            while True:

                if len(word_in_bytes) < 2:
                    break
                l = word_in_bytes[0]
                r = word_in_bytes[1]
                has_pair = False
                for idx in range(len(self._merges)):
                    if self._merges[idx] == (l, r):
                        has_pair = True
                        break
                if has_pair is False:
                    break

                merge_pair = l + r
                word_in_bytes = (merge_pair,) + word_in_bytes[2:]

            for i in range(len(word_in_bytes)):
                encode_words.append(vocab_reverse[word_in_bytes[i]])

        return encode_words


FIXTURES_PATH = (pathlib.Path(__file__).resolve().parent.parent) / "data"
tokenizer = Tokenizer.from_files(FIXTURES_PATH / "vocab.json", FIXTURES_PATH / "merges.txt")
print(tokenizer.encode("hello there is a cat."))



