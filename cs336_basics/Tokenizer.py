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

    def __init__(self, vocab: Dict[int, bytes], merges: list[tuple[bytes, bytes]], special_tokens: list[str] | None = None):
        self._vocab = vocab
        self._merges = merges
        self._special_tokens = special_tokens

    @classmethod
    def from_files(cls, vocab_filepath, merges_filepath, special_tokens: list[str] | None =None):
        _vocab = {}
        with open(vocab_filepath) as f:
            vocab = json.load(f)
            for k, v in vocab.items():
                if special_tokens is not None and k in special_tokens:
                    _vocab[v] = k.encode("utf-8")
                    continue
                _vocab[v] = unicode_str_to_bytes(k)

        _merge = []

        with open(merges_filepath) as f:
            for line in f:
                line = line.strip()
                k, v = line.split(' ')
                _merge.append(tuple([unicode_str_to_bytes(k), unicode_str_to_bytes(v)]))

        return cls(_vocab, _merge, special_tokens)


    def encode(self, text: str) -> List[int]:
        pre_token_words: List[str] = []
        encode_words: List[int] = []

        if self._special_tokens:
            pattern = "|".join(re.escape(t) for t in self._special_tokens)
            # the difference between re.split(pattern, text) and re.split(f"({pattern})", text) is the second way will keep the special token
            parts = re.split(f"({pattern})", text)
        else:
            parts = [text]

        for part in parts:
            if self._special_tokens and part in self._special_tokens:
                pre_token_words.append(part)
            else:
                PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
                for match in re.finditer(PAT, part):
                    pre_token_words.append(match.group())

        vocab_reverse: Dict[bytes, int] = {v: k for k, v in self._vocab.items()}
        merge_priorities: Dict[tuple[bytes, bytes], int] = {pair: i for i, pair in enumerate(self._merges)}

        for word in pre_token_words:

            if self._special_tokens is not None and word in self._special_tokens:
                encode_words.append(vocab_reverse[word.encode("utf-8")])
                continue

            word_in_bytes: tuple[bytes, ...] = tuple(bytes([b]) for b in word.encode("utf-8"))
            ## apply merge
            while True:

                merges_list = []
                for i in range(len(word_in_bytes) - 1):
                    pair = (word_in_bytes[i], word_in_bytes[i + 1])
                    if pair in merge_priorities:
                        merges_list.append((pair, merge_priorities[pair]))

                if len(merges_list) == 0:
                    break

                best_pair = min(merges_list, key = lambda p: (p[1], p[0]))  ## e.g. ((b'l', b'o'), 0)
                merge_pair = best_pair[0][0] + best_pair[0][1]  ## e.g. b'lo'
                new_word = []
                i = 0
                while i < len(word_in_bytes):
                    if i < len(word_in_bytes) - 1 and (word_in_bytes[i], word_in_bytes[i+1]) == best_pair[0]:
                        new_word.append(merge_pair)
                        i = i + 2
                        continue

                    new_word.append(word_in_bytes[i])
                    i += 1

                word_in_bytes = tuple(new_word)

            for i in range(len(word_in_bytes)):
                encode_words.append(vocab_reverse[word_in_bytes[i]])

        return encode_words