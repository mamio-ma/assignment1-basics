import os
import regex as re
import sys
from collections import defaultdict
from typing import BinaryIO, List, DefaultDict
from multiprocessing import Process, Queue


def read_chunks(path: str | os.PathLike,
                desired_num_chunks
) -> List[str]:
    """
    Read chunks from a file. Make sure chunks at end of the special token.
    (If chunks evenly, it will looks like ["I ha"], ["ve a"], some word might be separated and therefore lost the integrity)
    """
    chunks = []
    with open(path, "rb") as f:
        boundaries = find_chunk_boundaries(f, desired_num_chunks, b"<|endoftext|>")

        # The following is a serial implementation, but you can parallelize this
        # by sending each start/end pair to a set of processes.
        for start, end in zip(boundaries[:-1], boundaries[1:]):
            f.seek(start)
            chunk = f.read(end - start).decode("utf-8", errors="ignore")
            chunks.append(chunk)

    return chunks

def find_chunk_boundaries(
    file: BinaryIO,
    desired_num_chunks: int,
    split_special_token: bytes,
) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
    assert isinstance(split_special_token, bytes), "Must represent special token as a bytestring"

    # Get total file size in bytes
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks

    # Initial guesses for chunk boundary locations, uniformly spaced
    # Chunks start on previous index, don't include last index
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)  # Start at boundary guess
        while True:
            mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

            # If EOF, this boundary should be at the end of the file
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            # Find the special token in the mini chunk
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    return sorted(set(chunk_boundaries))

def sum_word_count(chunk: str, special_tokens: List[str], q: Queue):
    count: dict[str, int] = defaultdict(int)  ## store count of each words in str: e.g. {low: 5...}
    pattern = "|".join(re.escape(t) for t in special_tokens)
    chunk_without_special_tokens = re.split(pattern, chunk)
    for corpus in chunk_without_special_tokens:
        PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
        iter = re.finditer(PAT, corpus)
        for match in iter:
            count[match.group()] += 1

    q.put(count)

def pre_tokenization(
        path: str | os.PathLike,
        desired_num_chunks,
        special_tokens: List[str]) -> dict[tuple[bytes, ...], int]:

    chunks = read_chunks(path, desired_num_chunks)

    q = Queue()
    processes = []
    for chunk in chunks:
        p = Process(target=sum_word_count, args=(chunk, special_tokens, q))
        p.start()
        processes.append(p)

    count: dict[str, int] = defaultdict(int)  ## store count of each words in str: e.g. {low: 5...}
    for _ in processes:
        partial = q.get()
        for word, v in partial.items():
            count[word] += v

    for p in processes:
        p.join()

    counts: DefaultDict[tuple[bytes, ...], int] = defaultdict(int)  ## store the count of each word in bytes: e.g. {(l,o,w): 5 …}
    for words, v in count.items():
        counts[tuple(bytes([b]) for b in words.encode("utf-8"))] += v

    return counts

def train_bpe(
    indices: dict[tuple[bytes, ...], int],
    num_merges: int,
    special_tokens: list[str]
):
    merges: list[tuple[bytes, bytes]] = [] # index1, index2 => merged index
    vocab: dict[int, bytes] = {x: bytes([x]) for x in range(256)}  # index -> bytes

    for i in range(num_merges):
        # count the number of occurrences for each pair
        counts = defaultdict(int)
        for k, v in indices.items():
            for idx in range(len(k) - 1):
                counts[(k[idx], k[idx + 1])] += v

        pair = max(counts, key=lambda p: (counts[p], p))
        index1, index2 = pair
        new_index = 256 + i
        merges.append(pair)
        vocab[new_index] = index1 + index2

        # update the indices with the new merge
        new_indices: dict[tuple[bytes, ...], int] = {}
        merge_pair = index1 + index2
        for word_tuple, count in indices.items():
            new_word = []
            idx = 0
            while idx < len(word_tuple):
                if idx < len(word_tuple) - 1 and word_tuple[idx] == index1 and word_tuple[idx + 1] == index2:
                    new_word.append(merge_pair)
                    idx += 2
                else:
                    new_word.append(word_tuple[idx])
                    idx += 1

            new_indices[tuple(new_word)] = count

        indices = new_indices

    # add the special token at the bottom
    for i, token in enumerate(special_tokens):
        vocab[i + len(vocab)] = token.encode("utf-8")

    return vocab, merges