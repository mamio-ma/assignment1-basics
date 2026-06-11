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
    if special_tokens:
        pattern = "|".join(re.escape(t) for t in special_tokens)
        chunk_without_special_tokens = re.split(pattern, chunk)
    else:
        chunk_without_special_tokens = [chunk]
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
    merges: list[tuple[bytes, bytes]] = []
    vocab: dict[int, bytes] = {x: bytes([x]) for x in range(256)}

    # Convert to mutable lists indexed by word_id for O(1) update
    word_list = list(indices.keys())
    word_count_list = list(indices.values())
    words: list[list[bytes]] = [list(wt) for wt in word_list]

    # pair_counts[pair] = total frequency of this adjacent pair across all words
    pair_counts: dict[tuple[bytes, bytes], int] = defaultdict(int)
    # pair_to_word_ids[pair] = set of word IDs whose current form contains this pair
    pair_to_word_ids: dict[tuple[bytes, bytes], set] = defaultdict(set)

    # Initialize counts by scanning all words once (O(V * L) total)
    for word_id, (word, count) in enumerate(zip(words, word_count_list)):
        for i in range(len(word) - 1):
            pair = (word[i], word[i + 1])
            pair_counts[pair] += count
            pair_to_word_ids[pair].add(word_id)

    for merge_i in range(num_merges):
        if merge_i % 100 == 0:
            print(f"Merge {merge_i}/{num_merges}")

        if not pair_counts:
            break

        # Select the highest-frequency pair; break ties by lexicographic order
        best_pair = max(pair_counts, key=lambda p: (pair_counts[p], p))
        index1, index2 = best_pair
        new_token = index1 + index2

        merges.append(best_pair)
        vocab[256 + merge_i] = new_token

        # Only the words that contain this pair need updating
        affected = pair_to_word_ids.pop(best_pair)
        del pair_counts[best_pair]

        for word_id in affected:
            word = words[word_id]
            count = word_count_list[word_id]

            # Remove this word's contribution from every pair it currently contains.
            # We skip best_pair (already popped). A word may contain a non-best
            # pair multiple times, so we subtract the full word count per occurrence.
            # Bug that this avoids: a pair like (e, r) may appear at two positions
            # in the same word; if we only subtract the right-neighbor occurrence we
            # decrement pair_counts correctly but incorrectly discard word_id from
            # pair_to_word_ids, causing the word to be skipped in the future merge
            # step that selects that pair as best.
            for i in range(len(word) - 1):
                pair = (word[i], word[i + 1])
                if pair == best_pair:
                    continue
                pair_counts[pair] -= count
                if pair_counts[pair] <= 0:
                    del pair_counts[pair]
                pair_to_word_ids[pair].discard(word_id)

            # Rebuild the word, replacing every non-overlapping occurrence left-to-right
            new_word: list[bytes] = []
            i = 0
            while i < len(word):
                if i < len(word) - 1 and word[i] == index1 and word[i + 1] == index2:
                    new_word.append(new_token)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1

            words[word_id] = new_word

            # Re-add this word's contribution for every pair in new_word.
            # Re-adding is always safe: pairs that survived unchanged are
            # re-counted correctly; new pairs from new_token enter fresh.
            for i in range(len(new_word) - 1):
                pair = (new_word[i], new_word[i + 1])
                pair_counts[pair] += count
                pair_to_word_ids[pair].add(word_id)

    # Append special tokens at the end of the vocabulary
    for i, token in enumerate(special_tokens):
        vocab[i + len(vocab)] = token.encode("utf-8")

    return vocab, merges