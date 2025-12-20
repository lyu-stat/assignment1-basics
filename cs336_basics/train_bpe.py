from heapq import heappop, heappush, heapify
import os
import json
from typing import BinaryIO
from collections import Counter, defaultdict
from multiprocessing import Pool
import regex as re
from tqdm import tqdm


def find_chunk_boundaries(
    file: BinaryIO,
    desired_num_chunks: int,
    split_special_token: bytes,
) -> list[int]:
    """Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.

    Args:
        file (BinaryIO): Opened file object in binary mode
        desired_num_chunks (int): Desired number of chunks to split the file into
        split_special_token (bytes): Special token to use as a safe boundary

    Returns:
        list[int]: List of byte offsets representing chunk boundaries
    """
    assert isinstance(
        split_special_token, bytes
    ), "Must represent special token as a bytestring"

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

    # For each interior boundary guess (not the first or the last),
    # read ahead in 4KB blocks to find the nearest safe boundary.
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


def byte_to_unicode(
    keep: tuple[int, ...] = tuple(list(range(33, 127)) + [9, 10, 13, 32]),
    start_cp: int = 256,
) -> tuple[dict[int, str], dict[str, int]]:
    """This function generates a mapping of 256 single bytes to unicode characters.

    Args:
        keep (tuple[int, ...]): This list contains the visiable single-byte Unicode
            characters that we'll keep. Defaults to tuple(list(range(33, 127)) + [9,10,13,32]).
        start_cp (int): This is the starting code point of replacing the invisiable bytes.
            Defaults to 256.

    Returns:
        dict[int, str]: This is the mapping from 256 single bytes to unicode characters.
        dict[str, int]: This is the reverse mapping from unicode characters to bytes.
    """
    b2u = {}
    n = start_cp
    for b in range(256):
        if b in keep:
            b2u[b] = chr(b)
        else:
            b2u[b] = chr(n)
            n += 1
    u2b = {v: k for k, v in b2u.items()}
    return b2u, u2b


def pre_tokenize(
    file_path: str,
    start: int,
    end: int,
    special_tokens: list[str],
    byte_to_unicode_map: dict[int, str],
    pre_tokenizer_regex: re.Pattern,
) -> dict[tuple[str, ...], int]:
    """Pre-tokenize a single chunk of text

    Args:
        file_path (str): path of the text file
        start (int): starting point of the chunk within the text
        end (int): end point of the chunk within the text
        special_tokens (list[str]): the list of special tokens to separate documents
        byte_to_unicode_map (dict[int, str]): mapping from bytes to unicode characters
        pre_tokenizer_regex (re.Pattern): regex pattern for pre-tokenization

    Returns:
        dict[tuple[str], int]: counts of each of the pre-tokens where the pretokens
            split into characters
    """
    with open(file_path, "rb") as file:
        file.seek(start)
        chunk = file.read(end - start)

    # First split the chunk by the special tokens in bytes
    segments_bytes = re.split(
        b"|".join(re.escape(t.encode("utf-8")) for t in special_tokens), chunk
    )
    # regex works with bytes too. If subject is bytes, then pattern must be bytes.
    # If subject is str, then pattern must be str.

    # Count the frequency of pretokens
    counts = Counter()
    for seg in segments_bytes:
        # Map each byte to a unique Unicode character (byte-to-unicode trick)
        seg = "".join(byte_to_unicode_map[b] for b in seg)
        pretokens = pre_tokenizer_regex.finditer(seg)
        for pretoken in pretokens:
            counts[pretoken.group()] += 1

    # Split pre-tokens into characters for easier processing later
    new_counts = Counter()
    for pretoken, count in counts.items():
        new_counts[tuple(pretoken)] = count
    return new_counts


def heap_pop_best(
    pair_counts_heap: list[tuple[int, tuple[str, str]]],
    pair_counts: dict[tuple[str, str], int],
) -> tuple[str, str]:
    """Pop the best pair from the heap, ensuring it's up-to-date.

    Args:
        pair_counts_heap (list[tuple[int, tuple[str, str]]]):
            heap of pair counts
        pair_counts (dict[tuple[str, str], int]): current counts of pairs

    Returns:
        tuple[str, str]: the best pair of characters
    """
    # Find the top count that is valid
    while pair_counts_heap:
        neg_c, p = pair_counts_heap[0]
        c = pair_counts[p]
        if neg_c == -c:
            top_count = c
            break
        heappop(pair_counts_heap)
    else:
        raise ValueError("No valid pairs available in the heap.")

    # Collect all pairs with the top count
    top_pairs = set()
    while pair_counts_heap:
        neg_c, p = pair_counts_heap[0]
        c = pair_counts[p]
        if -neg_c == top_count and -neg_c == c:
            top_pairs.add(p)
            heappop(pair_counts_heap)
            continue
        if -neg_c == top_count and -neg_c != c:
            heappop(pair_counts_heap)
            continue
        break

    # Select the lexicographically largest pair among them
    best_pair = max(top_pairs)

    # Push back the other pairs into the heap
    for p in top_pairs:
        if p != best_pair:
            heappush(pair_counts_heap, (-pair_counts[p], p))

    return best_pair


def merge_tokens(
    pretoken_counts: Counter[tuple[str, ...]],
    pair_counts: Counter[tuple[str, str]],
    pair_counts_heap: list[tuple[int, tuple[str, str]]],
    pair_to_pretokens: dict[tuple[str, str], set[tuple[str, ...]]],
) -> tuple[
    tuple[str, str],
    Counter[tuple[str, ...]],
    Counter[tuple[str, str]],
    list[tuple[int, tuple[str, str]]],
    dict[tuple[str, str], set[tuple[str, ...]]],
]:
    """Merge characters of the pre-tokens to add to the vocabulary.

    Args:
        pretoken_counts (Counter[tuple[str]]): counts of each of the pre-tokens
        pair_counts (Counter[tuple[str, str]]): counts of each of the pairs
        pair_counts_heap (list[tuple[int, tuple[str, str]]]):
            to efficiently get the max pair
        pair_to_pretokens (dict[str, set[tuple[str]]]): mapping from pairs to pre-tokens

    Returns:
        tuple[str, str]: the pair of characters selected to be merged and added to vocab
        Counter[tuple[str, ...]]: counts of each of the updated pre-tokens
        Counter[tuple[str, str]]: counts of each of the updated pairs
        list[tuple[int, tuple[str, str]]]: updated heap for pair counts
        dict[tuple[str, str], set[tuple[str, ...]]]: updated mapping from pairs to pre-tokens
    """
    # Within each pre-token, merge the adjacent characters(tokens),
    # then count the pairs' frequencies.
    # This will only run once in the very first merging.
    if not pair_counts:
        pair_counts = Counter()
        pair_to_pretokens = defaultdict(set)
        for pretoken, count in pretoken_counts.items():
            for i in range(len(pretoken) - 1):
                pair = (pretoken[i], pretoken[i + 1])
                pair_counts[pair] += count
                pair_to_pretokens[pair].add(pretoken)
        # Use a heap to efficiently get the max pair later
        pair_counts_heap = [(-c, p) for p, c in pair_counts.items()]
        heapify(pair_counts_heap)

    # Find the merged pair with the highest frequency
    selected_pair = heap_pop_best(pair_counts_heap, pair_counts)

    # List the affected pre-tokens to take a snapshot
    # since we will modify the mapping during iteration
    affected_pretokens = list(pair_to_pretokens[selected_pair])

    # Process each affected pre-tokens to prepare for the next round of merging
    for pretoken in affected_pretokens:
        count = pretoken_counts[
            pretoken
        ]  # This is why you need to update pretoken_counts.

        # Collect old pairs of this pre-token
        old_pairs = []
        for i in range(len(pretoken) - 1):
            pair = (pretoken[i], pretoken[i + 1])
            old_pairs.append(pair)
        old_pairs_ctr = Counter(old_pairs)

        # Build new pre-token by merging the selected pair
        merged_pretoken = []
        i = 0
        while i < len(pretoken):
            if (
                i < len(pretoken) - 1
                and (pretoken[i], pretoken[i + 1]) == selected_pair
            ):
                merged_pretoken.append(pretoken[i] + pretoken[i + 1])
                i += 2  # Skip the next character as it's merged
            else:
                merged_pretoken.append(pretoken[i])
                i += 1
        merged_pretoken = tuple(merged_pretoken)

        # Collect new pairs of this pre-token after merging
        # When merged_pretoken has length 1, new_pairs will be empty.
        new_pairs = []
        for i in range(len(merged_pretoken) - 1):
            pair = (merged_pretoken[i], merged_pretoken[i + 1])
            new_pairs.append(pair)
        new_pairs_ctr = Counter(new_pairs)

        # More efficient way to update pair_counts and pair_counts_heap
        pair_delta = defaultdict(int)
        changed_pairs = set()

        # Collect changed pairs and their deltas
        for p, c in old_pairs_ctr.items():
            pair_delta[p] -= c * count
            changed_pairs.add(p)
        for p, c in new_pairs_ctr.items():
            pair_delta[p] += c * count
            changed_pairs.add(p)

        # Update changed pairs in pair_counts and pair_counts_heap with non-zero deltas
        for p in changed_pairs:
            newc = pair_counts[p] + pair_delta[p]
            if newc <= 0:
                pair_counts.pop(p, None)
            if newc > 0 and pair_delta[p] != 0:
                pair_counts[p] = newc
                heappush(pair_counts_heap, (-newc, p))

        # Update the mapping from pairs to pre-tokens
        for p in old_pairs_ctr.keys():
            pair_to_pretokens[p].discard(pretoken)
            if not pair_to_pretokens[p]:
                pair_to_pretokens.pop(p, None)

        for p in new_pairs_ctr.keys():
            pair_to_pretokens[p].add(merged_pretoken)

        # Update the pre-token counts
        pretoken_counts.pop(pretoken, None)
        pretoken_counts[merged_pretoken] += count

    return (
        selected_pair,
        pretoken_counts,
        pair_counts,
        pair_counts_heap,
        pair_to_pretokens,
    )


def train_bpe(
    input_path: str, vocab_size: int, special_tokens: list[str]
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """Train a BPE tokenizer on the given text document.

    Args:
        input_path (str): path of the text file
        vocab_size (int): maximum size of the vocabulary
        special_tokens (list[str]): list of special tokens to separate documents

    Returns:
        dict[int, bytes]: the vocabulary mapping
        list[tuple[bytes, bytes]]]: list of merges performed
    """
    with open(input_path, "rb") as f:
        # Split the entire text into chunks for parallel processing
        # special_tokens[0] is the token that represents boundary between documents.
        boundaries = find_chunk_boundaries(f, 12, special_tokens[0].encode("utf-8"))

    # Create single bytes to Unicode character mapping
    b2u, u2b = byte_to_unicode()

    # Use the regex-based GPT-2 pre-tokenizer to pre-tokenize each segment
    gpt2_pat = re.compile(
        r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    )

    # Use Python's multiprocessing module
    args = [
        (input_path, start, end, special_tokens, b2u, gpt2_pat)
        for start, end in zip(boundaries[:-1], boundaries[1:])
    ]
    with Pool(processes=12) as pool:
        results = pool.starmap(pre_tokenize, args)

    # Combine results from the multiple processes
    pretoken_counts = Counter()
    for result in results:
        # results are of the same Counter type.
        pretoken_counts += result

    ## Train BPE
    with tqdm(
        total=vocab_size - 256 - len(special_tokens), desc="BPE_training"
    ) as pbar:
        # Initialize vocabulary with single-byte tokens and special tokens
        vocab = {i: bytes([i]) for i in range(256)}
        for token in special_tokens:
            vocab[len(vocab)] = token.encode("utf-8")

        pair_counts = Counter()
        pair_counts_heap = []
        pair_to_pretokens = defaultdict(set)
        merges = []
        k = 0
        while k < vocab_size - 256 - len(special_tokens):
            (
                merged_pairs,
                pretoken_counts,
                pair_counts,
                pair_counts_heap,
                pair_to_pretokens,
            ) = merge_tokens(
                pretoken_counts, pair_counts, pair_counts_heap, pair_to_pretokens
            )
            if not pair_counts:
                break
            merged_pairs_bytes = (
                bytes(u2b[u] for u in merged_pairs[0]),
                bytes(u2b[u] for u in merged_pairs[1]),
            )
            merges.append(merged_pairs_bytes)
            vocab[len(vocab)] = merged_pairs_bytes[0] + merged_pairs_bytes[1]
            k += 1
            pbar.update(1)

    return vocab, merges


def save_vocab_and_merges(
    vocab: dict[int, bytes],
    merges: list[tuple[bytes, bytes]],
    output_dir: str,
    vocab_filename: str,
    merges_filename: str,
) -> None:
    """Save the vocabulary and merges to files.

    Args:
        vocab (dict[int, bytes]): vocabulary mapping
        merges (list[tuple[bytes, bytes]]): list of merges performed
        output_dir (str): directory to save the vocab and merges files
        vocab_filename (str): name of the vocab file
        merges_filename (str): name of the merges file
    """
    os.makedirs(output_dir, exist_ok=True)
    vocab_path = os.path.join(output_dir, vocab_filename)
    merges_path = os.path.join(output_dir, merges_filename)

    # Save vocab
    gpt2_vocab = {
        index: "".join(chr(b) for b in token) for index, token in vocab.items()
    }  # Latin-1 style encoding
    with open(vocab_path, "w", encoding="utf-8") as f:
        json.dump(gpt2_vocab, f, ensure_ascii=False, indent=4)

    # Save merges
    merges_list = [
        ["".join(chr(b1) for b1 in pair[0]), "".join(chr(b2) for b2 in pair[1])]
        for pair in merges
    ]  # Latin-1 style encoding
    with open(merges_path, "w", encoding="utf-8") as f:
        json.dump(merges_list, f, ensure_ascii=False, indent=4)


if __name__ == "__main__":
    # Special tokens
    END_TOKEN = "<|endoftext|>"
    # Maximum size of the vocabulary
    VOCAB_SIZE = 10000
    # Path of the document
    # TRAIN_FILE_PATH = "data/owt_train.txt"
    TRAIN_FILE_PATH = "data/TinyStoriesV2-GPT4-train.txt"

    OUTPUT_DIR = "cs336_basics"
    VOCAB_FILENAME = "trained_tokenizer/bpe_vocab_OWT.json"
    MERGES_FILENAME = "trained_tokenizer/bpe_merges_OWT.json"

    trained_vocab, trained_merges = train_bpe(TRAIN_FILE_PATH, VOCAB_SIZE, [END_TOKEN])
    # save_vocab_and_merges(
    #     trained_vocab, trained_merges, OUTPUT_DIR, VOCAB_FILENAME, MERGES_FILENAME
    # )
    # print(f"Saved BPE vocab to {os.path.join(OUTPUT_DIR, VOCAB_FILENAME)}")
    # print(f"Saved BPE merges to {os.path.join(OUTPUT_DIR, MERGES_FILENAME)}")
