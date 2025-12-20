import json
from typing import Iterable, Iterator
import regex as re
import random
import time
import os
from tqdm import tqdm
import numpy as np


class Tokenizer:
    """A simple Byte Pair Encoding (BPE) tokenizer."""

    def __init__(
        self,
        vocab: dict[int, bytes],
        merges: list[tuple[bytes, bytes]],
        special_tokens: list[str] | None = None,
    ) -> None:
        """Initializes the Tokenizer with vocabulary, merges, and special tokens.

        Args:
            vocab (dict[int, bytes]): vocabulary mapping token IDs to byte sequences
            merges (list[tuple[bytes, bytes]]): list of byte pair merges
            special_tokens (list[str] | None, optional): list of special tokens. Defaults to None.
        """
        self.vocab = vocab
        self.vocab_size = len(vocab)
        self.merges = merges
        self.merge_ranks = {p: i for i, p in enumerate(self.merges)}
        self.special_tokens = special_tokens or []
        self.gpt2_pat = re.compile(
            r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
        )
        if self.special_tokens:
            self.special_pat = re.compile(
                "("
                + "|".join(
                    re.escape(token)
                    for token in sorted(self.special_tokens, reverse=True)
                )
                + ")"
            )
        else:
            self.special_pat = None

        # Add special tokens to the vocabulary
        if self.special_tokens:
            i = 0
            max_id = max(self.vocab.keys())
            for token in self.special_tokens:
                b = token.encode("utf-8")
                if b not in self.vocab.values():
                    self.vocab[max_id + 1 + i] = b
                    i += 1

        self.token2id = {v: k for k, v in vocab.items()}

    @classmethod
    def from_files(
        cls,
        vocab_filepath: str,
        merges_filepath: str,
        special_tokens: list[str] | None = None,
    ) -> "Tokenizer":
        """Creates a Tokenizer instance from vocabulary and merges files.

        Args:
            vocab_filepath (str): filepath to the vocabulary file
            merges_filepath (str): filepath to the merges file
            special_tokens (list[str] | None, optional): list of special tokens. Defaults to None.

        Returns:
            Tokenizer: an instance of the Tokenizer class
        """
        with open(vocab_filepath, "r", encoding="utf-8") as vf:
            saved_vocab = json.load(vf)
        # Convert string keys back to integers, and string values back to bytes
        saved_vocab = {
            int(k): bytes(ord(ch) for ch in v) for k, v in saved_vocab.items()
        }  # Latin-1 style decoding

        with open(merges_filepath, "r", encoding="utf-8") as mf:
            saved_merges = json.load(mf)
        saved_merges = [
            (bytes(ord(ch) for ch in item[0]), bytes(ord(ch) for ch in item[1]))
            for item in saved_merges
        ]  # Latin-1 style decoding

        return cls(
            saved_vocab,
            saved_merges,
            special_tokens,
        )

    def _split_on_special_tokens(self, text: str) -> list[tuple[str, bool]]:
        """Splits the input text into chunks based on special tokens.

        Args:
            text (str): input text

        Returns:
            list[tuple[str, bool]]: list of tuples containing text chunks and a boolean indicator
                                     for whether the chunk is a special token
        """
        # Split the text using the special tokens pattern
        chunks = self.special_pat.split(text)

        # Identify special tokens in the chunks
        results = []
        for chunk in chunks:
            if chunk in self.special_tokens:
                results.append((chunk, True))
            else:
                results.append((chunk, False))
        return results

    def _get_pairs(self, tokens: list[bytes]) -> set[tuple[bytes, bytes]]:
        """Return set of adjacent token pairs.

        Args:
            tokens (list[bytes]): list of tokens in bytes

        Returns:
            set(tuple[bytes, bytes]): set of adjacent token pairs
        """
        return {(tokens[i], tokens[i + 1]) for i in range(len(tokens) - 1)}

    def _apply_bpe_merges(self, byte_sequence: bytes) -> list[bytes]:
        """Apply BPE merges to a byte sequence to create tokens.

        Args:
            byte_sequance (bytes): a byte sequence to be tokenized

        Returns:
            list[bytes]: the list of tokens generated after applying BPE merges
        """
        # Initialize the list of tokens with individual bytes
        tokens = [bytes([b]) for b in byte_sequence]

        # Localize self attributes for faster access
        merge_ranks = self.merge_ranks
        get_pairs = self._get_pairs

        pairs = get_pairs(tokens)

        while True:
            min_rank = float("inf")
            best_pair = None
            for p in pairs:
                rank = merge_ranks.get(p, None)
                if rank is not None and rank < min_rank:
                    # You can't do "if rank and rank < min_rank:" because this fails if rank is 0.
                    min_rank = rank
                    best_pair = p

            if not best_pair:
                break

            new_tokens = []
            i = 0
            while i < len(tokens):
                if i < len(tokens) - 1 and (tokens[i], tokens[i + 1]) == best_pair:
                    new_tokens.append(tokens[i] + tokens[i + 1])
                    i += 2
                else:
                    new_tokens.append(tokens[i])
                    i += 1

            tokens = new_tokens
            if len(tokens) == 1:
                break

            pairs = get_pairs(tokens)

        return tokens

    def encode(self, text: str) -> list[int]:
        """Encodes a string into a list of token IDs using BPE.

        Args:
            text (str): input string to encode

        Returns:
            list[int]: list of token IDs
        """
        # Handle special tokens
        if self.special_tokens:
            results = self._split_on_special_tokens(text)
        else:
            results = [(text, False)]

        # Pre-tokenization and BPE encoding
        gpt2_pat = self.gpt2_pat
        token2id = self.token2id
        apply_bpe_merges = self._apply_bpe_merges

        token_ids = []
        for result in results:
            chunk, is_special = result
            if is_special:
                # Encode special tokens directly
                token_ids.append(token2id[chunk.encode("utf-8")])
            else:
                # Pre-tokenize the chunk
                pretokens = gpt2_pat.finditer(chunk)

                # Encode the pre-tokens using BPE
                for pretoken in pretokens:
                    # Convert the input string to bytes
                    byte_sequence = pretoken.group().encode("utf-8")

                    # Apply BPE merges to get tokens
                    tokens = apply_bpe_merges(byte_sequence)

                    # Convert tokens to their corresponding IDs using the vocabulary
                    for token in tokens:
                        token_ids.append(token2id[token])

        return token_ids

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        """Encodes an iterable of strings into a flat iterator of token IDs.

        Args:
            iterable (Iterable[str]): iterable of input strings to encode

        Returns:
            Iterator[int]: flat iterator of token IDs
        """
        for text in iterable:
            for token_id in self.encode(text):
                yield token_id

    def decode(self, ids: list[int]) -> str:
        """Decodes a list of token IDs back into a string.

        Args:
            ids (list[int]): list of token IDs to decode

        Returns:
            str: decoded string
        """
        byte_sequence = bytearray()
        for token_id in ids:
            byte_sequence.extend(self.vocab[token_id])
        return byte_sequence.decode("utf-8", errors="ignore")


def encode_and_return_compression_ratio(
    tokenizer: Tokenizer, sample: list[str]
) -> float:
    """Encodes the input text and returns the compression ratio.

    Args:
        tokenizer (Tokenizer): the tokenizer to use
        text (str): input text to encode

    Returns:
        float: compression ratio (original length / encoded length)
    """
    original_length = len("".join(sample).encode("utf-8"))
    encoded_length = 0
    for _ in tokenizer.encode_iterable(sample):
        encoded_length += 1
    return original_length / encoded_length if encoded_length > 0 else 0.0


def sample_documents(filepath: str, special_token: str, n: int) -> list[str]:
    """Samples n documents from the path to the cohort file.

    Args:
        filepath (str): the path to the cohort file
        special_token (str): the token that separates documents
        n (int): number of documents to sample

    Returns:
        list[str]: list of sampled documents
    """
    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()
    documents = content.split(special_token)
    random.seed(42)
    return random.sample(documents, n)


def measure_throughput_bytes_per_sec(
    tokenizer: Tokenizer, filepath: str
) -> tuple[int, float]:
    """Measure the throughput of Tokenizer in bytes per second.

    Args:
        tokenizer (Tokenizer): BPE tokenizer instance
        filepath (str): path to the text file to be tokenized

    Returns:
        int: total bytes processed
        float: bytes processed per second
    """
    t0 = time.perf_counter()
    total_bytes = 0
    with open(filepath, "r", encoding="utf-8") as f:
        # Get total file size for progress bar
        f.seek(0, os.SEEK_END)
        total_size = f.tell()
        f.seek(0)
        with tqdm(total=total_size, desc="Tokenizing") as pbar:
            for line in f:
                line_bytes = line.encode("utf-8")
                total_bytes += len(line_bytes)
                _ = tokenizer.encode(line)
                pbar.update(len(line_bytes))
    t1 = time.perf_counter()
    elapsed_time = t1 - t0
    bytes_per_second = total_bytes / elapsed_time if elapsed_time > 0 else 0.0
    return total_bytes, bytes_per_second


def encode_file_stream(tokenizer: Tokenizer, filepath: str) -> np.ndarray:
    """Encode the contents of a file into a NumPy array of token IDs line by line.

    Args:
        tokenizer (Tokenizer): the tokenizer to use
        filepath (str): path to the text file to be encoded

    Returns:
        np.ndarray: NumPy array of token IDs
    """
    with open(filepath, "r", encoding="utf-8") as f:
        token_ids = list(tokenizer.encode_iterable(f))
    return np.array(token_ids, dtype=np.uint16)


def save_numpy_array(array: np.ndarray, output_filepath: str) -> None:
    """Save a NumPy array to a .npy file.

    Args:
        array (np.ndarray): the NumPy array to save
        output_filepath (str): path of the output .npy file
    """
    np.save(output_filepath, array)


if __name__ == "__main__":
    VOCAB_FILEPATH = "cs336_basics/trained_tokenizer/bpe_vocab_OWT.json"
    MERGES_FILEPATH = "cs336_basics/trained_tokenizer/bpe_merges_OWT.json"
    SPECIAL_TOKENS = ["<|endoftext|>"]
    ENCODE_FILEPATH = "data/owt_train.txt"
    TOKEN_IDS_FILEPATH = "cs336_basics/encoded_token_ids/OWT_train.npy"

    # Load the tokenizer
    bpe_tokenizer = Tokenizer.from_files(
        VOCAB_FILEPATH, MERGES_FILEPATH, SPECIAL_TOKENS
    )

    # Encode file and save token IDs
    t0_encode = time.perf_counter()
    encoded_token_ids = encode_file_stream(bpe_tokenizer, ENCODE_FILEPATH)
    save_numpy_array(encoded_token_ids, TOKEN_IDS_FILEPATH)
    t1_encode = time.perf_counter()
    print(f"Encoded {ENCODE_FILEPATH} in {t1_encode - t0_encode:.2f} seconds.")

    # # Estimate compression ratio
    # cohort = sample_documents("data/owt_valid.txt", "<|endoftext|>", 10)
    # compression_ratio = encode_and_return_compression_ratio(bpe_tokenizer, cohort)
    # print(f"OWT Compression Ratio: {compression_ratio:.2f}")

    # # Estimate throughput
    # tot_bytes, bps = measure_throughput_bytes_per_sec(
    #     bpe_tokenizer, "data/owt_valid.txt"
    # )
    # print(f"Processed {tot_bytes} bytes in OWT Valid at {bps:.2f} bytes/second")
