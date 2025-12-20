"""This script generate text from a trained transformer language model."""

import torch
from cs336_basics.transformer import TransformerLM, softmax
from cs336_basics.tokenizer import Tokenizer
import argparse


def generate_text(
    prompt: str,
    tokenizer: Tokenizer,
    model: TransformerLM,
    max_length: int,
    temperature: float = 1.0,
    p: float = 1.0,
    eos_token: bytes = b"<|endoftext|>",
) -> str:
    """Generate completions from the trained language model given
    the user-provided prompt.

    Args:
        prompt (str): user-provided prompt.
        tokenizer (Tokenizer): tokenizer to encode/decode text.
        model (TransformerLM): transformer language model.
        max_length (int): maximum number of the generated tokens.
        temperature (float): temperature parameter for sampling.
        p (float): threshold parameter for top-p sampling.

    Returns:
        str: decoded completions.
    """
    # Encode the prompt
    token_ids = tokenizer.encode(prompt)
    device = next(model.parameters()).device
    input_ids = torch.tensor(token_ids, dtype=torch.long, device=device).unsqueeze(
        0
    )  # (1, seq_len). Here seq_len is not fixed.

    # Generate tokens iteratively
    k = 0
    output_token = None
    completions = []
    while (
        k < max_length
        and input_ids.size(1) < model.context_length
        and output_token != eos_token
    ):
        # Get the probabilities of the next token
        logits = model(input_ids)  # (1, seq_len, vocab_size)
        next_token_logits = logits[0, -1, :]  # (vocab_size,)
        probs = softmax(
            next_token_logits, dim=-1, temperature=temperature
        )  # (vocab_size,)

        # Top-p sampling
        sorted_probs, sorted_indices = torch.sort(probs, descending=True)
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
        cutoff_index = torch.sum(cumulative_probs <= p).item()
        filtered_probs = sorted_probs[: cutoff_index + 1]
        filtered_probs /= torch.sum(filtered_probs)  # Re-normalize
        sampled_index = torch.multinomial(
            filtered_probs, num_samples=1
        ).item()  # Index in filtered_probs

        # Get the sampled token and append to input_ids
        output_token_index = sorted_indices[
            sampled_index
        ].item()  # Index in original vocab
        output_token = tokenizer.decode([output_token_index])

        # Append to completions
        completions.append(output_token)

        # Update input_ids
        input_ids = torch.cat(
            [input_ids, torch.tensor([[output_token_index]], device=device)], dim=1
        )

        k += 1

    return "".join(completions)


if __name__ == "__main__":
    VOCAB_FILEPATH = "cs336_basics/trained_tokenizer/bpe_vocab_TinyStories.json"
    MERGES_FILEPATH = "cs336_basics/trained_tokenizer/bpe_merges_TinyStories.json"
    SPECIAL_TOKENS = ["<|endoftext|>"]

    # Load tokenizer
    trained_tokenizer = Tokenizer.from_files(
        VOCAB_FILEPATH, MERGES_FILEPATH, SPECIAL_TOKENS
    )

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Generate text from a trained Transformer LM."
    )
    parser.add_argument(
        "--prompt", type=str, required=True, help="The prompt text to start generation."
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=100,
        help="Maximum number of tokens to generate.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Temperature parameter for sampling.",
    )
    parser.add_argument(
        "--top_p",
        type=float,
        default=1.0,
        help="Top-p threshold for nucleus sampling.",
    )
    parser.add_argument(
        "--model_checkpoint",
        type=str,
        required=True,
        help="Path to the trained model checkpoint.",
    )
    args = parser.parse_args()

    # Load model from checkpoint
    generation_device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available() else "cpu"
    )
    trained_model = TransformerLM(
        vocab_size=trained_tokenizer.vocab_size,
        context_length=128,
        d_model=64,
        num_layers=48,
        num_heads=8,
        theta=1e4,
    ).to(generation_device)
    checkpoint = torch.load(args.model_checkpoint, map_location=generation_device)
    trained_model.load_state_dict(checkpoint["model_state_dict"])

    # Generate text
    output_text = generate_text(
        prompt=args.prompt,
        tokenizer=trained_tokenizer,
        model=trained_model,
        max_length=args.max_length,
        temperature=args.temperature,
        p=args.top_p,
    )

    print("Generated Text:")
    print(output_text)
