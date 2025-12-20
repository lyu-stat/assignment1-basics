"""This script will take the tokenized data and train a transformer model on it."""

import argparse
import os
import numpy as np
import torch
import wandb
from cs336_basics.training_loop import (
    data_loading,
    save_checkpoint,
    load_checkpoint,
    make_checkpoint_dir,
    estimate_val_loss,
)
from cs336_basics.transformer import TransformerLM
from cs336_basics.optimizer import (
    compute_cross_entropy,
    AdamW,
    learning_rate_schedule,
    gradient_clipping,
)


INPUT_FILE = "cs336_basics/encoded_token_ids/TinyStories_train.npy"
VAL_FRAC = 0.1  # Fraction of data to use for validation
TRAINING_LOSS_LOGGING_INTERVAL = 100  # Log training loss every 100 steps
VALIDATION_LOSS_LOGGING_INTERVAL = 1000  # Log validation loss every 1000 steps
EVAL_ITERATIONS = 100  # Number of iterations to average validation loss over
MAX_STEP = 100000  # Total number of training steps

## Model and Optimizer hyperparameters from the command line arguments
parser = argparse.ArgumentParser(description="Train a Transformer Language Model.")
# Training data file
parser.add_argument(
    "--input_file", type=str, default=INPUT_FILE, help="Path to training data."
)
# Model hyperparameters
parser.add_argument(
    "--vocab_size", type=int, default=10000, help="Vocabulary size of the model."
)
parser.add_argument(
    "--batch_size", type=int, default=32, help="Batch size for training."
)
parser.add_argument(
    "--context_length", type=int, default=1024, help="Context length for the model."
)
parser.add_argument(
    "--n_layers", type=int, default=48, help="Number of transformer layers."
)
parser.add_argument(
    "--n_heads", type=int, default=25, help="Number of attention heads."
)
parser.add_argument("--d_model", type=int, default=1600, help="Dimension of the model.")
parser.add_argument(
    "--d_ff", type=int, default=6400, help="Dimension of the feedforward layer."
)
parser.add_argument(
    "--theta", type=float, default=1e4, help="Theta hyperparameter for RoPE."
)
# Optimizer hyperparameters
parser.add_argument(
    "--max_steps", type=int, default=MAX_STEP, help="Number of training steps."
)
parser.add_argument(
    "--weight_decay",
    type=float,
    default=1e-2,
    help="Weight decay for the optimizer.",
)
parser.add_argument(
    "--beta1", type=float, default=0.9, help="Beta1 hyperparameter for AdamW."
)
parser.add_argument(
    "--beta2", type=float, default=0.999, help="Beta2 hyperparameter for AdamW."
)
parser.add_argument("--eps", type=float, default=1e-8, help="Epsilon for AdamW.")
parser.add_argument(
    "--max_norm", type=int, default=1, help="Maximum L2-norm for gradient clipping."
)
parser.add_argument("--lr_max", type=float, default=3e-4, help="Maximum learning rate.")
parser.add_argument("--lr_min", type=float, default=3e-5, help="Minimum learning rate.")
parser.add_argument(
    "--w_step", type=int, default=0.03 * MAX_STEP, help="Warm-up steps."
)
parser.add_argument(
    "--c_step", type=int, default=MAX_STEP, help="Cosine annealing steps."
)
# Checkpoint location
parser.add_argument(
    "--checkpoint_to_load",
    type=str,
    default="",
    help="Checkpoint to load.",
)

args = parser.parse_args()

# Load dataset
dataset = np.load(args.input_file, mmap_mode="r")

# Set device
training_device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available() else "cpu"
)

# Initialize model and optimizer; Move model to appropriate device
transformer_model = TransformerLM(
    vocab_size=args.vocab_size,
    context_length=args.context_length,
    num_layers=args.n_layers,
    num_heads=args.n_heads,
    d_model=args.d_model,
    theta=args.theta,
)
transformer_model.to(training_device)
optimizer = AdamW(
    transformer_model.parameters(),
    lr=args.lr_max,  # Initial lr, will be updated in the loop
    weight_decay=args.weight_decay,
    betas=(args.beta1, args.beta2),
    eps=args.eps,
)

# Split training and validation datasets
split_idx = int(len(dataset) * (1 - VAL_FRAC))
train_dataset = dataset[:split_idx]
val_dataset = dataset[split_idx:]

# Load checkpoint if provided
start_step = 1
if args.checkpoint_to_load:
    start_step = (
        load_checkpoint(args.checkpoint_to_load, transformer_model, optimizer) + 1
    )
    assert start_step <= args.max_steps, "Checkpoint step exceeds max steps."

# Initialize Weights & Biases logging
run = wandb.init(
    project="cs336-basics-transformer-training",
    config=vars(args),
)
if args.checkpoint_to_load:
    checkpoint_path = os.path.dirname(args.checkpoint_to_load)
else:
    checkpoint_path = make_checkpoint_dir(
        run, "transformer", "TinyStories", "./cs336_basics/checkpoints"
    )

# Training loop
for t in range(start_step, args.max_steps + 1):
    # Get batch data
    inputs, targets = data_loading(
        train_dataset, args.batch_size, args.context_length, training_device
    )

    # Forward pass and compute loss
    logits = transformer_model(inputs)
    loss = compute_cross_entropy(logits, targets)

    # Backward pass
    optimizer.zero_grad()
    loss.backward()
    gradient_clipping(list(transformer_model.parameters()), args.max_norm)
    # Adjust learning rate
    lr = learning_rate_schedule(t, args.lr_max, args.lr_min, args.w_step, args.c_step)
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr
    optimizer.step()

    # Print training loss every 100 steps
    if t % 100 == 0:
        run.log({"training_loss": loss.item(), "step": t})
        print(f"Step {t}: Loss = {loss.item():.4f}")

    # Print validation loss every 1000 steps
    if t % 1000 == 0:
        validation_loss, perplexity = estimate_val_loss(
            EVAL_ITERATIONS,
            val_dataset,
            args.batch_size,
            args.context_length,
            transformer_model,
            training_device,
        )
        run.log(
            {
                "validation_loss": validation_loss,
                "perplexity": perplexity,
                "step": t,
            }
        )
        print(f"Step {t}: Validation Loss = {validation_loss:.4f}")

    # Save checkpoints and the final model
    if t % 1000 == 0:
        if t == args.max_steps:
            save_checkpoint(
                transformer_model,
                optimizer,
                t,
                f"{checkpoint_path}/final_model.pt",
            )
            print("Final model saved.")
        else:
            save_checkpoint(
                transformer_model,
                optimizer,
                t,
                f"{checkpoint_path}/checkpoint_step_{t}.pt",
            )

# Finish Weights & Biases run
run.finish()
