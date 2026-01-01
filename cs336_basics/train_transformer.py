"""This script will take the tokenized data and train a transformer model on it."""

import argparse
import os
import numpy as np
import torch
import wandb
from cs336_basics.training_loop import (
    gradient_clipping,
    data_loading,
    save_checkpoint,
    load_checkpoint,
    make_checkpoint_dir,
    estimate_val_loss,
    ActivationNormTracker,
    ActivationPerModuleNormTracker,
    compute_gradient_norms,
    compute_parameter_norms,
)
from cs336_basics.transformer import TransformerLM
from cs336_basics.optimizer import (
    compute_cross_entropy,
    AdamW,
    learning_rate_schedule,
)


## Model and Optimizer hyperparameters from the command line arguments
parser = argparse.ArgumentParser(description="Train a Transformer Language Model.")
# Training data file
parser.add_argument("--input_file", type=str, help="Path to training data.")
parser.add_argument(
    "--valid_file", type=str, default="", help="Path to validation data."
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
    "--max_steps", type=int, default=10000, help="Number of training steps."
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
parser.add_argument("--w_step", type=int, default=300, help="Warm-up steps.")
parser.add_argument("--c_step", type=int, default=10000, help="Cosine annealing steps.")
# Logging and checkpointing
parser.add_argument(
    "--val_fraction", type=float, default=0.1, help="Fraction of data for validation."
)
parser.add_argument(
    "--training_loss_interval",
    type=int,
    default=10,
    help="Interval for logging training loss.",
)
parser.add_argument(
    "--validation_loss_interval",
    type=int,
    default=100,
    help="Interval for logging validation loss.",
)
parser.add_argument(
    "--perplexity",
    action="store_true",
    help="Whether to log perplexity along with validation loss.",
)
parser.add_argument(
    "--checkpoint_interval",
    type=int,
    default=1000,
    help="Interval for saving model checkpoints.",
)
parser.add_argument(
    "--norm_interval",
    type=int,
    default=10,
    help="Interval for logging activation, gradient, and parameter norms.",
)
parser.add_argument(
    "--eval_iterations",
    type=int,
    default=100,
    help="Number of iterations to average validation loss over.",
)
parser.add_argument("--checkpoint_dir", type=str, help="Directory to save checkpoints.")
parser.add_argument(
    "--dataset_name", type=str, default="TinyStories", help="Name of the dataset."
)
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
if args.valid_file:
    val_dataset = np.load(args.valid_file, mmap_mode="r")
    train_dataset = dataset
else:
    split_idx = int(len(dataset) * (1 - args.val_fraction))
    train_dataset = dataset[:split_idx]
    val_dataset = dataset[split_idx:]

# Load checkpoint if provided
start_step = 1
run_id = ""
if args.checkpoint_to_load:
    start_step, run_id = load_checkpoint(
        args.checkpoint_to_load, transformer_model, optimizer
    )
    start_step = start_step + 1
    assert start_step <= args.max_steps, "Checkpoint step exceeds max steps."

# Initialize Weights & Biases logging
if run_id:
    run = wandb.init(
        project="cs336-basics-transformer-training",
        id=run_id,
        resume="must",
    )
else:
    run = wandb.init(
        project="cs336-basics-transformer-training",
        config=vars(args),
    )
if args.checkpoint_to_load:
    checkpoint_path = os.path.dirname(args.checkpoint_to_load)
else:
    checkpoint_path = make_checkpoint_dir(
        run, "transformer", args.dataset_name, args.checkpoint_dir
    )

# Initialize activation norm tracker
activation_norm_tracker = ActivationNormTracker(transformer_model)
activation_per_module_norm_tracker = ActivationPerModuleNormTracker(transformer_model)
model_parameters = list(transformer_model.parameters())

# Training loop
for t in range(start_step, args.max_steps + 1):
    # Get batch data
    inputs, targets = data_loading(
        train_dataset, args.batch_size, args.context_length, training_device
    )

    # Forward pass and compute loss
    activation_norm_tracker.reset()
    activation_per_module_norm_tracker.reset()
    logits = transformer_model(inputs)
    logits_rms_norm = torch.sqrt(torch.mean(logits**2)).item()
    activation_rms_norm = activation_norm_tracker.get_rms_norm()
    activation_per_module_rms_norms = activation_per_module_norm_tracker.get_rms_norms()
    loss = compute_cross_entropy(logits, targets)

    # Backward pass
    optimizer.zero_grad()
    loss.backward()
    gradient_l2_norm, gradient_rms_norm = compute_gradient_norms(model_parameters)
    gradient_clipping(model_parameters, args.max_norm)
    parameter_l2_norm, parameter_rms_norm = compute_parameter_norms(model_parameters)

    # Adjust learning rate
    lr = learning_rate_schedule(t, args.lr_max, args.lr_min, args.w_step, args.c_step)
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr

    # Log training loss
    if (t == 1) or (t % args.training_loss_interval == 0):
        run.log({"training_loss": loss.item()}, step=t)
        print(f"Step {t}: Loss = {loss.item():.4f}")

    # Log validation loss
    if (t == 1) or (t % args.validation_loss_interval == 0):
        validation_loss, perplexity = estimate_val_loss(
            args.eval_iterations,
            val_dataset,
            args.batch_size,
            args.context_length,
            transformer_model,
            training_device,
            args.perplexity,
        )
        run.log(
            {
                "validation_loss": validation_loss,
                "perplexity": perplexity,
            },
            step=t,
        )
        print(f"Step {t}: Validation Loss = {validation_loss:.4f}")

    # Log norms
    if (t == 1) or (t % args.norm_interval == 0):
        activation_per_module_rms_norms_payload = {
            f"activation_rms_norm/{module_name}": norm
            for module_name, norm in activation_per_module_rms_norms.items()
        }
        run.log(
            {
                "gradient_l2_norm": gradient_l2_norm,
                "gradient_rms_norm": gradient_rms_norm,
                "parameter_l2_norm": parameter_l2_norm,
                "parameter_rms_norm": parameter_rms_norm,
                "activation_rms_norm": activation_rms_norm,
                "logits_rms_norm": logits_rms_norm,
                **activation_per_module_rms_norms_payload,
            },
            step=t,
        )

    # Update model parameters
    optimizer.step()

    # Save checkpoints and the final model
    if t % args.checkpoint_interval == 0:
        if t == args.max_steps:
            save_checkpoint(
                transformer_model,
                optimizer,
                t,
                run.id,
                f"{checkpoint_path}/final_model.pt",
            )
            print("Final model saved.")
        else:
            save_checkpoint(
                transformer_model,
                optimizer,
                t,
                run.id,
                f"{checkpoint_path}/checkpoint_step_{t}.pt",
            )

# Remove the hooks and finish Weights & Biases run
activation_norm_tracker.close()
activation_per_module_norm_tracker.close()
run.finish()
