import os
import time
import math
import torch
import numpy as np
from torch import nn
from typing import BinaryIO, IO
from os import PathLike
from cs336_basics.optimizer import compute_cross_entropy


def data_loading(
    x: np.ndarray, batch_size: int, context_length: int, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    """Loads data into batches for training.

    Args:
        x (np.ndarray): the integer array with token IDs.
        batch_size (int): batch size.
        context_length (int): sequence length.
        device (torch.device): the device that where you plan to train the model.

    Returns:
        tuple[torch.Tensor, torch.Tensor]: a pair of tensors (inputs, targets).
    """
    inputs_array = np.zeros((batch_size, context_length), dtype=np.int64)
    target_array = np.zeros((batch_size, context_length), dtype=np.int64)
    # dtype np.int64 corresponds to torch.long

    for i in range(batch_size):
        start_idx = np.random.randint(
            0, len(x) - context_length - 1
        )  # len(x) - context_length is exclusive.
        inputs_array[i, :] = x[start_idx : start_idx + context_length]
        target_array[i, :] = x[start_idx + 1 : start_idx + context_length + 1]

    inputs = torch.tensor(inputs_array, device=device, dtype=torch.long)
    targets = torch.tensor(target_array, device=device, dtype=torch.long)

    return inputs, targets


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    iteration: int,
    out: str | PathLike | BinaryIO | IO[bytes],
) -> None:
    """Saves a model checkpoint.

    Args:
        model (nn.Module): the model to save.
        optimizer (torch.optim.Optimizer): the optimizer to save.
        iteration (int): the current training iteration.
        out (str): the output file path.
    """
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "iteration": iteration,
    }
    torch.save(checkpoint, out)
    print(f"Checkpoint saved to {out} at iteration {iteration}.")


def load_checkpoint(
    src: str | PathLike | BinaryIO | IO[bytes],
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
) -> int:
    """Loads a model checkpoint.

    Args:
        src (str): the source file path.
        model (nn.Module): the model to load.
        optimizer (torch.optim.Optimizer): the optimizer to load.

    Returns:
        int: the iteration when the checkpoint was taken.
    """
    checkpoint = torch.load(src)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    print(f"Checkpoint loaded from {src} at iteration {checkpoint['iteration']}.")
    return checkpoint["iteration"]


def make_checkpoint_dir(run, model_name: str, dataset: str, base_dir: str) -> str:
    """Creates a directory for saving checkpoints.

    Args:
        run: the wandb run object.
        model_name (str): the name of the model.
        dataset (str): the name of the dataset.
        base_dir (str): the base directory to save checkpoints.

    Returns:
        str: the path to the checkpoint directory.
    """
    date = time.strftime("%Y%m%d")
    checkpoint_dir = os.path.join(base_dir, f"{date}_{model_name}_{dataset}_{run.id}")
    os.makedirs(checkpoint_dir, exist_ok=True)
    return checkpoint_dir


def estimate_val_loss(
    eval_iters: int,
    val_dataset: np.ndarray,
    batch_size: int,
    context_length: int,
    model: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    """Estimate the validation loss over a number of iterations.
    Args:
        eval_iters (int): number of iterations to average the validation loss over.
        val_dataset (np.ndarray): the validation dataset.
        batch_size (int): batch size for evaluation.
        context_length (int): context length for evaluation.
        model (nn.Module): the model to evaluate.
        device (torch.device): device to perform evaluation on.
    Returns:
        tuple[float, float]: a tuple containing the average validation loss and perplexity.
    """
    val_inputs, val_targets = data_loading(
        val_dataset, batch_size, context_length, device
    )
    val_losses = []
    with torch.no_grad():
        for _ in range(eval_iters):
            val_logits = model(val_inputs)
            val_loss = compute_cross_entropy(val_logits, val_targets)
            val_losses.append(val_loss.item())
    return sum(val_losses) / len(val_losses), math.exp(
        sum(val_losses) / len(val_losses)
    )
