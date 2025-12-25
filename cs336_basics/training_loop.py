import os
from typing import BinaryIO, IO, Iterable
from os import PathLike
import time
import math
import numpy as np
import torch
from torch import nn
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
    run_id: str,
    out: str | PathLike | BinaryIO | IO[bytes],
) -> None:
    """Saves a model checkpoint.

    Args:
        model (nn.Module): the model to save.
        optimizer (torch.optim.Optimizer): the optimizer to save.
        iteration (int): the current training iteration.
        run_id (str): the wandb run id.
        out (str): the output file path.
    """
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "iteration": iteration,
        "run_id": run_id,
    }
    torch.save(checkpoint, out)
    print(f"Checkpoint saved to {out} at iteration {iteration}.")


def load_checkpoint(
    src: str | PathLike | BinaryIO | IO[bytes],
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
) -> tuple[int, str]:
    """Loads a model checkpoint.

    Args:
        src (str): the source file path.
        model (nn.Module): the model to load.
        optimizer (torch.optim.Optimizer): the optimizer to load.

    Returns:
        tuple[int, str]: a tuple containing the iteration and wandb run id.
    """
    checkpoint = torch.load(src)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    print(f"Checkpoint loaded from {src} at iteration {checkpoint['iteration']}.")
    return checkpoint["iteration"], checkpoint["run_id"]


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
    perplexity: bool,
) -> tuple[float, float]:
    """Estimate the validation loss over a number of iterations.
    Args:
        eval_iters (int): number of iterations to average the validation loss over.
        val_dataset (np.ndarray): the validation dataset.
        batch_size (int): batch size for evaluation.
        context_length (int): context length for evaluation.
        model (nn.Module): the model to evaluate.
        device (torch.device): device to perform evaluation on.
        perplexity (bool): whether or not to calculate perplexity.
    Returns:
        tuple[float, float]: a tuple containing the average validation loss and perplexity.
    """
    total_loss = 0.0
    with torch.no_grad():  # Ask PyTorch not to create computation graphs
        for _ in range(eval_iters):
            val_inputs, val_targets = data_loading(
                val_dataset, batch_size, context_length, device
            )
            val_logits = model(val_inputs)
            val_loss = compute_cross_entropy(val_logits, val_targets)
            total_loss += float(val_loss.item())
    if perplexity:
        return total_loss / eval_iters, math.exp(total_loss / eval_iters)
    return total_loss / eval_iters, 0.0


def gradient_clipping(params: list[nn.Parameter], max_norm: float) -> None:
    """Clip the gradients of the given parameters to have a maximum norm of max_norm.

    Args:
        params (list[torch.Tensor]): list of the model parameters
        max_norm (float): maximum allowed norm for the gradients
    """
    total_norm = 0.0
    for p in params:
        grad = p.grad
        if grad is not None:
            grad_norm = torch.norm(grad).item()
            total_norm += grad_norm**2
    total_norm = math.sqrt(total_norm)

    if total_norm > max_norm:
        clip_coef = max_norm / (total_norm + 1e-6)
        for p in params:
            if p.grad is not None:  # Be safe for in-place operations
                p.grad *= clip_coef


def _accumulate_tensor_stats(
    tensor: torch.Tensor, tss: float, total_numel: int
) -> tuple[float, int]:
    """Accumulate the total sum of squares of the elements of the tensor
    and accumulate the total number of elements.

    Args:
        tensor (torch.Tensor): input tensor.
        tss (float): total sum of squares accumulated so far.
        total_numel (int): total number of elements counted so far.

    Returns:
        tuple[float, int]: a tuple containing the updated total sum of squares
          and total number of elements.
    """
    # Detach tensor from computation graph and convert to float32 for numerical stability
    tensor_float = tensor.detach().to(torch.float32)
    tss += torch.sum(tensor_float**2).item()
    total_numel += tensor_float.numel()
    return tss, total_numel


def compute_parameter_norms(parameters: Iterable[nn.Parameter]) -> tuple[float, float]:
    """Compute L2 norm and RMS norm for all model parameters combined.

    Args:
        parameters (Iterable[nn.Parameter]): parameters of the model.

    Returns:
        tuple[float, float]: a tuple containing the L2 norm and RMS norm.
    """
    tss = 0.0
    total_numel = 0
    for param in parameters:
        tss, total_numel = _accumulate_tensor_stats(param, tss, total_numel)
    l2_norm = math.sqrt(tss)
    rms_norm = math.sqrt(tss / total_numel) if total_numel > 0 else 0.0
    return l2_norm, rms_norm


def compute_gradient_norms(parameters: Iterable[nn.Parameter]) -> tuple[float, float]:
    """Compute L2 norm and RMS norm for all gradients combined.

    Args:
        parameters (Iterable[nn.Parameter]): parameters of the model.

    Returns:
        tuple[float, float]: a tuple containing the L2 norm and RMS norm of the gradients.
    """
    tss = 0.0
    total_numel = 0
    for param in parameters:
        if param.grad is not None:
            tss, total_numel = _accumulate_tensor_stats(param.grad, tss, total_numel)
    l2_norm = math.sqrt(tss)
    rms_norm = math.sqrt(tss / total_numel) if total_numel > 0 else 0.0
    return l2_norm, rms_norm


def _accumulate_output_stats(
    output: object, tss: float, total_numel: int
) -> tuple[float, int]:
    """Accumulate the total sum of squares of all tensors within the output and
    accumulate total number of elements within the output

    Args:
        output (object): output from a forward hook.
        tss (float): total sum of squares accumulated so far.
        total_numel (int): total number of elements counted so far.

    Returns:
        tuple[float, int]: a tuple containing the updated total sum of squares
          and total number of elements.
    """
    if torch.is_tensor(output):
        return _accumulate_tensor_stats(output, tss, total_numel)
    if isinstance(output, (tuple, list)):
        for item in output:
            tss, total_numel = _accumulate_output_stats(item, tss, total_numel)
    if isinstance(output, dict):
        for item in output.values():
            tss, total_numel = _accumulate_output_stats(item, tss, total_numel)
    return tss, total_numel


class ActivationNormTracker:
    """Track RMS norm of all activations combined during training."""

    def __init__(self, model: nn.Module) -> None:
        """Initialize the activation norm tracker.

        Args:
            model (nn.Module): model to track activations for.
        """
        self._handles = []
        self._tss = 0.0
        self._total_numel = 0
        for name, module in model.named_modules():
            if not name:
                continue  # Skip the top-level module
            if len(list(module.children())) > 0:
                continue  # Skip non-leaf modules
            self._handles.append(module.register_forward_hook(self._hook))

    def _hook(self, module: nn.Module, input: tuple, output: torch.Tensor) -> None:
        """The hook function to register to the module.

        Args:
            module (nn.Module): module being hooked.
            input (tuple): inputs to the module.
            output (torch.Tensor): output from the module.
        """
        self._tss, self._total_numel = _accumulate_output_stats(
            output, self._tss, self._total_numel
        )

    def reset(self) -> None:
        """Reset the tracker statistics."""
        self._tss = 0.0
        self._total_numel = 0

    def get_rms_norm(self) -> float:
        """Get the RMS norm of the tracked activations.

        Returns:
            float: RMS norm of the tracked activations.
        """
        return (
            math.sqrt(self._tss / self._total_numel) if self._total_numel > 0 else 0.0
        )

    def close(self) -> None:
        """Remove all registered hooks."""
        for handle in self._handles:
            handle.remove()
        self._handles = []
