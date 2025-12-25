from typing import Callable, Iterable
import math
import torch


def compute_cross_entropy(
    logits: torch.Tensor, target_index: torch.Tensor
) -> torch.Tensor:
    """Compute the cross-entropy loss given logits and target index.

    Args:
        logits (torch.Tensor): logits tensor of shape (..., vocab_size).
            "..." can be any number of leading dimensions, aka batch-like dimensions.
        target_index (torch.Tensor): target index tensor of shape (...,).
            "..." must match the leading dimensions of logits.

    Returns:
        torch.Tensor: cross-entropy loss (scalar).
    """
    logits_max = torch.max(logits, dim=-1, keepdim=True).values
    stabilized_logits = logits - logits_max
    exp_logits = torch.exp(stabilized_logits)
    exp_logits_sum = torch.sum(exp_logits, dim=-1, keepdim=True)
    targeted_stabilized_logits = stabilized_logits.gather(
        dim=-1, index=target_index.unsqueeze(-1)
    )
    loss = -targeted_stabilized_logits + torch.log(
        exp_logits_sum + 1e-9
    )  # Adding a small constant for numerical stability
    return torch.mean(loss)  # Return the mean loss over all examples


class AdamW(torch.optim.Optimizer):
    """Implementation of the AdamW optimizer."""

    def __init__(
        self,
        params: Iterable[torch.Tensor],
        lr: float,
        weight_decay: float,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
    ):
        """Initialize the AdamW optimizer.

        Args:
            params (Itarable[torch.Tensor]): the parameters to optimize.
            lr (float): learning rate.
            weight_decay (float): weight decay coefficient.
            betas (tuple[float, float]): (first moment coefficient,
                second moment coefficient). Defaults to (0.9, 0.999).
            eps (float): small value used to improve numerical stability.
                Defaults to 1e-8.
        """
        if lr <= 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        defaults = {
            "lr": lr,
            "weight_decay": weight_decay,
            "betas": betas,
            "eps": eps,
        }
        super().__init__(params, defaults)
        # optimizer.param_groups = [
        #   {"params": list(model.parameters()), "lr": 3e-4, ...}
        # ]

    def step(self, closure: Callable | None = None) -> float | None:
        """Performs a single optimization step.
        Args:
            closure (Callable, optional): A closure that reevaluates the model
                and returns the loss. Defaults to None.
        Returns:
            float | None: loss value if closure is provided, else None.
        """
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"]
            weight_decay = group["weight_decay"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad
                state = self.state[p]
                t = state.get("step", 1)
                m = state.get("m", torch.zeros_like(p))
                v = state.get("v", torch.zeros_like(p))
                m = beta1 * m + (1 - beta1) * grad
                v = beta2 * v + (1 - beta2) * grad * grad
                lr_t = lr * math.sqrt(1 - beta2**t) / (1 - beta1**t)  # Bias correction
                p.data -= (lr_t * m) / (
                    torch.sqrt(v) + eps
                ) + lr * weight_decay * p.data
                state["m"] = m
                state["v"] = v
                state["step"] = t + 1

        return loss


def learning_rate_schedule(
    t: int, lr_max: float, lr_min: float, w_step: int, c_step: int
) -> float:
    """Compute the learning rate at step t using cosine annealing schedule.

    Args:
        t (int): current step of training
        lr_max (float): maximum learning rate
        lr_min (float): minimum (final) learning rate
        w_step (int): warm-up steps from the first step to w_step-1
        c_step (int): cosine annealing steps from w_step to c_step

    Returns:
        float: learning rate at step t
    """
    lr = lr_max
    if t < w_step:
        lr = lr_max * t / w_step
    if w_step <= t < c_step:
        cos_inner = math.pi * (t - w_step) / (c_step - w_step)
        lr = lr_min + 0.5 * (lr_max - lr_min) * (1 + math.cos(cos_inner))
    if t >= c_step:
        lr = lr_min
    return lr
