from transformer_resource_accounting import TransformerResourceAccounting


class AdamWResourceAccounting(TransformerResourceAccounting):
    """Compute total number of parameters, FLOPs, and memory usage for training
    a transformer model with AdamW."""

    def __init__(
        self,
        vocab_size: int,
        context_length: int,
        num_layers: int,
        d_model: int,
        num_heads: int,
    ):
        """Initialize the Transformer resource accounting class.

        Args:
            vocab_size (int): size of the vocabulary.
            context_length (int): sequence length.
            num_layers (int): number of Transformer layers.
            d_model (int): dimension of the model.
            num_heads (int): number of heads in multi-head self-attention.
        """
        self.vocab_size = vocab_size
        self.context_length = context_length
        self.num_layers = num_layers
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = 4 * d_model
        self.d_k = self.d_v = d_model // num_heads
        super().__init__(
            vocab_size,
            context_length,
            num_layers,
            d_model,
            num_heads,
            self.d_ff,
        )

    def get_params_momery(self) -> float:
        """Get total memory usage in bytes for storing model parameters,
        assuming each parameter is represented using single precision floating point.
        Assume B = batch_size, V = vocab_size, T = context_length,
        L = num_layers, D = d_model, H = num_heads, d_ff = 4 * D,
        memory of parameters = [2VD + (16D^2 + 2D)L + D] * 4 bytes."""
        return self.get_params() * 4 / (1024**3)  # 4 bytes for single precision float

    def get_gradients_memory(self) -> float:
        """Get total memory usage in bytes for storing gradients during training,
        assuming each gradient is represented using single precision floating point.
        Memory of gradients = Memory of parameters."""
        return self.get_params_momery()

    def get_optimizer_state_memory(self) -> float:
        """Get total memory usage in bytes for storing optimizer state during training,
        assuming each state value is represented using single precision floating point.

        optimizer.state_dict() returns a plain Python dict with two main keys:
            "state" : per-parameter optimizer state (e.g., Adam moments,
                        your t counter, etc.)
            "param_groups" : hyperparameters and parameter lists.
        Example :
        {
            'state': {
                0: {
                    'step': 1,
                    'exp_avg':   tensor(...same shape as param 0...),
                    'exp_avg_sq':tensor(...),
                },
                1: {
                    'step': 1,
                    'exp_avg':   tensor(...same shape as param 1...),
                    'exp_avg_sq':tensor(...),
                },
            },
            'param_groups': [
                {
                    'lr': 0.001,
                    'betas': (0.9, 0.999),
                    'eps': 1e-08,
                    'weight_decay': 0.01,
                    'amsgrad': False,
                    'params': [0, 1],
                }
            ]
        }
        """
        # Each parameter has two moment estimates in AdamW
        # that has the same size as the parameter.
        return 2 * self.get_params_momery()

    def get_total_param_related_memory(self) -> float:
        """Get total memory usage in bytes for storing model parameters,
        gradients, and optimizer state during training,
        assuming each value is represented using single precision floating point.
        Total parameter-related memory = Memory of parameters + Memory of gradients
        + Memory of optimizer state.
        """
        return (
            self.get_params_momery()
            + self.get_gradients_memory()
            + self.get_optimizer_state_memory()
        )

    def get_activations_memory(self, batch_size: int) -> float:
        """Get total memory usage in bytes for storing activations during training,
        assuming each activation is represented using single precision floating point.
        Assume B = batch_size, V = vocab_size, T = context_length,
        L = num_layers, D = d_model, H = num_heads, d_ff = 4 * D,
        memory of activations = [(24BTD + 2HBT^2)L + BTD + 2BTV + BT] * 4 bytes."""
        ## Transfomer blocks
        # 2 RMSNorms
        rmsnorm = 2 * batch_size * self.context_length * self.d_model

        # MHSA
        mhsa = 3 * batch_size * self.context_length * self.d_model  # Q, K, V
        mhsa += (
            self.num_heads * batch_size * self.context_length**2
        )  # Scaled dot-product attention
        mhsa += (
            self.num_heads * batch_size * self.context_length**2
        )  # Attention weights
        mhsa += batch_size * self.context_length * self.d_model  # Attention output
        mhsa += batch_size * self.context_length * self.d_model  # Output projection

        # FFN
        ffn = 4 * batch_size * self.context_length * self.d_model  # W1X
        ffn += 4 * batch_size * self.context_length * self.d_model  # W3X
        ffn += 4 * batch_size * self.context_length * self.d_model  # Swich activation
        ffn += 4 * batch_size * self.context_length * self.d_model  # GLU
        ffn += batch_size * self.context_length * self.d_model  # W2X

        transformer_blocks = self.num_layers * (rmsnorm + mhsa + ffn)

        ## Final RMSNorm
        final_rmsnorm = batch_size * self.context_length * self.d_model

        ## Output embedding
        output_embedding = batch_size * self.context_length * self.vocab_size

        ## Cross-entropy loss
        ce_loss = batch_size * self.context_length * self.vocab_size  # Softmax
        ce_loss += batch_size * self.context_length  # Output probabilities

        return (
            (transformer_blocks + final_rmsnorm + output_embedding + ce_loss)
            * 4
            / (1024**3)
        )

    def get_adamw_step_flops(self) -> int:
        """Get total number of FLOPs for a single step of AdamW."""
        # For each parameter, AdamW performs approximately 14 FLOPs per update
        # 3 FLOPs (2 multiplies, 1 add) for computing biased first moment estimate
        # 4 FLOPs (3 multiplies, 1 add) for computing biased second moment estimate
        # 7 FLOPs (2 multiplies, 1 sqrt, 1 division, 2 adds, 1 subtract) for
        # bias correction and parameter update
        return 14 * self.get_params()

    def get_training_time(
        self, mfu: float, peak_flops: float, batch_size: int, num_steps: int
    ) -> float:
        """Get total training time in days for training the model
        for a given number of steps with AdamW optimizer.

        Args:
            mfu (float): machine flops utilization (between 0 and 1).
            peak_flops (float): theoretical peak FLOPs per second of the machine.
            batch_size (int): batch size.
            num_steps (int): number of training steps.

        Returns:
            float: total training time in days.
        """
        # Use the standard Kaplan/Hoffmann approximation:
        # For a dense Transformer LM, total training FLOPs can be approximated as
        # 6 * number of parameters * number of tokens, which came from the fact that
        # forward FLOPs per token is approximately 2 * number of parameters
        # and backward pass requires approximately twice the FLOPs of the forward pass.
        num_tokens = batch_size * self.context_length * num_steps
        total_flops = 6 * self.get_params() * num_tokens
        return total_flops / (mfu * peak_flops * 86400)  # 86400 seconds in a day


if __name__ == "__main__":
    # GPT2-XL
    VOCAB_SIZE = 50257
    CONTEXT_LENGTH = 1024
    NUM_LAYERS = 48
    D_MODEL = 1600
    NUM_HEADS = 25

    resource_accounting = AdamWResourceAccounting(
        vocab_size=VOCAB_SIZE,
        context_length=CONTEXT_LENGTH,
        num_layers=NUM_LAYERS,
        d_model=D_MODEL,
        num_heads=NUM_HEADS,
    )

    print(
        f"""Total parameter-related memory (GB):
        {resource_accounting.get_total_param_related_memory():.2f}"""
    )
    print(
        f"""Total peak memory that AdamW requires for batch size 2 (GB):
        {resource_accounting.get_activations_memory(batch_size=2) +
         resource_accounting.get_total_param_related_memory():.2f}"""
    )
    print(
        "Estimated training time for 400K steps and batch size of 1024 "
        "on NVIDIA A100 GPU that has theoretical peak FLOP throughput of 19.5 "
        f"""teraFLOP/s for float32 operations, assuming 50% MFU (days):
        {resource_accounting.get_training_time(
            mfu=0.5,
            peak_flops=19.5*1e12,
            batch_size=1024,
            num_steps=400_000,
        ):.2f}"""
    )
