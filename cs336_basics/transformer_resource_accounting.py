class TransformerResourceAccounting:
    """Compute total number of parameters, FLOPs, and memory usage for a Transformer model."""

    def __init__(
        self,
        vocab_size: int,
        context_length: int,
        num_layers: int,
        d_model: int,
        num_heads: int,
        d_ff: int,
    ):
        """Initialize the Transformer resource accounting class.

        Args:
            vocab_size (int): size of the vocabulary.
            context_length (int): sequence length.
            num_layers (int): number of Transformer layers.
            d_model (int): dimension of the model.
            num_heads (int): number of heads in multi-head self-attention.
            d_ff (int): dimension of the feed-forward network.
        """
        self.vocab_size = vocab_size
        self.context_length = context_length
        self.num_layers = num_layers
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.d_k = self.d_v = d_model // num_heads

    def get_params(self) -> int:
        """Get total number of parameters in the model."""
        ## Embedding layer
        embedding_layer = self.vocab_size * self.d_model

        ## Transformation block
        # MHSA
        mhsa = 4 * self.d_model * self.d_model
        # FFN
        ffn = 3 * self.d_model * self.d_ff
        # Layer norm
        layer_norm = 2 * self.d_model
        transformation_block = self.num_layers * (mhsa + ffn + layer_norm)

        ## Final layer norm
        final_layer_norm = self.d_model

        ## Output projection
        output_projection = self.d_model * self.vocab_size

        ## Total parameters
        return (
            embedding_layer
            + transformation_block
            + final_layer_norm
            + output_projection
        )

    def get_memory_usage(self) -> int:
        """Get total memory usage in bytes for loading the model, assuming that each parameter
        is represted using single precision floating point."""
        return self.get_params() * 4  # 4 bytes for single precision float

    def get_flops(self) -> tuple[int, int, int, int, int]:
        """Get total number of FLOPs for a single forward pass through the model.
        This calculation only includes matrix multiplies. Given matrix A of shape (m, n)
        and matrix B of shape (n, p), the FLOPs for the matrix multiplication A @ B
        is 2 * m * n * p."""
        ## Transformer blocks
        # MHSA
        mhsa = (
            3 * 2 * self.context_length * self.d_model * self.d_model
        )  # Q, K, V projections
        mhsa += self.num_heads * (
            2 * self.context_length * self.d_k * self.context_length
            + 2 * self.context_length * self.context_length * self.d_v
        )  # Scaled dot-product attention
        mhsa += (
            2 * self.context_length * self.d_model * self.d_model
        )  # Output projection
        # FFN
        ffn = 2 * self.context_length * self.d_model * self.d_ff  # W1
        ffn += 2 * self.context_length * self.d_model * self.d_ff  # W3
        ffn += 2 * self.context_length * self.d_ff * self.d_model  # W2
        # FLOPsPerBlock​(T) = 8 * T * d_model^2 ​+ 4 * T^2 * d_model ​+ 6 * T * d_model * ​d_ff​
        # So overall, FLOPsPerBlock​(T) = O(T^2) + O(T)
        # As T grows large, the T^2 term dominates. So, when T is large,
        # as T increases, the total FLOPs increases quadratically.
        # Similar conclusion can be drawn for d_model, assuming d_ff = 4 * d_model.
        transformer_layers = self.num_layers * (mhsa + ffn)

        ## Ouptut projection
        output_projection = 2 * self.context_length * self.d_model * self.vocab_size

        ## Total FLOPs
        return (
            mhsa,
            ffn,
            transformer_layers,
            output_projection,
            transformer_layers + output_projection,
        )


def output_resource_accounting(
    vocab_size: int,
    context_length: int,
    num_layers: int,
    d_model: int,
    num_heads: int,
    d_ff: int,
    model_name: str,
) -> None:
    """Output resource accounting for GPT-2 XL model.

    Args:
        vocab_size (int): size of the vocabulary.
        context_length (int): sequence length.
        num_layers (int): number of Transformer layers.
        d_model (int): dimension of the model.
        num_heads (int): number of heads in multi-head self-attention.
        d_ff (int): dimension of the feed-forward network.
    """
    transformer = TransformerResourceAccounting(
        vocab_size=vocab_size,
        context_length=context_length,
        num_layers=num_layers,
        d_model=d_model,
        num_heads=num_heads,
        d_ff=d_ff,
    )
    total_params = transformer.get_params()
    total_memory = transformer.get_memory_usage()
    mhsa_flops, ffn_flops, transformer_layers_flops, output_flops, total_flops = (
        transformer.get_flops()
    )

    print(f"Resource Accounting for {model_name}:")
    print(f"  Total Parameters: {total_params / 1e9:.2f} Billion")
    print(f"  Total Memory Usage: {total_memory / (1024**3):.2f} GB")
    print(f"  Total FLOPs per forward pass: {total_flops / 1e12:.2f} Trillion")
    print(f"    - MHSA FLOPs: {mhsa_flops / 1e12:.2f} Trillion")
    print(f"    - FFN FLOPs: {ffn_flops / 1e12:.2f} Trillion")
    print(
        f"    - Transformer Layers FLOPs: {transformer_layers_flops / 1e12:.2f} Trillion"
    )
    print(f"    - Output Projection FLOPs: {output_flops / 1e12:.2f} Trillion")
    print("  Component Percentage of FLOPs:")
    print(f"    - MHSA: {mhsa_flops * num_layers / total_flops * 100:.2f}%")
    print(f"    - FFN: {ffn_flops * num_layers / total_flops * 100:.2f}%")
    print(f"    - Output Projection: {output_flops / total_flops * 100:.2f}%")


if __name__ == "__main__":
    output_resource_accounting(
        vocab_size=50257,
        context_length=16384,
        num_layers=48,
        d_model=1600,
        num_heads=25,
        d_ff=6400,
        model_name="GPT-2 XL - very long context",
    )
    output_resource_accounting(
        vocab_size=50257,
        context_length=2048,
        num_layers=48,
        d_model=1600,
        num_heads=25,
        d_ff=6400,
        model_name="GPT-2 XL - longer context",
    )
    output_resource_accounting(
        vocab_size=50257,
        context_length=1024,
        num_layers=48,
        d_model=1600,
        num_heads=25,
        d_ff=6400,
        model_name="GPT-2 XL",
    )
    output_resource_accounting(
        vocab_size=50257,
        context_length=1024,
        num_layers=36,
        d_model=1280,
        num_heads=20,
        d_ff=6400,
        model_name="GPT-2 large",
    )
    output_resource_accounting(
        vocab_size=50257,
        context_length=1024,
        num_layers=24,
        d_model=1024,
        num_heads=16,
        d_ff=6400,
        model_name="GPT-2 medium",
    )
    output_resource_accounting(
        vocab_size=50257,
        context_length=1024,
        num_layers=12,
        d_model=768,
        num_heads=12,
        d_ff=6400,
        model_name="GPT-2 small",
    )
