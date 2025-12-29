import torch
from torch import nn
from einops import rearrange
import numpy as np


class Linear(nn.Module):
    """Perform a linear transformation y = xW^T.
    Mimics the interface of nn.Linear but without bias.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
    ):
        """Initialize the Linear layer.

        Args:
            in_features (int): dimension of the input features.
            out_features (int): dimension of the output features.
        """
        super().__init__()
        self.weight = nn.Parameter(torch.empty((out_features, in_features)))
        std = np.sqrt(2 / (in_features + out_features))
        nn.init.trunc_normal_(self.weight, mean=0, std=std, a=-3 * std, b=3 * std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the Linear layer.

        Args:
            x (torch.Tensor): input tensor of shape (..., in_features).

        Returns:
            torch.Tensor: output tensor of shape (..., out_features).
        """
        return x @ self.weight.T


class Embedding(nn.Module):
    """Perform an embedding lookup.
    Mimics the interface of nn.Embedding.
    """

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
    ):
        """Initialize the Embedding layer.

        Args:
            num_embeddings (int): size of the vocabulary.
            embedding_dim (int): dimension of the embedding vectors.
        """
        super().__init__()
        self.weight = nn.Parameter(torch.empty((num_embeddings, embedding_dim)))
        nn.init.trunc_normal_(self.weight, mean=0, std=1, a=-3, b=3)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """Lookup embeddings for the given token IDs.

        Args:
            token_ids (torch.Tensor): token ids of shape (batch_size, sequence_length).

        Returns:
            torch.Tensor: embeddings of shape (batch_size, sequence_length, embedding_dim).
        """
        token_ids_long = token_ids.to(torch.long)
        return self.weight[
            token_ids_long
        ]  # Advanced indexing. For each entry in token_ids_long, treat it as a
        # row index into self.weight. PyTorch indexing operator onlly accepts torch.long (int64).


class RMSNorm(nn.Module):
    """RMSNorm layer."""

    def __init__(
        self,
        d_model: int,
        eps: float = 1e-5,
    ):
        """Initialize the RMSNorm layer.

        Args:
            d_model (int): dimension of the model.
            eps (float, optional): epsilon value for numerical stability. Defaults to 1e-5.
        """
        super().__init__()
        self.eps = eps
        self.gain = nn.Parameter(torch.ones((d_model,)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the RMSNorm layer.

        Args:
            x (torch_Tensor): input tensor of shape (batch_size, sequence_length, d_model).

        Returns:
            torch.Tensor: output tensor of shape (batch_size, sequence_length, d_model).
        """
        in_dtype = x.dtype
        x = x.to(torch.float32)  # Prevent overflow when squaring
        x_squared = x**2
        x_mean_squared = torch.mean(x_squared, dim=-1, keepdim=True)
        rms = torch.sqrt(x_mean_squared + self.eps)
        x_normed = x / rms * self.gain
        return x_normed.to(in_dtype)


class PositionWiseFeedForward(nn.Module):
    """Position-wise Feedforward layer, employing SwiGLU activation function.
    FFN(x) = SwiGLU(x, W1, W2, W3) = W2(SiLU(W1x)⊙W3x)
    """

    def __init__(
        self,
        d_model: int,
        d_ff: int,
    ):
        """Initialize the Positionwise_Feedforward layer.

        Args:
            d_model (int): dimension of the model.
            d_ff (int): dimension of the feedforward layer.
        """
        super().__init__()
        self.d_ff = d_ff
        self.weight1 = nn.Parameter(torch.empty((self.d_ff, d_model)))
        self.weight2 = nn.Parameter(torch.empty((d_model, self.d_ff)))
        self.weight3 = nn.Parameter(torch.empty((self.d_ff, d_model)))
        std = np.sqrt(2 / (d_model + self.d_ff))
        nn.init.trunc_normal_(self.weight1, mean=0, std=std, a=-3 * std, b=3 * std)
        nn.init.trunc_normal_(self.weight2, mean=0, std=std, a=-3 * std, b=3 * std)
        nn.init.trunc_normal_(self.weight3, mean=0, std=std, a=-3 * std, b=3 * std)

    def _silu(self, x: torch.Tensor) -> torch.Tensor:
        """SiLU activation function.

        Args:
            x (torch.Tensor): input tensor of any shape.

        Returns:
            torch.Tensor: output tensor of the same size as input.
        """
        return x * torch.sigmoid(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass on the Positionwise_Feedforward layer.

        Args:
            x (torch.Tensor): input tensor of shape (batch_size, sequence_length, d_model).

        Returns:
            torch.Tensor: output tensor of shape (batch_size, sequence_length, d_model).
        """
        z1 = x @ self.weight1.T
        z3 = x @ self.weight3.T
        a1 = self._silu(z1) * z3
        return a1 @ self.weight2.T


class RotaryPositionalEmbedding(nn.Module):
    """RoPE."""

    def __init__(
        self,
        theta: float,
        d_k: int,
        max_seq_len: int,
    ):
        """Construct the RoPE module and create buffers.

        Args:
            theta (float): Θ value for RoPE.
            d_k (int): dimension of the key/query vectors.
            max_seq_len (int): maximum sequence length that can be handled.
        """
        super().__init__()
        freq_seq = torch.arange(0, d_k, 2.0) / d_k
        inv_freq = 1.0 / (theta**freq_seq)
        t = torch.arange(max_seq_len)
        freqs = torch.einsum("i,j->ij", t, inv_freq)  # Outer product of 2 vectors
        self.register_buffer("cos_cached", torch.cos(freqs), persistent=False)
        self.register_buffer("sin_cached", torch.sin(freqs), persistent=False)

    def forward(
        self, x: torch.Tensor, token_positions: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Forward pass of RoPE.

        Args:
            x (torch.Tensor): input tensor of shape (..., seq_len, d_k).
            token_positions (torch.Tensor | None): tensor of shape (..., seq_len)
                specifying the token positions of x along the sequence dimension.

        Returns:
            torch.Tensor: output tensor of shape (..., seq_len, d_k).
        """
        # It can be case where the full sequence is length 2048,
        # but in this forward pass you are feeding only a chunk of it.
        # For example, seq_len = 4, token_positions = [1000, 1001, 1002, 1003]
        if token_positions is not None:
            cos_cached = self.cos_cached[
                token_positions.squeeze()
            ]  # Might not be necessary, just to pass the tests
            sin_cached = self.sin_cached[token_positions.squeeze()]
        else:
            seq_len = x.shape[-2]
            cos_cached = self.cos_cached[
                torch.arange(seq_len, device=self.cos_cached.device)
            ]
            sin_cached = self.sin_cached[
                torch.arange(seq_len, device=self.sin_cached.device)
            ]
            # The default type of torch.arange(int) is torch.int64,
            # which is compatible with buffer indexing.

        x1, x2 = x[..., ::2], x[..., 1::2]  # Split last dim into even and odd parts
        x_rotated = torch.stack([-x2, x1], dim=-1).reshape_as(
            x
        )  # Rotate X11, X12 to -X12, X11
        # Shape of cos_cached, sin_cached: (max_seq_len, d_k/2).
        # Repeat for even and odd to match the shape of x and allow
        # two consecutive features in x share the same cos/sin.
        cos_cached_rep = cos_cached.repeat_interleave(2, dim=1)
        sin_cached_rep = sin_cached.repeat_interleave(2, dim=1)
        return x * cos_cached_rep + x_rotated * sin_cached_rep


def softmax(x: torch.Tensor, dim: int, temperature: float) -> torch.Tensor:
    """Compute the softmax of the input tensor along the specified dimension.

    Args:
        x (torch.Tensor): input tensor.
        dim (int): the dimension along which to compute the softmax.
        temperature (float): temperature parameter for scaling.

    Returns:
        torch.Tensor: output tensor after applying softmax with the same shape as input.
    """
    x_max = torch.max(x, dim=dim, keepdim=True).values
    x_subtract_max = x - x_max
    x_exp = torch.exp(x_subtract_max / temperature)
    x_exp_sum = torch.sum(x_exp, dim=dim, keepdim=True)
    return x_exp / x_exp_sum


class MultiHeadSelfAttention(nn.Module):
    """Compute causal muti-head self-attention with RoPE."""

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        theta: float,
        max_seq_len: int,
    ):
        """Initialize the MultiHeadSelfAttention layer.

        Args:
            d_model (int): dimension of the model.
            num_heads (int): number of heads.
            theta (float): Θ value for RoPE.
            max_seq_len (int): maximum sequence length that can be handled.
        """
        super().__init__()
        assert d_model % num_heads == 0
        self.num_heads = num_heads
        self.d_k = self.d_v = d_model // num_heads
        self.w_q = Linear(d_model, d_model)  # Registered submodules
        self.w_k = Linear(d_model, d_model)
        self.w_v = Linear(d_model, d_model)
        self.w_o = Linear(d_model, d_model)
        self.rope = RotaryPositionalEmbedding(
            theta=theta, d_k=self.d_k, max_seq_len=max_seq_len
        )

    def _scaled_dot_product_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute the scaled dot-product attention.

        Args:
            q (torch.Tensor): query tensor of shape (..., seq_len, d_k).
            k (torch.Tensor): key tensor of shape (..., seq_len, d_k).
            v (torch.Tensor): value tensor of shape (..., seq_len, d_v).
            mask (torch.Tensor | None, optional): boolean mask of shape (seq_len, seq_len)
                indicating which keys the query should attend to. Defaults to None.

        Returns:
            torch.Tensor: scaled dot-product attention output of shape (..., seq_len, d_v).
        """
        d_k = k.shape[-1]
        dot_product = q @ k.transpose(-2, -1)
        if mask is not None:
            dot_product = dot_product.masked_fill(~mask, float("-inf"))
        scaled_dot_product = dot_product / np.sqrt(d_k)
        attention_weights = softmax(scaled_dot_product, dim=-1, temperature=1.0)
        return attention_weights @ v

    def forward(
        self, x: torch.Tensor, token_positions: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Forward pass of the MultiHeadSelfAttention layer.

        Args:
            x (torch.Tensor): input tensor of shape (batch_size, sequence_length, d_model).
            token_positions (torch.Tensor | None): tensor of shape (sequence_length, )
                specifying the token positions of x along the sequence dimension.

        Returns:
            torch.Tensor: output tensor of shape (batch_size, sequence_length, d_model).
        """
        # Create causal mask
        seq_len = x.shape[1]
        causal_mask = torch.tril(
            torch.ones((seq_len, seq_len), dtype=torch.bool, device=x.device)
        )

        # Compute query, key, value projections
        q = self.w_q(x)
        k = self.w_k(x)
        v = self.w_v(x)

        # Head dimension split
        q_heads = rearrange(q, "b s (h d_k) -> b h s d_k", h=self.num_heads)
        k_heads = rearrange(k, "b s (h d_k) -> b h s d_k", h=self.num_heads)
        v_heads = rearrange(v, "b s (h d_v) -> b h s d_v", h=self.num_heads)

        # Apply RoPE to query and key
        # Q means "What am I looking for?", K means "Where is the relevant info?",
        # V means "What is the relevant info?"
        # When apply RoPE to both Q and K, their relative positional information
        # is encoded in the dot product QK^T, via the mathematical properties of RoPE -
        # (Rtqt)^T(Rsks) = qt^TR(t-s)ks.
        q_heads = self.rope(q_heads, token_positions)
        k_heads = self.rope(k_heads, token_positions)

        # Compute scaled dot-product attention for each head
        attention_outputs = self._scaled_dot_product_attention(
            q_heads, k_heads, v_heads, causal_mask
        )

        # Concatenate heads
        concat_attention = rearrange(attention_outputs, "b h s d_v -> b s (h d_v)")

        # Final linear projection
        return self.w_o(concat_attention)


class TransformerBlock(nn.Module):
    """Pre-norm transformer block with multi-head self-attention and position-wise
    feedforward network."""

    def __init__(
        self, d_model: int, num_heads: int, d_ff: int, theta: float, max_seq_len: int
    ):
        """Initialize the transformer block.

        Args:
            d_model (int): dimension of the model.
            num_heads (int): number of heads.
            d_ff (int): dimension of the feedforward layer.
            theta (float): Θ value for RoPE.
            max_seq_len (int): maximum sequence length that can be handled.
        """
        super().__init__()
        self.rmsnorm1 = RMSNorm(d_model)
        self.mhsa = MultiHeadSelfAttention(
            d_model=d_model,
            num_heads=num_heads,
            theta=theta,
            max_seq_len=max_seq_len,
        )
        self.rmsnorm2 = RMSNorm(d_model)
        self.ffn = PositionWiseFeedForward(d_model, d_ff)

    def forward(
        self, x: torch.Tensor, token_positions: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Forward pass of one transformer block.

        Args:
            x (torch.Tensor): input tensor of shape (batch_size, sequence_length, d_model).
            token_positions (torch.Tensor | None, optional): tensor of shape (..., seq_len)
            specifying the positions of x along the sequence dimension. Defaults to None.

        Returns:
            torch.Tensor: output tensor of shape (batch_size, sequence_length, d_model).
        """
        # Multi-head self-attention sublayer
        # x_normed1 = self.rmsnorm1(x)
        # mhsa_output = self.mhsa(x_normed1, token_positions)
        mhsa_output = self.mhsa(x, token_positions)
        x = x + mhsa_output

        # Position-wise feedforward sublayer
        # x_normed2 = self.rmsnorm2(x)
        # ffn_output = self.ffn(x_normed2)
        ffn_output = self.ffn(x)
        x = x + ffn_output

        return x


class TransformerLM(nn.Module):
    """Transformer language model."""

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        theta: float,
        vocab_size: int,
        context_length: int,
        num_layers: int,
        d_ff: int | None = None,
    ):
        """Initialize the Transformer language model.

        Args:
            d_model (int): dimension of the model.
            num_heads (int): number of self-attention heads.
            d_ff (int): dimension of the feedforward layer.
            theta (float): Θ value for RoPE.
            vocab_size (int): size of the vocabulary.
            context_length (int): maximum context length.
            num_layers (int): number of transformer blocks.
        """
        super().__init__()
        self.context_length = context_length
        self.token_embedding = Embedding(vocab_size, d_model)
        if d_ff is None:
            self.d_ff = d_model * 4
        else:
            self.d_ff = d_ff
        self.layers = nn.ModuleList(
            [
                TransformerBlock(
                    d_model=d_model,
                    num_heads=num_heads,
                    d_ff=self.d_ff,
                    theta=theta,
                    max_seq_len=self.context_length,
                )
                for _ in range(num_layers)
            ]
        )
        self.rmsnorm_final = RMSNorm(d_model)
        self.output_projection = Linear(d_model, vocab_size)

    def forward(
        self, x: torch.Tensor, token_positions: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Forward pass of the Transformer language model.

        Args:
            x (torch.Tensor): input tensor of shape (batch_size, sequence_length).
            token_positions (torch.Tensor): tensor of shape (batch_size, sequence_length)
                specifying the token positions of x along the sequence dimension.

        Returns:
            torch.Tensor: output tensor of shape (batch_size, sequence_length, vocab_size).
        """
        # Token embedding
        x = self.token_embedding(x)

        # Transformer blocks
        for layer in self.layers:
            x = layer(x, token_positions)

        # Final RMSNorm
        # x = self.rmsnorm_final(x)

        # Output projection to vocabulary size
        return self.output_projection(x)
