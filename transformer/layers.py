from typing import Optional

from sublayers import *


def clones(module, N):
    """Produce N identical layers."""
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


class LayerNorm(nn.Module):
    """Construct a layer-norm module"""

    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        x = x.float()
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


class SublayerConnection(nn.Module):
    """
    A residual connection followed by a layer norm.
    """

    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))


class EncoderLayer(nn.Module):
    """
    From Section 3.1 of Attention is All You Need:
    Encoder Layer Consists of Two Sub-layers
    - multi-head self-attention mechanism
    - position-wise fully connected feed-forward network

    We employ a residual connection around each of the two sub-layers,
    followed by layer normalization (Directly from the Paper)

    Residual Connection: SublayerConnection
    Layer Normalization: LayerNorm
    """

    def __init__(self, size: int, self_attn: MultiHeadAttention,
                 feed_forward: PositionwiseFeedForward, dropout: float = 0.1):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        self.size = size

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        return self.sublayer[1](x, self.feed_forward)


class DecoderLayer(nn.Module):
    """
    From Section 3.1 of Attention is All You Need:
    The Decoder consists of N identical layers (DecoderLayer). Each layer has three sub-layers:
    1. A multi-head self-attention mechanism
    2. A multi-head attention over the output of the encoder stack
    3. A position-wise fully connected feed-forward network

    The decoder layers also employ residual connections and layer normalization.
    See the above functions.
    """

    def __init__(self, size: int, self_attn: MultiHeadAttention, src_attn: MultiHeadAttention,
                 feed_forward: PositionwiseFeedForward, dropout: float = 0.1):
        super(DecoderLayer, self).__init__()
        self.size = size
        self.self_attn = self_attn
        self.src_attn = src_attn  # Encoder-Decoder attention
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 3)

    def forward(self, x: torch.Tensor, enc_output: torch.Tensor,
                self_attn_mask: Optional[torch.Tensor] = None,
                src_attn_mask: Optional[torch.Tensor] = None) -> torch.Tensor:

        m = enc_output
        # Self attention
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, self_attn_mask))

        # Encoder-Decoder attention
        x = self.sublayer[1](x, lambda x: self.src_attn(x, m, m, src_attn_mask))

        # Feedforward
        return self.sublayer[2](x, self.feed_forward)


