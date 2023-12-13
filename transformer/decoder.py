from transformer.layers import *


class Decoder(nn.Module):
    """
    The Decoder for a standard Transformer model as described in "Attention Is All You Need".
    """

    def __init__(self, layer: DecoderLayer, N: int):
        super(Decoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x: torch.Tensor, memory: torch.Tensor,
                src_mask: Optional[torch.Tensor], tgt_mask: Optional[torch.Tensor]) -> torch.Tensor:
        """
        Pass the input through each layer in turn.

        :param x: The sequence from the target.
        :param memory: The sequence from the source (output of the encoder).
        :param src_mask The mask for the source sequence.
        :param tgt_mask: The mask for the target sequence.
        :return: torch.Tensor: The output of the decoder.
        """

        for layer in self.layers:
            x = layer(x, memory, tgt_mask, src_mask)

        return self.norm(x)


def make_decoder(N=6, d_model=768, d_ff=2048, h=8, dropout=0.1):
    c = copy.deepcopy
    self_attn = MultiHeadAttention(h=h, d_model=d_model)
    src_attn = MultiHeadAttention(h=h, d_model=d_model)
    ff = PositionwiseFeedForward(d_model, d_ff, dropout)
    layer = DecoderLayer(d_model, c(self_attn), c(src_attn), c(ff), dropout)
    return Decoder(layer, N)

