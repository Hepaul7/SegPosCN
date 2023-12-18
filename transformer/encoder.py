from transformer.layers import *


class Encoder(nn.Module):
    """
    An encoder with self-attention mechanism.

    Why?
    Sequence modelling (RNN and CNNs), suffer from long term dependency problems.
    It cannot extract non-local interactions in a sentence (Qiu et al. 2020). Hence, we adapt the
    Transformer encoder (Vaswani et al. 2017)

    Adapting a similar structure as (Qiu et al. 2020). The encoder consists of
    six multi-head self-attention layers and fully connected layers.
    """

    def __init__(self, layer, N):
        super(Encoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x,  mask):
        """
        Pass the input through each layer in turn.
        """
        mask = mask.byte().unsqueeze(-2)
        for layer in self.layers:
            # print(mask.shape)
            x = layer(x, mask)
        return self.norm(x)


def make_encoder(N=12, d_model=768, d_ff=3072, h=12, dropout=0.1):
    c = copy.deepcopy
    attn = MultiHeadAttention(h=h, d_model=d_model)
    ff = PositionwiseFeedForward(d_model, d_ff, dropout)
    return Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout), N)
