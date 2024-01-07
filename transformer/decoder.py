from transformer.layers import *

CLS = '<s>'
def get_subsequent_mask(seq):
    ''' For masking out the subsequent info. '''
    len_s = seq.size(1)
    subsequent_mask = torch.tril(torch.ones((len_s, len_s), dtype=torch.uint8))
    subsequent_mask = subsequent_mask.bool()
    return subsequent_mask.unsqueeze(0)


class Decoder(nn.Module):
    """
    The Decoder for a standard Transformer model as described in "Attention Is All You Need".
    """

    def __init__(self, layer: DecoderLayer, N: int):
        # print(N)
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
        # assert src_mask.shape == tgt_mask.shape if src_mask is not None and tgt_mask is not None else True
        # print(f'output shape: {x.shape}, encoder_output shape: {memory.shape} ')
        # print(x.shape, memory.shape, src_mask.shape, tgt_mask.shape)
        # print(f'Subsequent Mask: {get_subsequent_mask(x).shape} \n')
        # print(f'Output Mask: {tgt_mask.shape} \n')
        tgt_mask = tgt_mask.byte().unsqueeze(-2) & get_subsequent_mask(x)
        # print(f'With Subsequent: {tgt_mask.shape} \n')
        # print(tgt_mask)
        # print(tgt_mask.shape)
        # print(f'input mask: {src_mask.shape} \n')
        src_mask = src_mask.byte().unsqueeze(-2)
        # print(f'input mask: {src_mask.shape} \n')
        # print(tgt_mask)
        # print(get_subsequent_mask(x))
        for layer in self.layers:
            # print(f'x shape {x.shape}, memory shape {memory.shape}')
            x = layer(x, memory, tgt_mask, src_mask)
        return self.norm(x)


def make_decoder(N=3, d_model=768, d_ff=1024, h=8, dropout=0.1):
    c = copy.deepcopy
    self_attn = MultiHeadAttention(h=h, d_model=d_model)
    src_attn = MultiHeadAttention(h=h, d_model=d_model)
    ff = PositionwiseFeedForward(d_model, d_ff, dropout)
    layer = DecoderLayer(d_model, c(self_attn), c(src_attn), c(ff), dropout)
    return Decoder(layer, N)

