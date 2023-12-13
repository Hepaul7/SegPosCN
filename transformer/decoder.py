from layers import *


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
