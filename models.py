import torch
import torch.nn as nn
from transformer.embeddings import *
from transformer.encoder import Encoder
from transformer.decoder import Decoder
from transformer.output_embeddings import OutputEmbedder, PositionalEncoder


class CWSPOSTransformer(nn.Module):
    """
    # output_size = 33 (tags) x 4 (B, M, E, S)
    input corresponds to sentences
    output corresponds to 33 tags x BMES
    Basically, the diagram in Attention is all you need
    """
    def __init__(self, encoder: Encoder, decoder: Decoder, output_size: int, d_model: int,
                 output_embedder: OutputEmbedder, positional_encoder: PositionalEncoder):
        super(CWSPOSTransformer, self).__init__()

        self.tgt_embedder = output_embedder
        self.position = positional_encoder
        # Transformer Encoder and Decoder
        self.encoder = encoder
        self.decoder = decoder
        self.d_model = d_model
        self.output_size = output_size

        # Output projection layer -> 33 x 4 + 3 (we dont need beam search, output length known)
        self.output = nn.Linear(d_model, output_size)

    def forward(self, input_embeddings, attention_mask, output_tensor, decoder_attention_mask):
        """

        :param input_embeddings: inputs
        :param attention_mask: attention_mask for inputs
        :param output_tensor: corresponds to target symbols
        :param decoder_attention_mask: attention for decoder
        :return:
        """
        # embeddings for target
        assert type(output_tensor) == torch.Tensor
        # print('decoder shape', decoder_attention_mask.shape)
        decoder_attention_mask = decoder_attention_mask.clone()
        output_embeddings = self.tgt_embedder(output_tensor)
        output_embeddings = self.position(output_embeddings)
        # Pass embeddings through the encoder
        encoder_output = self.encoder(input_embeddings, attention_mask.float())
        # subsequent mask applied in decoder layer
        decoder_output = self.decoder(output_embeddings, encoder_output, attention_mask, decoder_attention_mask.float())
        output = self.output(decoder_output)

        return output


class CWSPOSCRF(nn.Module):
    """
    Transformer Encoder, CRF Decoder
    # output_size = 33 (tags) x 4 (B, M, E, S)
    input corresponds to sentences
    output corresponds to 33 tags x BMES
    Basically, the diagram in Attention is all you need
    """
    def __init__(self, encoder: Encoder, decoder: Decoder, output_size: int, d_model: int,
                 output_embedder: OutputEmbedder, positional_encoder: PositionalEncoder):
        super(CWSPOSCRF, self).__init__()

        self.tgt_embedder = output_embedder
        self.position = positional_encoder
        # Transformer Encoder and Decoder
        self.encoder = encoder
        self.decoder = decoder
        self.d_model = d_model
        self.output_size = output_size

        # Output projection layer -> 33 x 4 + 3 (we dont need beam search, output length known)
        self.output = nn.Linear(d_model, output_size)

    def forward(self, input_embeddings, attention_mask, output_tensor, decoder_attention_mask):
        """

        :param input_embeddings: inputs
        :param attention_mask: attention_mask for inputs
        :param output_tensor: corresponds to target symbols
        :param decoder_attention_mask: attention for decoder
        :return:
        """
        # embeddings for target
        assert type(output_tensor) == torch.Tensor
        # print('decoder shape', decoder_attention_mask.shape)
        decoder_attention_mask = decoder_attention_mask.clone()
        output_embeddings = self.tgt_embedder(output_tensor)
        output_embeddings = self.position(output_embeddings)
        # Pass embeddings through the encoder
        encoder_output = self.encoder(input_embeddings, attention_mask.float())
        # subsequent mask applied in decoder layer
        decoder_output = self.decoder(output_embeddings, encoder_output, attention_mask, decoder_attention_mask.float())
        output = self.output(decoder_output)

        return output
