import torch.nn as nn
from transformer.embeddings import *
from transformer.encoder import Encoder
from transformer.decoder import Decoder


class CWSPOSTransformer(nn.Module):
    """
    # output_size = 33 (tags) x 4 (B, M, E, S)
    input corresponds to sentences
    output corresponds to 33 tags x BMES
    Basically, the diagram in Attention is all you need
    """
    def __init__(self, encoder: Encoder, decoder: Decoder, output_size: int, d_model: int):
        super(CWSPOSTransformer, self).__init__()
        # note d_model refers to the BERT dimension, TODO: make this clear

        # self.bert = BertModel.from_pretrained(bert_model_name)

        # Transformer Encoder and Decoder
        self.encoder = encoder
        self.decoder = decoder

        # Output layer -> 33 x 4 + 3 (OUTPUTSIZE)
        self.output = nn.Linear(d_model, output_size)

    def forward(self, input_embeddings, attention_mask, output_embeddings, decoder_attention_mask):
        """

        :param input_embeddings: inputs
        :param attention_mask: attention_mask for inputs
        :param output_embeddings: corresponds to target symbols
        :param decoder_attention_mask: attention for decoder
        :return:
        """
        # Generate embeddings from BERT
        # model = load_model('bert-base-chinese')
        # input_embeddings = get_bert_embeddings(model, input_ids, attention_mask)
        # output_embeddings = get_bert_embeddings(model, decoder_input_ids, decoder_attention_mask)

        # Pass embeddings through the encoder
        # print(f'input embeddings shape: {input_embeddings.shape}')
        # print(f'attention mask shape: {attention_mask.shape}, decoder mask shape {decoder_attention_mask.shape}')
        encoder_output = self.encoder(input_embeddings, attention_mask)
        # print('encoder success!')
        # pass encoder output into decoder, along with decoder input
        # print(f'output embeddings shape: {output_embeddings.shape}, encoder_out shape: {encoder_output.shape}')
        # print(f'input mask shape: {attention_mask.shape}, decoder mask shape: {decoder_attention_mask.shape}')
        decoder_output = self.decoder(output_embeddings, encoder_output, attention_mask, decoder_attention_mask)
        # print('decoder success!')
        output = self.output(decoder_output)

        return output
