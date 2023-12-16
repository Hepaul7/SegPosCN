import torch.nn as nn
from transformer.embedding_layer import *
from transformer.encoder import Encoder
from transformer.decoder import Decoder


class CWSPOSTransformer(nn.Module):
    """
    # output_size = 33 (tags) x 4 (B, M, E, S)
    input corresponds to sentences
    output corresponds to 33 tags x BMES
    Basically, the diagram in Attention is all you need
    """
    def __init__(self, bert_model_name: str, encoder: Encoder, decoder: Decoder, output_size: int, d_model: int):
        super(CWSPOSTransformer, self).__init__()

        self.bert = BertModel.from_pretrained(bert_model_name)

        # Transformer Encoder and Decoder
        self.encoder = encoder
        self.decoder = decoder

        # Output layer -> 33 x 4
        self.output = nn.Linear(d_model, output_size)

    def forward(self, input_ids, attention_mask, decoder_input_ids, decoder_attention_mask):
        """

        :param input_ids: inputs
        :param attention_mask: attention_mask for inputs
        :param decoder_input_ids: corresponds to target symbols
        :param decoder_attention_mask: attention for decoder
        :return:
        """
        # Generate embeddings from BERT
        model = load_model('bert-base-chinese')
        input_embeddings = get_bert_embeddings(model, input_ids, attention_mask)
        output_embeddings = get_bert_embeddings(model, decoder_input_ids, decoder_attention_mask)

        # Pass embeddings through the encoder
        encoder_output = self.encoder(input_embeddings, attention_mask)
        # pass encoder output into decoder, along with decoder input
        decoder_output = self.decoder(decoder_input_ids, encoder_output, decoder_attention_mask, attention_mask)

        output = self.output(decoder_output)

        return output
