import torch.nn as nn
from transformer.embedding_layer import *
from transformer.encoder import Encoder
from transformer.decoder import Decoder


class CWSPOSTransformer(nn.Module):
    def __init__(self, bert_model_name: str, encoder: Encoder, decoder: Decoder, pos_vocab_size: int, cws_vocab_size: int, d_model: int):
        super(CWSPOSTransformer, self).__init__()

        self.bert = BertModel.from_pretrained(bert_model_name)

        # Transformer Encoder and Decoder
        self.encoder = encoder
        self.decoder = decoder

        # Output layers for POS tagging and CWS
        self.pos_output_layer = nn.Linear(d_model, pos_vocab_size)
        self.cws_output_layer = nn.Linear(d_model, cws_vocab_size)

    def forward(self, input_ids, attention_mask, decoder_input_ids, decoder_attention_mask):
        # Generate embeddings from BERT
        embeddings = self.bert(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state

        # Pass embeddings through the encoder
        encoder_output = self.encoder(embeddings, attention_mask)

        # Decoder forward pass
        decoder_output = self.decoder(decoder_input_ids, encoder_output, decoder_attention_mask, attention_mask)

        pos_logits = self.pos_output_layer(decoder_output)
        cws_logits = self.cws_output_layer(decoder_output)

        return pos_logits, cws_logits
