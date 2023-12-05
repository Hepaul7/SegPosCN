import torch.nn as nn
from transformers import BertModel, BertTokenizer, BertConfig, EncoderDecoderModel

BERT_MODEL = 'bert-base-chinese'


class Encoder(nn.Module):
    """
    An encoder with self-attention mechanism.

    Why?
    Sequence modelling (RNN and CNNs), suffer from long term dependency problems.
    It cannot extract non-local interactions in a sentence (Qiu et al. 2020). Hence, we adapt the
    Transformer encoder (Vaswani et al. 2017)

    The following are based on (Vaswani et al. 2017) section 3.2.1
    Scaled Dot Product Attention
    We are given sequence of vectors H \in \mathbb{R}^{(T+1) \times d_{model}}. A single
    head, self-attention projects H into:
        - Q: The Query Matrix \in \mathbb{R}^{(T+1) \times d_{k}}
        - K: The Key Matrix \in \mathbb{R}^{(T+1) \times d_{k}}
        - V: The Value Matrix \mathbb{R}^{(T+1) \times d_{v}}

    Attention(Q, K, V) = softmax(\frac{QK^T}{\sqrt{d_k}})V

    Adapting a similar structure as (Qiu et al. 2020). The encoder consists of
    six multi-head self-attention layers and fully connected layers.
    """
    def __init__(self, bert_model_name: str = BERT_MODEL, num_layers: int = 6):
        """
        FOR NOW, I will just use a pre-trained BERT transformer to get things to work.
        :param bert_model_name: Model name for BERT model to be used
        :param num_layers: Number of Encoder Layers
        """
        super(Encoder, self).__init__()
        config = BertConfig.from_pretrained(bert_model_name, num_hidden_layers=num_layers)
        self.bert = BertModel.from_pretrained(bert_model_name, config=config)

    def forward(self, attention_mask, embeddings):
        return self.bert.encoder(embeddings, attention_mask=attention_mask)

