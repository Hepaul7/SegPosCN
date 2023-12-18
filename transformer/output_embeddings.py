import torch
import torch.nn as nn
import math


# draft
class OutputEmbedder(torch.nn.Module):
    def __init__(self, num_labels, embedding_dim):
        super(OutputEmbedder, self).__init__()
        self.embedding = torch.nn.Embedding(num_labels, embedding_dim)

    def forward(self, label_ids):
        return self.embedding(label_ids)


class PositionalEncoder(nn.Module):
    """
    Since transformer models contain no recurrence or convolution, we need
    to inject some information about relative or absolute position of the
    sequence (Vaswani et al. 2017)

    Add positional encodings to the input embeddings before the encoder
    layer.

    Implementation of this class is based on section 3.5 of Attention is
    all you need (Vaswani et al. 2017)
    """
    def __init__(self, d_model: int, drop_out: float, max_len: int = 5000):
        """
        :param d_model: dimension of the model, if BERT it would be 768
        :param drop_out: drop out rate
        :param max_len: max length
        """
        super(PositionalEncoder, self).__init__()
        self.dropout = nn.Dropout(p=drop_out)

        pe = torch.zeros(max_len, d_model).float()
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        # Apply sine function to even indices (pos, 2i)
        pe[:, 0::2] = torch.sin(position * div_term)
        # Apply cosine function to odd indices (pos, 2i + 1)
        pe[:, 1::2] = torch.cos(position * div_term)

        # For batch processing
        pe = pe.unsqueeze(0)

        self.register_buffer('pe', pe)

    def forward(self, x: torch.tensor):
        x += self.pe[:, x.size(0), :]
        return self.dropout(x)


def get_output_embeddings(output_tensor: torch.tensor) -> torch.tensor:
    """

    :param output_tensor: size batch size x max_length in batch
    :return: embeddings size batch size x max_length x embedding dimension
    """
    output_embedder = OutputEmbedder(768, 768)
    # p_e = PositionalEncoder(768, drop_out=0.1, max_len=64)
    output_embeddings = output_embedder(output_tensor)
    return output_embeddings

