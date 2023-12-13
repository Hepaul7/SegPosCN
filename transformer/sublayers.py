import copy
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def attention(query, key, value, mask=None, dropout=None):
    """Compute 'Scaled Dot Product Attention' """
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        # print(scores.size(),mask.size())
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = F.softmax(scores, dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn


class MultiHeadAttention(nn.Module):
    """
    Implementation based on Attention is All You Need section 3.2.2
    Implementation based on a Pytorch Implementation of Paper and MCCWS
    See Citation for details
    d_k: dimension of Q and K
    d_v: dimension of Values
    h: head, = 8 by paper
    d_model: dimension of the model
    """

    def __init__(self, d_model: int, h: int = 8, dropout: float = 0.1):
        super(MultiHeadAttention, self).__init__()
        self.h = h
        # d_k = d_model / h = 64 (on paper)
        self.d_k = d_model // h
        self.d_v = d_model // h

        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.w_k = nn.Linear(d_model, d_model, bias=False)
        self.w_v = nn.Linear(d_model, d_model, bias=False)
        self.w_o = nn.Linear(d_model, d_model, bias=False)

        self.attention = None
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)

    def forward(self, q, k, v, mask=None):
        """
        Forward, based on paper
        """
        # for head axis broadcasting
        if mask is not None:
            mask = mask.unsqueeze(1)

        batch_size = q.size(0)
        len_q = q.size(1)
        len_k = k.size(1)
        len_v = v.size(1)
        # head_i = Attention(QW_q, KW_k, VW_v)
        # Note to self: nn.Linear.view()
        # Returns a new tensor with the same data as the self tensor but of a different shape

        # also need to transpose into b x n x lq x dv
        q = self.w_q(q).view(batch_size, len_q, self.h, self.d_k).transpose(1, 2)
        k = self.w_k(k).view(batch_size, len_k, self.h, self.d_k).transpose(1, 2)
        v = self.w_v(v).view(batch_size, len_v, self.h, self.d_k).transpose(1, 2)

        # Apply Attention to each to get head_i
        q, self.attention = attention(q, k, v, mask=mask)

        # Concat Each one to get MultiHead(Q, K, V)
        q = q.transpose(1, 2).contiguous().view(batch_size, -1, self.h * self.d_k)
        return self.w_o(q)


class PositionwiseFeedForward(nn.Module):
    """
    See section 3.3 in Attention is All You Need Paper
    FFN(x) = max(0, xW1 + b1 )W2 + b2

    d_model: dimension of model
    d_ff: dimension of feed-forward layers
    """
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(nn.functional.relu(self.w_1(x))))

