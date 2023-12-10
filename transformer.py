import copy
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def attention(query, key, value, mask=None, dropout=None):
    "Compute 'Scaled Dot Product Attention'"
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
    def __init__(self, d_model: int, h: int = 8, dropout = 0.1):
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
        See citation on paper for details
        """
        # for head axis broadcasting
        if mask is not None:
            mask = mask.unsqueeze(1)

        batch_size = q.size(0)
        len_q = q.size(1)   # = 1 for MCCWS
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


class LayerNorm(nn.Module):
    """Construct a layernorm module (See citation for details)."""

    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        x = x.float()
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


class PositionwiseFeedForward(nn.Module):
    """Position-wise Feed-Forward Networks, See Section 3.3 of AttnAllYouNeed"""

    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))


def clones(module, N):
    """Produce N identical layers."""
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


class SublayerConnection(nn.Module):
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    """

    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        "Apply residual connection to any sublayer with the same size."
        return x + self.dropout(sublayer(self.norm(x)))


class EncoderLayer(nn.Module):
    "Encoder is made up of self-attn and feed forward (defined below)"

    def __init__(self, size, self_attn, feed_forward, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        self.size = size

    def forward(self, x, mask):
        "Follow Figure 1 (left) for connections."
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        return self.sublayer[1](x, self.feed_forward)


class Encoder(nn.Module):
    "Core encoder is a stack of N layers"

    def __init__(self, layer, N):
        super(Encoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, mask):
        # print(x.size(),mask.size())
        "Pass the input (and mask) through each layer in turn."
        mask = mask.byte().unsqueeze(-2)
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


def make_encoder(N=6, d_model=768, d_ff=2048, h=8, dropout=0.1):
    c = copy.deepcopy
    attn = MultiHeadAttention(h=h, d_model=d_model)
    ff = PositionwiseFeedForward(d_model, d_ff, dropout)
    return Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout), N)