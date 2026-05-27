import math
import torch
import torch.nn as nn


class InputEmbeddings(nn.Module):
    def __init__(self, d_model, vocab):
        super().__init__()
        self.d_model = d_model
        self.embeddings = nn.Embedding(vocab, d_model)

    def forward(self, src):
        return self.embeddings(src) * math.sqrt(self.d_model)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, seq_len, dropout):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.d_model = d_model
        self.seq_len = seq_len

        pe = torch.zeros(seq_len, d_model)
        pos = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
        denom = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * denom)
        pe[:, 1::2] = torch.cos(pos * denom)

        pe = pe.unsqueeze(0)

        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + (self.pe[:, :x.shape[1], : ]).requires_grad_(False)
        return self.dropout(x)


class LayerNorm(nn.Module):
    def __init__(self, d_model, eps=10e-6):
        super().__init__()
        self.d_model = d_model
        self.eps = eps
        self.alpha = nn.Parameter(torch.ones(1))
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.alpha * (x - mean) / (std + self.eps) + self.bias

class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        return self.linear2(self.dropout(torch.relu(self.linear1(x))))

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        assert d_model % num_heads == 0

        self.d_k = d_model // num_heads
        self.d_v = d_model // num_heads
        self.wq = nn.Linear(d_model, d_model)
        self.wk = nn.Linear(d_model, d_model)
        self.wv = nn.Linear(d_model, d_model)
        self.wout = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(p=dropout)

    @staticmethod
    def self_attention(query, key, value, mask=None, dropout=None):
        d_k = query.shape[-1]

        attention = (query * key.transpose(-2, 1)) / math.sqrt(d_k)
        if mask is not None:
            attention = attention.masked_fill(mask == 0, -1e9)

        attention = attention.softmax(attention, dim=-1)

        if dropout is not None:
            attention = dropout(attention)

        return (attention @ value), attention

    def forward(self, q, k, v, mask=None):
        query = self.wq(q)
        key = self.wk(k)
        value = self.wv(v)

        query = query.view(query.shape[0], query.shape[1], self.num_heads, self.d_k).transpose(1, 2)
        key = key.view(key.shape[0], key.shape[1], self.d_k).transpose(1, 2)
        value = value.view(value.shape[0], value.shape[1], self.d_v).transpose(1, 2)

        x, self.attention_scores = self.self_attention(query, key, value, mask, self.dropout)

        x = x.transpose(1, 2).contiguous().view(x.shape[0], -1, self.num_heads * self.d_k)
        return self.wout(x)

class ResidualConnection(nn.Module):
    def __init__(self, features, dropout):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm = LayerNorm(features)

    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))

class EncoderLayer(nn.Module):
    def __init__(self, self_attn_block, feed_forward_block, dropout):
        super().__init__()
        self.self_attn_block = self_attn_block
        self.feed_forward = feed_forward_block
        self.residual_connection = nn.ModuleList([ResidualConnection(dropout) for _ in range(2)])

    def forward(self, x, src_mask):
        x = self.residual_connection[0](x, lambda x: self.self_attn_block(x, x, x, src_mask))
        x = self.residual_connection[1](x, self.feed_forward)

        return x

class Encoder(nn.Module):
    def __init__(self, features, num_layers: nn.ModuleList):
        super().__init__()
        self.num_layers = num_layers
        self.norm = nn.LayerNorm(features)

    def forward(self, x, mask):
        for layer in self.num_layers:
            x = layer(x, mask)

        return self.norm(x)



