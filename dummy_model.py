import torch
import math
from torch import optim, nn

class InputEmbedding(nn.Module):
    def __init__(self, vocab_size, d_model):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)

    def forward(self, x):
        return self.embedding(x) * math.sqrt(self.d_model)

class PositionalEncoding(nn.Module):

    def __init__(self, d_model, seq_len, dropout):
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(seq_len, d_model)
        pos = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
        denominator = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(pos * denominator)
        pe[:, 1::2] = torch.cos(pos * denominator)

        pe = pe.unsqueeze(0)

        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + (self.pe[:, :x.shape[1], :]).requires_grad_(False)
        return self.dropout(x)

class LayerNorm(nn.Module):
    def __init__(self, features, eps=10e-6):
        super().__init__()
        self.eps = eps
        self.alpha = nn.Parameter(torch.ones(features))
        self.bias = nn.Parameter(torch.zeros(features))


    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.alpha * (x - mean) / math.sqrt(std ** 2 + self.eps) + self.bias

class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout):
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.dropout = nn.Dropout(dropout)
        self.linear_1 = nn.Linear(d_model, d_ff)
        self.linear_2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        return self.linear_2(self.dropout(torch.relu(self.linear_1(x))))

class MultiheadAttention(nn.Module):

     def __init__(self, d_model, h, dropout):
         super().__init__()
         self.h = h
         self.d_model = d_model
         assert d_model % h == 0, "d_model must be divisible by h"
         self.dropout = nn.Dropout(dropout)

         self.d_k = d_v = d_model // h
         self.w_q = nn.Linear(d_model, d_model)
         self.w_k = nn.Linear(d_model, d_model)
         self.w_v = nn.Linear(d_model, d_model)

         self.w_o = nn.Linear(d_model, d_model)

     def forward(self, query, key, value, mask):
         q = self.w_q(query)
         k = self.w_k(key)
         v = self.w_v(value)

         q = q.view(q.shape[0], q.shape[1], self.h, self.d_k).transpose(1, 2)
         k = k.view(k.shape[0], k.shape[1], self.h, self.d_k).transpose(1, 2)
         v = v.view(v.shape[0], v.shape[1], self.h, self.d_k).transpose(1, 2)

