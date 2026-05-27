import torch
from torch import nn
import torch.nn.functional as F


class InputEmbedding(nn.Module):
    def __init__(self, vocab_size, embed_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)

    def forward(self, x):
        return self.embedding(x)


class RMSNorm(nn.Module):
    def __init__(self, embed_dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(embed_dim))

    def forward(self, x):
        rms = torch.sqrt(x.pow(2).mean(dim=1, keepdim=True) + self.eps)
        x_norm = x * rms * self.weight
        return x_norm


class RoPE(nn.Module):
    def __init__(self, embed_dim, seq_len):
        super().__init__()
        self.seq_len = seq_len
        N = 10000
        inv_freq = 1. / (N ** (torch.arange(0, embed_dim, 2).float() / embed_dim))
        inv_freq = torch.cat((inv_freq, inv_freq), dim=-1)
        pos = torch.arange(seq_len).float()
        sinusoid_inp = torch.outer(pos, inv_freq)
        self.register_buffer('sin', sinusoid_inp.sin())
        self.register_buffer('cos', sinusoid_inp.cos())

    def rotate_half(self, x):
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat((-x2, x1), dim=-1)

    def apply_rotary_pos_emb(self, x, cos, sin):
        return (x * cos) + (self.rotate_half(x) * sin)

    def forward(self, x):
        cos = self.cos[:self.seq_len].view(1, self.seq_len, 1, -1)
        sin = self.sin[:self.seq_len].view(1, self.seq_len, 1, -1)
        return self.apply_rotary_pos_emb(x, cos, sin)


class GroupedQueryAttention(nn.Module):
    def __init__(self, d_model, num_heads, num_groups, seq_len):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.num_groups = num_groups
        self.group_size = num_heads // num_groups
        self.head_dim = d_model // num_heads
        self.rope_embed = RoPE(d_model, seq_len)

        self.w_q = nn.Linear(d_model, num_heads * self.head_dim)
        self.w_k = nn.Linear(d_model, num_heads * self.head_dim)
        self.w_v = nn.Linear(d_model, num_heads * self.head_dim)
        self.w_o = nn.Linear(num_heads * self.head_dim, d_model)

    @staticmethod
    def compute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
        freqs= 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
        t = torch.arange(end, device=freqs.device)
        freqs_cis = torch.polar(torch.ones_like(torch.outer(t, freqs)), torch.outer(t, freqs))
        return freqs_cis

    def _repeat_kv(self, x: torch.Tensor, rep: int) -> torch.Tensor:
        batch, seq_len, n_kv_heads, head_dim = x.shape
        if rep == 1:
            return x
        return (
            x[:, :, :, None, :]
            .expand(batch, seq_len, n_kv_heads, rep, head_dim)
            .reshape(batch, seq_len, n_kv_heads * rep, head_dim)
        )

    def forward(self, x, mask):
        batch_size, seq_len = x.size(0), x.size(1)

        q = self.w_q(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        k = self.w_k(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        v = self.w_v(x).view(batch_size, seq_len, self.num_heads, self.head_dim)

        q = self.rope_embed(q)
        k = self.rope_embed(k)

        k = self._repeat_kv(k, self.group_size)
        v = self._repeat_kv(v, self.group_size)

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("-inf"))

        attn_weights = F.softmax(scores, dim=-1)
        output = torch.matmul(attn_weights, v)

        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        return self.out_proj(output)


class SkipConnection(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()

    def forward(self, x1, x2):
        return x1 + x2


class FeedForwardNetwork(nn.Module):
    def __init__(self, embed_dim, hidden_dim, ):
        super().__init__()
        self.ffl1 = nn.Linear(embed_dim, hidden_dim, bias=False)
        self.ffl2 = nn.Linear(embed_dim, hidden_dim, bias=False)
        self.ffl3 = nn.Linear(hidden_dim, embed_dim, bias=False)

    def forward(self, x):
        ffl1_out = self.ffl1(x)
        ffl2_out = self.ffl2(x)
        x = nn.SiLU(ffl1_out) + ffl2_out
        return self.ffl3(x)


class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()


class LLAMA3Model(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_heads):
        super().__init__()