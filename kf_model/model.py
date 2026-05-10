import torch
import torch.nn as nn
import math

import torch.utils.checkpoint as checkpoint

from transformers.modeling_utils import PreTrainedModel
from transformers.modeling_outputs import CausalLMOutput, BaseModelOutput
from kf_model.config import KiteFishConfig


class KiteFishAttention(nn.Module):

    def __init__(self, config: KiteFishConfig):
        super().__init__()

        self.num_heads = config.num_attention_heads
        self.head_dim = config.hidden_size // config.num_attention_heads

        self.qkv = nn.Linear(config.hidden_size, config.hidden_size * 3)
        self.out = nn.Linear(config.hidden_size, config.hidden_size)

    def forward(self, x, attention_mask=None):
        B, T, C = x.shape

        qkv = self.qkv(x)
        qkv = qkv.view(B, T, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)

        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)

        # ← REMOVE the causal mask for embedding model
        # mask = torch.tril(torch.ones(T, T, device=x.device))
        # attn = attn.masked_fill(mask == 0, float("-inf"))

        # Use padding mask from attention_mask if provided
        if attention_mask is not None:
            # attention_mask: (B, T) → (B, 1, 1, T)
            pad_mask = (1.0 - attention_mask.float()).unsqueeze(1).unsqueeze(2) * -1e9
            attn = attn + pad_mask

        attn = torch.softmax(attn, dim=-1)
        out = attn @ v
        out = out.transpose(1, 2).contiguous().view(B, T, C)

        return self.out(out)


class KiteFishMLP(nn.Module):

    def __init__(self, config: KiteFishConfig):
        super().__init__()

        self.fc1 = nn.Linear(config.hidden_size, config.intermediate_size)
        self.fc2 = nn.Linear(config.intermediate_size, config.hidden_size)

        self.act = nn.GELU()

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))


class KitefishBlock(nn.Module):

    def __init__(self, config):
        super().__init__()

        self.ln1 = nn.LayerNorm(config.hidden_size)
        self.ln2 = nn.LayerNorm(config.hidden_size)

        self.attn = KiteFishAttention(config)
        self.mlp = KiteFishMLP(config)
        self.gradient_checkpointing = False

    def forward(self, x, attention_mask=None):
        def forward_fn(x):
            x = x + self.attn(self.ln1(x), attention_mask=attention_mask)
            x = x + self.mlp(self.ln2(x))
            return x

        if self.gradient_checkpointing and self.training:
            x = checkpoint.checkpoint(forward_fn, x, use_reentrant=False)
        else:
            x = forward_fn(x)
        return x


class KiteFishModel(PreTrainedModel):

    config_class = KiteFishConfig
    base_model_prefix = "kitefish"
    supports_gradient_checkpointing = True

    def __init__(self, config: KiteFishConfig):
        super().__init__(config)

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.embed_positions = nn.Embedding(
            config.max_position_embeddings,
            config.hidden_size,
        )

        self.layers = nn.ModuleList(
            [KitefishBlock(config) for _ in range(config.num_hidden_layers)]
        )

        self.final_layernorm = nn.LayerNorm(config.hidden_size)

        self.post_init()

    def forward(
        self,
        input_ids,
        attention_mask=None,
        return_dict=None,          # ST/HF always passes True
        output_hidden_states=None, # absorbed
        **kwargs,
    ):
        B, T = input_ids.shape
        pos = torch.arange(0, T, device=input_ids.device)
        x = self.embed_tokens(input_ids) + self.embed_positions(pos)

        for layer in self.layers:
            x = layer(x, attention_mask=attention_mask)

        x = self.final_layernorm(x)

        # ST indexes the output as a dict with "last_hidden_state" / "token_embeddings".
        # Return BaseModelOutput so both HF and sentence-transformers work correctly.
        if return_dict is False:
            return x
        return BaseModelOutput(last_hidden_state=x)


class KiteFishForCausalLM(PreTrainedModel):

    config_class = KiteFishConfig
    base_model_prefix = "model"       # checkpoint keys are model.layers.* etc.
    supports_gradient_checkpointing = True
    # When ST wraps the backbone directly, lm_head keys won't be present — that's fine.
    _keys_to_ignore_on_load_unexpected = [r"lm_head\..*"]

    def __init__(self, config):
        super().__init__(config)

        self.model = KiteFishModel(config)

        self.lm_head = nn.Linear(
            config.hidden_size,
            config.vocab_size,
            bias=False,
        )

        self.post_init()

    def forward(self, input_ids, attention_mask=None, labels=None, return_dict=None, **kwargs):

        out = self.model(input_ids, attention_mask=attention_mask, return_dict=True)
        hidden_states = out.last_hidden_state

        logits = self.lm_head(hidden_states)

        loss = None

        if labels is not None:

            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()

            loss_fct = nn.CrossEntropyLoss()

            loss = loss_fct(
                shift_logits.view(-1, self.config.vocab_size),
                shift_labels.view(-1),
            )

        return CausalLMOutput(
            loss=loss,
            logits=logits,
        )

    def encode(self, input_ids, attention_mask):
        # Get raw hidden states directly from the backbone
        out = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
        )
        last_hidden = out.last_hidden_state  # (B, T, hidden_size)

        # Mean pooling over non-padding tokens
        mask = attention_mask.unsqueeze(-1).float()  # (B, T, 1)
        summed = (last_hidden * mask).sum(1)          # (B, hidden_size)
        counts = mask.sum(1).clamp(min=1e-9)          # (B, 1) — avoid div by zero
        embedding = summed / counts                    # (B, hidden_size)

        return torch.nn.functional.normalize(embedding, p=2, dim=1)

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        self.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs)

    def _set_gradient_checkpointing(self, module, value=False):
        if hasattr(self.model, "_set_gradient_checkpointing"):
            self.model._set_gradient_checkpointing(module, value)