# ============================================================================
# IMPORTS - Bringing in the tools we need
# ============================================================================
import torch                          # PyTorch: the main deep learning library
import torch.nn as nn                 # nn = "neural network" building blocks (layers, etc.)
import math                           # Standard math library (we use it for sqrt)

import torch.utils.checkpoint as checkpoint  # Tool to save GPU memory during training
                                             # (re-computes activations instead of storing them)

# These are HuggingFace Transformers helpers — they make our model compatible
# with the HuggingFace ecosystem (so we can save/load like any HF model)
from transformers.modeling_utils import PreTrainedModel
from transformers.modeling_outputs import CausalLMOutput, BaseModelOutput

# Our own config class — holds all the hyperparameters (hidden size, num layers, etc.)
from kf_model.config import KiteFishConfig


# ============================================================================
# ATTENTION LAYER — The "heart" of a Transformer
# This lets each token "look at" other tokens in the sequence.
# ============================================================================
class KiteFishAttention(nn.Module):

    def __init__(self, config: KiteFishConfig):
        super().__init__()  # Always call parent's __init__ first in PyTorch

        # How many attention heads we have (parallel attention computations)
        self.num_heads = config.num_attention_heads
        # Each head works on a smaller slice of the hidden dimension
        # e.g. hidden_size=768, num_heads=12 → head_dim=64
        self.head_dim = config.hidden_size // config.num_attention_heads

        # ONE linear layer that produces Query, Key, and Value all at once
        # Output is 3x the hidden size (one chunk for Q, one for K, one for V)
        self.qkv = nn.Linear(config.hidden_size, config.hidden_size * 3)
        # Final linear projection after attention is computed
        self.out = nn.Linear(config.hidden_size, config.hidden_size)

    def forward(self, x, attention_mask=None):
        # x has shape (Batch, Time/Tokens, Channels/hidden_size)
        B, T, C = x.shape

        # Step 1: Project input into Q, K, V combined → shape (B, T, 3*C)
        qkv = self.qkv(x)
        # Reshape so we can split into 3 parts and multiple heads
        # New shape: (B, T, 3, num_heads, head_dim)
        qkv = qkv.view(B, T, 3, self.num_heads, self.head_dim)
        # Rearrange dimensions so we can easily grab Q, K, V
        # New shape: (3, B, num_heads, T, head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)

        # Pull out Query, Key, and Value tensors separately
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Step 2: Compute attention scores = Q @ K^T / sqrt(head_dim)
        # The sqrt scaling keeps values stable (otherwise softmax becomes too sharp)
        # Shape after: (B, num_heads, T, T)
        attn = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)

        # Step 3: Apply CAUSAL mask (lower-triangular)
        # This means each token can only attend to itself and previous tokens
        # NOTE: comment says "REMOVE causal mask for embedding model" but it's still here
        # tril = lower triangular matrix of 1s
        mask = torch.tril(torch.ones(T, T, device=x.device))
        # Where mask is 0 (future positions), set attention score to -infinity
        # so softmax will turn those into 0 (i.e., ignore future tokens)
        attn = attn.masked_fill(mask == 0, float("-inf"))

        # Step 4: Apply PADDING mask (ignore <PAD> tokens in batch)
        if attention_mask is not None:
            # attention_mask is (B, T): 1 = real token, 0 = padding
            # Convert: padding positions get a huge negative number, real tokens get 0
            # Reshape to (B, 1, 1, T) so it broadcasts across heads and query positions
            pad_mask = (1.0 - attention_mask.float()).unsqueeze(1).unsqueeze(2) * -1e9
            attn = attn + pad_mask  # add to attention scores

        # Step 5: Softmax → turn scores into probabilities (sum to 1 across last dim)
        attn = torch.softmax(attn, dim=-1)
        # Step 6: Multiply attention weights by Values to get output
        # Shape: (B, num_heads, T, head_dim)
        out = attn @ v
        # Step 7: Re-merge heads back together → (B, T, C)
        out = out.transpose(1, 2).contiguous().view(B, T, C)

        # Step 8: Final projection
        return self.out(out)


# ============================================================================
# MLP (Feed-Forward Network) — The "thinking" part after attention
# Each token is processed independently here.
# ============================================================================
class KiteFishMLP(nn.Module):

    def __init__(self, config: KiteFishConfig):
        super().__init__()

        # Expand from hidden_size → intermediate_size (usually 4x bigger)
        self.fc1 = nn.Linear(config.hidden_size, config.intermediate_size)
        # Project back down to hidden_size
        self.fc2 = nn.Linear(config.intermediate_size, config.hidden_size)

        # GELU = smooth activation function (modern alternative to ReLU)
        self.act = nn.GELU()

    def forward(self, x):
        # Pipeline: expand → activate → shrink back
        return self.fc2(self.act(self.fc1(x)))


# ============================================================================
# TRANSFORMER BLOCK — One full layer = LayerNorm + Attention + LayerNorm + MLP
# We stack many of these to build the full model.
# ============================================================================
class KitefishBlock(nn.Module):

    def __init__(self, config):
        super().__init__()

        # Two LayerNorms: one before attention, one before MLP (Pre-Norm style)
        self.ln1 = nn.LayerNorm(config.hidden_size)
        self.ln2 = nn.LayerNorm(config.hidden_size)

        # The two main sub-modules
        self.attn = KiteFishAttention(config)
        self.mlp = KiteFishMLP(config)
        # Flag for memory-saving trick during training (off by default)
        self.gradient_checkpointing = False

    def forward(self, x, attention_mask=None):
        # Define what one block does:
        # 1) Normalize → Attention → ADD back to input (residual connection)
        # 2) Normalize → MLP → ADD back (residual connection)
        # Residual connections help gradients flow through deep networks
        def forward_fn(x):
            x = x + self.attn(self.ln1(x), attention_mask=attention_mask)
            x = x + self.mlp(self.ln2(x))
            return x

        # If gradient checkpointing is on AND we're training,
        # use the memory-saving version (recomputes during backward pass)
        if self.gradient_checkpointing and self.training:
            x = checkpoint.checkpoint(forward_fn, x, use_reentrant=False)
        else:
            # Normal forward pass
            x = forward_fn(x)
        return x


# ============================================================================
# MAIN BACKBONE MODEL — Embeddings + stack of Transformer blocks
# This is the "encoder" part. It outputs hidden states (no prediction head).
# ============================================================================
class KiteFishModel(PreTrainedModel):

    # HuggingFace boilerplate so save/load works correctly
    config_class = KiteFishConfig
    base_model_prefix = "kitefish"
    supports_gradient_checkpointing = True

    def __init__(self, config: KiteFishConfig):
        super().__init__(config)

        # Token embedding: turns token IDs (integers) into vectors
        # Shape: (vocab_size, hidden_size)
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        # Position embedding: tells the model WHERE each token is in the sequence
        # (Transformers have no built-in sense of order, so we add this info)
        self.embed_positions = nn.Embedding(
            config.max_position_embeddings,
            config.hidden_size,
        )

        # Build a stack of N Transformer blocks (e.g., 12 layers)
        self.layers = nn.ModuleList(
            [KitefishBlock(config) for _ in range(config.num_hidden_layers)]
        )

        # Final normalization at the very end of the stack
        self.final_layernorm = nn.LayerNorm(config.hidden_size)

        # HF helper: initializes weights properly (calls _init_weights)
        self.post_init()

    def forward(
        self,
        input_ids,                  # (B, T): the token IDs
        attention_mask=None,        # (B, T): 1=real token, 0=padding
        return_dict=None,           # whether to return a structured output
        output_hidden_states=None,  # (we ignore this, just absorbed)
        **kwargs,                   # catch-all for extra unused arguments
    ):
        B, T = input_ids.shape
        # Create position indices: [0, 1, 2, ..., T-1]
        pos = torch.arange(0, T, device=input_ids.device)
        # Combined input = token embedding + position embedding
        x = self.embed_tokens(input_ids) + self.embed_positions(pos)

        # Pass through each transformer block one by one
        for layer in self.layers:
            x = layer(x, attention_mask=attention_mask)

        # Final layer norm
        x = self.final_layernorm(x)

        # If caller wants raw tensor, return that
        if return_dict is False:
            return x
        # Otherwise wrap in HF's BaseModelOutput (a fancy dict)
        # This makes it work with sentence-transformers and HF pipelines
        return BaseModelOutput(last_hidden_state=x)


# ============================================================================
# FULL MODEL FOR LANGUAGE MODELING — backbone + LM head
# This adds a prediction head on top to predict the next token.
# Used for training a CAUSAL language model (GPT-style).
# ============================================================================
class KiteFishForCausalLM(PreTrainedModel):

    config_class = KiteFishConfig
    # checkpoint keys saved as "model.layers.*", so prefix is "model"
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    # If someone loads ONLY the backbone (e.g. for embeddings),
    # ignore missing lm_head weights — don't error out
    _keys_to_ignore_on_load_unexpected = [r"lm_head\..*"]

    def __init__(self, config):
        super().__init__(config)

        # The backbone (embeddings + transformer blocks)
        self.model = KiteFishModel(config)

        # LM head: maps hidden state → vocabulary logits
        # No bias (common practice for LM heads)
        self.lm_head = nn.Linear(
            config.hidden_size,
            config.vocab_size,
            bias=False,
        )

        # Initialize weights
        self.post_init()

    def forward(self, input_ids, attention_mask=None, labels=None, return_dict=None, **kwargs):

        # Step 1: Run the backbone to get hidden states
        out = self.model(input_ids, attention_mask=attention_mask, return_dict=True)
        hidden_states = out.last_hidden_state  # (B, T, hidden_size)

        # Step 2: Project to vocabulary → logits (B, T, vocab_size)
        # Each position now has a score for every possible next token
        logits = self.lm_head(hidden_states)

        loss = None  # default if no labels provided (inference mode)

        # Step 3: If labels are given, compute the training loss
        if labels is not None:

            # Shift so that we predict token[t+1] from position t
            # logits at position t should match labels at position t+1
            shift_logits = logits[..., :-1, :].contiguous()  # drop last logit
            shift_labels = labels[..., 1:].contiguous()      # drop first label

            # Standard cross-entropy loss for classification (over vocab)
            loss_fct = nn.CrossEntropyLoss()

            loss = loss_fct(
                # Flatten: (B*T, vocab_size)
                shift_logits.view(-1, self.config.vocab_size),
                # Flatten: (B*T,)
                shift_labels.view(-1),
            )

        # Return both loss and logits in HF's standard wrapper
        return CausalLMOutput(
            loss=loss,
            logits=logits,
        )

    def encode(self, input_ids, attention_mask):
        """
        Used to turn a sentence into a single embedding vector.
        This is the 'embedding model' usage (for similarity search, retrieval, etc.)
        """
        # Get hidden states from the backbone
        out = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
        )
        last_hidden = out.last_hidden_state  # (B, T, hidden_size)

        # MEAN POOLING:
        # Average all token embeddings, but ignore padding tokens
        mask = attention_mask.unsqueeze(-1).float()   # (B, T, 1) for broadcasting
        summed = (last_hidden * mask).sum(1)          # zero-out pads, then sum → (B, hidden_size)
        counts = mask.sum(1).clamp(min=1e-9)          # how many real tokens per row
                                                       # clamp prevents divide-by-zero
        embedding = summed / counts                    # average → (B, hidden_size)

        # L2 normalize so all embeddings have length 1
        # This makes cosine similarity = dot product (faster, standard practice)
        return torch.nn.functional.normalize(embedding, p=2, dim=1)

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        # Forwarding the call to the backbone (HF compatibility)
        self.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs)

    def _set_gradient_checkpointing(self, module, value=False):
        # Same — delegate to backbone if it has the method
        if hasattr(self.model, "_set_gradient_checkpointing"):
            self.model._set_gradient_checkpointing(module, value)