from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


class LayerNorm(nn.Module):
    def __init__(self, ndim: int, bias: bool) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.layer_norm(x, self.weight.shape, self.weight, self.bias, 1e-5)


class CausalSelfAttention(nn.Module):
    def __init__(self, n_embd: int, n_head: int, dropout: float, bias: bool, block_size: int) -> None:
        super().__init__()
        if n_embd % n_head != 0:
            raise ValueError(f"embedding dim {n_embd} must be divisible by number of heads {n_head}")
        self.n_head = n_head
        self.n_embd = n_embd
        self.dropout = dropout
        self.c_attn = nn.Linear(n_embd, 3 * n_embd, bias=bias)
        self.c_proj = nn.Linear(n_embd, n_embd, bias=bias)
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)
        self.flash = hasattr(torch.nn.functional, "scaled_dot_product_attention")
        if not self.flash:
            mask = torch.tril(torch.ones(block_size, block_size)).view(1, 1, block_size, block_size)
            self.register_buffer("bias", mask)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, sequence_length, channels = x.shape
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        head_dim = channels // self.n_head
        q = q.view(batch_size, sequence_length, self.n_head, head_dim).transpose(1, 2)
        k = k.view(batch_size, sequence_length, self.n_head, head_dim).transpose(1, 2)
        v = v.view(batch_size, sequence_length, self.n_head, head_dim).transpose(1, 2)

        if self.flash:
            y = torch.nn.functional.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=None,
                dropout_p=self.dropout if self.training else 0.0,
                is_causal=True,
            )
        else:
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(head_dim))
            att = att.masked_fill(self.bias[:, :, :sequence_length, :sequence_length] == 0, float("-inf"))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v

        y = y.transpose(1, 2).contiguous().view(batch_size, sequence_length, channels)
        return self.resid_dropout(self.c_proj(y))


class MLP(nn.Module):
    def __init__(self, n_embd: int, dropout: float, bias: bool) -> None:
        super().__init__()
        self.c_fc = nn.Linear(n_embd, 4 * n_embd, bias=bias)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4 * n_embd, n_embd, bias=bias)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return self.dropout(x)


class Block(nn.Module):
    def __init__(self, n_embd: int, n_head: int, dropout: float, bias: bool, block_size: int) -> None:
        super().__init__()
        self.ln_1 = LayerNorm(n_embd, bias=bias)
        self.attn = CausalSelfAttention(n_embd, n_head, dropout, bias, block_size)
        self.ln_2 = LayerNorm(n_embd, bias=bias)
        self.mlp = MLP(n_embd, dropout, bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


@dataclass
class TransformerConfig:
    input_dim: int
    block_size: int
    n_layer: int
    n_head: int
    n_embd: int
    dropout: float
    bias: bool


class ContinuousTransformer(nn.Module):
    def __init__(self, config: TransformerConfig) -> None:
        super().__init__()
        self.config = config
        self.input_proj = nn.Linear(config.input_dim, config.n_embd, bias=config.bias)
        self.position_embedding = nn.Embedding(config.block_size, config.n_embd)
        self.dropout = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList(
            [
                Block(
                    n_embd=config.n_embd,
                    n_head=config.n_head,
                    dropout=config.dropout,
                    bias=config.bias,
                    block_size=config.block_size,
                )
                for _ in range(config.n_layer)
            ]
        )
        self.final_norm = LayerNorm(config.n_embd, bias=config.bias)
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, sequence_length, _ = x.shape
        if sequence_length > self.config.block_size:
            raise ValueError(
                f"input sequence length {sequence_length} exceeds transformer block size {self.config.block_size}"
            )
        positions = torch.arange(sequence_length, dtype=torch.long, device=x.device)
        x = self.input_proj(x) + self.position_embedding(positions)
        x = self.dropout(x)
        for block in self.blocks:
            x = block(x)
        return self.final_norm(x)

