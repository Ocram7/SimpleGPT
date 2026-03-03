# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
#
# Llama 2 is licensed under the LLAMA 2 Community License,
# Copyright (c) Meta Platforms, Inc. All Rights Reserved.


from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn, norm
from torchtitan.models.norms import create_norm


@dataclass
class ModelArgs:
    dim: int = 4096
    n_layers: int = 32
    n_heads: int = 32
    n_kv_heads: Optional[int] = None
    vocab_size: int = -1  # defined later by tokenizer
    multiple_of: int = 256  # make SwiGLU hidden layer size multiple of large power of 2
    ffn_dim_multiplier: Optional[float] = None
    norm_eps: float = 1e-5
    rope_theta: float = 10000

    max_batch_size: int = 32
    max_seq_len: int = 2048
    # If `True`, then each transformer block init uses its layer ID, and if
    # `False`, each uses the total number of transformer blocks
    depth_init: bool = True
    norm_type: str = "rmsnorm"


def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0) -> torch.Tensor:
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device)
    freqs = torch.outer(t, freqs).float()
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
    return freqs_cis


def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    ndim = x.ndim
    assert 0 <= 1 < ndim
    seqlen = x.shape[1]
    freqs_cis = freqs_cis[0:seqlen]
    assert freqs_cis.shape == (seqlen, x.shape[-1])
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(*shape)


def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    freqs_cis = reshape_for_broadcast(freqs_cis, xq_)
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """torch.repeat_interleave(x, dim=2, repeats=n_rep)"""
    bs, slen, n_kv_heads, head_dim = x.shape
    if n_rep == 1:
        return x
    return (
        torch.unsqueeze(x, dim=3)
        .expand(bs, slen, n_kv_heads, n_rep, head_dim)
        .reshape(bs, slen, n_kv_heads * n_rep, head_dim)
    )


class SimpleNorm(nn.Module):
    """
    SimpleNorm module.

    After every linear layer, normalization is applied to the output.

    Args:
        in_features (int): The number of input features for the linear layer.
        out_features (int): The number of output features for the linear layer.
        bias (bool): Whether to include a bias term in the linear layer.
        norm_type (str): The type of normalization layer.
            Supported types: 1. rmsnorm 2. fused_rmsnorm 3. layernorm 4. np_layernorm
        norm_eps (float, optional): The norm epsilon value for numerical stability. Defaults to 1e-6.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool,
        norm_type: str,
        norm_eps: float = 1e-6,
    ):
        super().__init__()
        self.lin = nn.Linear(in_features, out_features, bias=bias)
        self.norm = create_norm(norm_type, out_features, eps=norm_eps)

    def forward(self, x):
        return self.norm(self.lin(x))

    def init_weights(self, init_std: float = 0.02):
        self.norm.reset_parameters()
        nn.init.trunc_normal_(self.lin.weight, mean=0.0, std=init_std)


class Attention(nn.Module):
    def __init__(self, model_args: ModelArgs):
        super().__init__()
        self.n_heads = model_args.n_heads
        self.n_kv_heads = (
            model_args.n_heads
            if model_args.n_kv_heads is None
            else model_args.n_kv_heads
        )
        self.n_rep = self.n_heads // self.n_kv_heads
        self.head_dim = model_args.dim // model_args.n_heads

        self.snq = SimpleNorm(
            model_args.dim,
            model_args.n_heads * self.head_dim,
            bias=False,
            norm_type=model_args.norm_type,
            norm_eps=model_args.norm_eps,
        )

        self.snk = SimpleNorm(
            model_args.dim,
            self.n_kv_heads * self.head_dim,
            bias=False,
            norm_type=model_args.norm_type,
            norm_eps=model_args.norm_eps,
        )

        self.snv = SimpleNorm(
            model_args.dim,
            self.n_kv_heads * self.head_dim,
            bias=False,
            norm_type=model_args.norm_type,
            norm_eps=model_args.norm_eps,
        )

        self.sno = SimpleNorm(
            self.n_heads * self.head_dim,
            model_args.dim,
            bias=False,
            norm_type=model_args.norm_type,
            norm_eps=model_args.norm_eps,
        )

    def init_weights(self, init_std: float):
        for sn in (self.snq, self.snk, self.snv):
            sn.init_weights(init_std=0.02)
        self.sno.init_weights(init_std=init_std)

    def forward(
        self,
        x: torch.Tensor,
        freqs_cis: torch.Tensor,
    ):
        bs, seqlen, _ = x.shape
        xq, xk, xv = self.snq(x), self.snk(x), self.snv(x)

        xq = xq.view(bs, seqlen, self.n_heads, self.head_dim)
        xk = xk.view(bs, seqlen, self.n_kv_heads, self.head_dim)
        xv = xv.view(bs, seqlen, self.n_kv_heads, self.head_dim)

        xq, xk = apply_rotary_emb(xq, xk, freqs_cis=freqs_cis)

        # repeat k/v heads if n_kv_heads < n_heads
        keys = repeat_kv(xk, self.n_rep)  # (bs, seqlen, n_local_heads, head_dim)
        values = repeat_kv(xv, self.n_rep)  # (bs, seqlen, n_local_heads, head_dim)

        xq = xq.transpose(1, 2)  # (bs, n_local_heads, seqlen, head_dim)
        xk = keys.transpose(1, 2)  # (bs, n_local_heads, seqlen, head_dim)
        xv = values.transpose(1, 2)  # (bs, n_local_heads, seqlen, head_dim)

        # we use casual mask for training
        output = F.scaled_dot_product_attention(xq, xk, xv, is_causal=True)
        output = output.transpose(
            1, 2
        ).contiguous()  # (bs, seqlen, n_local_heads, head_dim)
        output = output.view(bs, seqlen, -1)
        return self.sno(output)


class FeedForward(nn.Module):
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        multiple_of: int,
        ffn_dim_multiplier: Optional[float],
        model_args: ModelArgs,
    ):
        super().__init__()
        hidden_dim = int(2 * hidden_dim / 3)
        # custom dim factor multiplier
        if ffn_dim_multiplier is not None:
            hidden_dim = int(ffn_dim_multiplier * hidden_dim)
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)

        self.sn1 = SimpleNorm(
            dim,
            hidden_dim,
            bias=False,
            norm_type=model_args.norm_type,
            norm_eps=model_args.norm_eps,
        )
        self.sn2 = SimpleNorm(
            hidden_dim,
            dim,
            bias=False,
            norm_type=model_args.norm_type,
            norm_eps=model_args.norm_eps,
        )
        self.sn3 = SimpleNorm(
            dim,
            hidden_dim,
            bias=False,
            norm_type=model_args.norm_type,
            norm_eps=model_args.norm_eps,
        )

    def forward(self, x):
        return self.sn2(F.silu(self.sn1(x)) * self.sn3(x))

    def init_weights(self, init_std: float):
        self.sn1.init_weights(init_std=0.02)
        for sn in (self.sn2, self.sn3):
            sn.init_weights(init_std=init_std)

class TransformerBlock(nn.Module):
    def __init__(self, layer_id: int, model_args: ModelArgs):
        super().__init__()
        self.n_heads = model_args.n_heads
        self.dim = model_args.dim
        self.attention = Attention(model_args)
        self.feed_forward = FeedForward(
            dim=model_args.dim,
            hidden_dim=4 * model_args.dim,
            multiple_of=model_args.multiple_of,
            ffn_dim_multiplier=model_args.ffn_dim_multiplier,
            model_args=model_args,
        )
        self.layer_id = layer_id
        self.num_layers = model_args.n_layers

        if model_args.depth_init:
            self.weight_init_std = 0.02 / (2 * (self.layer_id + 1)) ** 0.5
        else:
            self.weight_init_std = 0.02 / (2 * self.num_layers) ** 0.5

    def forward(
        self,
        x: torch.Tensor,
        freqs_cis: torch.Tensor,
    ):
        h = x + self.attention(x, freqs_cis)
        out = h + self.feed_forward(h)
        return out

    def init_weights(self):
        self.attention.init_weights(self.weight_init_std)
        self.feed_forward.init_weights(self.weight_init_std)


class SimpleGPT(nn.Module):
    def __init__(self, model_args: ModelArgs):
        super().__init__()
        self.model_args = model_args
        self.vocab_size = model_args.vocab_size
        self.n_layers = model_args.n_layers

        self.tok_embeddings = nn.Embedding(model_args.vocab_size, model_args.dim)

        self.register_buffer("freqs_cis", self._precompute_freqs_cis(), persistent=True)

        self.layers = torch.nn.ModuleDict()
        for layer_id in range(model_args.n_layers):
            self.layers[str(layer_id)] = TransformerBlock(layer_id, model_args)

        self.norm = create_norm(
            model_args.norm_type, dim=model_args.dim, eps=model_args.norm_eps
        )

        self.output = nn.Linear(model_args.dim, model_args.vocab_size, bias=False)
        self.init_weights()

    def init_weights(self):
        with torch.device(self.freqs_cis.device):
            self.freqs_cis = self._precompute_freqs_cis()
        nn.init.normal_(self.tok_embeddings.weight)
        for layer in self.layers.values():
            layer.init_weights()
        self.norm.reset_parameters()
        final_out_std = self.model_args.dim**-0.5
        cutoff_factor = 3
        nn.init.trunc_normal_(
            self.output.weight,
            mean=0.0,
            std=final_out_std,
            a=-cutoff_factor * final_out_std,
            b=cutoff_factor * final_out_std,
        )

    def _precompute_freqs_cis(self) -> torch.Tensor:
        return precompute_freqs_cis(
            self.model_args.dim // self.model_args.n_heads,
            # Need to compute until at least the max token limit for generation
            # (use 2x max sequence length to be safe)
            self.model_args.max_seq_len * 2,
            self.model_args.rope_theta,
        )

    def forward(self, tokens: torch.Tensor):
        # passthrough for nonexistent layers, allows easy configuration of pipeline parallel stages
        h = self.tok_embeddings(tokens) if self.tok_embeddings else tokens

        for layer in self.layers.values():
            h = layer(h, self.freqs_cis)

        h = self.norm(h) if self.norm else h
        output = self.output(h).float() if self.output else h
        return output

    @classmethod
    def from_model_args(cls, model_args: ModelArgs) -> "SimpleGPT":
        return cls(model_args)
