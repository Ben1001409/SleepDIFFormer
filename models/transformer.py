import math
from collections import OrderedDict
from functools import partial
from typing import Any, Callable, List, NamedTuple, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy

def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """torch.repeat_interleave(x, dim=1, repeats=n_rep)"""
    bs, n_kv_heads, slen, head_dim = x.shape
    if n_rep == 1:
        return x
    return (
        x[:, :, None, :, :]
        .expand(bs, n_kv_heads, n_rep, slen, head_dim)
        .reshape(bs, n_kv_heads * n_rep, slen, head_dim)
    )

def lambda_init_fn(depth):
    return 0.8 - 0.6 * math.exp(-0.3 * depth)

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6, elementwise_affine=True, memory_efficient=False):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        if self.elementwise_affine:
            self.weight = nn.Parameter(torch.ones(dim))
        else:
            self.register_parameter('weight', None)

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        if self.weight is not None:
            output = output * self.weight
        return output

    def extra_repr(self) -> str:
        return f'dim={self.dim}, eps={self.eps}, elementwise_affine={self.elementwise_affine}'

def apply_rope(x):
    # x: (B, H, L, D), where D is 2 * head_dim
    B, H, L, D = x.shape
    head_dim = D // 2
    theta = 10000.0 ** (-torch.arange(0, head_dim, dtype=torch.float32) / head_dim).to(x.device)
    t = torch.arange(L, dtype=torch.float32).to(x.device)
    freqs = torch.einsum("i,j->ij", t, theta)
    sin, cos = freqs.sin(), freqs.cos()

    sin = sin[None, None, :, :]
    cos = cos[None, None, :, :]

    x1, x2 = x[..., :head_dim], x[..., head_dim:]
    return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)


class MultiheadDiffAttn(nn.Module):
    def __init__(
        self,
        embed_dim,
        depth, #current layer index
        num_heads,
        return_attention=True,
        num_kv_heads=None,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        

        self.num_heads = num_heads
        
        self.num_kv_heads = num_kv_heads if num_kv_heads is not None else num_heads
        self.n_rep = self.num_heads // self.num_kv_heads
        
        self.head_dim = embed_dim // num_heads // 2 
        self.scaling = self.head_dim ** -0.5
        
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.k_proj = nn.Linear(embed_dim, embed_dim // self.n_rep, bias=False)
        self.v_proj = nn.Linear(embed_dim, embed_dim // self.n_rep, bias=False)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=False)

    
        self.lambda_init = lambda_init_fn(depth)
        self.lambda_q1 = nn.Parameter(torch.zeros(self.head_dim, dtype=torch.float32).normal_(mean=0,std=0.02))
        self.lambda_k1 = nn.Parameter(torch.zeros(self.head_dim, dtype=torch.float32).normal_(mean=0,std=0.02))
        self.lambda_q2 = nn.Parameter(torch.zeros(self.head_dim, dtype=torch.float32).normal_(mean=0,std=0.02))
        self.lambda_k2 = nn.Parameter(torch.zeros(self.head_dim, dtype=torch.float32).normal_(mean=0,std=0.02))

        self.subln = RMSNorm(2 * self.head_dim, eps=1e-5, elementwise_affine=True)
        self.return_attention=return_attention
        
       
    def forward(
        self,
        x,
        #rel_pos,
        attn_mask=None,
    ):
        bsz, tgt_len, embed_dim = x.size()
        #print(x.size())
        src_len = tgt_len

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        q = q.view(bsz, tgt_len, 2 * self.num_heads, self.head_dim)
        k = k.view(bsz, src_len, 2 * self.num_kv_heads, self.head_dim)
        v = v.view(bsz, src_len, self.num_kv_heads, 2 * self.head_dim)


        offset = src_len - tgt_len
        q = q.transpose(1, 2)


        k = repeat_kv(k.transpose(1, 2), self.n_rep)
        v = repeat_kv(v.transpose(1, 2), self.n_rep)
        #print(k.shape)
        #print(q.shape)
        q *= self.scaling
        attn_weights = (torch.matmul(q, k.transpose(-1, -2)))  
        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).type_as(
            attn_weights
        )

        lambda_1 = torch.exp(torch.sum(self.lambda_q1 * self.lambda_k1, dim=-1).float()).type_as(q)
        lambda_2 = torch.exp(torch.sum(self.lambda_q2 * self.lambda_k2, dim=-1).float()).type_as(q)
        lambda_full = lambda_1 - lambda_2 + self.lambda_init
        attn_weights = attn_weights.view(bsz, self.num_heads, 2, tgt_len, src_len)
        attn_weights = attn_weights[:, :, 0] - lambda_full * attn_weights[:, :, 1]
        

        attn = torch.matmul(attn_weights, v)
        attn = self.subln(attn)
        attn = attn * (1 - self.lambda_init)
        attn = attn.transpose(1, 2).reshape(bsz, tgt_len, self.num_heads * 2 * self.head_dim)

        attn = self.out_proj(attn)
        #print("attn_shape: ",attn.shape)
        if self.return_attention:
            return attn,attn_weights
        else:
            return attn
        # return attn,attn_weights
    
class MLPBlock(nn.Sequential):
    """Transformer MLP block."""

    def __init__(self, in_dim: int, mlp_dim: int, dropout: float):
        super().__init__()
        self.linear_1 = nn.Linear(in_dim, mlp_dim)
        self.act = nn.GELU()
        self.dropout_1 = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(mlp_dim, in_dim)
        self.dropout_2 = nn.Dropout(dropout)

        nn.init.xavier_uniform_(self.linear_1.weight)
        nn.init.xavier_uniform_(self.linear_2.weight)
        nn.init.normal_(self.linear_1.bias, std=1e-6)
        nn.init.normal_(self.linear_2.bias, std=1e-6)


class EncoderBlock(nn.Module):
    """Transformer encoder block."""

    def __init__(
        self,
        num_heads: int,
        hidden_dim: int,
        mlp_dim: int,
        depth:int,
        dropout: float,
        attention_dropout: float,
        return_attention:bool,
        norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
    ):
        super().__init__()
        self.num_heads = num_heads
        self.return_attention=return_attention
        print("return attention: ",self.return_attention)
        # Attention block
        self.ln_1 = norm_layer(hidden_dim)
        self.self_attention = nn.MultiheadAttention(hidden_dim, num_heads, dropout=attention_dropout, batch_first=True)
        self.diff_attention=MultiheadDiffAttn(embed_dim=hidden_dim,depth=depth,num_heads=self.num_heads//2,return_attention=self.return_attention)
        self.dropout = nn.Dropout(dropout)

        # MLP block
        self.ln_2 = norm_layer(hidden_dim)
        self.mlp = MLPBlock(hidden_dim, mlp_dim, dropout)

    def forward(self, input: torch.Tensor):
        #print(input.shape)
        torch._assert(input.dim() == 3, f"Expected (seq_length, batch_size, hidden_dim) got {input.shape}")
        x = self.ln_1(input)
        #x=input
        if self.return_attention:
            x, attn_weights = self.self_attention(query=x, key=x, value=x, need_weights=True,average_attn_weights=False)
            #x, attn_weights=self.diff_attention(x)
        else:
            x,_ = self.self_attention(query=x, key=x, value=x, need_weights=False)
            #x=self.diff_attention(x)
        #attn_weights = attn_weights.mean(0).mean(0)

        #print(attn_weights.shape) 
        x = self.dropout(x)
        x = x + input
        #print(x.shape)
        y = self.ln_2(x)
        y = self.mlp(y)
        if self.return_attention:
            return x + y,attn_weights
        else:
            return x + y


class TransformerEncoder(nn.Module):
    """Transformer Model Encoder for sequence to sequence translation."""

    def __init__(
        self,
        seq_length: int,
        num_layers: int,
        num_heads: int,
        hidden_dim: int,
        mlp_dim: int,
        depth:int,
        dropout: float,
        attention_dropout: float,
        return_attention:bool,
        norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
    ):
        super().__init__()

        self.pos_embedding = nn.Parameter(torch.empty(1, seq_length, hidden_dim).normal_(std=0.02))  # from BERT
        self.dropout = nn.Dropout(dropout)
        layers: OrderedDict[str, nn.Module] = OrderedDict()
        for i in range(num_layers):

            layers[f"encoder_layer_{i}"] = EncoderBlock(
                num_heads,
                hidden_dim,
                mlp_dim,
                i,
                dropout,
                attention_dropout,
                return_attention,
                norm_layer,
            )
        #self.layers = _get_clones(encoder_block, num_layers)
        self.layers = nn.Sequential(layers)
        # self.layers = nn.ModuleList([
        #     EncoderBlock(
        #         num_heads,
        #         hidden_dim,
        #         mlp_dim,
        #         dropout,
        #         attention_dropout,
        #         depth=i,
        #         norm_layer=norm_layer,
        #     ) for i in range(num_layers)
        # ])
        #self.layers = nn.ModuleList(layers.values())
        self.ln = norm_layer(hidden_dim)

        self.return_attention=return_attention

    def forward(self, input: torch.Tensor):
        torch._assert(input.dim() == 3, f"Expected (batch_size, seq_length, hidden_dim) got {input.shape}")
        #print(input.shape) 
        input = input + self.pos_embedding 



        x = self.dropout(input)

        if self.return_attention:  
            for layer in self.layers:
                x, attn_weights = layer(x)  
        else:
            for layer in self.layers:
                x = layer(x)
        x = self.ln(x)
        if self.return_attention:
            attn_weights=attn_weights.mean(dim=1).mean(dim=0)

        if self.return_attention:
            return x,attn_weights
        else:
            return x

