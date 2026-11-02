import torch
import torch.nn as nn
from dataclasses import dataclass
from mmengine.registry import MODELS
from mmengine.model import BaseModel
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


@dataclass
class LLaMAHFConfig:
    block_size: int = 4096
    vocab_size: int = 32000
    n_layer: int = 32
    n_head: int = 32
    n_embd: int = 4096
    condition_dim: int = 512

@MODELS.register_module()
class LLaMAHF(BaseModel):
    def __init__(self, 
                 lm_head_cnt: int, 
                 block_size: int = 4096, 
                 vocab_size: int = 32000, 
                 n_layer: int = 32, 
                 n_head: int = 32, 
                 n_embd: int = 4096,
                 condition_dim = 512, 
                 **kwargs) -> None:
        super().__init__(**kwargs)
        config = LLaMAHFConfig(
            block_size=block_size, vocab_size=vocab_size, n_layer=n_layer, 
            n_head=n_head, n_embd=n_embd, condition_dim=condition_dim)
        assert config.vocab_size is not None
        assert config.block_size is not None
        self.config = config

        self.lm_heads = nn.ModuleList()
        for _ in range(lm_head_cnt):
            self.lm_heads.append(nn.Linear(config.n_embd, config.vocab_size, bias=False))

        wtes = {f'wte{i}': nn.Embedding(config.vocab_size, config.n_embd) for i in range(lm_head_cnt)}
        self.transformer = nn.ModuleDict(
            dict(
                h=nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
                ln_f=RMSNorm(config.n_embd),
                **wtes
            )
        )

    def forward(self, idx, non_pad_mask, text_feat, text_mask, audio_feat, audio_mask, traj_feat, traj_mask):
        t_token_lens = text_mask.sum(1)
        traj_token_lens = traj_mask.sum(1)
        m_token_lens = non_pad_mask.sum(1)
        a_token_lens = audio_mask.sum(1)

        x, mask = self.prepare_input(idx, text_feat, audio_feat, traj_feat, t_token_lens, m_token_lens, a_token_lens, traj_token_lens)
        for block in self.transformer.h: # type: ignore
            x = block(x, mask)
        x = self.transformer.ln_f(x) # type: ignore
        return self.post_process(idx, x, m_token_lens, mask)

    def post_process(self, idx, x, m_token_lens, mask):
        '''
        # new_logits = torch.zeros((B, T, self.config.vocab_size)).to(logits)
        # for i in range(B):
        #     new_logits[i, :m_token_lens[i]] = logits[i, mask[i].to(torch.bool)][-m_token_lens[i]:]
        '''
        T = idx[0].size(1)
        all_logits = []
        for lm_head in self.lm_heads:
            logits = lm_head(x) # (b, t, vocab_size)
            new_logits = torch.zeros_like(logits)
            valid_lens = mask.to(torch.int64).sum(dim=1).to(torch.int64)   # [B]
            pos = torch.arange(logits.shape[1], device=logits.device).unsqueeze(0)  # [1, T]
            select = (pos >= (valid_lens - m_token_lens).unsqueeze(1)) & \
                    (pos < valid_lens.unsqueeze(1))  # [B, T]
            target = pos < m_token_lens.unsqueeze(1)  # [B, T]
            new_logits[target] = logits[select]
            all_logits.append(new_logits[:, :T])
        return all_logits

    def prepare_input(self, idx, text_feat, audio_feat, traj_feat, t_token_lens, m_token_lens, a_token_lens, traj_token_lens):
        B, T = idx[0].size()
        length = int(T + t_token_lens.max() + a_token_lens.max() + traj_token_lens.max())

        x = torch.zeros((B, length, self.config.n_embd)).to(text_feat)
        mask = torch.zeros((B, length)).to(text_feat)
        # 1. 写入 text token
        pos = torch.arange(x.size(1), device=x.device).unsqueeze(0)  # [1, T]

        text_mask = pos < t_token_lens.unsqueeze(1)
        text_feat = F.pad(text_feat, (0, 0, 0, x.shape[1] - text_feat.shape[1]), 'constant', 0)
        x[text_mask] = text_feat[text_mask]
        offset = t_token_lens

        # 2. 写入 traj token
        traj_mask = (
            (pos >= (offset).unsqueeze(1)) &
            (pos < (offset + traj_token_lens).unsqueeze(1))
        )
        # traj_mask1 = torch.arange(traj_token_lens.max(), device=x.device).unsqueeze(0) < traj_token_lens.unsqueeze(1)
        traj_mask1 = torch.arange(traj_feat.shape[1], device=x.device).unsqueeze(0) < traj_token_lens.unsqueeze(1)

        if traj_mask.sum() > 0:
            x[traj_mask] = traj_feat[traj_mask1]

        # 3. 写入 audio token
        audio_mask = (
            (pos >= (offset + traj_token_lens).unsqueeze(1)) &
            (pos < (offset + traj_token_lens + a_token_lens).unsqueeze(1))
        )
        # audio_mask1 = torch.arange(a_token_lens.max(), device=x.device).unsqueeze(0) < a_token_lens.unsqueeze(1)
        audio_mask1 = torch.arange(audio_feat.shape[1], device=x.device).unsqueeze(0) < a_token_lens.unsqueeze(1)
        if audio_mask.sum() > 0:
            x[audio_mask] = audio_feat[audio_mask1]

        # 4. 写入 motion token（最重要）
        motion_mask = (
            (pos >= (offset + traj_token_lens + a_token_lens).unsqueeze(1)) &
            (pos < (offset + traj_token_lens + a_token_lens + m_token_lens).unsqueeze(1))
        )
        motion_mask1 = torch.arange(m_token_lens.max(), device=x.device).unsqueeze(0) < m_token_lens.unsqueeze(1)

        motion_embeds = []
        for i in range(len(idx)):
            motion_embed = self.transformer.get_submodule(f'wte{i}')(idx[i]).to(x)
            motion_embeds.append(motion_embed)
        motion_embed = sum(motion_embeds)
        x[motion_mask] = motion_embed[motion_mask1]

        # 5. mask
        total_len = offset + traj_token_lens + a_token_lens + m_token_lens
        mask[pos < total_len.unsqueeze(1)] = 1.0
        return x, mask

class Block(nn.Module):
    def __init__(self, config: LLaMAHFConfig) -> None: # , use_qkNorm=False, use_moe=False) -> None:
        super().__init__()
        self.rms_1 = RMSNorm(config.n_embd)
        self.attn = LengthCausalSelfAttention(config) # , use_qkNorm)
        self.rms_2 = RMSNorm(config.n_embd)
        
        self.mlp = MLP(config)

    def forward(self, x: torch.Tensor, y_mask: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.rms_1(x), y_mask)
        x = x + self.mlp(self.rms_2(x))

        return x


class CausalSelfAttention(nn.Module):
    def __init__(self, config: LLaMAHFConfig) -> None:
        super().__init__()
        assert config.n_embd % config.n_head == 0

        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=False)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=False)

        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.block_size = config.block_size
        self.rope_cache = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.size()  # batch size, sequence length, embedding dimensionality (n_embd)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)

        head_size = C // self.n_head
        k = k.view(B, T, self.n_head, head_size).transpose(1, 2)  # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, head_size).transpose(1, 2)  # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, head_size).transpose(1, 2)  # (B, nh, T, hs)

        if self.rope_cache is None:
            # cache for future forward calls
            self.rope_cache = build_rope_cache(
                seq_len=self.block_size,
                n_elem=self.n_embd // self.n_head, 
                dtype=x.dtype,
                device=x.device,
            )

        q = apply_rope(q, self.rope_cache)
        k = apply_rope(k, self.rope_cache)

        # efficient attention using Flash Attention CUDA kernels
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=0.0, is_causal=True)

        y = y.transpose(1, 2).contiguous().view(B, T, C)  # re-assemble all head outputs side by side

        # output projection
        y = self.c_proj(y)

        return y

class LengthCausalSelfAttention(nn.Module):
    def __init__(self, config: LLaMAHFConfig) -> None: # , use_qkNorm=False) -> None:
        super().__init__()
        assert config.n_embd % config.n_head == 0

        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=False)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=False)

        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.block_size = config.block_size
        self.rope_cache = None 
    def forward(self, x: torch.Tensor, y_mask: torch.Tensor) -> torch.Tensor:
        B, T, C = x.size()  # batch size, sequence length, embedding dimensionality (n_embd)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        # q, k, v = checkpoint(lambda x: self.c_attn(x), x, use_reentrant=False).split(self.n_embd, dim=2)

        head_size = C // self.n_head
        k = k.view(B, T, self.n_head, head_size).transpose(1, 2)  # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, head_size).transpose(1, 2)  # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, head_size).transpose(1, 2)  # (B, nh, T, hs)
        
        if self.rope_cache is None:
            # cache for future forward calls
            self.rope_cache = build_rope_cache(
                seq_len=self.block_size,
                n_elem=self.n_embd // self.n_head, 
                dtype=x.dtype,
                device=x.device,
            )
            
        q = apply_rope(q, self.rope_cache)
        k = apply_rope(k, self.rope_cache)

        # create attention mask
        attn_mask = torch.ones(T, T, dtype=torch.bool, device=x.device)
        attn_mask = torch.tril(attn_mask)
        attn_mask = attn_mask.unsqueeze(0).expand(B, -1, -1)

        text_mask = y_mask.unsqueeze(2)*y_mask.unsqueeze(1)
        text_mask = F.pad(text_mask, (0, T-y_mask.shape[1], 0, T-y_mask.shape[1]), mode='constant', value=0)
        attn_mask = torch.logical_or(attn_mask, text_mask)
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask.unsqueeze(1), dropout_p=0.0, is_causal=False)

        y = y.transpose(1, 2).contiguous().view(B, T, C)

        y = self.c_proj(y)
        # y = checkpoint(lambda y: self.c_proj(y), y, use_reentrant=False)

        return y


class MLP(nn.Module):
    def __init__(self, config: LLaMAHFConfig) -> None:
        super().__init__()
        hidden_dim = 4 * config.n_embd
        n_hidden = int(2 * hidden_dim / 3)
        N = 256
        # ensure n_hidden is multiple of N
        n_hidden = ((n_hidden - 1) // N) * N + N

        self.c_fc1 = nn.Linear(config.n_embd, n_hidden, bias=False)
        self.c_fc2 = nn.Linear(config.n_embd, n_hidden, bias=False)
        self.c_proj = nn.Linear(n_hidden, config.n_embd, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        
        x = F.silu(self.c_fc1(x)) * self.c_fc2(x)
        x = self.c_proj(x)

        # x = checkpoint(lambda x: F.silu(self.c_fc1(x)) * self.c_fc2(x), x, use_reentrant=False)
        # x = checkpoint(lambda x: self.c_proj(x), x, use_reentrant=False)
        return x


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization.

    Derived from https://github.com/bzhangGo/rmsnorm/blob/master/rmsnorm_torch.py. BSD 3-Clause License:
    https://github.com/bzhangGo/rmsnorm/blob/master/LICENSE.
    """

    def __init__(self, size: int, dim: int = -1, eps: float = 1e-5) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.ones(size))
        self.eps = eps
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # NOTE: the original RMSNorm paper implementation is not equivalent
        # norm_x = x.norm(2, dim=self.dim, keepdim=True)
        # rms_x = norm_x * d_x ** (-1. / 2)
        # x_normed = x / (rms_x + self.eps)
        norm_x = torch.mean(x * x, dim=self.dim, keepdim=True)
        x_normed = x * torch.rsqrt(norm_x + self.eps)
        return self.scale * x_normed


def build_rope_cache(seq_len: int, n_elem: int, dtype: torch.dtype, device: torch.device, base: int = 10000) -> torch.Tensor:
    """Enhanced Transformer with Rotary Position Embedding.

    Derived from: https://github.com/labmlai/annotated_deep_learning_paper_implementations/blob/master/labml_nn/
    transformers/rope/__init__.py. MIT License:
    https://github.com/labmlai/annotated_deep_learning_paper_implementations/blob/master/license.
    """
    # $\Theta = {\theta_i = 10000^{\frac{2(i-1)}{d}}, i \in [1, 2, ..., \frac{d}{2}]}$
    theta = 1.0 / (base ** (torch.arange(0, n_elem, 2, dtype=dtype, device=device) / n_elem))

    # Create position indexes `[0, 1, ..., seq_len - 1]`
    seq_idx = torch.arange(seq_len, dtype=dtype, device=device)

    # Calculate the product of position index and $\theta_i$
    idx_theta = torch.outer(seq_idx, theta)

    # Compute cache. Because polar only takes float32 or float64, we need to cast
    # when working with 16 bit floats (float16 or bfloat16)
    dtypes_requiring_casting = [torch.float16, torch.bfloat16, torch.int8]
    working_dtype = (
        torch.float32 if dtype in dtypes_requiring_casting else dtype
    )
    complex_dtype = (
        torch.complex32 if dtype in dtypes_requiring_casting else torch.complex64
    )
    cache = torch.polar(
        torch.ones_like(idx_theta).to(working_dtype), idx_theta.to(working_dtype)
    ).to(complex_dtype)
    return cache


def apply_rope(x: torch.Tensor, rope_cache: torch.Tensor) -> torch.Tensor:
    x = x.transpose(1, 2)

    # truncate to support variable sizes
    T = x.size(1)
    rope_cache = rope_cache[:T]
    
    # cast because `view_as_complex` does not support 16 bit tensors
    xc = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
    rope_cache = rope_cache.view(1, xc.size(1), 1, xc.size(3))
    x_out = torch.view_as_real(xc * rope_cache).flatten(3)
    return x_out.transpose(1, 2).type_as(x)
