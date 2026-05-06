# ------------------------------------------------------------------------------------
# 3D MambaVision T2 for IVF — Joint Space-Time
#
# Architecture: Concatenate T×H×W tokens into a single sequence, Mamba2/Attention processes the whole
#
# Stage 1: 1× ConvBlock3D → Downsample
# Stage 2: 3× ConvBlock3D → Downsample
# Stage 3: 11× Block3D (6 Mamba2 + 5 Attention) on (B, T*H*W, C) → Downsample
# Stage 4:  4× Block3D (2 Mamba2 + 2 Attention) on (B, T*H*W, C)
# Stage 5: mean(T) → (B, 640, H, W)
# Stage 6: BN → GAP → Linear → (B, num_classes)
#
# Why this is true 3D:
#   Mamba2 and Attention ACTUALLY see all 5 frames in one sequence
#   Order: [frame0_tok0..tok(HW-1), frame1_tok0..tok(HW-1), ..., frame4_tok(HW-1)]
#   Mamba2 causal: frame0 → frame1 → frame2 → frame3 → frame4 (correct temporal order)
#   Attention: sees the entire spatial-temporal context simultaneously
#
# Sequence length với 224×224, T=5:
#   Stage 3: 5×14×14 = 980 tokens, d_state=16, chunk_size=128 → OK
#   Stage 4: 5×7×7   = 245 tokens, d_state=16, chunk_size=128 → OK
# ------------------------------------------------------------------------------------

import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F
import math
from torch.utils.checkpoint import checkpoint as grad_checkpoint

try:
    from timm.models import register_model
except ImportError:
    from timm.models.registry import register_model

try:
    from .registry import register_pip_model
except ImportError:
    def register_pip_model(cls): return cls

from timm.models.layers import DropPath, trunc_normal_
from timm.models.vision_transformer import Mlp
from einops import rearrange

try:
    from causal_conv1d import causal_conv1d_fn
except ImportError:
    causal_conv1d_fn = None

try:
    from mamba_ssm.ops.triton.layernorm_gated import RMSNorm as RMSNormGated
except ImportError:
    RMSNormGated = None

from mamba_ssm.ops.triton.ssd_combined import (
    mamba_chunk_scan_combined,
    mamba_split_conv1d_scan_combined,
)


# ------------------------------------------------------------------------------------
# 3D PatchEmbed — matches 2D: Conv→BN→ReLU→Conv→BN→ReLU, spatial /4, T preserved
# ------------------------------------------------------------------------------------

class PatchEmbed3D(nn.Module):
    def __init__(self, in_chans: int = 3, in_dim: int = 32, dim: int = 80):
        super().__init__()
        self.conv_down = nn.Sequential(
            nn.Conv3d(in_chans, in_dim, kernel_size=(3, 3, 3),
                      stride=(1, 2, 2), padding=(1, 1, 1), bias=False),
            nn.BatchNorm3d(in_dim, eps=1e-4),
            nn.ReLU(inplace=True),
            nn.Conv3d(in_dim, dim, kernel_size=(3, 3, 3),
                      stride=(1, 2, 2), padding=(1, 1, 1), bias=False),
            nn.BatchNorm3d(dim, eps=1e-4),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.conv_down(x)


# ------------------------------------------------------------------------------------
# 3D ConvBlock — matches 2D ConvBlock but with Conv3d
# ------------------------------------------------------------------------------------

class ConvBlock3D(nn.Module):
    def __init__(self, dim: int, drop_path: float = 0.,
                 layer_scale=None, kernel_size: int = 3, temporal_kernel: int = 3):
        super().__init__()
        pad = kernel_size // 2
        t_pad = temporal_kernel // 2
        self.conv1 = nn.Conv3d(dim, dim,
                               kernel_size=(temporal_kernel, kernel_size, kernel_size),
                               stride=1, padding=(t_pad, pad, pad))
        self.norm1 = nn.BatchNorm3d(dim, eps=1e-5)
        self.act1 = nn.GELU(approximate='tanh')
        self.conv2 = nn.Conv3d(dim, dim,
                               kernel_size=(temporal_kernel, kernel_size, kernel_size),
                               stride=1, padding=(t_pad, pad, pad))
        self.norm2 = nn.BatchNorm3d(dim, eps=1e-5)
        self.layer_scale = False
        if layer_scale is not None and type(layer_scale) in [int, float]:
            self.gamma = nn.Parameter(layer_scale * torch.ones(dim))
            self.layer_scale = True
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        residual = x
        x = self.act1(self.norm1(self.conv1(x)))
        x = self.norm2(self.conv2(x))
        if self.layer_scale:
            x = x * self.gamma.view(1, -1, 1, 1, 1)
        return residual + self.drop_path(x)


# ------------------------------------------------------------------------------------
# Downsample3D — spatial only, doubles channels
# ------------------------------------------------------------------------------------

class Downsample3D(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.reduction = nn.Conv3d(dim, dim * 2,
                                   kernel_size=(1, 3, 3),
                                   stride=(1, 2, 2),
                                   padding=(0, 1, 1), bias=False)

    def forward(self, x: Tensor) -> Tensor:
        return self.reduction(x)


# ------------------------------------------------------------------------------------
# Mamba2Mixer — Mamba2 SSM block for vision tokens
#
# Key params vs Mamba1:
#   expand=2   → d_inner = d_model × 2  (required: d_inner % headdim == 0)
#   headdim=64 → nheads = d_inner / headdim
#                stage3: d_model=320 → d_inner=640 → nheads=10
#                stage4: d_model=640 → d_inner=1280 → nheads=20
#   d_state=64 → 8× larger than Mamba1's 8, better long-range memory
#   d_conv=4   → slightly wider local conv than Mamba1's 3
#   chunk_size=256 → processes 5120-token sequences in 20 chunks (memory efficient)
#   A: (nheads,) scalar per head — simpler than Mamba1's (d_inner/2, d_state)
#   B, C: (B, L, ngroups, d_state) — grouped projection
# ------------------------------------------------------------------------------------

def _headdim3d(d_inner: int, cap: int = 64) -> int:
    """Largest divisor of d_inner not exceeding cap."""
    h = min(cap, d_inner)
    while h > 0 and d_inner % h != 0:
        h -= 1
    return h


class Mamba2Mixer(nn.Module):
    """
    Mamba2 SSM — lighter config compared to v1:
      d_state=16 (instead of 64), expand=1 (instead of 2), headdim=dynamic
    Closer to 2D config → less overfitting on IVF dataset.
    """
    def __init__(self, d_model, d_state=16, d_conv=3, expand=1,
                 headdim=None, ngroups=1,
                 A_init_range=(1, 16), dt_min=0.001, dt_max=0.1,
                 dt_init_floor=1e-4, dt_limit=(0.0, float("inf")),
                 conv_bias=True, bias=False,
                 chunk_size=64, use_mem_eff_path=False,
                 device=None, dtype=None, **kwargs):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        assert RMSNormGated is not None, "mamba_ssm.ops.triton.layernorm_gated not available"

        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.headdim = headdim if headdim is not None else _headdim3d(self.d_inner)
        self.ngroups = ngroups
        assert self.d_inner % self.headdim == 0, \
            f"d_inner={self.d_inner} must be divisible by headdim={self.headdim}"
        self.nheads = self.d_inner // self.headdim
        self.dt_limit = dt_limit
        self.chunk_size = chunk_size
        self.use_mem_eff_path = use_mem_eff_path

        # in_proj: [z, x, B, C, dt]
        d_in_proj = 2 * self.d_inner + 2 * self.ngroups * self.d_state + self.nheads
        self.in_proj = nn.Linear(self.d_model, d_in_proj, bias=bias, **factory_kwargs)

        # Conv over x+B+C channels
        conv_dim = self.d_inner + 2 * self.ngroups * self.d_state
        self.conv1d = nn.Conv1d(
            conv_dim, conv_dim, bias=conv_bias,
            kernel_size=d_conv, groups=conv_dim,
            padding=d_conv - 1, **factory_kwargs)

        self.act = nn.SiLU()

        # dt bias init
        dt = torch.exp(
            torch.rand(self.nheads, **factory_kwargs)
            * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        self.dt_bias = nn.Parameter(inv_dt)
        self.dt_bias._no_weight_decay = True

        # A: scalar per head
        assert A_init_range[0] > 0 and A_init_range[1] >= A_init_range[0]
        A = torch.empty(self.nheads, dtype=torch.float32,
                        device=device).uniform_(*A_init_range)
        self.A_log = nn.Parameter(torch.log(A).to(dtype=dtype))
        self.A_log._no_weight_decay = True

        # D skip
        self.D = nn.Parameter(torch.ones(self.nheads, device=device))
        self.D._no_weight_decay = True

        # Output norm + proj
        self.norm = RMSNormGated(self.d_inner, eps=1e-5,
                                 norm_before_gate=False, **factory_kwargs)
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)

    def forward(self, hidden_states):
        """(B, L, D) → (B, L, D)"""
        batch, seqlen, _ = hidden_states.shape
        A = -torch.exp(self.A_log)
        zxbcdt = self.in_proj(hidden_states)
        dt_limit_kwargs = {} if self.dt_limit == (0.0, float("inf")) \
            else dict(dt_limit=self.dt_limit)

        if self.use_mem_eff_path:
            out = mamba_split_conv1d_scan_combined(
                zxbcdt,
                rearrange(self.conv1d.weight, "d 1 w -> d w"),
                self.conv1d.bias,
                self.dt_bias,
                A,
                D=self.D,
                chunk_size=self.chunk_size,
                seq_idx=None,
                activation="swish",
                rmsnorm_weight=self.norm.weight,
                rmsnorm_eps=self.norm.eps,
                outproj_weight=self.out_proj.weight,
                outproj_bias=self.out_proj.bias,
                headdim=self.headdim,
                ngroups=self.ngroups,
                norm_before_gate=False,
                initial_states=None,
                **dt_limit_kwargs,
            )
        else:
            z, xBC, dt = torch.split(
                zxbcdt,
                [self.d_inner,
                 self.d_inner + 2 * self.ngroups * self.d_state,
                 self.nheads],
                dim=-1)
            dt = F.softplus(dt + self.dt_bias)
            if causal_conv1d_fn is None:
                xBC = self.act(
                    self.conv1d(xBC.transpose(1, 2)).transpose(1, 2)[:, :seqlen, :])
            else:
                xBC = causal_conv1d_fn(
                    x=xBC.transpose(1, 2),
                    weight=rearrange(self.conv1d.weight, "d 1 w -> d w"),
                    bias=self.conv1d.bias,
                    activation="swish",
                ).transpose(1, 2)
            x, B, C = torch.split(
                xBC,
                [self.d_inner,
                 self.ngroups * self.d_state,
                 self.ngroups * self.d_state],
                dim=-1)
            y = mamba_chunk_scan_combined(
                rearrange(x, "b l (h p) -> b l h p", p=self.headdim),
                dt, A,
                rearrange(B, "b l (g n) -> b l g n", g=self.ngroups),
                rearrange(C, "b l (g n) -> b l g n", g=self.ngroups),
                chunk_size=self.chunk_size,
                D=self.D, z=None, seq_idx=None,
                **dt_limit_kwargs,
            )
            y = rearrange(y, "b l h p -> b l (h p)")
            y = self.norm(y, z)
            out = self.out_proj(y)
        return out


# ------------------------------------------------------------------------------------
# Attention — same as 2D MambaVision
# ------------------------------------------------------------------------------------

class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_norm=False,
                 attn_drop=0., proj_drop=0., norm_layer=nn.LayerNorm):
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)
        x = F.scaled_dot_product_attention(q, k, v, dropout_p=self.attn_drop.p)
        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


# ------------------------------------------------------------------------------------
# Block3D — matches 2D Block exactly: either MambaVisionMixer or Attention
# ------------------------------------------------------------------------------------

class Block3D(nn.Module):
    def __init__(self, dim, num_heads, counter, transformer_blocks,
                 mlp_ratio=4., qkv_bias=False, qk_scale=False,
                 drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm,
                 layer_scale=None):
        super().__init__()
        self.norm1 = norm_layer(dim)
        if counter in transformer_blocks:
            self.mixer = Attention(
                dim, num_heads=num_heads, qkv_bias=qkv_bias,
                qk_norm=qk_scale, attn_drop=attn_drop, proj_drop=drop,
                norm_layer=norm_layer)
        else:
            # Mamba2 — following Mamba2Simple author config but adjusted for IVF:
            # - d_state=16 (author=64): lighter, suitable for dataset with ~320 embryos
            # - d_conv=4 (same as author): wider local conv
            # - expand=2, headdim=64: increased capacity compared to expand=1
            # - chunk_size=256 (author=256): suitable for 980-token sequences
            # - use_mem_eff_path=True if causal_conv1d is available
            # stage3: d_model=320 → d_inner=640 → nheads=10 (headdim=64)
            # stage4: d_model=640 → d_inner=1280 → nheads=20 (headdim=64)
            self.mixer = Mamba2Mixer(
                d_model=dim, d_state=16, d_conv=4,
                expand=2, headdim=64, ngroups=1, chunk_size=256,
                use_mem_eff_path=(causal_conv1d_fn is not None))

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio),
                        act_layer=act_layer, drop=drop)
        use_layer_scale = layer_scale is not None and type(layer_scale) in [int, float]
        self.gamma_1 = nn.Parameter(layer_scale * torch.ones(dim)) if use_layer_scale else 1
        self.gamma_2 = nn.Parameter(layer_scale * torch.ones(dim)) if use_layer_scale else 1

    def forward(self, x):
        x = x + self.drop_path(self.gamma_1 * self.mixer(self.norm1(x)))
        x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
        return x


# ------------------------------------------------------------------------------------
# MambaVisionLayer3D — Conv stages (Stage 1 & 2)
# ------------------------------------------------------------------------------------

class MambaVisionLayer3D_Conv(nn.Module):
    def __init__(self, dim, depth, drop_path, downsample=True,
                 layer_scale_conv=None, temporal_kernel=3):
        super().__init__()
        self.blocks = nn.ModuleList([
            ConvBlock3D(dim=dim,
                        drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                        layer_scale=layer_scale_conv,
                        temporal_kernel=temporal_kernel)
            for i in range(depth)
        ])
        self.downsample = Downsample3D(dim) if downsample else None

    def forward(self, x: Tensor) -> Tensor:
        for blk in self.blocks:
            x = blk(x)
        if self.downsample is not None:
            x = self.downsample(x)
        return x


# ------------------------------------------------------------------------------------
# MambaVisionLayer3D_Mamba — Joint Space-Time (Stage 3 & 4)
#
# Method 2: Concatenate T×H×W tokens into a long sequence, Mamba2/Attention processes everything.
#
# Token order: [frame0_tok0..tok(HW-1), frame1_tok0..tok(HW-1), ..., frame4_tok(HW-1)]
#   → Mamba2 causal scan: processes frame0 first → frame1 → ... → frame4
#     Mamba2 naturally learns temporal order because SSM is a causal sequence model
#   → Attention: sees all T×H×W tokens at once → joint spatial-temporal
#
# Sequence length with 224×224 input:
#   Stage 3: T×H×W = 5×14×14 = 980 tokens, chunk_size=128 → 7.6 chunks ✓
#   Stage 4: T×H×W = 5×7×7   = 245 tokens, chunk_size=128 → 1.9 chunks ✓
#
# Why this is true 3D:
#   Mamba2 and Attention ACTUALLY see all 5 frames simultaneously in one sequence
#   → Not "2D × 5 + aggregation"
#   → Mamba2 learns: "token (i,j) of frame t depends on all previous tokens
#     (both spatial and temporal)" — this is the core of SSM
# ------------------------------------------------------------------------------------

class MambaVisionLayer3D_Mamba(nn.Module):
    def __init__(self, dim, depth, num_heads,
                 downsample=True, mlp_ratio=4., qkv_bias=True, qk_scale=None,
                 drop=0., attn_drop=0., drop_path=0.,
                 layer_scale=None, transformer_blocks=None,
                 use_grad_ckpt=True):
        super().__init__()
        if transformer_blocks is None:
            transformer_blocks = []
        self.use_grad_ckpt = use_grad_ckpt

        # Blocks processing joint T×H×W sequence
        # Input: (B, T*H*W, C) — all tokens from 5 frames concatenated
        # Mamba2: causal scan in order frame0→frame1→...→frame4
        # Attention: sees entire spatial-temporal context
        self.blocks = nn.ModuleList([
            Block3D(dim=dim, counter=i, transformer_blocks=transformer_blocks,
                    num_heads=num_heads, mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias, qk_scale=qk_scale,
                    drop=drop, attn_drop=attn_drop,
                    drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                    layer_scale=layer_scale)
            for i in range(depth)
        ])

        # Temporal Positional Encoding — learnable, shape (1, T, 1, 1, dim)
        # Purpose: model knows which token belongs to which frame (frame 0 vs frame 4)
        # Spatial PE not needed because Conv stages 1-2 already learned spatial locality
        # Only needs T=5 vectors → very lightweight (5 × dim params)
        self.temporal_pe = nn.Parameter(torch.zeros(1, 5, 1, 1, dim))
        # Small init to avoid noise in features from conv stages
        nn.init.trunc_normal_(self.temporal_pe, std=0.02)

        self.downsample = Downsample3D(dim) if downsample else None

    def forward(self, x: Tensor) -> Tensor:
        B, C, T, H, W = x.shape

        # Add Temporal PE before flattening
        # x: (B, C, T, H, W) → permute → (B, T, H, W, C)
        # temporal_pe: (1, T, 1, 1, C) broadcast over B, H, W
        x = x.permute(0, 2, 3, 4, 1).contiguous()   # (B, T, H, W, C)
        x = x + self.temporal_pe[:, :T, :, :, :]     # broadcast: model knows which frame it is

        # Flatten T×H×W → 1 sequence in temporal order
        # Order: frame0_all_tokens, frame1_all_tokens, ..., frame4_all_tokens
        x = x.reshape(B, T * H * W, C)               # (B, T*H*W, C)

        for blk in self.blocks:
            if self.use_grad_ckpt and self.training:
                x = grad_checkpoint(blk, x, use_reentrant=False)
            else:
                x = blk(x)

        # Restore: (B, T*H*W, C) → (B, C, T, H, W)
        x = x.reshape(B, T, H, W, C)                 # (B, T, H, W, C)
        x = x.permute(0, 4, 1, 2, 3).contiguous()    # (B, C, T, H, W)

        if self.downsample is not None:
            x = self.downsample(x)
        return x


# ------------------------------------------------------------------------------------
# Main Model — follows 2D MambaVision T2 exactly
#
# 2D T2 config: depths=[1,3,11,4], dim=80, in_dim=32, num_heads=[2,4,8,16]
#               mlp_ratio=4, drop_path=0.2
#               Stage 0,1 = ConvBlock, Stage 2,3 = Block (Mamba+Attention)
#               transformer_blocks: half Mamba, half Attention per stage
#               Head: BN → AdaptiveAvgPool → Linear
#
# 3D adaptation (512×512 input):
#   patch_embed  → (B, 80,  T, 128, 128)
#   level_0 down → (B, 160, T,  64,  64)
#   level_1 down → (B, 320, T,  32,  32)   ← global Mamba on 32×32×T tokens
#   level_2 down → (B, 640, T,  16,  16)   ← global Mamba on 16×16×T tokens
#   level_3      → (B, 640, T,  16,  16)
#   temporal mean→ (B, 640,     16,  16)
#   BN→GAP→head → (B, num_classes)
# ------------------------------------------------------------------------------------

@register_pip_model
class Mamba2VisionMorph_Model(nn.Module):
    """
    3D MambaVision T2 for IVF — simplified to match 2D architecture.

    Input:  (B, T, H, W), (B, T, 3, H, W), or (B, 3, T, H, W)
    Output: (B, num_classes)
    """

    def __init__(self, num_classes: int = 9, num_frames: int = 5,
                 drop_path_rate: float = 0.2, drop_rate: float = 0.,
                 attn_drop_rate: float = 0., **kwargs):
        super().__init__()

        if drop_path_rate is None:
            drop_path_rate = 0.0

        # ── Config — same as 2D MambaVision T2 ──────────────────────────────
        depths     = [1, 3, 11, 4]
        num_heads  = [2, 4, 8, 16]
        dim        = 80
        in_dim     = 32
        mlp_ratio  = 4
        self.T     = num_frames
        num_features = int(dim * 2 ** (len(depths) - 1))  # 640

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

        # ── Stem ─────────────────────────────────────────────────────────────
        self.patch_embed = PatchEmbed3D(in_chans=3, in_dim=in_dim, dim=dim)

        # ── Stage 1: ConvBlock3D ×1 + Downsample ────────────────────────────
        # (B, 80, T, 56, 56) → (B, 160, T, 28, 28)
        self.level_0 = MambaVisionLayer3D_Conv(
            dim=dim, depth=depths[0],
            drop_path=dpr[0:sum(depths[:1])],
            downsample=True, temporal_kernel=3,
        )

        # ── Stage 2: ConvBlock3D ×3 + Downsample ────────────────────────────
        # (B, 160, T, 28, 28) → (B, 320, T, 14, 14)
        self.level_1 = MambaVisionLayer3D_Conv(
            dim=dim * 2, depth=depths[1],
            drop_path=dpr[sum(depths[:1]):sum(depths[:2])],
            downsample=True, temporal_kernel=3,
        )

        # ── Stage 3: Mamba/Attn ×11 + Downsample ────────────────────────────
        # SimAM3D removed — computing mean/var on T×H×W doesn't match original SimAM meaning
        # (Original SimAM only computes on spatial H×W, no temporal)
        self.level_2 = MambaVisionLayer3D_Mamba(
            dim=dim * 4, depth=depths[2], num_heads=num_heads[2],
            downsample=True,
            mlp_ratio=mlp_ratio,
            drop=drop_rate, attn_drop=attn_drop_rate,
            drop_path=dpr[sum(depths[:2]):sum(depths[:3])],
            transformer_blocks=list(range(depths[2] // 2 + 1, depths[2])),
            use_grad_ckpt=True,
        )

        # ── Stage 4: Mamba/Attn ×4 + No Downsample ──────────────────────────
        # SimAM3D removed
        self.level_3 = MambaVisionLayer3D_Mamba(
            dim=dim * 8, depth=depths[3], num_heads=num_heads[3],
            downsample=False,
            mlp_ratio=mlp_ratio,
            drop=drop_rate, attn_drop=attn_drop_rate,
            drop_path=dpr[sum(depths[:3]):sum(depths[:4])],
            transformer_blocks=list(range(depths[3] // 2, depths[3])),
            use_grad_ckpt=True,
        )

        # ── Head — BN → GAP → Linear ────────────────────────────────────────
        # Temporal Fusion: mean pool T frames
        # Lý do dùng mean thay vì learned weights:
        #   - Temporal Mamba2 trong level_2/level_3 đã học đầy đủ temporal dynamics
        #     tại từng spatial position (B*H*W, T, C)
        #   - Sau temporal pass, mỗi frame đã "biết" về các frame khác
        #   - Mean pool là đủ để aggregate — không cần bias về frame cuối
        #     vì thông tin đã được propagate qua Mamba2 causal scan
        # (Không cần temporal_pos_enc hay temporal_weights phức tạp)

        self.norm    = nn.BatchNorm2d(num_features)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.head    = nn.Linear(num_features, num_classes) if num_classes > 0 else nn.Identity()

        self.apply(self._init_weights)

    # -------------------------------------------------------------------------
    # Grad-CAM target layer property
    # -------------------------------------------------------------------------
    @property
    def gradcam_target_layer(self):
        """
        Target layer for Grad-CAM:
        - Stage 5: norm (BatchNorm2d) - feature map 7x7 before GAP
        - Stage 4: level_3.blocks (Attention blocks in joint layer)
        Returns list of nn.Modules for GradCAM to hook into.
        """
        # Primary target: norm (Stage 5) - feature map 7x7
        targets = [self.norm]
        return targets

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm3d)):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)

    def forward_features(self, x: Tensor) -> Tensor:
        """Returns features before head — mean pool T frames.
        Used for classification inference."""
        # Input normalization (giống forward)
        if x.dim() == 4:
            x = x.unsqueeze(1).expand(-1, 3, -1, -1, -1)
        elif x.dim() == 5:
            if x.shape[1] == 3 and x.shape[2] == self.T:
                pass
            elif x.shape[1] == self.T and x.shape[2] == 3:
                x = x.permute(0, 2, 1, 3, 4).contiguous()
            elif x.shape[1] == 1:
                x = x.expand(-1, 3, -1, -1, -1)
            else:
                x = x[:, :3, ...]

        x = self.patch_embed(x)
        x = self.level_0(x)
        x = self.level_1(x)
        x = self.level_2(x)
        x = self.level_3(x)
        x = x.mean(dim=2)        # (B, 640, H, W)
        x = self.norm(x)
        x = self.avgpool(x)      # (B, 640, 1, 1)
        x = torch.flatten(x, 1) # (B, 640)
        return x

    def forward_last_frame_feat(self, x: Tensor) -> Tensor:
        """Returns features of the LAST frame (t) in the clip — used for Phase 2.

        Different from forward_features() which uses mean(T frames):
          - mean pool: feat_t ≈ feat_{t+1} (cosine sim ~0.99) → weak temporal signal
          - last frame: feat_t represents specific frame t → stronger temporal signal

        Clip [t-4, t-3, t-2, t-1, t] → backbone → (B, 640, T, H, W)
        → take the last frame [:, :, -1, :, :] → BN → GAP → (B, 640)

        Mamba2 in level_2/level_3 processed causal scan over T frames,
        so the features of the last frame have "seen" the context from t-4..t-1.
        → Causal, no future leakage.
        """
        if x.dim() == 4:
            x = x.unsqueeze(1).expand(-1, 3, -1, -1, -1)
        elif x.dim() == 5:
            if x.shape[1] == 3 and x.shape[2] == self.T:
                pass
            elif x.shape[1] == self.T and x.shape[2] == 3:
                x = x.permute(0, 2, 1, 3, 4).contiguous()
            elif x.shape[1] == 1:
                x = x.expand(-1, 3, -1, -1, -1)
            else:
                x = x[:, :3, ...]

        x = self.patch_embed(x)
        x = self.level_0(x)
        x = self.level_1(x)
        x = self.level_2(x)
        x = self.level_3(x)          # (B, 640, T, H, W)
        x = x[:, :, -1, :, :]       # lấy frame cuối → (B, 640, H, W)
        x = self.norm(x)             # BN2d
        x = self.avgpool(x)          # (B, 640, 1, 1)
        x = torch.flatten(x, 1)      # (B, 640)
        return x

    def forward(self, x: Tensor):
        # ── Input normalization ──────────────────────────────────────────────
        if x.dim() == 4:
            # (B, T, H, W) → (B, 3, T, H, W)
            x = x.unsqueeze(1).expand(-1, 3, -1, -1, -1)
        elif x.dim() == 5:
            if x.shape[1] == 3 and x.shape[2] == self.T:
                pass  # (B, 3, T, H, W) — correct
            elif x.shape[1] == self.T and x.shape[2] == 3:
                x = x.permute(0, 2, 1, 3, 4).contiguous()  # → (B, 3, T, H, W)
            elif x.shape[1] == 1:
                x = x.expand(-1, 3, -1, -1, -1)
            else:
                x = x[:, :3, ...]

        # ── Forward ──────────────────────────────────────────────────────────
        x = self.patch_embed(x)     # (B, 80,  T, 56, 56)
        x = self.level_0(x)         # (B, 160, T, 28, 28)
        x = self.level_1(x)         # (B, 320, T, 14, 14)
        x = self.level_2(x)         # (B, 640, T,  7,  7)
        x = self.level_3(x)         # (B, 640, T,  7,  7)

        # ── Temporal Fusion: mean pool T frames ─────────────────────────────
        # x: (B, 640, T, 7, 7)
        # Temporal Mamba2 in level_2/level_3 has processed (B*H*W, T, C)
        # → each frame "knows" about other frames via causal scan
        # → mean pool is enough to aggregate T frames
        x = x.mean(dim=2)            # (B, 640, 7, 7)

        # ── Head: BN → GAP → Linear ─────────────────────────────────────────
        x = self.norm(x)             # BN2d
        x = self.avgpool(x)          # (B, 640, 1, 1)
        x = torch.flatten(x, 1)      # (B, 640)
        x = self.head(x)             # (B, num_classes)
        return x


@register_model
def Mamba2VisionMorph(pretrained: bool = False, **kwargs):
    model = Mamba2VisionMorph_Model(**kwargs)
    return model


# ------------------------------------------------------------------------------------
# Test — run directly: python training/models/ivf_3d_mamba2.py
# ------------------------------------------------------------------------------------
if __name__ == "__main__":
    import sys

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # Diagnose causal_conv1d
    try:
        from causal_conv1d import causal_conv1d_fn as _test_ccf
        print(f"causal_conv1d: INSTALLED")
    except ImportError:
        print("causal_conv1d: NOT installed — fallback path (OK)")

    T = 5
    errors = []

    def check(name, cond, detail=""):
        mark = "✅" if cond else "❌"
        print(f"  {mark} {name}" + (f"  [{detail}]" if detail else ""))
        if not cond:
            errors.append(name)

    # ── 1. Initialization ──────────────────────────────────────────────────
    print("\n[1] Initializing model...")
    model = Mamba2VisionMorph(
        num_classes=7, num_frames=T,
        drop_path_rate=0.0, drop_rate=0.0,
    ).to(device)
    model.eval()

    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {total_params / 1e6:.2f}M")
    check("Params > 30M", total_params > 30e6, f"{total_params/1e6:.1f}M")
    check("Params < 80M (not too heavy)", total_params < 80e6, f"{total_params/1e6:.1f}M")

    # ── 2. Structure Check ──────────────────────────────────────────────────
    print("\n[2] Checking structure...")
    check("norm is BatchNorm2d", isinstance(model.norm, nn.BatchNorm2d),
          type(model.norm).__name__)
    check("head output = 7", model.head.out_features == 7)
    check("head input = 640", model.head.in_features == 640)
    check("norm là BatchNorm2d", isinstance(model.norm, nn.BatchNorm2d),
          type(model.norm).__name__)
    check("head output = 7", model.head.out_features == 7)
    check("head input = 640", model.head.in_features == 640)
    check("No temporal_pos_enc/temporal_weights (removed)",
          not hasattr(model, 'temporal_pos_enc') and not hasattr(model, 'temporal_weights'))
    check("No BatchNorm1d",
          not any(isinstance(m, nn.BatchNorm1d) for m in model.modules()))
    check("No MultiheadAttention",
          not any(isinstance(m, nn.MultiheadAttention) for m in model.modules()))
    check("No legacy causal_mask buffer", not hasattr(model, 'causal_mask'))

    # ── 3. Mamba2Mixer config ─────────────────────────────────────────────────
    print("\n[3] Mamba2Mixer config...")
    mamba_mods = [(n, m) for n, m in model.named_modules() if isinstance(m, Mamba2Mixer)]
    print(f"  Total Mamba2Mixer: {len(mamba_mods)}")
    for name, mod in mamba_mods[:2]:
        print(f"  [{name}] d_model={mod.d_model}, d_state={mod.d_state}, "
              f"expand={mod.expand}, d_inner={mod.d_inner}, "
              f"nheads={mod.nheads}, headdim={mod.headdim}")
    for name, mod in mamba_mods:
        check(f"[{name}] d_state=16", mod.d_state == 16, f"got {mod.d_state}")
        check(f"[{name}] expand=1", mod.expand == 1, f"got {mod.expand}")
        check(f"[{name}] d_inner % headdim == 0",
              mod.d_inner % mod.headdim == 0,
              f"{mod.d_inner} % {mod.headdim}")

    # ── 4. Joint layers ───────────────────────────────────────────────────────
    print("\n[4] Joint Space-Time layers...")
    fact_layers = [(n, m) for n, m in model.named_modules()
                   if isinstance(m, MambaVisionLayer3D_Mamba)]
    check("Correctly has 2 Joint layers (level_2, level_3)", len(fact_layers) == 2,
          f"got {len(fact_layers)}")
    for name, layer in fact_layers:
        check(f"[{name}] has blocks (joint T*H*W)",
              hasattr(layer, 'blocks') and len(layer.blocks) > 0,
              f"{len(layer.blocks)} blocks")
        check(f"[{name}] NO separate spatial_blocks (merged into blocks)",
              not hasattr(layer, 'spatial_blocks'))
        check(f"[{name}] NO separate temporal_mamba (merged into blocks)",
              not hasattr(layer, 'temporal_mamba'))
        check(f"[{name}] has temporal_pe (1, 5, 1, 1, dim)",
              hasattr(layer, 'temporal_pe') and layer.temporal_pe.shape[1] == T,
              str(layer.temporal_pe.shape))
        print(f"  [{name}] sequence length = T*H*W = 5*H*W (joint)")

    # ── 5. Forward pass — 3 input formats ────────────────────────────────────
    print("\n[5] Forward pass (224×224)...")
    with torch.no_grad():
        x1 = torch.randn(2, T, 224, 224).to(device)
        out1 = model(x1)
        print(f"  (B,T,H,W)   224 -> {out1.shape}")
        check("(B,T,H,W) → (2,7)", out1.shape == (2, 7), str(out1.shape))

        x2 = torch.randn(2, T, 3, 224, 224).to(device)
        out2 = model(x2)
        print(f"  (B,T,3,H,W) 224 -> {out2.shape}")
        check("(B,T,3,H,W) → (2,7)", out2.shape == (2, 7), str(out2.shape))

        x3 = torch.randn(2, 3, T, 224, 224).to(device)
        out3 = model(x3)
        print(f"  (B,3,T,H,W) 224 -> {out3.shape}")
        check("(B,3,T,H,W) → (2,7)", out3.shape == (2, 7), str(out3.shape))

    # ── 6. Spatial flow shapes ────────────────────────────────────────────────
    print("\n[6] Spatial flow shapes (hook)...")
    shapes = {}
    hooks = []
    for layer_name in ['patch_embed', 'level_0', 'level_1', 'level_2', 'level_3']:
        layer = getattr(model, layer_name)
        def _hook(m, inp, out, n=layer_name):
            if isinstance(out, torch.Tensor):
                shapes[n] = tuple(out.shape)
        hooks.append(layer.register_forward_hook(_hook))

    with torch.no_grad():
        _ = model(torch.randn(1, T, 3, 224, 224).to(device))
    for h in hooks:
        h.remove()

    expected_shapes = {
        'patch_embed': (1, 80,  T, 56, 56),
        'level_0':     (1, 160, T, 28, 28),
        'level_1':     (1, 320, T, 14, 14),
        'level_2':     (1, 640, T,  7,  7),
        'level_3':     (1, 640, T,  7,  7),
    }
    for name, exp in expected_shapes.items():
        got = shapes.get(name)
        print(f"  {name:<12}: {str(exp):<28} got {got}")
        check(f"{name} shape is correct", got == exp, f"exp={exp}, got={got}")

    # ── 7. Gradient flow ─────────────────────────────────────────────────────
    print("\n[7] Gradient flow...")
    model.train()
    x = torch.randn(1, T, 3, 224, 224).to(device)
    out = model(x)
    out.sum().backward()
    # Check gradient flow through blocks of joint layer
    for name, layer in fact_layers:
        first_block = layer.blocks[0]
        has_grad = (first_block.norm1.weight.grad is not None and
                    first_block.norm1.weight.grad.abs().sum().item() > 0)
        check(f"[{name}] blocks[0] received gradient", has_grad)
    model.eval()

    # ── 8. Temporal sensitivity ───────────────────────────────────────────────
    print("\n[8] Temporal sensitivity (order matters)...")
    with torch.no_grad():
        base = torch.randn(1, 3, 224, 224).to(device)
        clip_fwd = torch.stack([base * (0.8 + 0.1*i) for i in range(T)], dim=1)
        clip_bwd = torch.stack([base * (0.8 + 0.1*(T-1-i)) for i in range(T)], dim=1)
        diff = (model(clip_fwd) - model(clip_bwd)).abs().max().item()
        check("Temporal order affects output (diff > 1e-4)", diff > 1e-4,
              f"max_diff={diff:.6f}")

        # Deterministic
        clip = torch.randn(1, T, 3, 224, 224).to(device)
        diff_same = (model(clip) - model(clip)).abs().max().item()
        check("Deterministic (same input → same output)", diff_same < 1e-5,
              f"diff={diff_same:.2e}")

    # ── 9. Kiểm tra joint sequence length ────────────────────────────────────
    print("\n[9] Joint sequence length...")
    shapes_joint = {}
    hooks_j = []
    for name, layer in fact_layers:
        def _hook_joint(m, inp, out, n=name):
            # inp[0] is forward input — after reshape it is (B, T*H*W, C)
            pass
        hooks_j.append(layer.blocks[0].register_forward_hook(
            lambda m, inp, out, n=name: shapes_joint.update({n: tuple(inp[0].shape)})
        ))
    with torch.no_grad():
        _ = model(torch.randn(1, T, 3, 224, 224).to(device))
    for h in hooks_j:
        h.remove()
    for name, shape in shapes_joint.items():
        B_, L_, C_ = shape
        expected_L_stage3 = T * 14 * 14  # 980
        expected_L_stage4 = T * 7 * 7    # 245
        print(f"  [{name}] input shape: {shape}  (L={L_})")
        if 'level_2' in name:
            check(f"[{name}] L = T*14*14 = {expected_L_stage3}",
                  L_ == expected_L_stage3, f"got {L_}")
        elif 'level_3' in name:
            check(f"[{name}] L = T*7*7 = {expected_L_stage4}",
                  L_ == expected_L_stage4, f"got {L_}")

    # ── Results ─────────────────────────────────────────────────────────────
    print(f"\n{'='*55}")
    if errors:
        print(f"❌ {len(errors)} FAILED:")
        for e in errors:
            print(f"   - {e}")
        sys.exit(1)
    else:
        print(f"✅ All tests PASSED!")
        print(f"   Model : Mamba2VisionMorph (Joint Space-Time)")
        print(f"   Params: {total_params/1e6:.2f}M")
        print(f"   Input : (B,T,H,W) | (B,T,3,H,W) | (B,3,T,H,W)")
        print(f"   Output: (B, 7)")
        print(f"   Stage3: (B, 5*14*14=980, 320) — Mamba2+Attn sees 5 frames")
        print(f"   Stage4: (B, 5*7*7=245,   640) — Mamba2+Attn sees 5 frames")
