"""
ivf_phase2_mamba2MIXatteniton.py — Phase 2: MambaVision-style Mamba2 + Attention
==================================================================================
Architecture: instead of Mamba2 (streaming) + CausalAttnBuffer (separate KV-cache),
this model processes the **entire history sequence** (sliding window W frames) via:

    feat(640) → input_proj → PE
    → accumulate into sliding window buffer (max W = 128 frames)
    → [Mamba2 × N_MAMBA](buffer sequence B, L, D)   # global context
    → [Attention × N_ATTN](same sequence B, L, D)    # local refinement
    → last-token → head → logits

Different from EmbryoTemporalNet (ivf_phase2_cnn.py):
  - Mamba2 here processes the ENTIRE sequence (batch mode), no streaming
  - Attention attends on the OUTPUT of Mamba2 (same sequence),
    following MambaVision principles [18]
  - "Cache" = sliding window buffer (B, L, D), not SSM state

Interface remains fully compatible with train_phase2_3d.py:
  make_initial_hidden(B, device) → [None]   (empty buffer)
  forward(feat, _, cache, frame_idx) → (logits, cache_new, dummy_attn)
  _detach_cache compatible: cache is list[tensor | None]
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

try:
    from mamba_ssm.ops.triton.ssd_combined import mamba_chunk_scan_combined
    from mamba_ssm.ops.triton.layernorm_gated import RMSNorm as RMSNormGated
    _MAMBA2_AVAILABLE = True
except ImportError:
    _MAMBA2_AVAILABLE = False
    RMSNormGated = None


# ---------------------------------------------------------------------------
# OrdinalProgressionLoss — identical to ivf_phase2_cnn.py (required by train script)
# ---------------------------------------------------------------------------

class OrdinalProgressionLoss(nn.Module):
    """CE + backward-regression penalty + forward-jump penalty."""

    def __init__(self, num_classes=7, alpha=0.3, forward_alpha=0.2,
                 consistency_alpha=0.0, max_forward_jump=2,
                 label_smoothing=0.1, class_weight=None):
        super().__init__()
        self.num_classes      = num_classes
        self.alpha            = alpha
        self.forward_alpha    = forward_alpha
        self.max_forward_jump = max_forward_jump
        self.label_smoothing  = label_smoothing

        if class_weight is not None:
            self.register_buffer('class_weight', class_weight.float())
        else:
            self.class_weight = None

        dist_back = torch.zeros(num_classes, num_classes)
        for i in range(num_classes):
            for j in range(num_classes):
                if j < i:
                    dist_back[i, j] = float((i - j) ** 2)
        self.register_buffer('ordinal_dist', dist_back)

        dist_fwd = torch.zeros(num_classes, num_classes)
        for i in range(num_classes):
            for j in range(num_classes):
                excess = j - i - max_forward_jump
                if excess > 0:
                    dist_fwd[i, j] = float(excess ** 2)
        self.register_buffer('forward_dist', dist_fwd)

    def forward(self, logits, targets, prev_pred=None):
        ce_loss = F.cross_entropy(logits, targets,
                                  weight=self.class_weight,
                                  label_smoothing=self.label_smoothing)
        probs = torch.softmax(logits, dim=-1)
        total_loss = ce_loss

        if self.alpha > 0.0:
            dist_row   = self.ordinal_dist[targets]
            back_pen   = (probs * dist_row).sum(dim=-1)
            if self.class_weight is not None:
                back_pen = back_pen * self.class_weight[targets]
            total_loss = total_loss + self.alpha * back_pen.mean()

        if self.forward_alpha > 0.0:
            fwd_row  = self.forward_dist[targets]
            fwd_pen  = (probs * fwd_row).sum(dim=-1)
            if self.class_weight is not None:
                fwd_pen = fwd_pen * self.class_weight[targets]
            total_loss = total_loss + self.forward_alpha * fwd_pen.mean()

        return total_loss


# ---------------------------------------------------------------------------
# Mamba2SeqBlock — Mamba2 processes entire sequence (B, L, D) batch mode
# No streaming state — similar to Stage 3/4 in ivf_3d_mamba2.py
# ---------------------------------------------------------------------------

class Mamba2SeqBlock(nn.Module):
    """
    Mamba2 block for sequence (B, L, D) — batch mode, no streaming.
    Pre-Norm → Mamba2 SSM → residual → Pre-Norm → MLP → residual.

    Config matches EmbryoTemporalNet for easy comparison:
      d_state=64, expand=1, headdim=64, d_conv=4, chunk_size=128
    """

    def __init__(self, d_model=640, d_state=64, d_conv=4,
                 expand=1, headdim=64, mlp_ratio=1,
                 dropout=0.1, chunk_size=128):
        super().__init__()
        assert _MAMBA2_AVAILABLE, "cần mamba_ssm >= 2.0"
        self.d_model    = d_model
        self.d_state    = d_state
        self.d_conv     = d_conv
        self.d_inner    = int(expand * d_model)
        self.headdim    = headdim
        self.ngroups    = 1
        self.chunk_size = chunk_size
        self.nheads     = self.d_inner // headdim

        self.norm   = nn.LayerNorm(d_model)
        d_in_proj   = 2 * self.d_inner + 2 * self.ngroups * d_state + self.nheads
        self.in_proj = nn.Linear(d_model, d_in_proj, bias=False)

        conv_dim   = self.d_inner + 2 * self.ngroups * d_state
        self.conv1d = nn.Conv1d(conv_dim, conv_dim, bias=True,
                                kernel_size=d_conv, groups=conv_dim,
                                padding=d_conv - 1)
        self.act = nn.SiLU()

        dt = torch.exp(
            torch.rand(self.nheads) * (math.log(0.1) - math.log(0.001)) + math.log(0.001)
        ).clamp(min=1e-4)
        self.dt_bias = nn.Parameter(dt + torch.log(-torch.expm1(-dt)))
        self.dt_bias._no_weight_decay = True

        A = torch.empty(self.nheads).uniform_(1, 16)
        self.A_log = nn.Parameter(torch.log(A))
        self.A_log._no_weight_decay = True
        self.D  = nn.Parameter(torch.ones(self.nheads))
        self.D._no_weight_decay = True

        self.out_norm = RMSNormGated(self.d_inner, eps=1e-5, norm_before_gate=False)
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)
        self.drop     = nn.Dropout(dropout)

        # MLP
        hidden = int(d_model * mlp_ratio)
        self.norm2 = nn.LayerNorm(d_model)
        self.mlp   = nn.Sequential(
            nn.Linear(d_model, hidden), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(hidden, d_model), nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, L, D) → (B, L, D)"""
        x = x.contiguous()
        B, L, D = x.shape
        residual = x

        xn     = self.norm(x)
        zxbcdt = self.in_proj(xn)
        z, xBC, dt = torch.split(
            zxbcdt,
            [self.d_inner, self.d_inner + 2 * self.ngroups * self.d_state, self.nheads],
            dim=-1,
        )
        dt  = F.softplus(dt + self.dt_bias)
        xBC = self.act(self.conv1d(xBC.transpose(1, 2)).transpose(1, 2)[:, :L, :])

        x_ssm, B_ssm, C_ssm = torch.split(
            xBC,
            [self.d_inner, self.ngroups * self.d_state, self.ngroups * self.d_state],
            dim=-1,
        )
        A = -torch.exp(self.A_log)

        # Batch-mode SSM: no initial_states, no return_final_states
        y, _ = mamba_chunk_scan_combined(
            rearrange(x_ssm, "b l (h p) -> b l h p", p=self.headdim),
            dt, A,
            rearrange(B_ssm, "b l (g n) -> b l g n", g=self.ngroups),
            rearrange(C_ssm, "b l (g n) -> b l g n", g=self.ngroups),
            chunk_size=self.chunk_size,
            D=self.D, z=None, seq_idx=None,
            initial_states=None, return_final_states=True,
        )
        y   = rearrange(y, "b l h p -> b l (h p)")
        y   = self.out_norm(y, z)
        out = self.drop(self.out_proj(y)) + residual

        # MLP
        out = out + self.mlp(self.norm2(out))
        return out


# ---------------------------------------------------------------------------
# AttentionSeqBlock — Standard self-attention (B, L, D) → (B, L, D)
# Attends on Mamba2 OUTPUT — following MambaVision principles
# ---------------------------------------------------------------------------

class AttentionSeqBlock(nn.Module):
    """
    Standard multi-head self-attention + MLP.
    Non-causal within window (causality enforced by buffer itself).
    """

    def __init__(self, d_model=640, num_heads=10, mlp_ratio=1, dropout=0.1):
        super().__init__()
        assert d_model % num_heads == 0
        self.num_heads = num_heads
        self.head_dim  = d_model // num_heads
        self.scale     = self.head_dim ** -0.5

        self.norm1 = nn.LayerNorm(d_model)
        self.qkv   = nn.Linear(d_model, d_model * 3, bias=False)
        self.proj  = nn.Linear(d_model, d_model, bias=False)
        self.drop  = nn.Dropout(dropout)
        self.last_attn_weights = None  # Store weights for visualization

        hidden = int(d_model * mlp_ratio)
        self.norm2 = nn.LayerNorm(d_model)
        self.mlp   = nn.Sequential(
            nn.Linear(d_model, hidden), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(hidden, d_model), nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, L, D) → (B, L, D)"""
        B, L, C = x.shape
        residual = x

        xn  = self.norm1(x)
        qkv = self.qkv(xn).reshape(B, L, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)

        # ── Extract attention weights for the last token ────────────────
        if not self.training:
            # Manually compute scores to get weights (B, H, L, L)
            with torch.no_grad():
                scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
                weights = torch.softmax(scores, dim=-1) # (B, H, L, L)
                # Take attention of last token (query -1) looking back
                # Mean over heads: (B, H, L) -> (B, L)
                self.last_attn_weights = weights[:, :, -1, :].mean(dim=1).detach()

        # Main flow uses Flash Attention (if available) for speed
        out = F.scaled_dot_product_attention(q, k, v, dropout_p=self.drop.p if self.training else 0.0)
        out = out.transpose(1, 2).reshape(B, L, C)
        out = self.proj(out) + residual

        # MLP
        out = out + self.mlp(self.norm2(out))
        return out


# ---------------------------------------------------------------------------
# EmbryoMambaAttn — Main Model
# ---------------------------------------------------------------------------

class EmbryoTemporalNet(nn.Module):
    """
    Phase 2 Sequential Refiner — MambaVision-style.

    At each step t:
      1. Append feat_t into sliding window buffer (B, L<=W, D)
      2. Run Mamba2 × N_MAMBA on the entire buffer → (B, L, D)
      3. Run Attention × N_ATTN on Mamba2 output → (B, L, D)
      4. Take the last token → Head → logits

    Cache = [buffer_tensor | None] — compatible with _detach_cache
    in train_phase2_3d.py (just .detach() is enough).
    """

    # ── Fixed Config (matches EmbryoTemporalNet for comparison) ─────────
    FEAT_DIM    = 640
    D_MODEL     = 640
    D_STATE     = 64
    D_CONV      = 4
    EXPAND      = 1
    HEADDIM     = 64
    N_MAMBA     = 2
    N_ATTN      = 2
    N_LAYERS    = 4       # = N_MAMBA + N_ATTN
    WINDOW_SIZE = 128     # max buffer length (frame)
    CHUNK_SIZE  = 128
    DROPOUT     = 0.1
    MAX_PE_LEN  = 2000
    NUM_CLASSES = 7
    NUM_HEADS   = 10      # D_MODEL // HEADDIM = 640 // 64

    def __init__(self, **kwargs):
        super().__init__()
        self.num_classes  = self.NUM_CLASSES
        self.d_model      = self.D_MODEL
        self.window_size  = self.WINDOW_SIZE  # backward compat attr

        # Input projection
        self.input_norm = nn.LayerNorm(self.FEAT_DIM)
        self.input_proj = nn.Linear(self.FEAT_DIM, self.D_MODEL, bias=False)

        # Sinusoidal PE
        pe  = torch.zeros(self.MAX_PE_LEN, self.D_MODEL)
        pos = torch.arange(0, self.MAX_PE_LEN, dtype=torch.float).unsqueeze(1)
        div = torch.exp(
            torch.arange(0, self.D_MODEL, 2, dtype=torch.float)
            * (-math.log(10000.0) / self.D_MODEL)
        )
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer('pe', pe)

        # Mamba2 blocks (process full sequence)
        self.mamba_blocks = nn.ModuleList([
            Mamba2SeqBlock(
                d_model=self.D_MODEL, d_state=self.D_STATE, d_conv=self.D_CONV,
                expand=self.EXPAND, headdim=self.HEADDIM, mlp_ratio=1,
                dropout=self.DROPOUT, chunk_size=self.CHUNK_SIZE,
            ) for _ in range(self.N_MAMBA)
        ])

        # Attention blocks (attend on Mamba2 output)
        self.attn_blocks = nn.ModuleList([
            AttentionSeqBlock(
                d_model=self.D_MODEL, num_heads=self.NUM_HEADS,
                mlp_ratio=1, dropout=self.DROPOUT,
            ) for _ in range(self.N_ATTN)
        ])

        # Head
        self.head = nn.Sequential(
            nn.LayerNorm(self.D_MODEL),
            nn.Dropout(self.DROPOUT),
            nn.Linear(self.D_MODEL, self.NUM_CLASSES),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def make_initial_hidden(self, batch_size: int, device) -> list:
        """Returns empty cache — buffer contains nothing."""
        return [None]  # [buffer_tensor | None]

    def forward(
        self,
        feat:      torch.Tensor,          # (B, 640)
        _unused1:  object = None,
        cache:     list   = None,         # [buffer (B,L,D) | None]
        frame_idx: object = None,         # int | list[int] | None
        _unused2:  object = None,
    ):
        """
        Returns:
            logits:    (B, 7)
            cache_new: [buffer_new (B, L_new, D)]
            dummy_attn: zeros (B, 1) — backward compat
        """
        if cache is None or len(cache) == 0:
            cache = [None]

        B = feat.shape[0]
        buf: torch.Tensor | None = cache[0]  # (B, L_prev, D) or None

        B = feat.shape[0]
        buf: torch.Tensor | None = cache[0]  # (B, L_prev, D) or None

        # ── Step 1: project feat → (B, D) + PE ──────────────────────────
        x = self.input_proj(self.input_norm(feat.float().contiguous()))  # (B, D)

        if frame_idx is not None:
            if isinstance(frame_idx, (list, torch.Tensor)):
                idxs = (frame_idx if isinstance(frame_idx, torch.Tensor)
                        else torch.tensor(frame_idx, dtype=torch.long))
                idxs = idxs.to(self.pe.device) % self.pe.shape[0]
                x = x + self.pe[idxs]       # (B, D)
            else:
                x = x + self.pe[int(frame_idx) % self.pe.shape[0]]

        x = x.unsqueeze(1)   # (B, 1, D) — token of current frame

        # ── Step 2: append into sliding window buffer ─────────────────────
        if buf is None:
            buf_new = x                                    # (B, 1, D)
        else:
            buf_new = torch.cat([buf, x], dim=1)           # (B, L+1, D)

        # Trim về WINDOW_SIZE
        if buf_new.shape[1] > self.WINDOW_SIZE:
            buf_new = buf_new[:, -self.WINDOW_SIZE:, :]    # (B, W, D)

        # ── Step 3: Mamba2 → Attention on the entire buffer ───────────────
        h = buf_new  # (B, L, D)
        for blk in self.mamba_blocks:
            h = blk(h)
        for blk in self.attn_blocks:
            h = blk(h)

        # ── Step 4: take the last token → head ────────────────────────────────
        logits = self.head(h[:, -1, :])   # (B, 7)

        # ── Extract average attention weights from blocks ───────────────
        attn_list = [blk.last_attn_weights for blk in self.attn_blocks if blk.last_attn_weights is not None]
        if attn_list:
            # (N_blocks, B, L) -> mean -> (B, L)
            attn_mean = torch.stack(attn_list).mean(dim=0)
            # Ensure sum = 1 (normalize)
            attn_mean = attn_mean / (attn_mean.sum(dim=-1, keepdim=True) + 1e-8)
        else:
            # Fallback if no data (e.g., during training)
            attn_mean = torch.zeros(B, h.shape[1], device=feat.device)
            attn_mean[:, -1] = 1.0  # default look at itself

        return logits, [buf_new], attn_mean




# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    if not _MAMBA2_AVAILABLE:
        print("⚠️  mamba_ssm not found — smoke test skipped.")
        exit(0)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model  = EmbryoTemporalNet().to(device)
    print(f"Params: {sum(p.numel() for p in model.parameters() if p.requires_grad)/1e3:.1f}K")

    cache = model.make_initial_hidden(2, device)
    for t in range(70):
        feat   = torch.randn(2, 640, device=device)
        logits, cache, _ = model(feat, None, cache, frame_idx=t)
        assert logits.shape == (2, 7)
        assert isinstance(cache, list) and len(cache) == 1
        if t in [0, 5, 32, 69]:
            print(f"  t={t:3d}: logits={tuple(logits.shape)}, buf={tuple(cache[0].shape)}")
    print("✅ Smoke test passed.")
