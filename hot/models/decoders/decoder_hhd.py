"""HybridHOT Decoder (HHD).

Multi-scale pixel decoder + depth / person-mask conditioned feature modulation
+ person-aware mask transformer decoder (one query per class), with optional
fusion of the encoder's spatial prior S_map and deep supervision.

Usage (selected in the config, built by ModelBuilder.build_decoder):

    MODEL:
      arch_decoder: "hhd"

    pred = decoder(x4, x3, x2, x1, x0, prior_logits,
                   person_mask, total_person, depth, S_map=S_map)
    # pred: (B, num_class, H/4, W/4) raw logits
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _sinusoidal_pos_2d(H: int, W: int, dim: int,
                       temperature: float = 10000.0,
                       device=None, dtype=torch.float32) -> torch.Tensor:
    """2D sinusoidal positional encoding. Returns (H*W, dim)."""
    assert dim % 4 == 0, "dim must be divisible by 4 for 2D sin-cos pos enc"
    half = dim // 2
    y = torch.arange(H, device=device, dtype=dtype).unsqueeze(1).expand(H, W).flatten()
    x = torch.arange(W, device=device, dtype=dtype).unsqueeze(0).expand(H, W).flatten()
    dim_t = torch.arange(half // 2, device=device, dtype=dtype) * 2.0
    div = temperature ** (dim_t / half)
    pe_y = torch.zeros(H * W, half, device=device, dtype=dtype)
    pe_x = torch.zeros(H * W, half, device=device, dtype=dtype)
    pe_y[:, 0::2] = torch.sin(y.unsqueeze(1) / div)
    pe_y[:, 1::2] = torch.cos(y.unsqueeze(1) / div)
    pe_x[:, 0::2] = torch.sin(x.unsqueeze(1) / div)
    pe_x[:, 1::2] = torch.cos(x.unsqueeze(1) / div)
    return torch.cat([pe_y, pe_x], dim=1)


class DropPath(nn.Module):
    def __init__(self, p: float = 0.0):
        super().__init__()
        self.p = p

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or self.p == 0.0:
            return x
        keep = 1.0 - self.p
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        mask = x.new_empty(shape).bernoulli_(keep).div_(keep)
        return x * mask


class PixelTransformerLayer(nn.Module):
    """Pre-norm Transformer encoder layer for the multi-scale pixel decoder."""

    def __init__(self, dim: int, n_heads: int,
                 ffn_ratio: int = 4, dropout: float = 0.1,
                 drop_path: float = 0.1):
        super().__init__()
        self.norm_attn = nn.LayerNorm(dim)
        self.self_attn = nn.MultiheadAttention(
            dim, n_heads, dropout=dropout, batch_first=True
        )
        self.drop_path_attn = DropPath(drop_path)

        self.norm_ffn = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, ffn_ratio * dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_ratio * dim, dim),
            nn.Dropout(dropout),
        )
        self.drop_path_ffn = DropPath(drop_path)

    def forward(self, x: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
        # x, pos: (B, N, C). Pos added to Q & K only.
        xn = self.norm_attn(x)
        q = k = xn + pos
        out, _ = self.self_attn(q, k, xn, need_weights=False)
        x = x + self.drop_path_attn(out)

        xn = self.norm_ffn(x)
        x = x + self.drop_path_ffn(self.ffn(xn))
        return x


class MaskTransformerLayer(nn.Module):
    """Mask2Former-style decoder layer (pre-norm):
        MaskedCrossAttn(Q -> KV) -> SelfAttn(Q) -> FFN(Q).
    """

    def __init__(self, dim: int, n_heads: int,
                 ffn_ratio: int = 4, dropout: float = 0.1,
                 drop_path: float = 0.1):
        super().__init__()
        self.n_heads = n_heads

        self.norm_q_ca = nn.LayerNorm(dim)
        self.cross_attn = nn.MultiheadAttention(
            dim, n_heads, dropout=dropout, batch_first=True
        )
        self.drop_path_ca = DropPath(drop_path)

        self.norm_q_sa = nn.LayerNorm(dim)
        self.self_attn = nn.MultiheadAttention(
            dim, n_heads, dropout=dropout, batch_first=True
        )
        self.drop_path_sa = DropPath(drop_path)

        self.norm_q_ffn = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, ffn_ratio * dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_ratio * dim, dim),
            nn.Dropout(dropout),
        )
        self.drop_path_ffn = DropPath(drop_path)

    def forward(self,
                Q: torch.Tensor,           # (B, Qd, C)
                kv: torch.Tensor,          # (B, N, C)
                pos_kv: torch.Tensor,      # (B, N, C)
                attn_bias: torch.Tensor    # (B, Qd, N) additive float bias
                ) -> torch.Tensor:
        B, Qd, _ = Q.shape
        _, S, _ = kv.shape

        # Cross-attention with masked / biased attention
        Qn = self.norm_q_ca(Q)
        K = kv + pos_kv
        # Broadcast (B, Qd, N) bias to per-head (B*nhead, Qd, N) as expected by MHA
        ca_mask = attn_bias.unsqueeze(1).expand(B, self.n_heads, Qd, S).reshape(
            B * self.n_heads, Qd, S
        )
        out, _ = self.cross_attn(Qn, K, kv, attn_mask=ca_mask, need_weights=False)
        Q = Q + self.drop_path_ca(out)

        # Self-attention among queries
        Qn = self.norm_q_sa(Q)
        out, _ = self.self_attn(Qn, Qn, Qn, need_weights=False)
        Q = Q + self.drop_path_sa(out)

        # FFN
        Qn = self.norm_q_ffn(Q)
        Q = Q + self.drop_path_ffn(self.ffn(Qn))
        return Q


# ---------------------------------------------------------------------------
# HybridHOTDecoder
# ---------------------------------------------------------------------------

class HybridHOTDecoder(nn.Module):
    """Drop-in replacement for `hot.models.decoder.Decoder`.

    See module docstring for block-level description.
    """

    # Variant glue checks this attribute to decide whether to forward S_map.
    accepts_smap: bool = True

    def __init__(self,
                 in_channel: int = 2048,
                 output_channel: int = 18,
                 dim: int = 256,
                 n_heads: int = 8,
                 n_pixel_layers: int = 2,
                 n_mask_layers: int = 3,
                 n_queries: int = 18,
                 dropout: float = 0.1,
                 drop_path: float = 0.1,
                 ffn_ratio: int = 4,
                 smap_alpha_init: float = 0.0):
        super().__init__()
        assert output_channel == n_queries, (
            f"output_channel ({output_channel}) must equal n_queries "
            f"({n_queries}); query 0 = BG, queries 1..K = body parts."
        )
        self.dim = dim
        self.n_heads = n_heads
        self.n_queries = n_queries
        self.output_channel = output_channel
        self.n_mask_layers = n_mask_layers

        # Sapiens FPN adapter feeds these channel dims (in_channel = 2048 by default).
        c4 = in_channel
        c3 = in_channel // 2
        c2 = in_channel // 4
        c1 = in_channel // 8

        # ----- Block A: lateral 1x1 projections -----
        def lateral(c_in):
            return nn.Sequential(
                nn.Conv2d(c_in, dim, kernel_size=1, bias=False),
                nn.GroupNorm(32, dim),
                nn.GELU(),
            )
        self.lat_x4 = lateral(c4)
        self.lat_x3 = lateral(c3)
        self.lat_x2 = lateral(c2)
        self.lat_x1 = lateral(c1)

        # ----- Block A: per-scale level embedding (3 scales: x4, x3, x2) -----
        self.level_emb = nn.Parameter(torch.zeros(3, dim))

        # ----- Block A: dense multi-scale Transformer encoder -----
        self.pixel_layers = nn.ModuleList([
            PixelTransformerLayer(
                dim, n_heads, ffn_ratio=ffn_ratio,
                dropout=dropout, drop_path=drop_path,
            )
            for _ in range(n_pixel_layers)
        ])

        # ----- Block A: FPN top-down conv refinements -----
        def fpn_conv():
            return nn.Sequential(
                nn.Conv2d(dim, dim, kernel_size=3, padding=1, bias=False),
                nn.GroupNorm(32, dim),
                nn.GELU(),
            )
        self.fpn_p3 = fpn_conv()
        self.fpn_p2 = fpn_conv()
        self.fpn_p1 = fpn_conv()

        # ----- Block B: DCFM (depth + person aggregate -> FiLM gamma, beta) -----
        self.dcfm = nn.Sequential(
            nn.Conv2d(2, 64, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(8, 64),
            nn.GELU(),
            nn.Conv2d(64, 2 * dim, kernel_size=3, padding=1),
        )

        # ----- Block C: queries + S_cls bootstrap + mask transformer -----
        # Convention: query index 0 = BG; queries 1..K = FG classes (K = n_queries - 1).
        self.queries = nn.Parameter(torch.randn(n_queries, dim) * 0.02)
        # Per-class S_cls bias projection: scalar presence prob -> dim
        self.scls_proj = nn.Linear(1, dim)

        self.mask_layers = nn.ModuleList([
            MaskTransformerLayer(
                dim, n_heads, ffn_ratio=ffn_ratio,
                dropout=dropout, drop_path=drop_path,
            )
            for _ in range(n_mask_layers)
        ])

        # Project query before dot-product mask prediction (Mask2Former style).
        self.mask_head = nn.Sequential(
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Linear(dim, dim),
        )

        # ----- Block D: learnable prior fusion weight -----
        # Init 0 -> mask_logits dominates early; SGD warms it up organically.
        self.smap_alpha = nn.Parameter(torch.tensor(float(smap_alpha_init)))

        # ----- Block E: cache for intermediate mask logits (deep supervision) -----
        self._aux_mask_logits: list = []

        self._init_weights()

    # ------------------------------------------------------------------ init

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.GroupNorm, nn.LayerNorm)):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        # Keep small init for queries and zero init for level_emb (overrides
        # any side effect of constructor-time module iteration).
        nn.init.normal_(self.queries, std=0.02)
        nn.init.zeros_(self.level_emb)

    # ----------------------------------------------------------- aux helpers

    def _mask_logits(self, Q: torch.Tensor, pix_tok: torch.Tensor,
                     H: int, W: int) -> torch.Tensor:
        """mask_logits = (W_q Q) . F_pix    -> (B, n_queries, H, W)."""
        Qm = self.mask_head(Q)                                  # (B, Qd, C)
        logits = torch.einsum("bqc,bnc->bqn", Qm, pix_tok)      # (B, Qd, N)
        return logits.reshape(Qm.shape[0], Qm.shape[1], H, W)

    def _attn_bias_from_mask(self,
                             mask_logits_2d: torch.Tensor,
                             person_log_bias: torch.Tensor) -> torch.Tensor:
        """Construct additive cross-attention bias for the next decoder layer.

        - Block positions where the previous layer's mask logit < 0
          (Mask2Former masked-attention).
        - If a (query, batch) row would be fully blocked (-> NaN after softmax),
          unblock it so the layer can still attend to everything.
        - Add log(person aggregate) on FG queries; BG query stays unbiased.
        """
        B, Qd, H, W = mask_logits_2d.shape
        flat = mask_logits_2d.detach().reshape(B, Qd, H * W)
        blocked = (flat < 0)
        all_blocked = blocked.all(dim=-1, keepdim=True)
        blocked = blocked & ~all_blocked

        bias = torch.zeros_like(flat)
        # -1e4 is safe in both fp32 and fp16 (does not produce -inf).
        bias = bias.masked_fill(blocked, -1e4)
        bias = bias + person_log_bias                            # broadcast (B, Qd, N)
        return bias

    # ----------------------------------------------------------- public API

    def deep_sup_loss(self, seg_label: torch.Tensor, crit) -> torch.Tensor:
        """Auxiliary CE over the intermediate mask logits cached during the
        last `forward()`. Off by default; flipped on by `cfg.MODEL.decoder_deep_sup`
        in the variant SegmentationModule (Stage S5).
        """
        if not self._aux_mask_logits:
            return torch.zeros((), device=seg_label.device)
        loss = sum(crit(m, seg_label) for m in self._aux_mask_logits)
        return loss / len(self._aux_mask_logits)

    # ------------------------------------------------------------- forward

    def forward(self,
                x4: torch.Tensor, x3: torch.Tensor, x2: torch.Tensor,
                x1: torch.Tensor, x0: torch.Tensor,
                logits_per_image: torch.Tensor,
                person_mask: torch.Tensor,
                total_person: torch.Tensor,
                depth: torch.Tensor,
                S_map: torch.Tensor = None) -> torch.Tensor:
        # x0 and total_person are accepted only for interface parity with the
        # legacy decoder; the new design does not consume them.
        del x0, total_person
        self._aux_mask_logits = []

        B = x4.shape[0]
        device = x4.device
        H4, W4 = x1.shape[-2:]                                  # = 56, 56 at 224

        # ====================== Block A: MS-PD ======================

        p4 = self.lat_x4(x4)                                    # (B, C,  7,  7)
        p3 = self.lat_x3(x3)                                    # (B, C, 14, 14)
        p2 = self.lat_x2(x2)                                    # (B, C, 28, 28)
        p1 = self.lat_x1(x1)                                    # (B, C, 56, 56)

        scales = [p4, p3, p2]
        sizes  = [s.shape[-2:] for s in scales]                 # [(7,7),(14,14),(28,28)]
        tokens, pos_embs = [], []
        for i, p in enumerate(scales):
            Hi, Wi = p.shape[-2:]
            t = p.flatten(2).transpose(1, 2)                    # (B, Hi*Wi, C)
            t = t + self.level_emb[i].view(1, 1, -1)
            pos = _sinusoidal_pos_2d(Hi, Wi, self.dim,
                                     device=device, dtype=t.dtype)
            tokens.append(t)
            pos_embs.append(pos.unsqueeze(0).expand(B, -1, -1))

        tok = torch.cat(tokens, dim=1)                          # (B, 1029, C) at 224
        pos = torch.cat(pos_embs, dim=1)
        for layer in self.pixel_layers:
            tok = layer(tok, pos)

        # Split refined tokens back to per-scale maps.
        offsets = [0]
        for Hi, Wi in sizes:
            offsets.append(offsets[-1] + Hi * Wi)
        p4_ref = tok[:, offsets[0]:offsets[1]].transpose(1, 2).reshape(B, self.dim, *sizes[0])
        p3_ref = tok[:, offsets[1]:offsets[2]].transpose(1, 2).reshape(B, self.dim, *sizes[1])
        p2_ref = tok[:, offsets[2]:offsets[3]].transpose(1, 2).reshape(B, self.dim, *sizes[2])

        # FPN top-down with high-res lateral on x1.
        p3_td = self.fpn_p3(
            F.interpolate(p4_ref, size=sizes[1], mode="bilinear", align_corners=False)
            + p3_ref
        )
        p2_td = self.fpn_p2(
            F.interpolate(p3_td, size=sizes[2], mode="bilinear", align_corners=False)
            + p2_ref
        )
        F_pix = self.fpn_p1(
            F.interpolate(p2_td, size=(H4, W4), mode="bilinear", align_corners=False)
            + p1
        )                                                       # (B, C, H/4, W/4)

        # ====================== Block B: DCFM ======================

        p_agg = person_mask.sum(dim=1, keepdim=True).clamp(0.0, 1.0)  # (B,1,H/4,W/4)
        depth_in = torch.cat([depth.unsqueeze(1), p_agg], dim=1)      # (B,2,H/4,W/4)
        film = self.dcfm(depth_in)                                    # (B,2C,H/4,W/4)
        gamma, beta = film.chunk(2, dim=1)
        gamma = torch.tanh(gamma)                                     # scale in (0, 2)
        F_pix = (1.0 + gamma) * F_pix + beta

        # ====================== Block C: PMTD ======================

        pix_tok = F_pix.flatten(2).transpose(1, 2)                    # (B, N, C)
        pix_pos = _sinusoidal_pos_2d(H4, W4, self.dim,
                                     device=device, dtype=pix_tok.dtype)
        pix_pos = pix_pos.unsqueeze(0).expand(B, -1, -1)              # (B, N, C)

        # Initialize queries; bootstrap FG queries with S_cls presence prior.
        Q = self.queries.unsqueeze(0).expand(B, -1, -1).clone()       # (B, Qd, C)
        # logits_per_image: (B, K_fg) with K_fg = n_queries - 1
        scls_bias = self.scls_proj(logits_per_image.unsqueeze(-1))    # (B, K_fg, C)
        K_fg = scls_bias.shape[1]
        Q[:, 1:1 + K_fg, :] = Q[:, 1:1 + K_fg, :] + scls_bias

        # Person attention log-bias for FG queries; BG query stays unbiased.
        log_p = torch.log(p_agg.clamp_min(1e-6))                      # (B,1,H/4,W/4)
        log_p_flat = log_p.flatten(2).squeeze(1)                      # (B, N)
        person_log_bias = log_p_flat.unsqueeze(1).expand(
            B, self.n_queries, -1
        ).clone()                                                     # (B, Qd, N)
        person_log_bias[:, 0, :] = 0.0                                # BG: no bias

        # Initial mask logits M_0 (used as attn mask for layer 1).
        mask_logits = self._mask_logits(Q, pix_tok, H4, W4)           # (B, Qd, H, W)
        all_logits = [mask_logits]                                    # M_0

        # L mask-transformer decoder layers.
        for layer in self.mask_layers:
            attn_bias = self._attn_bias_from_mask(mask_logits, person_log_bias)
            Q = layer(Q, pix_tok, pos_kv=pix_pos, attn_bias=attn_bias)
            mask_logits = self._mask_logits(Q, pix_tok, H4, W4)
            all_logits.append(mask_logits)

        final_logits = all_logits[-1]                                 # M_L
        self._aux_mask_logits = all_logits[:-1]                       # M_0..M_{L-1}

        # ====================== Block D: prior fusion ======================

        if S_map is not None:
            # S_map covers FG classes 1..K only; pad a zero BG channel so
            # shape and class convention match `final_logits` (BG at idx 0).
            zeros_bg = torch.zeros(
                B, 1, H4, W4, device=S_map.device, dtype=S_map.dtype
            )
            S_map_pad = torch.cat([zeros_bg, S_map], dim=1)           # (B, Qd, H, W)
            pred = final_logits + self.smap_alpha * S_map_pad
        else:
            pred = final_logits

        return pred
