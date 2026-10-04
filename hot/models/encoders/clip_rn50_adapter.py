"""CLIP RN50 encoder adapter (P3HOT baseline).

Wraps the CLIP model so encode_image(img) returns the same dict shape used by
all other encoders in this package:
    {"prior_logits", "x4", "x3", "x2", "x1", "x0"}

The "prior_logits" tensor is the per-class CLIP image-text similarity
(equivalent to the legacy `logits_per_image` in raw P3HOT). Text embeddings
are computed lazily on the first call and cached.

Variable input size:
    The RN50 checkpoint's AttentionPool2d has a fixed (7*7+1, C) positional
    embedding tied to the 224-input CLIP. Feeding a larger crop (e.g. 348)
    yields an 11x11 feature map at layer4 and the original pool head raises
    a shape mismatch. `_patch_attnpool_for_variable_input` swaps the pool's
    forward for a version that bicubic-interpolates the positional embedding
    to the current HxW on the fly (cached per shape).
"""

import os
import sys

# CLIP lives at raw/P3HOT/CLIP (reference: the original P3HOT project). Ensure
# `raw/P3HOT` is on sys.path so `import CLIP.clip` resolves regardless of how
# train.py is invoked.
_THIS = os.path.dirname(os.path.abspath(__file__))
# hyhot/hot/models/encoders -> HybridHOT
_REPO_ROOT = os.path.normpath(os.path.join(_THIS, "..", "..", "..", ".."))
_P3HOT_DIR = os.path.join(_REPO_ROOT, "raw", "P3HOT")
if _P3HOT_DIR not in sys.path and os.path.isdir(_P3HOT_DIR):
    sys.path.insert(0, _P3HOT_DIR)

import torch
import torch.nn as nn
import torch.nn.functional as F

import CLIP.clip as clip


_TEXT_PROMPTS = [
    f"A {name.lower()} of the human body is in contact with an object."
    for name in [
        "Head", "Chest", "Left Upper Arm", "Left Fore Arm", "Left Hand",
        "Right Upper Arm", "Right Fore Arm", "Right Hand", "Buttocks", "Hip",
        "Back", "Left Thigh", "Left Calf", "Left Foot",
        "Right Thigh", "Right Calf", "Right Foot",
    ]
]


def _patch_attnpool_for_variable_input(attnpool: nn.Module) -> None:
    """Make CLIP RN50 AttentionPool2d tolerant of non-7x7 feature maps.

    The checkpoint's positional embedding is (49+1, C). At each call we
    interpolate its spatial portion to the incoming (H, W) so the (H*W+1, C)
    add succeeds. Interpolated tensors are cached per (H, W, device, dtype).
    """
    if getattr(attnpool, "_variable_input_patched", False):
        return

    attnpool._pe_cache = {}

    def _resample(pos_embed: torch.Tensor, h: int, w: int) -> torch.Tensor:
        cls_pe = pos_embed[:1]
        spatial_pe = pos_embed[1:]
        n, c = spatial_pe.shape
        g = int(round(n ** 0.5))
        assert g * g == n, f"positional_embedding is not square: {n}"
        spatial_pe = spatial_pe.reshape(1, g, g, c).permute(0, 3, 1, 2)
        spatial_pe = F.interpolate(
            spatial_pe.float(), size=(h, w), mode="bicubic", align_corners=False,
        ).to(pos_embed.dtype)
        spatial_pe = spatial_pe.permute(0, 2, 3, 1).reshape(h * w, c)
        return torch.cat([cls_pe, spatial_pe], dim=0)

    def forward(x: torch.Tensor) -> torch.Tensor:
        h, w = x.shape[-2], x.shape[-1]
        x = x.flatten(start_dim=2).permute(2, 0, 1)  # NCHW -> (HW)NC
        x = torch.cat([x.mean(dim=0, keepdim=True), x], dim=0)  # (HW+1)NC

        pe = attnpool.positional_embedding
        if pe.shape[0] == h * w + 1:
            pe_hw = pe
        else:
            key = (h, w, pe.device, pe.dtype)
            pe_hw = attnpool._pe_cache.get(key)
            if pe_hw is None:
                pe_hw = _resample(pe, h, w)
                attnpool._pe_cache[key] = pe_hw

        x = x + pe_hw[:, None, :].to(x.dtype)
        x, _ = F.multi_head_attention_forward(
            query=x[:1], key=x, value=x,
            embed_dim_to_check=x.shape[-1],
            num_heads=attnpool.num_heads,
            q_proj_weight=attnpool.q_proj.weight,
            k_proj_weight=attnpool.k_proj.weight,
            v_proj_weight=attnpool.v_proj.weight,
            in_proj_weight=None,
            in_proj_bias=torch.cat(
                [attnpool.q_proj.bias, attnpool.k_proj.bias, attnpool.v_proj.bias]
            ),
            bias_k=None,
            bias_v=None,
            add_zero_attn=False,
            dropout_p=0,
            out_proj_weight=attnpool.c_proj.weight,
            out_proj_bias=attnpool.c_proj.bias,
            use_separate_proj_weight=True,
            training=attnpool.training,
            need_weights=False,
        )
        return x.squeeze(0)

    attnpool.forward = forward
    attnpool._variable_input_patched = True


class CLIPRN50Adapter(nn.Module):
    """CLIP RN50 visual + text similarity head wrapped in a single nn.Module.

    Notes:
      - `self.visual` is the CLIP image encoder (used by the encoder optimizer
        in raw P3HOT). Exposed as an attribute so train.py can still build the
        optimizer with `net_encoder.visual.parameters()` if desired.
      - The CLIP model is held in `self.model` (kept on its native device by
        `clip.load`); `.encode_text` is invoked only the first time and the
        result is cached on `self._t_e`.
    """

    accepts_smap: bool = False

    def __init__(self, num_class: int = 18, weights: str = ""):
        super().__init__()
        clip_model, _ = clip.load("RN50", device="cpu")
        self.model = clip_model
        self.visual = clip_model.visual
        _patch_attnpool_for_variable_input(self.visual.attnpool)
        self.num_class = num_class
        self._t_e = None

        if weights:
            print("Loading weights for CLIPRN50Adapter")
            state = torch.load(weights, map_location="cpu")
            self.load_state_dict(state, strict=False)

    def _ensure_text_embed(self, device):
        if self._t_e is None or self._t_e.device != device:
            tokens = clip.tokenize(_TEXT_PROMPTS).to(device)
            t_e = self.model.encode_text(tokens)
            t_e = t_e / (t_e.norm(dim=1, keepdim=True) + 1e-6)
            # Detach so the cached embedding does not retain the text-encoder
            # graph across iterations (matches raw P3HOT's `.detach()` in the
            # train-branch matmul).
            self._t_e = t_e.detach()

    def encode_image(self, img: torch.Tensor) -> dict:
        self._ensure_text_embed(img.device)
        i_e, x4, x3, x2, x1, x0 = self.model.encode_image(img)
        image_feat = i_e / (i_e.norm(dim=1, keepdim=True) + 1e-6)
        prior_logits = F.relu(image_feat @ self._t_e.t())  # (B, 17)
        return {
            "prior_logits": prior_logits,
            "x4": x4, "x3": x3, "x2": x2, "x1": x1, "x0": x0,
        }

    def forward(self, img: torch.Tensor) -> dict:
        return self.encode_image(img)
