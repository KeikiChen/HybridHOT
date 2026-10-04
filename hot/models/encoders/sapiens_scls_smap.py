"""Sapiens-Scls-Smap encoder.

Wraps a Sapiens2 ViT backbone and returns the dict format shared by every
encoder in this package. The Sapiens2 code lives under `hyhot/sapiens2/`;
this file adds that folder to sys.path so the `sapiens` package can be
imported.
"""

import os
import sys

_THIS = os.path.dirname(os.path.abspath(__file__))
# hot/models/encoders -> project root
_HYHOT_ROOT = os.path.normpath(os.path.join(_THIS, "..", "..", ".."))
_SAP2_DIR = os.path.join(_HYHOT_ROOT, "sapiens2")
if _SAP2_DIR not in sys.path and os.path.isdir(_SAP2_DIR):
    sys.path.insert(0, _SAP2_DIR)

import torch
import torch.nn as nn
import torch.nn.functional as F

import torch.distributed.fsdp as _fsdp_compat
if not hasattr(_fsdp_compat, "MixedPrecisionPolicy"):
    _fsdp_compat.MixedPrecisionPolicy = getattr(_fsdp_compat, "MixedPrecision", None)

from sapiens.backbones.standalone.sapiens2 import Sapiens2


_ARCH_CONFIGS = {
    "sapiens2_0.1b": {"embed_dims": 768,  "num_layers": 12, "out_indices": [2,  5,  8,  11]},
    "sapiens2_0.4b": {"embed_dims": 1024, "num_layers": 24, "out_indices": [5, 11, 17,  23]},
    "sapiens2_0.8b": {"embed_dims": 1280, "num_layers": 32, "out_indices": [7, 15, 23,  31]},
    "sapiens2_1b":   {"embed_dims": 1536, "num_layers": 40, "out_indices": [9, 19, 29,  39]},
    "sapiens2_5b":   {"embed_dims": 2432, "num_layers": 56, "out_indices": [13, 27, 41, 55]},
}


class SapiensSclsSmapAdapter(nn.Module):
    """Sapiens2 ViT + FPN adapter with S_cls and S_map heads.

    encode_image(img) returns a dict:
        prior_logits: (B, K)               class presence prior (S_cls)
        x4           : (B, 2048, H/32, W/32)
        x3           : (B, 1024, H/16, W/16)
        x2           : (B,  512, H/8,  W/8)
        x1           : (B,  256, H/4,  W/4)
        x0           : (B,  256, H/4,  W/4)  same tensor as x1
        S_map        : (B,    K, H/4,  W/4)  spatial prior logits
    """

    accepts_smap: bool = True

    def __init__(
        self,
        arch: str = "sapiens2_0.4b",
        img_size: tuple = (224, 224),
        num_classes: int = 17,
        pretrained: str = "",
        use_cls_head: bool = True,
        use_spatial_head: bool = True,
    ):
        super().__init__()
        assert arch in _ARCH_CONFIGS, (
            f"Unknown arch '{arch}'. Choose from: {list(_ARCH_CONFIGS)}"
        )

        cfg_a = _ARCH_CONFIGS[arch]
        embed_dim = cfg_a["embed_dims"]

        self.num_classes = num_classes
        self.use_cls_head = use_cls_head
        self.use_spatial_head = use_spatial_head

        self.backbone = Sapiens2(
            arch=arch, img_size=img_size, patch_size=16,
            out_indices=cfg_a["out_indices"], out_type="featmap",
            final_norm=True, with_cls_token=True,
        )

        self.proj_x1 = nn.Sequential(nn.Conv2d(embed_dim, 256,  1, bias=False), nn.BatchNorm2d(256),  nn.ReLU(inplace=True))
        self.proj_x2 = nn.Sequential(nn.Conv2d(embed_dim, 512,  1, bias=False), nn.BatchNorm2d(512),  nn.ReLU(inplace=True))
        self.proj_x3 = nn.Sequential(nn.Conv2d(embed_dim, 1024, 1, bias=False), nn.BatchNorm2d(1024), nn.ReLU(inplace=True))
        self.proj_x4 = nn.Sequential(nn.Conv2d(embed_dim, 2048, 1, bias=False), nn.BatchNorm2d(2048), nn.ReLU(inplace=True))

        if use_cls_head:
            self.cls_head = nn.Sequential(
                nn.Linear(embed_dim, embed_dim // 4),
                nn.ReLU(inplace=True),
                nn.Linear(embed_dim // 4, num_classes),
                nn.Sigmoid(),
            )

        if use_spatial_head:
            self.spatial_head = nn.Sequential(
                nn.Conv2d(embed_dim, 256, 3, padding=1, bias=False),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
                nn.Conv2d(256, num_classes, 1),
            )

        self._init_weights()
        if pretrained:
            self._load_pretrained(pretrained)

    def _init_weights(self):
        conv_modules = [self.proj_x1, self.proj_x2, self.proj_x3, self.proj_x4]
        if self.use_spatial_head:
            conv_modules.append(self.spatial_head)
        for m in conv_modules:
            for layer in m.modules():
                if isinstance(layer, nn.Conv2d):
                    nn.init.kaiming_normal_(layer.weight, mode="fan_out", nonlinearity="relu")
                    if layer.bias is not None:
                        nn.init.constant_(layer.bias, 0)
                elif isinstance(layer, nn.BatchNorm2d):
                    nn.init.constant_(layer.weight, 1)
                    nn.init.constant_(layer.bias, 0)
        if self.use_cls_head:
            for layer in self.cls_head.modules():
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_normal_(layer.weight)
                    if layer.bias is not None:
                        nn.init.constant_(layer.bias, 0)

    def _load_pretrained(self, ckpt_path: str):
        if ckpt_path.endswith(".safetensors"):
            from safetensors.torch import load_file
            state = load_file(ckpt_path)
        else:
            state = torch.load(ckpt_path, map_location="cpu")
            if "state_dict" in state:
                state = state["state_dict"]
        PREFIX = "backbone."
        if any(k.startswith(PREFIX) for k in state):
            state = {k[len(PREFIX):]: v for k, v in state.items() if k.startswith(PREFIX)}
        missing, unexpected = self.backbone.load_state_dict(state, strict=False)
        n_loaded = len(state) - len(unexpected)
        print(f"[SapiensSclsSmapAdapter] Loaded pretrained backbone: {ckpt_path}")
        print(f"  Matched/loaded: {n_loaded}/{len(self.backbone.state_dict())} keys")
        if missing:
            print(f"  Missing  keys: {len(missing)}")
        if unexpected:
            print(f"  Unexpected keys: {len(unexpected)}")

    def encode_image(self, img: torch.Tensor) -> dict:
        feats = self.backbone(img)
        f0, f1, f2, f3 = feats

        x1 = F.interpolate(self.proj_x1(f0), scale_factor=4.0, mode="bilinear", align_corners=False)
        x2 = F.interpolate(self.proj_x2(f1), scale_factor=2.0, mode="bilinear", align_corners=False)
        x3 = self.proj_x3(f2)
        x4 = F.avg_pool2d(self.proj_x4(f3), kernel_size=2, stride=2)

        if self.use_cls_head:
            S_cls = self.cls_head(f3.mean(dim=(-2, -1)))
        else:
            # No cls head: return ones of shape (B, K) so the decoder's
            # per-class multiply is a no-op but the tensor shape still
            # matches what downstream decoders expect.
            S_cls = torch.ones(
                img.shape[0], self.num_classes,
                device=x4.device, dtype=x4.dtype,
            )

        out = {
            "prior_logits": S_cls,
            "x4": x4, "x3": x3, "x2": x2, "x1": x1, "x0": x1,
        }
        if self.use_spatial_head:
            out["S_map"] = F.interpolate(
                self.spatial_head(f3), scale_factor=4.0,
                mode="bilinear", align_corners=False,
            )
        return out

    def forward(self, img: torch.Tensor) -> dict:
        return self.encode_image(img)


def build_sapiens_scls_smap(cfg, weights: str = "") -> SapiensSclsSmapAdapter:
    arch = getattr(cfg.MODEL, "sapiens_arch", "sapiens2_0.4b")
    num_classes = cfg.DATASET.num_class - 1
    pretrained = getattr(cfg.MODEL, "pretrained", "") if not weights else ""
    net = SapiensSclsSmapAdapter(
        arch=arch, img_size=(224, 224),
        num_classes=num_classes, pretrained=pretrained,
        use_cls_head=getattr(cfg.MODEL, "use_cls_head", True),
        use_spatial_head=getattr(cfg.MODEL, "use_spatial_head", True),
    )
    if weights:
        print("Loading weights for SapiensSclsSmapAdapter")
        state = torch.load(weights, map_location="cpu")
        net.load_state_dict(state, strict=False)
    return net
