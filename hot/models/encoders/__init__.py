# Encoder adapters. Each adapter exposes encode_image(img) returning a dict:
#     {
#         "prior_logits": (B, K),
#         "x4": (B, C4, H/32, W/32),
#         "x3": (B, C3, H/16, W/16),
#         "x2": (B, C2, H/8,  W/8),
#         "x1": (B, C1, H/4,  W/4),
#         "x0": (B, C0, H/4,  W/4),
#         "S_map": optional (B, K, H/4, W/4),
#     }
