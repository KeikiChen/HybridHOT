# Encoder adapters. encode_image(img) returns a dict with "prior_logits" (B, K),
# multi-scale features "x4".."x0" (H/32 .. H/4) and optional "S_map" (B, K, H/4, W/4).
