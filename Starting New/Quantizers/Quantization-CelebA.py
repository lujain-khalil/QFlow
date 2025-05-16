#!/usr/bin/env python3
# ------------------------------------------------------------
# PTQ for FlowTransformer: uniform, OT
# ------------------------------------------------------------
import os
import copy
import torch
import numpy as np
from torch import nn
import math

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# ----------------------------
# Configuration
# ----------------------------
device           = 'cuda' if torch.cuda.is_available() else 'cpu'
float_model_path = os.path.join("./models/celebA_model.pt")
out_root         = "./quantized_models/celebA"
if not os.path.exists(out_root):
    os.makedirs(out_root)
bits_full        = [16, 8, 4, 2]

for sub in ["uniform", "ot"]:
    os.makedirs(os.path.join(out_root, sub), exist_ok=True)


image_size       = 64
patch_size       = 8
n_patches        = (image_size // patch_size) ** 2
embed_dim        = 768
n_heads          = 16
n_layers         = 8
class_drop_prob  = 0.1
attr_dim         = 40

class GaussianFourierProjection(nn.Module):
    def __init__(self, embed_dim, scale=10.0):
        super().__init__()
        # fixed random projection
        self.W = nn.Parameter(torch.randn(embed_dim) * scale, requires_grad=False)

    def forward(self, x):
        # x: (B,) ? proj: (B, embed_dim)
        proj = x.unsqueeze(-1) * self.W.unsqueeze(0) * 2 * math.pi
        return torch.cat([proj.sin(), proj.cos()], dim=-1)

class FlowTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.patch_proj = nn.Linear(patch_size * patch_size * 3, embed_dim)
        self.attr_emb   = nn.Linear(attr_dim, embed_dim)
        self.time_mlp   = nn.Sequential(
            GaussianFourierProjection(embed_dim),
            nn.Linear(embed_dim * 2, embed_dim),
            nn.SiLU(),
            nn.Linear(embed_dim, embed_dim),
        )
        self.pos_emb    = nn.Parameter(
            torch.randn(1, n_patches, embed_dim) * (1.0 / math.sqrt(embed_dim))
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=n_heads,
            dim_feedforward=embed_dim * 4, dropout=0.1,
            activation='gelu', batch_first=True
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=n_layers
        )
        self.out_proj   = nn.Linear(embed_dim, patch_size * patch_size * 3)
        self.class_drop_prob = class_drop_prob

    def forward(self, x, t, y):
        B = x.size(0)
        # split into patches
        patches = (
            x.unfold(2, patch_size, patch_size)
             .unfold(3, patch_size, patch_size)
             .permute(0,2,3,1,4,5)
             .contiguous()
             .view(B, n_patches, -1)
        )
        h = self.patch_proj(patches)

        # attribute embedding with dropout
        attr_emb = self.attr_emb(y.float())
        mask     = torch.rand(B, device=x.device) < self.class_drop_prob
        attr_emb[mask] *= 0
        attr_emb = attr_emb.unsqueeze(1)

        # time embedding
        tm = self.time_mlp(t)

        # combine and run transformer
        h = h + attr_emb + tm.unsqueeze(1) + self.pos_emb
        h = self.transformer(h)

        # project back to image
        out = self.out_proj(h)
        p   = image_size // patch_size
        out = out.view(B, p, p, 3, patch_size, patch_size)
        out = (
            out.permute(0,3,1,4,2,5)
               .contiguous()
               .view(B, 3, image_size, image_size)
        )
        return out
    
# ------------------------------------------------------------
# Tensor‑wise quantisers
# ------------------------------------------------------------
def _skip_small(t: torch.Tensor) -> bool:
    """Skip tiny tensors (≤4 elements)."""
    return t.numel() <= 4

# ---- Uniform ------------------------------------------------
def quantize_tensor_uniform(t: torch.Tensor, bits: int) -> torch.Tensor:
    if bits >= 32 or _skip_small(t):
        return t.clone()

    flat  = t.detach().cpu()
    mn, mx = flat.min(), flat.max()
    levels = 2 ** bits
    scale  = (mx - mn) / (levels - 1)
    q = torch.round((flat - mn) / scale) * scale + mn
    return q.to(t.device)

def quantize_model_uniform(model_fp: nn.Module, bits: int) -> nn.Module:
    q_model = copy.deepcopy(model_fp).cpu()
    for p in q_model.parameters():
        with torch.no_grad():
            p.copy_(quantize_tensor_uniform(p.data, bits))
    return q_model.to(device)

# ---- Optimal‑transport equal‑mass ---------------------------
def quantize_tensor_ot(t: torch.Tensor, bits: int) -> torch.Tensor:
    if bits >= 32 or _skip_small(t):
        return t.clone()

    flat = t.detach().cpu().numpy().reshape(-1)
    K = min(2 ** bits, flat.size)           # prevent empty chunks

    idx  = np.argsort(flat)
    flat_sorted = flat[idx]
    chunks   = np.array_split(flat_sorted, K)
    centroids = np.array([c.mean() for c in chunks], dtype=flat.dtype)
    quant_sorted = np.empty_like(flat_sorted)
    start = 0
    for k, chunk in enumerate(chunks):
        L = len(chunk)
        quant_sorted[start:start + L] = centroids[k]
        start += L
    inv = np.argsort(idx)
    return torch.from_numpy(quant_sorted[inv].reshape(t.shape)).to(t.device)

def quantize_model_ot(model_fp: nn.Module, bits: int) -> nn.Module:
    q_model = copy.deepcopy(model_fp).cpu()
    for p in q_model.parameters():
        with torch.no_grad():
            p.copy_(quantize_tensor_ot(p.data, bits))
    return q_model.to(device)


# ----------------------------
# Load FP model
# ----------------------------
flow_fp = FlowTransformer().to(device)
flow_fp.load_state_dict(torch.load(float_model_path, map_location=device))
flow_fp.eval()

# ----------------------------
# Quantise & save
# ----------------------------
for quantizer in ["uniform", "ot"]:
    bit_loop = bits_full
    for bits in bit_loop:
        if quantizer == "uniform":
            q_model = quantize_model_uniform(flow_fp, bits)
        elif quantizer == "ot":
            q_model = quantize_model_ot(flow_fp, bits)

        fname = f"{quantizer}_{bits}bit.pt"
        save_path = os.path.join(out_root, quantizer, fname)
        torch.save(q_model.state_dict(), save_path)
        print(f"Saved {quantizer:7s} {bits:2d}-bit model → {save_path}")
