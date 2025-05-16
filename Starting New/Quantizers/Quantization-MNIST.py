#!/usr/bin/env python3
# ------------------------------------------------------------
# PTQ for FlowTransformer: uniform, OT, K‑means (2/4/8‑bit)
# ------------------------------------------------------------
import os
import copy
import torch
import numpy as np
from torch import nn
from sklearn.cluster import KMeans          # pip install scikit‑learn

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# ----------------------------
# Configuration
# ----------------------------
device           = 'cuda' if torch.cuda.is_available() else 'cpu'
float_model_path = os.path.join("./models/mnist_model.pt")
out_root         = "./quantized_models/mnist"
if not os.path.exists(out_root):
    os.makedirs(out_root)
bits_full        = [16, 8, 4, 2]            # uniform & OT

for sub in ["uniform", "ot"]:
    os.makedirs(os.path.join(out_root, sub), exist_ok=True)

# ----------------------------
# FlowTransformer definition
# ----------------------------
class FlowTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.patch_size = 4
        self.embed_dim  = 384
        self.n_heads    = 16
        self.n_layers   = 10
        self.n_patches  = (28 // self.patch_size) ** 2

        self.patch_proj = nn.Linear(self.patch_size * self.patch_size, self.embed_dim)
        self.class_emb  = nn.Embedding(10, self.embed_dim)
        self.time_mlp   = nn.Sequential(
            nn.Linear(1, self.embed_dim), 
            nn.SiLU(), 
            nn.Linear(self.embed_dim, self.embed_dim)
        )
        self.pos_emb = nn.Parameter(torch.randn(1, self.n_patches, self.embed_dim))

        layer = nn.TransformerEncoderLayer(d_model=self.embed_dim,
                                           nhead=self.n_heads,
                                           batch_first=True)
        self.transformer = nn.TransformerEncoder(layer, num_layers=self.n_layers)
        self.out_proj = nn.Linear(self.embed_dim, self.patch_size * self.patch_size)

    def forward(self, x, t, y):
        B = x.size(0)

        patches = (
            x.unfold(2, self.patch_size, self.patch_size)
             .unfold(3, self.patch_size, self.patch_size)
             .contiguous()
             .view(B, self.n_patches, -1)
        )
        h  = self.patch_proj(patches)
        h += self.class_emb(y).unsqueeze(1)
        h += self.time_mlp(t.unsqueeze(-1)).unsqueeze(1)
        h += self.pos_emb
        h  = self.transformer(h)
        out = self.out_proj(h)
        p   = 28 // self.patch_size
        out = out.view(B, 1, p, p, self.patch_size, self.patch_size)
        out = (
            out.permute(0, 1, 2, 4, 3, 5)
               .contiguous()
               .view(B, 1, 28, 28)
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
