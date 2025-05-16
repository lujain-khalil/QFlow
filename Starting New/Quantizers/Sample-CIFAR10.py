import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.utils import save_image
import glob

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# ---------------------------------------------------------
# Config
# ---------------------------------------------------------
device      = 'cuda' if torch.cuda.is_available() else 'cpu'

# sampling parameters
num_per_cls = 100    # per class → 1000 total
sigma       = 0.5
num_steps   = 1000
dt          = 1.0 / num_steps

# model hyper-parameters (must match training)
image_size       = 32
patch_size       = 4
n_patches        = (image_size // patch_size) ** 2
embed_dim        = 768
n_heads          = 16
n_layers         = 8
class_drop_prob  = 0.0    # disable class dropout at sampling time
attr_dim         = 10     # CIFAR-10 classes

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

        # attribute embedding
        attr_emb = self.attr_emb(y)
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

# ---------------------------------------------------------
# Gaussian Fourier Projection
# ---------------------------------------------------------
class GaussianFourierProjection(nn.Module):
    def __init__(self, embed_dim, scale=10.0):
        super().__init__()
        self.W = nn.Parameter(torch.randn(embed_dim) * scale, requires_grad=False)

    def forward(self, x):
        proj = x.unsqueeze(-1) * self.W.unsqueeze(0) * 2 * math.pi
        return torch.cat([proj.sin(), proj.cos()], dim=-1)
    

ckpt_paths = glob.glob("./quantized_models/cifar10/**/*.pt", recursive=True)

for ckpt_path in ckpt_paths:
# for ckpt_path in ["./quantized_models/cifar10/uniform/uniform_8bit.pt"]:
    parts = ckpt_path.split("/")
    quantizer = parts[3]
    model_name = parts[4].replace(".pt", "")

    if not os.path.exists(ckpt_path):
        print(f"Checkpoint `{ckpt_path}` not found. Exiting.")
        exit(1)
    
    base_dir    = "./sampling/cifar10"
    if not os.path.exists(base_dir):
        os.makedirs(base_dir, exist_ok=True)
    gen_dir     = os.path.join(base_dir, quantizer, model_name)

    # ---------------------------------------------------------
    # Load model
    # ---------------------------------------------------------
    print(f"Loading checkpoint `{ckpt_path}`...")
    model = FlowTransformer().to(device)
    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state)
    model.eval()

    for c in range(10):
        os.makedirs(os.path.join(gen_dir, str(c)), exist_ok=True)

    # ---------------------------------------------------------
    # Generate samples
    # ---------------------------------------------------------
    with torch.no_grad():
        for c in range(attr_dim):
            for i in range(num_per_cls):
                print(f"Generating {num_per_cls} images for class {c}...")
                y_idx    = torch.tensor([c], device=device)
                y_onehot = F.one_hot(y_idx, num_classes=attr_dim).float()
                x        = torch.randn(1, 3, image_size, image_size, device=device) * sigma

                # Euler-forward sampling with 1000 steps
                for step in range(num_steps):
                    t_val = 1.0 - step * dt
                    t     = torch.full((1,), t_val, device=device)
                    v     = model(x, t, y_onehot)
                    x     = x + v * dt

                img = (x.clamp(-1, 1) + 1) * 0.5
                save_image(img, os.path.join(gen_dir, str(c), f"gen_c{c}_{i:03d}.png"))

    print(f"Generated {attr_dim * num_per_cls} images in `{gen_dir}`")
    print(f"Finished processing checkpoint `{ckpt_path}`")
    print("---------------------------------------------------------")
print("All done!")