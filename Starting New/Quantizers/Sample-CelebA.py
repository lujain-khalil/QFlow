#!/usr/bin/env python
import os
import math
import torch
import torch.nn as nn
from torchvision.utils import save_image
import glob 

os.environ["CUDA_VISIBLE_DEVICES"] = "1"  # or remove if you want default
device     = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# model + sampling hyperparameters - make sure these match your training run
image_size      = 64
patch_size      = 8
n_patches       = (image_size // patch_size) ** 2
embed_dim       = 768
n_heads         = 16
n_layers        = 8
class_drop_prob = 0.1
attr_dim        = 40

sigma      = 0.5     # same s used at train time
num_steps  = 1000
dt         = 1.0 / num_steps
n_samples  = 1000

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


def main():
    # ckpt_paths = glob.glob("./quantized_models/celebA/**/*.pt", recursive=True)
    # ckpt_paths.append("./models/celebA_model.pt")

    # for ckpt_path in ckpt_paths:
    for ckpt_path in ["./models/celebA_model.pt"]:
        if ckpt_path.endswith("celebA_model.pt"):
            method = "full"
        else:
            parts = ckpt_path.split("/")
            method = parts[3]
            quantized_model_name = parts[4].replace(".pt", "")

        if not os.path.exists(ckpt_path):
            print(f"Checkpoint `{ckpt_path}` not found. Exiting.")
            exit(1)
        
        base_dir = "./sampling/celebA"
        if not os.path.exists(base_dir):
            os.makedirs(base_dir, exist_ok=True)

        if method == "full":
            gen_dir = os.path.join(base_dir, method)
        else:
            gen_dir = os.path.join(base_dir, method, quantized_model_name)
        if not os.path.exists(gen_dir):
            os.makedirs(gen_dir, exist_ok=True)

        # load model
        print(f"Loading checkpoint `{ckpt_path}`...")
        model = FlowTransformer().to(device)
        state = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(state)
        model.eval()

        # unconditional zeros
        ys = torch.zeros(n_samples, attr_dim, device=device)

        # start from noise at t=1.0
        x_gen = torch.randn(n_samples, 3, image_size, image_size, device=device) * sigma

        # integrate dx/dt = -v_pred(t) backwards from t=1?0
        with torch.no_grad():
            print(f"Generating {n_samples} images...")
            for step in range(num_steps):
                t_val = 1.0 - step * dt
                t     = torch.full((n_samples,), t_val, device=device)
                v     = model(x_gen, t, ys)
                x_gen = x_gen + v * dt

        # rescale to [0,1]
        samples = (x_gen.clamp(-1, 1) + 1) * 0.5

        for i, img in enumerate(samples):
            save_image(img, os.path.join(gen_dir, f"sample_{i:02d}.png"))

    print(f"Saved {n_samples} samples to ./{base_dir}/")
    print(f"Finished processing checkpoint `{ckpt_path}`")
    print("---------------------------------------------------------")

print("All done!")

if __name__ == "__main__":
    main()
