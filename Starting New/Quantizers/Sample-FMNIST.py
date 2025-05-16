import os
import torch
import torch.nn.functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image
import glob

device      = 'cuda' if torch.cuda.is_available() else 'cpu'
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# ---------------------------------------------------------
# Config
# ---------------------------------------------------------

num_per_cls = 100   # 100 images per class ? 1 000 total
sigma       = 0.5
num_steps   = 500
dt          = 1.0 / num_steps

# Transformer patch dims (must match training)
patch_size  = 4
n_patches   = (28 // patch_size) ** 2
embed_dim   = 384
n_heads     = 12
n_layers    = 10

# ---------------------------------------------------------
# Model Definition (same as training)
# ---------------------------------------------------------
class FlowTransformer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.patch_proj = torch.nn.Linear(patch_size * patch_size, embed_dim)
        self.class_emb  = torch.nn.Embedding(10, embed_dim)
        self.time_mlp   = torch.nn.Sequential(
            torch.nn.Linear(1, embed_dim),
            torch.nn.SiLU(),
            torch.nn.Linear(embed_dim, embed_dim)
        )
        self.pos_emb    = torch.nn.Parameter(torch.randn(1, n_patches, embed_dim))
        layer = torch.nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=n_heads, batch_first=True
        )
        self.transformer = torch.nn.TransformerEncoder(layer, num_layers=n_layers)
        self.out_proj   = torch.nn.Linear(embed_dim, patch_size * patch_size)

    def forward(self, x, t, y):
        B = x.size(0)
        patches = (
            x.unfold(2, patch_size, patch_size)
             .unfold(3, patch_size, patch_size)
             .contiguous()
             .view(B, n_patches, -1)
        )
        h  = self.patch_proj(patches)
        c  = self.class_emb(y).unsqueeze(1)
        tm = self.time_mlp(t.unsqueeze(-1))
        h  = h + c + tm.unsqueeze(1) + self.pos_emb
        h  = self.transformer(h)
        out = self.out_proj(h)
        p   = 28 // patch_size
        out = out.view(B, 1, p, p, patch_size, patch_size)
        out = out.permute(0,1,2,4,3,5).contiguous().view(B,1,28,28)
        return out

ckpt_paths = glob.glob("./quantized_models/fmnist/**/*.pt", recursive=True)

for ckpt_path in ckpt_paths:
    parts = ckpt_path.split("/")
    quantizer = parts[3]
    model_name = parts[4].replace(".pt", "")

    if not os.path.exists(ckpt_path):
        print(f"Checkpoint `{ckpt_path}` not found. Exiting.")
        exit(1)
    
    base_dir    = "./sampling/fmnist"
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

    ## ---------------------------------------------------------
    ## Sample per class
    ## ---------------------------------------------------------
    with torch.no_grad():
       for c in range(10):
        print(f"Generating {num_per_cls} images for class {c}...")
        cls_dir = os.path.join(gen_dir, str(c))
        for i in range(num_per_cls):
            y = torch.tensor([c], device=device)
            x = torch.randn(1,1,28,28, device=device) * sigma
            for step in range(num_steps):
                t      = torch.full((1,), step * dt, device=device)
                v_pred = model(x, t, y)
                x      = x + v_pred * dt
            img = (x.clamp(-1, 1) + 1) * 0.5
            save_image(img, os.path.join(cls_dir, f"gen_c{c}_{i:03d}.png"))
    print(f"Generated {10*num_per_cls} images in class subfolders under `{gen_dir}`")
    
    print(f"Finished processing checkpoint `{ckpt_path}`")
    print("---------------------------------------------------------")
print("All done!")
