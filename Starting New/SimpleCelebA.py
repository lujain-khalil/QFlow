import os
import time
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import save_image
from tqdm import tqdm

# make sure to pick the right GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# ---------------------------------------------------------
# Config
# ---------------------------------------------------------
device      = 'cuda' if torch.cuda.is_available() else 'cpu'
base_dir    = "NEW_NEWCelebA64_FlowTrans"
data_dir    = "../"
os.makedirs(base_dir, exist_ok=True)
os.makedirs(os.path.join(base_dir, "checkpoints"), exist_ok=True)
os.makedirs(os.path.join(base_dir, "samples"), exist_ok=True)

batch_size       = 32
epochs           = 250
learning_rate    = 1e-4
weight_decay     = 1e-4

sigma            = 0.5
grad_clip        = 1.0

image_size       = 64
patch_size       = 8
n_patches        = (image_size // patch_size) ** 2
embed_dim        = 768
n_heads          = 16
n_layers         = 8
class_drop_prob  = 0.1
attr_dim         = 40

# ---------------------------------------------------------
# Data
# ---------------------------------------------------------
transform = transforms.Compose([
    transforms.CenterCrop(178),
    transforms.Resize(image_size),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

train_ds = datasets.CelebA(
    root=data_dir, split='train', target_type='attr',
    download=True, transform=transform
)
val_ds = datasets.CelebA(
    root=data_dir, split='valid', target_type='attr',
    download=True, transform=transform
)

train_loader = DataLoader(
    train_ds, batch_size=batch_size, shuffle=True,
    num_workers=8, pin_memory=True
)
val_loader = DataLoader(
    val_ds, batch_size=batch_size, shuffle=False,
    num_workers=8, pin_memory=True
)

steps_per_epoch = len(train_loader)
total_steps     = epochs * steps_per_epoch
warmup_steps    = steps_per_epoch * 2

# ---------------------------------------------------------
# Learning rate schedule
# ---------------------------------------------------------
def cosine_warmup_schedule(step):
    if step < warmup_steps:
        return step / float(max(1, warmup_steps))
    progress = (step - warmup_steps) / float(max(1, total_steps - warmup_steps))
    return 0.5 * (1.0 + math.cos(math.pi * progress))

# ---------------------------------------------------------
# Early stopping
# ---------------------------------------------------------
class EarlyStopping:
    def __init__(self, patience, save_path):
        self.patience  = patience
        self.save_path = save_path
        self.best_loss = float('inf')
        self.counter   = 0

    def __call__(self, val_loss, model):
        if val_loss < self.best_loss:
            self.best_loss = val_loss
            self.counter   = 0
            torch.save(model.state_dict(), self.save_path)
        else:
            self.counter += 1
        return self.counter >= self.patience

# ---------------------------------------------------------
# Gaussian Fourier Projection
# ---------------------------------------------------------
class GaussianFourierProjection(nn.Module):
    def __init__(self, embed_dim, scale=10.0):
        super().__init__()
        # fixed random projection
        self.W = nn.Parameter(torch.randn(embed_dim) * scale, requires_grad=False)

    def forward(self, x):
        # x: (B,) ? proj: (B, embed_dim)
        proj = x.unsqueeze(-1) * self.W.unsqueeze(0) * 2 * math.pi
        return torch.cat([proj.sin(), proj.cos()], dim=-1)

# ---------------------------------------------------------
# FlowTransformer
# ---------------------------------------------------------
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

def flow_loss(model, x0, x1, t, y):
    xt       = x0 + (1 - t).view(-1,1,1,1) * (x1 - x0)
    v_target = (x1 - x0)
    v_pred   = model(xt, t, y)
    return F.mse_loss(v_pred, v_target)

# ---------------------------------------------------------
# Instantiate everything
# ---------------------------------------------------------
model     = FlowTransformer().to(device)
optimizer = optim.AdamW(
    model.parameters(), lr=learning_rate, weight_decay=weight_decay
)
scheduler = LambdaLR(optimizer, lr_lambda=cosine_warmup_schedule)
stop      = EarlyStopping(
    patience=10,
    save_path=os.path.join(base_dir, "checkpoints", "best.pt")
)

# sampling parameters
num_steps = 1000
dt        = 1.0 / num_steps

# ---------------------------------------------------------
# Training Loop
# ---------------------------------------------------------
global_step   = 0
best_val_loss = float('inf')

for epoch in range(1, epochs + 1):
    t_start = time.time()
    model.train()
    train_loss = 0.0

    for x, y in tqdm(train_loader, desc=f"Epoch {epoch} Train"):
        x, y   = x.to(device), y.to(device)
        x1, x0 = x, torch.randn_like(x) * sigma
        B      = x.size(0)
        t_rand = torch.rand(B, device=device)

        loss = flow_loss(model, x0, x1, t_rand, y)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        scheduler.step()
        global_step += 1

        train_loss += loss.item() * B
    train_loss /= len(train_loader.dataset)

    # validation
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for x, y in tqdm(val_loader, desc="Validation"):
            x, y   = x.to(device), y.to(device)
            x1, x0 = x, torch.randn_like(x) * sigma
            B      = x.size(0)
            t_rand = torch.rand(B, device=device)
            val_loss += flow_loss(model, x0, x1, t_rand, y).item() * B
    val_loss /= len(val_loader.dataset)

    elapsed = time.time() - t_start
    print(f"Epoch {epoch}: Train {train_loss:.4e}  Val {val_loss:.4e}  Time {elapsed:.1f}s")

    # save best
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        ckpt_path = os.path.join(base_dir, "checkpoints", f"best_epoch{epoch}.pt")
        torch.save(model.state_dict(), ckpt_path)
        print(f"  ? saved best checkpoint at epoch {epoch}")

    # -----------------------------------------------------
    # Unconditional backward sampling from t=1 ? t=0
    # -----------------------------------------------------
    n_samples = 1
    ys        = torch.zeros(n_samples, attr_dim, device=device)
    x_gen     = torch.randn(n_samples, 3, image_size, image_size, device=device) * sigma

    with torch.no_grad():
        for step in range(num_steps):
            t_val = 1.0 - step * dt
            t     = torch.full((n_samples,), t_val, device=device)
            v     = model(x_gen, t, ys)
            x_gen = x_gen + v * dt

    samples = (x_gen.clamp(-1, 1) + 1) * 0.5
    save_image(
        samples,
        os.path.join(base_dir, "samples", f"epoch{epoch:03d}.png")
    )

    if stop(val_loss, model):
        print(f"Early stopping at epoch {epoch}")
        break
