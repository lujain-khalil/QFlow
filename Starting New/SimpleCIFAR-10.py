import os
import time
import math
import torch
import torch.nn.functional as F
from torch import optim, nn
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingWarmRestarts, SequentialLR
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from torchvision.utils import save_image
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')

# ---------------------------------------------------------
# Config
# ---------------------------------------------------------
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
device     = 'cuda' if torch.cuda.is_available() else 'cpu'
base_dir   = "CIFAR10_FlowTrans_fwd"
os.makedirs(base_dir, exist_ok=True)
os.makedirs(os.path.join(base_dir, "checkpoints"), exist_ok=True)
os.makedirs(os.path.join(base_dir, "samples"), exist_ok=True)

batch_size   = 128
epochs       = 100
base_lr      = 5e-4
patience     = 10
sigma        = 0.5

# Transformer / patch dims for 32x32 CIFAR
patch_size = 4
n_patches  = (32 // patch_size) ** 2   # 8x8 = 64
embed_dim  = 768                       # bumped up from 576
n_heads    = 16
n_layers   = 12

# LR scheduling
warmup_steps = 1000

# ---------------------------------------------------------
# Data (CIFAR-10)
# ---------------------------------------------------------
transform = transforms.Compose([
    transforms.ToTensor(),                              # [0,1]
    transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5)),   # [-1,1]
])

full_train = datasets.CIFAR10('.', train=True, download=True, transform=transform)
train_ds, val_ds = random_split(full_train, [45000, 5000])
train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=4, pin_memory=True)
val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

# ---------------------------------------------------------
# EarlyStopping helper
# ---------------------------------------------------------
class EarlyStopping:
    def __init__(self, patience, path):
        self.patience, self.path = patience, path
        self.best_loss, self.counter = float('inf'), 0

    def __call__(self, val_loss, model):
        if val_loss < self.best_loss:
            self.best_loss, self.counter = val_loss, 0
            torch.save(model.state_dict(), self.path)
        else:
            self.counter += 1
        return self.counter >= self.patience

# ---------------------------------------------------------
# Sinusoidal time-embedding
# ---------------------------------------------------------
def sinusoidal_embedding(t, dim):
    half = dim // 2
    freq = torch.exp(-math.log(10000) * torch.arange(half, device=t.device) / (half - 1))
    args = t.unsqueeze(1) * freq.unsqueeze(0)
    return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)

# ---------------------------------------------------------
# FlowTransformer model
# ---------------------------------------------------------
class FlowTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.patch_proj = nn.Linear(patch_size * patch_size * 3, embed_dim)
        self.class_emb  = nn.Embedding(10, embed_dim)
        self.time_mlp   = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.SiLU(),
            nn.Linear(embed_dim, embed_dim),
        )
        self.pos_emb    = nn.Parameter(torch.randn(1, n_patches, embed_dim))

        layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=n_heads, batch_first=True, norm_first=True
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.out_proj   = nn.Linear(embed_dim, patch_size * patch_size * 3)

    def forward(self, x, t, y):
        B = x.size(0)
        patches = (
            x.unfold(2, patch_size, patch_size)
             .unfold(3, patch_size, patch_size)
             .contiguous()
             .view(B, n_patches, -1)
        )  # B x 64 x (4*4*3)
        h     = self.patch_proj(patches)                # B x 64 x embed_dim
        c     = self.class_emb(y).unsqueeze(1)           # B x 1 x embed_dim
        t_emb = sinusoidal_embedding(t, embed_dim)       # B x embed_dim
        tm    = self.time_mlp(t_emb).unsqueeze(1)        # B x 1 x embed_dim
        h     = h + c + tm + self.pos_emb
        h     = self.transformer(h)                      # B x 64 x embed_dim

        out = self.out_proj(h)                           # B x 64 x (4*4*3)
        p   = 32 // patch_size                           # 8
        out = out.view(B, 3, p, p, patch_size, patch_size)
        out = out.permute(0,1,2,4,3,5).contiguous().view(B,3,32,32)
        return out

def flow_loss(model, x0, x1, t, y):
    xt       = (1 - t).view(-1,1,1,1) * x0 + t.view(-1,1,1,1) * x1
    v_target = x1 - x0
    v_pred   = model(xt, t, y)
    return F.mse_loss(v_pred, v_target)

# ---------------------------------------------------------
# Instantiate model, optimizer, scheduler
# ---------------------------------------------------------
model = FlowTransformer().to(device)
opt   = optim.AdamW(model.parameters(), lr=base_lr)

warmup_scheduler = LinearLR(opt, start_factor=0.1, total_iters=warmup_steps)
cosine_scheduler = CosineAnnealingWarmRestarts(opt, T_0=10, T_mult=2)
scheduler = SequentialLR(
    opt,
    schedulers=[warmup_scheduler, cosine_scheduler],
    milestones=[warmup_steps]
)

stop     = EarlyStopping(patience, os.path.join(base_dir, "checkpoints", "best.pt"))
best_vl  = float('inf')

# ---------------------------------------------------------
# Training loop + forward sampling
# ---------------------------------------------------------
for epoch in range(1, epochs+1):
    t0 = time.perf_counter()
    model.train()
    tr_loss = 0.0

    for x, y in tqdm(train_loader, desc=f"Epoch {epoch} Train", leave=False):
        x, y = x.to(device), y.to(device)
        x1   = x
        x0   = torch.randn_like(x) * sigma
        B    = x1.size(0)
        t    = torch.rand(B, device=device)

        loss = flow_loss(model, x0, x1, t, y)
        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        opt.step()
        scheduler.step()

        tr_loss += loss.item() * B
    tr_loss /= len(train_loader.dataset)

    model.eval()
    vl_loss = 0.0
    with torch.no_grad():
        for x, y in tqdm(val_loader, desc=f"Epoch {epoch} Val  ", leave=False):
            x, y = x.to(device), y.to(device)
            x1   = x
            x0   = torch.randn_like(x) * sigma
            B    = x1.size(0)
            t    = torch.rand(B, device=device)
            vl_loss += flow_loss(model, x0, x1, t, y).item() * B
    vl_loss /= len(val_loader.dataset)

    dur = time.perf_counter() - t0
    print(f"Epoch {epoch}: Train {tr_loss:.4e}  Val {vl_loss:.4e}  Time {dur:.1f}s")

    if vl_loss < best_vl:
        best_vl = vl_loss
        ckpt = os.path.join(base_dir, "checkpoints", f"best_epoch{epoch:03d}.pt")
        torch.save(model.state_dict(), ckpt)
        print(f"  --> Saved new best checkpoint: {ckpt}")

    # -- forward sampling (0 --> 1) adapted from FMNIST --
    with torch.no_grad():
        y_samp = torch.randint(0, 10, (1,), device=device)
        x_s    = torch.randn(1, 3, 32, 32, device=device) * sigma

        num_steps = 500
        dt        = 1.0 / num_steps

        for step in range(num_steps):
            t_val = torch.full((1,), step * dt, device=device)
            v_pred = model(x_s, t_val, y_samp)
            x_s = x_s + v_pred * dt

        img = (x_s.clamp(-1, 1) + 1) * 0.5
        save_image(
            img,
            os.path.join(
                base_dir, "samples",
                f"epoch{epoch:03d}_cls{y_samp.item()}.png"
            )
        )

    if stop(vl_loss, model):
        print(f"Early stopping at epoch {epoch}")
        break
