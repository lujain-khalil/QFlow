import os
import time
import torch
import torch.nn.functional as F
from torch import optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from torchvision.utils import save_image
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# ---------------------------------------------------------
# Config
# ---------------------------------------------------------
device = 'cuda' if torch.cuda.is_available() else 'cpu'
base_dir = "FMNIST_FlowTrans" # Changed from MNIST_FlowTrans
data_dir = "."
os.makedirs(base_dir, exist_ok=True)
os.makedirs(os.path.join(base_dir, "checkpoints"), exist_ok=True)
os.makedirs(os.path.join(base_dir, "samples"), exist_ok=True)

batch_size      = 128
epochs          = 100
lr              = 2e-4
patience        = 10
sigma           = 0.5

# Transformer patch dims
patch_size = 4
n_patches  = (28 // patch_size) ** 2
embed_dim  = 384
n_heads    = 12
n_layers   = 10

# ---------------------------------------------------------
# Data
# ---------------------------------------------------------
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Lambda(lambda t: t * 2 - 1),
])
# Changed to FashionMNIST
full_train   = datasets.FashionMNIST(data_dir, train=True,  download=True, transform=transform)
train_ds, val_ds = random_split(full_train, [55000, 5000])
train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=4)
val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=4)

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
# FlowTransformer model
# ---------------------------------------------------------
class FlowTransformer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.patch_proj = torch.nn.Linear(patch_size * patch_size, embed_dim)
        self.class_emb  = torch.nn.Embedding(10, embed_dim) # FMNIST also has 10 classes
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
        out = out.permute(0,1,2,4,3,5).contiguous().view(B,1,28,28) # FMNIST images are 28x28
        return out

def flow_loss(model, x0, x1, t, y):
    xt       = (1 - t).view(-1,1,1,1) * x0 + t.view(-1,1,1,1) * x1
    v_target = x1 - x0
    v_pred   = model(xt, t, y)
    return F.mse_loss(v_pred, v_target)

# ---------------------------------------------------------
# Instantiate
# ---------------------------------------------------------
model     = FlowTransformer().to(device)
opt       = optim.AdamW(model.parameters(), lr=lr)
scheduler = ReduceLROnPlateau(opt, mode='min', factor=0.5, patience=3, verbose=True)
stop      = EarlyStopping(patience, os.path.join(base_dir, "checkpoints", "best.pt"))
train_logs = []

# ---------------------------------------------------------
# Training Loop (with single-shot sampling at t=0)
# ---------------------------------------------------------
best_vl_loss = float('inf')

for epoch in range(1, epochs+1):
    t_epoch = time.perf_counter()

    # -- TRAIN --
    model.train()
    tr_loss = 0.0
    for x, y in tqdm(train_loader, desc=f"Epoch {epoch} --> Train", leave=False):
        x, y   = x.to(device), y.to(device)
        x1, x0 = x, torch.randn_like(x) * sigma
        B      = x1.size(0)
        t      = torch.rand(B, device=device)
        loss   = flow_loss(model, x0, x1, t, y)
        opt.zero_grad(); loss.backward(); opt.step()
        tr_loss += loss.item() * B
    tr_loss /= len(train_loader.dataset)

    # -- VALIDATE --
    model.eval()
    vl_loss = 0.0
    with torch.no_grad():
        for x, y in tqdm(val_loader, desc=f"Epoch {epoch} --> Val  ", leave=False):
            x, y   = x.to(device), y.to(device)
            x1, x0 = x, torch.randn_like(x) * sigma
            B      = x1.size(0)
            t      = torch.rand(B, device=device)
            vl_loss += flow_loss(model, x0, x1, t, y).item() * B
    vl_loss /= len(val_loader.dataset)

    dur = time.perf_counter() - t_epoch
    train_logs.append({
        "epoch": epoch,
        "train_loss": tr_loss,
        "val_loss": vl_loss,
        "epoch_time_s": dur
    })
    print(f"Epoch {epoch}: Train {tr_loss:.4e}  Val {vl_loss:.4e}  Time {dur:.1f}s")

    # -- LR SCHEDULER STEP --
    scheduler.step(vl_loss)

    # -- SAVE BEST CHECKPOINT --
    if vl_loss < best_vl_loss:
        best_vl_loss = vl_loss
        ckpt = os.path.join(
            base_dir, "checkpoints", f"best_epoch{epoch:03d}.pt"
        )
        torch.save(model.state_dict(), ckpt)
        print(f"  --> Saved new best checkpoint: {ckpt} (val_loss={vl_loss:.4e})")

       # -- 500-STEP SAMPLING --
    with torch.no_grad():
        y_samp = torch.randint(0, 10, (1,), device=device) # FMNIST also has 10 classes
        x = torch.randn(1, 1, 28, 28, device=device) * sigma # FMNIST images are 28x28
        
        num_steps = 500
        dt = 1.0 / num_steps
        
        for step in range(num_steps):
            t_val = torch.full((1,), step * dt, device=device) # Renamed t to t_val to avoid conflict
            v_pred = model(x, t_val, y_samp)
            x = x + v_pred * dt
        
        img = (x.clamp(-1, 1) + 1) * 0.5
        save_image(
            img,
            os.path.join(
                base_dir, "samples",
                f"epoch{epoch:03d}_cls{y_samp.item()}.png"
            )
        )

    # -- EARLY STOPPING --
    if stop(vl_loss, model):
        print(f"Early stopping at epoch {epoch}")
        break