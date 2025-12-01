# -*- coding: utf-8 -*-
# 3-class classification (faster version): mini-batches + reduced dataset
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from qru import make_qru_torchlayer

# ------------------ "Fast" settings ------------------
SEED = 0
N = 300                # was 1200
L = 4                  # was 6
EVAL_EVERY = 10        # more frequent logs
STEPS = 120            # optimization steps (instead of full epochs)
BATCH_SIZE = 64        # mini-batches
LR = 5e-3              # slightly more stable for mini-batches
# -----------------------------------------------------

# Light determinism
os.environ["PYTHONHASHSEED"] = str(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
# On Windows/CPU, limiting threads can reduce overhead
try:
    torch.set_num_threads(1)
except Exception:
    pass

# 1D data → 3 classes based on thresholds over x: {0,1,2}
xs = np.random.uniform(-1.5, 1.5, size=N).astype(np.float32)

def label_fn(x):
    if x < -0.5: return 0  # class 0 (left)
    if x >  0.5: return 2  # class 2 (right)
    return 1               # class 1 (center)

ys = np.array([label_fn(x) for x in xs], dtype=np.int64)

x_full = torch.tensor(xs).unsqueeze(-1)          # (N,1)
y_full = torch.tensor(ys)                        # (N,)

# Continuous target to train a single output z01 ∈ [0,1]
# 0.0 -> class 0 ; 0.5 -> class 1 ; 1.0 -> class 2
target_full = torch.zeros_like(x_full, dtype=torch.float32)
target_full[y_full == 0] = 0.0
target_full[y_full == 1] = 0.5
target_full[y_full == 2] = 1.0

# Z-score normalization + ×π scaling; output mapped to [0,1]
model = make_qru_torchlayer(
    L=L,
    input_norm="zscore",
    input_stats={"mean": float(x_full.mean()), "std": float(x_full.std()) + 1e-8},
    input_angle_scale="pi",
    output_range=(0.0, 1.0),
    ry_scale_max=1.0,
)

opt = optim.Adam(model.parameters(), lr=LR)
loss_fn = nn.MSELoss()

def discretize01(z01):
    """map [0,1] -> classes {0,1,2} via thresholds 0.33 and 0.66"""
    z = z01.detach()
    cls = torch.zeros_like(z, dtype=torch.long)
    cls[z > 0.66] = 2
    cls[(z >= 0.33) & (z <= 0.66)] = 1
    cls[z < 0.33] = 0
    return cls

# Training loop with mini-batches
for step in range(1, STEPS + 1):
    # sample a mini-batch
    idx = torch.randint(0, N, (BATCH_SIZE,))
    xb = x_full[idx]             # (B,1)
    yb_target = target_full[idx] # (B,1)

    # optimization step
    opt.zero_grad()
    z01b = model(xb)             # (B,1) ∈ [0,1]
    loss = loss_fn(z01b, yb_target)
    loss.backward()
    opt.step()
    model.constrain_()

    if step % EVAL_EVERY == 0:
        # Evaluate on the full dataset (to check global accuracy)
        with torch.no_grad():
            z01_full = model(x_full)                   # (N,1)
            preds = discretize01(z01_full).squeeze(-1) # (N,)
            acc = (preds == y_full).float().mean().item()
        print(f"[{step:03d}] loss(batch)={loss.item():.4f}  acc(full)={acc:.3f}")
