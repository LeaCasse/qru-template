# -*- coding: utf-8 -*-
# 3-class classification with CrossEntropy (fast, stable)
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from qru import make_qru_torchlayer

# ------------------ "Fast" settings ------------------
SEED = 0
N = 300                # reduced dataset for speed
L = 4                  # QRU depth
EVAL_EVERY = 10
STEPS = 150            # optimization iterations
BATCH_SIZE = 64
LR = 5e-3
# -----------------------------------------------------

# Light determinism + reduce CPU overhead on Windows
os.environ["PYTHONHASHSEED"] = str(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
try:
    torch.set_num_threads(1)
except Exception:
    pass

# ----- Data (3 classes by thresholds) -----
xs = np.random.uniform(-1.5, 1.5, size=N).astype(np.float32)

def label_fn(x):
    if x < -0.5: return 0
    if x >  0.5: return 2
    return 1

ys = np.array([label_fn(x) for x in xs], dtype=np.int64)

x_full = torch.tensor(xs).unsqueeze(-1)   # (N,1)
y_full = torch.tensor(ys)                 # (N,)

# ----- Model: QRU + linear head (1 -> 3 logits) -----
qru = make_qru_torchlayer(
    L=L,
    input_norm="zscore",
    input_stats={"mean": float(x_full.mean()), "std": float(x_full.std()) + 1e-8},
    input_angle_scale="pi",
    output_range=None,     # we want features in [-1,1], not [0,1]
    ry_scale_max=1.0,
)

head = nn.Linear(1, 3, bias=True)  # 1 feature (⟨Z⟩) -> 3 classes
model = nn.Sequential(qru, head)

opt = optim.Adam(model.parameters(), lr=LR)
loss_fn = nn.CrossEntropyLoss()

def evaluate():
    with torch.no_grad():
        z = qru(x_full)               # (N,1)
        logits = head(z)              # (N,3)
        preds = logits.argmax(dim=1)  # (N,)
        acc = (preds == y_full).float().mean().item()
    return acc

# ----- Mini-batch training -----
for step in range(1, STEPS + 1):
    idx = torch.randint(0, N, (BATCH_SIZE,))
    xb = x_full[idx]
    yb = y_full[idx]

    opt.zero_grad()
    feats = qru(xb)           # (B,1) ∈ [-1,1]
    logits = head(feats)      # (B,3)
    loss = loss_fn(logits, yb)
    loss.backward()
    opt.step()
    qru.constrain_()          # angular constraint on the quantum part

    if step % EVAL_EVERY == 0:
        acc = evaluate()
        print(f"[{step:03d}] loss(batch)={loss.item():.4f}  acc(full)={acc:.3f}")
