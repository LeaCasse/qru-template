# -*- coding: utf-8 -*-
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from qru import make_qru_torchlayer

# 1D data → 3 classes based on thresholds over x: {-1, 0, +1}
N = 1200
xs = np.random.uniform(-1.5, 1.5, size=N).astype(np.float32)

def label_fn(x):
    if x < -0.5: return 0  # class 0
    if x >  0.5: return 2  # class 2
    return 1               # class 1 (center)

ys = np.array([label_fn(x) for x in xs], dtype=np.int64)

x = torch.tensor(xs).unsqueeze(-1)
y = torch.tensor(ys)

# We keep a single output ⟨Z⟩ → rescaled to [0,1] for BCE
# Then we learn 3 thresholds via a small linear + softmax head OVR (simple & stable)
# Simplification: train the quantum output to approximate a “continuous target”
# (here, we push the output toward 0.0, 0.5, 1.0 for the 3 classes).
target_cont = torch.zeros_like(x, dtype=torch.float32)
target_cont[y == 0] = 0.0
target_cont[y == 1] = 0.5
target_cont[y == 2] = 1.0

model = make_qru_torchlayer(
    L=6,
    input_norm="zscore",                      # stabilizes the input
    input_stats={"mean": float(x.mean()), "std": float(x.std()) + 1e-8},
    input_angle_scale="pi",                   # map → ×π to stay within a sensitive range
    output_range=(0.0, 1.0),                  # ⟨Z⟩∈[-1,1] → [0,1]
    ry_scale_max=1.0,
)

opt = optim.Adam(model.parameters(), lr=1e-2)
loss_fn = nn.MSELoss()

def discretize01(z):
    # map [0,1] -> classes {0,1,2} via thresholds at ~0.33 and ~0.66
    z = z.detach()
    cls = torch.zeros_like(z, dtype=torch.long)
    cls[z > 0.66] = 2
    cls[(z >= 0.33) & (z <= 0.66)] = 1
    cls[z < 0.33] = 0
    return cls

for epoch in range(100):
    opt.zero_grad()
    z01 = model(x)                 # (N,1) in [0,1]
    loss = loss_fn(z01, target_cont)
    loss.backward()
    opt.step()
    model.constrain_()

    if (epoch + 1) % 20 == 0:
        with torch.no_grad():
            preds = discretize01(z01).squeeze(-1)
            acc = (preds == y).float().mean().item()
        print(f"[{epoch+1:03d}] loss={loss.item():.4f}  acc={acc:.3f}")
