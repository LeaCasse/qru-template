# -*- coding: utf-8 -*-
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from qru import make_qru_torchlayer

# Data: y = sin(x) over [-π, π]
N = 256
xs = torch.linspace(-np.pi, np.pi, N).unsqueeze(-1)
ys = torch.sin(xs)  # already in [-1,1] → no need to rescale the output

# Here, no input normalization (inputs already in radians), no output_range
model = make_qru_torchlayer(
    L=6,
    input_norm="identity",
    input_angle_scale="none",   # we don’t multiply (x is already in radians)
    output_range=None,          # output ⟨Z⟩ ~ [-1,1]
    ry_scale_max=1.0,
)

criterion = nn.MSELoss()
opt = optim.Adam(model.parameters(), lr=1e-2)

for epoch in range(100):
    opt.zero_grad()
    y_pred = model(xs)
    loss = criterion(y_pred, ys)
    loss.backward()
    opt.step()
    model.constrain_()
    if (epoch + 1) % 20 == 0:
        print(f"[{epoch+1:03d}] MSE={loss.item():.5f}")
