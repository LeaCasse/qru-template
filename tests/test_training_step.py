# tests/test_training_step.py
# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.optim as optim
from qru.qru_pennylane import make_qru_torchlayer

def test_single_step_decreases_loss():
    torch.manual_seed(0)
    model = make_qru_torchlayer(L=4)
    x = torch.linspace(-1, 1, 64).unsqueeze(-1)
    y = torch.sin(x * 3.1415)  #  non triviale cible
    loss_fn = nn.MSELoss()
    opt = optim.SGD(model.parameters(), lr=0.05)

    y_pred0 = model(x); loss0 = loss_fn(y_pred0, y).item()
    opt.zero_grad(); loss_fn(y_pred0, y).backward(); opt.step()
    y_pred1 = model(x); loss1 = loss_fn(y_pred1, y).item()

    assert loss1 <= loss0 or abs(loss1 - loss0) < 1e-6
