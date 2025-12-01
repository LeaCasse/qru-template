# tests/test_shapes.py
# -*- coding: utf-8 -*-
import torch
from qru.qru_pennylane import make_qru_torchlayer

def test_forward_shape():
    model = make_qru_torchlayer(L=3)
    x = torch.randn(10, 1)
    y = model(x)
    assert y.shape == (10, 1)
    assert torch.isfinite(y).all()
