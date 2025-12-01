import torch, numpy as np
from qru import make_qru_torchlayer

# Example data: y in [0, 1]
N = 256
xs = torch.linspace(-3, 3, N).unsqueeze(-1)    # non-normalized distribution
ys = ((torch.sin(xs) + 1.0) * 0.5)             # target ∈ [0,1]

# Input stats (z-score)
mu  = float(xs.mean())
std = float(xs.std())

model = make_qru_torchlayer(
    L=6,
    input_norm="zscore",
    input_stats={"mean": mu, "std": std},
    input_angle_scale="pi",     # normalized x × π
    output_range=(0.0, 1.0),    # map ⟨Z⟩→[0,1]
    ry_scale_max=1.0,           # upper bound for RY scale
)

opt = torch.optim.Adam(model.parameters(), lr=1e-2)
loss_fn = torch.nn.MSELoss()

for epoch in range(100):
    opt.zero_grad()
    y_pred = model(xs)
    loss = loss_fn(y_pred, ys)
    loss.backward()
    opt.step()
    model.constrain_()          # <<< ANGULAR CONSTRAINT AFTER step
    if (epoch+1) % 20 == 0:
        print(f"[{epoch+1:03d}] loss={loss.item():.5f}")
