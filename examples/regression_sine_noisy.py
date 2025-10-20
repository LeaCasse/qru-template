# -*- coding: utf-8 -*-
"""
Regression of y = sin(x) using a 1-qubit Quantum Re-Uploading Unit (QRU)
under realistic NISQ-like noise conditions (default.mixed + finite shots).

This script:
- Builds a noisy QRU layer with Depolarizing + Damping channels
- Trains on a simple sine dataset
- Logs final MSE to a CSV file for later analysis

Usage:
    python examples/regression_sine_noisy.py --p 0.001 --g 0.001
    python examples/regression_sine_noisy.py --sweep quick
"""

import os
import csv
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import pennylane as qml

# ---------------- Configuration ----------------
SEED = 0
N = 128              # dataset size
L = 4                # circuit depth
EPOCHS = 60
LR = 1e-2
SHOTS = 1000
RESULTS_DIR = "results"
RESULTS_CSV = os.path.join(RESULTS_DIR, "noise_qru.csv")
# ------------------------------------------------

# Determinism and CPU-friendly setup
os.environ["PYTHONHASHSEED"] = str(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
try:
    torch.set_num_threads(1)
except Exception:
    pass


def build_noisy_qru_torchlayer(L, p, g, shots=SHOTS):
    """Build a Torch-compatible QRU layer with quantum noise injected after each layer."""
    dev = qml.device("default.mixed", wires=1, shots=shots)

    @qml.qnode(dev, interface="torch")
    def circuit(inputs, weights):
        x = inputs
        for l in range(len(weights)):
            qml.RX(weights[l, 0], wires=0)

            # Ensure scalar angle for RY
            angle = weights[l, 1] * x
            try:
                angle = qml.math.squeeze(angle)
            except Exception:
                pass
            qml.RY(angle, wires=0)
            qml.RZ(weights[l, 2], wires=0)

            # Inject noise channels after each layer
            if p > 0.0:
                qml.DepolarizingChannel(p, wires=0)
            if g > 0.0:
                qml.AmplitudeDamping(g, wires=0)
                qml.PhaseDamping(g, wires=0)

        return qml.expval(qml.PauliZ(0))

    weight_shapes = {"weights": (L, 3)}
    qlayer = qml.qnn.TorchLayer(circuit, weight_shapes)

    class NoisyQRU(nn.Module):
        def __init__(self):
            super().__init__()
            self.layer = qlayer
            self._ry_scale_max = 1.0

        @torch.no_grad()
        def constrain_(self):
            """Post-update parameter constraints:
            - Wrap RX/RZ angles within (-π, π)
            - Clamp RY scales to [0, ry_scale_max]
            """
            for p_ in self.parameters():
                if p_.data.ndim == 2 and p_.data.shape == (L, 3):
                    p_.data[:, 0] = ((p_.data[:, 0] + np.pi) % (2 * np.pi)) - np.pi
                    p_.data[:, 2] = ((p_.data[:, 2] + np.pi) % (2 * np.pi)) - np.pi
                    p_.data[:, 1].clamp_(min=0.0, max=self._ry_scale_max)

        def forward(self, x):
            """Batch-safe forward loop for PL ≤0.36 (no vectorized evaluation)."""
            if hasattr(x, "ndim") and x.ndim == 2 and x.shape[0] > 1:
                outs = []
                for i in range(x.shape[0]):
                    xi = x[i:i + 1]
                    yi = self.layer(xi)
                    if hasattr(yi, "ndim") and yi.ndim == 1:
                        yi = yi.unsqueeze(-1)
                    outs.append(yi)
                return torch.cat(outs, dim=0)
            out = self.layer(x)
            if hasattr(out, "ndim") and out.ndim == 1:
                out = out.unsqueeze(-1)
            return out

    return NoisyQRU()


def run_single(p, g):
    """Train QRU on sin(x) for given noise parameters p and g."""
    xs = torch.linspace(-np.pi, np.pi, N).unsqueeze(-1)
    ys = torch.sin(xs)

    model = build_noisy_qru_torchlayer(L=L, p=p, g=g, shots=SHOTS)
    opt = optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.MSELoss()

    for epoch in range(1, EPOCHS + 1):
        opt.zero_grad()
        y_pred = model(xs)
        loss = loss_fn(y_pred, ys)
        loss.backward()
        opt.step()
        model.constrain_()
        if epoch % 10 == 0:
            print(f"[{epoch:03d}] p={p} g={g} shots={SHOTS}  MSE={loss.item():.5f}")

    return float(loss.item())


def ensure_results_csv_header(path):
    """Create the CSV file with header if it doesn't exist."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if not os.path.exists(path):
        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["p", "gamma", "shots", "L", "N", "epochs", "lr", "seed", "train_mse"])


def append_result(path, p, g, mse):
    """Append one experiment result to the CSV."""
    with open(path, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([p, g, SHOTS, L, N, EPOCHS, LR, SEED, mse])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--p", type=float, default=None, help="Depolarizing probability")
    parser.add_argument("--g", type=float, default=None, help="Gamma for damping channels")
    parser.add_argument("--sweep", type=str, default=None, help="quick | full")
    args = parser.parse_args()

    ensure_results_csv_header(RESULTS_CSV)

    if args.sweep is None:
        if args.p is None or args.g is None:
            raise SystemExit("Specify --p and --g, or use --sweep quick/full")
        mse = run_single(args.p, args.g)
        append_result(RESULTS_CSV, args.p, args.g, mse)
    else:
        if args.sweep == "quick":
            Ps = [0.0, 0.001, 0.01]
            Gs = [0.0, 0.001, 0.01]
        elif args.sweep == "full":
            Ps = [0.0, 0.001, 0.01, 0.05]
            Gs = [0.0, 0.001, 0.01, 0.05]
        else:
            raise SystemExit("Invalid value for --sweep (quick|full)")

        for p in Ps:
            for g in Gs:
                print(f"\n=== Run p={p}, g={g} ===")
                mse = run_single(p, g)
                append_result(RESULTS_CSV, p, g, mse)


if __name__ == "__main__":
    main()
