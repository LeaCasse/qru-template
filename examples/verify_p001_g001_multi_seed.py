# -*- coding: utf-8 -*-
"""
Multi-seed verification that the noisy QRU (p = g = 0.01) achieves low MSE on y = sin(x).

Usage (defaults: seeds=5, epochs=60, shots=1000, L=4, N=128, lr=1e-2, threshold=0.02):
    python examples/verify_p001_g001_multi_seed.py
    python examples/verify_p001_g001_multi_seed.py --seeds 10 --epochs 60 --shots 1000 --threshold 0.02

It prints per-seed final MSEs and a summary (mean/std/min/max),
writes results to results/noise_qru_p001_g001_seeds.csv,
and exits with non-zero code if mean MSE > threshold.
"""

import os
import csv
import math
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import pennylane as qml


# ---------------- Defaults (aligned with your base script) ----------------
DEFAULT_N = 128         # dataset size
DEFAULT_L = 4           # circuit depth
DEFAULT_EPOCHS = 60
DEFAULT_LR = 1e-2
DEFAULT_SHOTS = 1000
DEFAULT_THRESHOLD = 0.02
P_FIXED = 0.01
G_FIXED = 0.01

RESULTS_DIR = "results"
RESULTS_CSV = os.path.join(RESULTS_DIR, "noise_qru_p001_g001_seeds.csv")
# -------------------------------------------------------------------------


def set_all_seeds(seed: int):
    """Deterministic setup per-seed."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    try:
        torch.set_num_threads(1)
    except Exception:
        pass


def build_noisy_qru_torchlayer(L: int, p: float, g: float, shots: int):
    """Build a Torch-compatible QRU layer with noise after each layer."""
    dev = qml.device("default.mixed", wires=1, shots=shots)

    @qml.qnode(dev, interface="torch")
    def circuit(inputs, weights):
        x = inputs
        for l in range(len(weights)):
            qml.RX(weights[l, 0], wires=0)

            # ensure scalar angle for RY on PL 0.36
            angle = weights[l, 1] * x
            try:
                angle = qml.math.squeeze(angle)
            except Exception:
                pass
            qml.RY(angle, wires=0)
            qml.RZ(weights[l, 2], wires=0)

            # noise per layer
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
            # wrap RX/RZ into (-pi, pi], clamp RY scale
            for p_ in self.parameters():
                if p_.data.ndim == 2 and p_.data.shape == (L, 3):
                    p_.data[:, 0] = ((p_.data[:, 0] + math.pi) % (2 * math.pi)) - math.pi
                    p_.data[:, 2] = ((p_.data[:, 2] + math.pi) % (2 * math.pi)) - math.pi
                    p_.data[:, 1].clamp_(min=0.0, max=self._ry_scale_max)

        def forward(self, x):
            # batch-safe forward loop for PL ≤0.36
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


def run_single_seed(seed: int, N: int, L: int, epochs: int, lr: float, shots: int) -> float:
    """Train the QRU on y=sin(x) for a given seed; return final MSE."""
    set_all_seeds(seed)

    xs = torch.linspace(-math.pi, math.pi, N).unsqueeze(-1)
    ys = torch.sin(xs)

    model = build_noisy_qru_torchlayer(L=L, p=P_FIXED, g=G_FIXED, shots=shots)
    opt = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    final_mse = None
    for epoch in range(1, epochs + 1):
        opt.zero_grad()
        y_pred = model(xs)
        loss = loss_fn(y_pred, ys)
        loss.backward()
        opt.step()
        model.constrain_()

        if epoch % 10 == 0:
            print(f"[seed={seed:03d} epoch={epoch:03d}] p={P_FIXED} g={G_FIXED} shots={shots}  MSE={loss.item():.5f}")

        final_mse = float(loss.item())

    return final_mse


def ensure_results_csv_header(path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if not os.path.exists(path):
        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["p", "gamma", "shots", "L", "N", "epochs", "lr", "seed", "train_mse"])


def append_result(path: str, seed: int, mse: float, N: int, L: int, epochs: int, lr: float, shots: int):
    with open(path, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([P_FIXED, G_FIXED, shots, L, N, epochs, lr, seed, mse])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=5, help="Number of seeds to run (default: 5)")
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--shots", type=int, default=DEFAULT_SHOTS)
    parser.add_argument("--L", type=int, default=DEFAULT_L)
    parser.add_argument("--N", type=int, default=DEFAULT_N)
    parser.add_argument("--lr", type=float, default=DEFAULT_LR)
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD,
                        help="Assert mean MSE <= threshold; exit non-zero if violated")
    args = parser.parse_args()

    ensure_results_csv_header(RESULTS_CSV)

    mses = []
    for i in range(args.seeds):
        seed = i  # seeds: 0..seeds-1
        mse = run_single_seed(seed=seed, N=args.N, L=args.L, epochs=args.epochs, lr=args.lr, shots=args.shots)
        mses.append(mse)
        append_result(RESULTS_CSV, seed=seed, mse=mse, N=args.N, L=args.L, epochs=args.epochs, lr=args.lr, shots=args.shots)

    # summary
    mean_mse = float(np.mean(mses))
    std_mse = float(np.std(mses, ddof=0))
    min_mse = float(np.min(mses))
    max_mse = float(np.max(mses))

    print("\n=== Multi-seed summary (p=0.01, g=0.01) ===")
    print(f"seeds         : {args.seeds}")
    print(f"epochs/shots  : {args.epochs} / {args.shots}")
    print(f"L / N / lr    : {args.L} / {args.N} / {args.lr}")
    print(f"MSEs          : {', '.join(f'{m:.5f}' for m in mses)}")
    print(f"mean ± std    : {mean_mse:.5f} ± {std_mse:.5f}")
    print(f"min .. max    : {min_mse:.5f} .. {max_mse:.5f}")
    print(f"threshold     : {args.threshold:.5f}")

    # fail fast if mean MSE is not low enough
    if mean_mse > args.threshold:
        print(f"\n[FAIL] mean MSE {mean_mse:.5f} > threshold {args.threshold:.5f}")
        raise SystemExit(1)
    else:
        print(f"\n[OK] mean MSE {mean_mse:.5f} ≤ threshold {args.threshold:.5f}")


if __name__ == "__main__":
    main()
