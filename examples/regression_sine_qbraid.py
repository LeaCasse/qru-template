# -*- coding: utf-8 -*-
"""
regression_sine_qbraid.py

Demo:
1) Train a 1-qubit QRU (RY re-upload) on y = sin(x) over [-π, π] using PennyLane + Torch.
2) Export the trained QRU circuit to OpenQASM 2.0 with qBraid’s PennyLane converter.
3) Run a few test points on an IonQ backend (simulator or hardware) through qBraid’s IonQProvider.

Requirements (conda env already created by you):
    pip install torch --index-url https://download.pytorch.org/whl/cpu
    pip install pennylane~=0.36
    pip install qbraid "qbraid[ionq]"
    pip install -e .

Environment variables:
    set QBRAID_API_KEY=...
    set IONQ_API_KEY=...

Usage:
    python regression_sine_qbraid.py --mode C      # train + CPU preview only
    python regression_sine_qbraid.py --mode Q      # train + CPU preview + IonQ simulator via qBraid
"""

import argparse
import math
from typing import List

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import pennylane as qml

from qru import make_qru_torchlayer, qru_template

# qBraid imports (only needed when mode Q is enabled)
try:
    from qbraid.transpiler.conversions.pennylane import pennylane_to_qasm2
    from qbraid.runtime import IonQProvider
    _QBRAID_AVAILABLE = True
except Exception:
    _QBRAID_AVAILABLE = False


# ---------------------------------------------------------------------------
# 1. Dataset: y = sin(x) on [-π, π]
# ---------------------------------------------------------------------------

def make_sine_dataset(n_points: int = 256):
    xs = torch.linspace(-math.pi, math.pi, n_points).unsqueeze(-1)  # (N,1)
    ys = torch.sin(xs)  # already in [-1,1]
    return xs, ys


# ---------------------------------------------------------------------------
# 2. QRU model (PennyLane + TorchLayer) – implemented in qru_pennylane.py
# ---------------------------------------------------------------------------

def build_model(L: int = 6) -> nn.Module:
    """
    Build a 1-qubit QRU TorchLayer suitable for sine regression.

    - input_norm = "identity"  (x already provided in radians)
    - input_angle_scale = "none" (no multiplication by π)
    - output_range = None (use raw ⟨Z⟩ ∈ [-1,1])
    """
    model = make_qru_torchlayer(
        L=L,
        dev_name="default.qubit",
        wires=1,
        observable=None,            # defaults to PauliZ
        input_norm="identity",
        input_stats=None,
        input_angle_scale="none",
        output_range=None,
        ry_scale_max=1.0,
    )
    return model


# ---------------------------------------------------------------------------
# 3. Training (CPU, default.qubit)
# ---------------------------------------------------------------------------

def train_sine(
    model: nn.Module,
    xs: torch.Tensor,
    ys: torch.Tensor,
    epochs: int = 100,
    lr: float = 1e-2,
    print_every: int = 20,
):
    criterion = nn.MSELoss()
    opt = optim.Adam(model.parameters(), lr=lr)

    model.train()
    for epoch in range(epochs):
        opt.zero_grad()
        y_pred = model(xs)
        loss = criterion(y_pred, ys)
        loss.backward()
        opt.step()
        # Enforce periodic constraints on QRU parameters
        model.constrain_()

        if (epoch + 1) % print_every == 0:
            print(f"[train] epoch={epoch+1:03d}  MSE={loss.item():.5f}")

    return model


# ---------------------------------------------------------------------------
# 4. CPU preview: evaluate a few test points
# ---------------------------------------------------------------------------

def cpu_preview(model: nn.Module, n_points: int = 6):
    model.eval()
    with torch.no_grad():
        xs_test = torch.linspace(-math.pi, math.pi, n_points).unsqueeze(-1)
        ys_true = torch.sin(xs_test)
        ys_pred = model(xs_test)

    print("\n[CPU preview]")
    for x, yt, yp in zip(xs_test, ys_true, ys_pred):
        print(
            f"x = {float(x.item()):+7.3f}   "
            f"sin(x) = {float(yt.item()):+7.4f}   "
            f"QRU(x) = {float(yp.item()):+7.4f}"
        )

    return xs_test, ys_true, ys_pred


# ---------------------------------------------------------------------------
# 5. Extract trained weights for QNode export
# ---------------------------------------------------------------------------

def extract_weights_numpy(model: nn.Module) -> np.ndarray:
    """
    Extract the TorchLayer weights as a numpy array of shape (L,3).
    Assumes the PennyLane layer is stored under 'layer.weights'.
    """
    state = model.state_dict()
    if "layer.weights" not in state:
        raise RuntimeError("Missing 'layer.weights' in state_dict.")
    w = state["layer.weights"].detach().cpu().numpy().astype(np.float32)
    return w


# ---------------------------------------------------------------------------
# 6. QNode for qBraid export (probabilities → counts)
# ---------------------------------------------------------------------------

def make_export_qnode(L: int = 6):
    """
    Build a PennyLane QNode that applies qru_template
    and returns the output probabilities.
    """
    dev = qml.device("default.qubit", wires=1, shots=None)

    @qml.qnode(dev)
    def qnode_export(x, weights):
        qru_template(x, weights)
        return qml.probs(wires=0)

    return qnode_export


# ---------------------------------------------------------------------------
# 7. PennyLane → OpenQASM 2.0 conversion via qBraid
# ---------------------------------------------------------------------------

def strip_measurements_from_qasm2(qasm: str) -> str:
    """
    IonQ/qBraid recommend QASM without explicit 'measure' commands.
    Measurements will be inserted automatically.
    """
    lines: List[str] = []
    for line in qasm.splitlines():
        if "measure" in line:
            continue
        lines.append(line)
    return "\n".join(lines)


def pl_qnode_to_qasm2(qnode_export, x_val: float, weights: np.ndarray) -> str:
    """
    Build the circuit for a given x and convert it to OpenQASM 2.0.
    Compatible with multiple PennyLane versions (tape / qtape / _tape).
    """
    if not _QBRAID_AVAILABLE:
        raise RuntimeError("qbraid is not installed or cannot be imported.")

    # Execute QNode once to build its tape
    _ = qnode_export(x_val, weights)

    # Retrieve the tape depending on PL version
    if hasattr(qnode_export, "tape"):
        tape = qnode_export.tape
    elif hasattr(qnode_export, "qtape"):
        tape = qnode_export.qtape
    elif hasattr(qnode_export, "_tape"):
        tape = qnode_export._tape
    else:
        raise RuntimeError("Cannot access QNode tape (tape/qtape/_tape missing).")

    # qBraid conversion helper
    qasm2 = pennylane_to_qasm2(tape)
    qasm2 = strip_measurements_from_qasm2(qasm2)
    return qasm2


# ---------------------------------------------------------------------------
# 8. Run QASM on an IonQ backend via qBraid
# ---------------------------------------------------------------------------

def run_on_ionq_simulator(
    qasm_list: List[str],
    shots: int = 1000,
    device_id: str = "simulator",
):
    """
    Submit a list of OpenQASM 2.0 circuits to an IonQ backend through qBraid.
    Returns a list of count dictionaries.
    """
    if not _QBRAID_AVAILABLE:
        raise RuntimeError("qbraid is not installed or cannot be imported.")

    provider = IonQProvider()  # reads IONQ_API_KEY from environment
    device = provider.get_device(device_id)

    print(f"\n[qBraid] Using IonQ device: {device_id}")
    print(f"[qBraid] Submitting {len(qasm_list)} circuit(s) with {shots} shots each...")

    job = device.run(qasm_list, shots=shots, name="QRU sine regression (preview)")
    result = job.result()

    counts_batch = result.data.get_counts()
    return counts_batch


def hardware_preview_with_qbraid(
    weights: np.ndarray,
    n_points: int = 3,
    L: int = 6,
    shots: int = 1000,
    device_id: str = "simulator",
):
    """
    Export a few QRU circuits to QASM and execute them on an IonQ backend
    (simulator or hardware) via qBraid.
    """
    if not _QBRAID_AVAILABLE:
        raise RuntimeError("qbraid is not available. Install 'qbraid[ionq]'.")

    qnode_export = make_export_qnode(L=L)

    xs_test = np.linspace(-math.pi, math.pi, n_points)

    qasm_batch = []
    for x in xs_test:
        qasm2 = pl_qnode_to_qasm2(qnode_export, float(x), weights)
        qasm_batch.append(qasm2)

    counts_batch = run_on_ionq_simulator(
        qasm_batch,
        shots=shots,
        device_id=device_id,
    )

    print(f"\n[IonQ via qBraid] device_id = {device_id}")
    for x, counts in zip(xs_test, counts_batch):
        print(f"  x={float(x):+7.3f}  counts={counts}")

    return xs_test, counts_batch


# ---------------------------------------------------------------------------
# 9. Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="QRU sine regression with PennyLane + Torch + qBraid IonQ preview."
    )
    p.add_argument(
        "--mode",
        choices=["C", "Q"],
        default="C",
        help="C = CPU only, Q = CPU + IonQ backend via qBraid",
    )
    p.add_argument("--epochs", type=int, default=100, help="Training epochs")
    p.add_argument("--L", type=int, default=6, help="QRU depth")
    p.add_argument(
        "--device",
        type=str,
        default="simulator",
        help="IonQ device id (simulator, simulator_aria1, simulator_harmony, qpu.harmony, ...)",
    )
    p.add_argument(
        "--shots",
        type=int,
        default=1000,
        help="Shots for the IonQ backend (used only in mode Q)",
    )
    return p.parse_args()


def main():
    args = parse_args()

    # 1) Data
    xs, ys = make_sine_dataset(n_points=256)

    # 2) Model
    model = build_model(L=args.L)

    # 3) Train
    model = train_sine(model, xs, ys, epochs=args.epochs, lr=1e-2, print_every=20)

    # 4) CPU preview
    cpu_preview(model, n_points=6)

    if args.mode.upper() == "Q":
        if not _QBRAID_AVAILABLE:
            raise RuntimeError(
                "Mode Q selected but qbraid is not available. "
                "Install 'qbraid[ionq]' and set IONQ_API_KEY."
            )

        # 5) Export weights and evaluate a few points on IonQ backend
        weights_np = extract_weights_numpy(model)
        hardware_preview_with_qbraid(
            weights=weights_np,
            n_points=1,   # fewer hardware calls
            L=args.L,
            shots=50,     # keep hardware cost low
            device_id=args.device,
        )


if __name__ == "__main__":
    main()
