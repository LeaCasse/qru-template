# -*- coding: utf-8 -*-
"""
Minimal 1-qubit QRU compatible with PennyLane 0.36 + TorchLayer.

- Circuit template: RX(θ_l0) → RY(θ_l1 * x) → RZ(θ_l2) for l=0..L-1
- Scalar RY angles (squeeze) for PL 0.36 compatibility
- Batch handled on the forward side (loop over N samples)
"""

import numpy as np
import pennylane as qml
import torch
import torch.nn as nn

__all__ = ["qru_template", "make_qru_qnode", "make_qru_torchlayer"]


# ---------------------------------------------------------------------------
# QRU circuit template
# ---------------------------------------------------------------------------

def qru_template(inputs, weights):
    """Minimal QRU(1q); supports scalar or shape-(1,1) inputs.

    Args:
        inputs: Data input x (scalar-like, possibly tensor of shape (1,1)).
        weights: Array-like of shape (L, 3) with:
            - weights[l, 0] → RX angle
            - weights[l, 1] → scale factor for RY * x
            - weights[l, 2] → RZ angle
    """
    x = inputs
    L = len(weights)
    for l in range(L):
        qml.RX(weights[l, 0], wires=0)

        angle = weights[l, 1] * x
        # PennyLane 0.36: RY expects a scalar, so we squeeze it.
        try:
            angle = qml.math.squeeze(angle)
        except Exception:
            pass
        qml.RY(angle, wires=0)

        qml.RZ(weights[l, 2], wires=0)


# ---------------------------------------------------------------------------
# Raw QNode factory (no Torch)
# ---------------------------------------------------------------------------

def make_qru_qnode(
    dev_name: str = "default.qubit",
    L: int = 6,
    wires: int = 1,
    observable=None,
    dev_kwargs: dict | None = None,
):
    """Create a plain PennyLane QNode implementing the QRU.

    Returns:
        qnode(inputs, weights), init_weights(rng)
    """
    if observable is None:
        observable = qml.PauliZ(0)

    dev_kwargs = {} if dev_kwargs is None else dict(dev_kwargs)
    dev = qml.device(dev_name, wires=wires, **dev_kwargs)

    @qml.qnode(dev, interface="torch")
    def qnode(inputs, weights):
        qru_template(inputs, weights)
        return qml.expval(observable)

    def init_weights(rng=None):
        rng = np.random.RandomState() if rng is None else rng
        w = np.zeros((L, 3), dtype=np.float32)
        # RX and RZ in [-pi, pi]
        w[:, 0] = rng.uniform(-np.pi, np.pi, size=L)
        w[:, 2] = rng.uniform(-np.pi, np.pi, size=L)
        # RY scale in [0, 1]
        w[:, 1] = rng.uniform(0.0, 1.0, size=L)
        return w

    return qnode, init_weights


# ---------------------------------------------------------------------------
# TorchLayer wrapper with preprocessing / postprocessing / constraints
# ---------------------------------------------------------------------------

def make_qru_torchlayer(
    L: int = 6,
    dev_name: str = "default.qubit",
    wires: int = 1,
    observable=None,
    *,
    dev_kwargs: dict | None = None,
    # -------- Input preprocessing --------
    input_norm: str = "identity",     # "identity" | "zscore" | "minmax"
    input_stats: dict | None = None,  # {"mean":..., "std":...} or {"min":..., "max":...}
    input_angle_scale: str = "pi",    # "pi" (→ ×π) | "2pi" (→ ×2π) | "none"
    # -------- Output postprocessing --------
    output_range: tuple | None = None,  # (a,b) to map ⟨Z⟩∈[-1,1] → [a,b]
    # -------- Parameter constraints --------
    ry_scale_max: float = 1.0,        # upper bound for column 1 (RY scale)
):
    """Create a torch.nn.Module wrapping a QRU TorchLayer.

    Features:
      - Input normalization (zscore / minmax) + angular scaling (π or 2π).
      - Optional output remapping ⟨Z⟩∈[-1,1] → [a,b].
      - Parameter constraints after opt.step():
          * RX/RZ wrapped into (-π,π]
          * RY_scale clamped to [0, ry_scale_max]
    """
    if observable is None:
        observable = qml.PauliZ(0)

    dev_kwargs = {} if dev_kwargs is None else dict(dev_kwargs)
    dev = qml.device(dev_name, wires=wires, **dev_kwargs)

    @qml.qnode(dev, interface="torch")
    def circuit(inputs, weights):
        qru_template(inputs, weights)
        return qml.expval(observable)

    weight_shapes = {"weights": (L, 3)}
    qlayer = qml.qnn.TorchLayer(circuit, weight_shapes)  # input_arg="inputs" by default

    class QRUTorchModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.layer = qlayer

            # Store preprocessing/postprocessing config
            self._input_norm = input_norm
            self._input_stats = input_stats or {}
            self._input_angle_scale = input_angle_scale
            self._output_range = output_range
            self._ry_scale_max = float(ry_scale_max)

        # --------- input preprocessing utilities ----------
        @staticmethod
        def _safe_div(x, d, eps: float = 1e-8):
            return x / (d + eps)

        def _norm_inputs(self, x: torch.Tensor) -> torch.Tensor:
            """Apply normalization + angular scaling for rotations."""
            x_norm = x

            # Normalization
            if self._input_norm == "zscore":
                mu = self._input_stats.get("mean", 0.0)
                sd = self._input_stats.get("std", 1.0)
                x_norm = (x - float(mu)) / (float(sd) + 1e-8)
            elif self._input_norm == "minmax":
                xmin = self._input_stats.get("min", 0.0)
                xmax = self._input_stats.get("max", 1.0)
                rng = max(float(xmax) - float(xmin), 1e-8)
                x_norm = (x - float(xmin)) / rng  # → [0,1]
            elif self._input_norm == "identity":
                pass
            else:
                raise ValueError(f"invalid input_norm: {self._input_norm}")

            # Angular scaling
            if self._input_angle_scale == "pi":
                x_norm = x_norm * np.pi
            elif self._input_angle_scale == "2pi":
                x_norm = x_norm * (2.0 * np.pi)
            elif self._input_angle_scale == "none":
                pass
            else:
                raise ValueError(f"invalid input_angle_scale: {self._input_angle_scale}")

            return x_norm

        # --------- output postprocessing ----------
        @staticmethod
        def _map_to_range(z: torch.Tensor, a: float, b: float) -> torch.Tensor:
            """Map ⟨Z⟩∈[-1,1] → [a,b]."""
            z01 = (z + 1.0) * 0.5   # [-1,1] → [0,1]
            return z01 * (b - a) + a

        # --------- parameter constraints ----------
        @torch.no_grad()
        def constrain_(self):
            """To be called AFTER opt.step().

            - Wrap RX/RZ in (-π,π].
            - Clamp RY_scale∈[0, ry_scale_max].
            """
            for p in self.parameters():
                if p.data.ndim == 2 and p.data.shape == (L, 3):
                    # RX (col 0) and RZ (col 2): wrap modulo 2π into (-π, π]
                    p.data[:, 0] = ((p.data[:, 0] + np.pi) % (2.0 * np.pi)) - np.pi
                    p.data[:, 2] = ((p.data[:, 2] + np.pi) % (2.0 * np.pi)) - np.pi
                    # RY scale (col 1): clamp [0, ry_scale_max]
                    p.data[:, 1].clamp_(min=0.0, max=self._ry_scale_max)

        # --------- forward (batch handled sample by sample) ----------
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            x = self._norm_inputs(x)

            # Batch (N,1) → evaluate sample by sample (PL 0.36)
            if hasattr(x, "ndim") and x.ndim == 2 and x.shape[0] > 1:
                outs = []
                for i in range(x.shape[0]):
                    xi = x[i : i + 1]       # (1,1)
                    yi = self.layer(xi)     # often (1,)
                    if hasattr(yi, "ndim") and yi.ndim == 1:
                        yi = yi.unsqueeze(-1)  # -> (1,1)
                    outs.append(yi)
                y = torch.cat(outs, dim=0)     # (N,1)
            else:
                y = self.layer(x)
                if hasattr(y, "ndim") and y.ndim == 1:
                    y = y.unsqueeze(-1)        # -> (1,1)

            # Optional remap to [a,b]
            if self._output_range is not None:
                a, b = map(float, self._output_range)
                y = self._map_to_range(y, a, b)
            return y

    return QRUTorchModel()
