# 📘 QRU Template (1-Qubit Quantum Re-Uploading Unit)

A minimal, transparent, **hardware-ready QRU template** for PennyLane, with clean PyTorch integration, noise-simulation tools, and now **qBraid + IonQ backend support** (OpenQASM 2.0 export + execution on IonQ simulators).

This repo provides:

* a reusable 1-qubit **QRU block** implemented in PennyLane
* a configurable **TorchLayer wrapper** (normalization, scaling, clamping, batching)
* clean **regression/classification examples**
* NISQ **noise protocols**
* **hardware-ready pipeline**: train → export QASM → run via qBraid/IonQ

---

# 🆕 What’s New (Dec 2025)

The repo now includes:

### ✔️ **`regression_sine_qbraid.py`**

A full demonstration of:

1. training a QRU(1q) on CPU (PennyLane + Torch)
2. exporting the trained circuit to **OpenQASM 2.0** with qBraid
3. running selected inference points on **IonQ simulated backends**:

   * `simulator` (ideal)
   * `simulator_aria1` (Aria-1 noise model)
   * `simulator_harmony` (Harmony noise model)

The workflow is **identical** to what would be required for running on a real IonQ QPU.

---

# 🧠 QRU Overview

Each QRU block applies:

```
RX(w[l, 0])
RY(w[l, 1] * x)   ← data re-uploading
RZ(w[l, 2])
```

For depth **L**, the circuit returns a single quantum feature:

[
\langle Z \rangle \in [-1,1].
]

The PyTorch wrapper adds:

* input normalization (`identity`, `zscore`, `minmax`)
* angle rescaling (`none`, `pi`, `2pi`)
* output mapping (`[-1,1] → [a,b]`)
* periodic parameter constraints (wrap RX/RZ, clamp RY scale)
* batch loop (required with PennyLane ≤ 0.36)

---

# 📂 Repository Structure

```
qru-template/
├─ qru/
│  ├─ __init__.py
│  ├─ qru_pennylane.py            # Core QRU + TorchLayer
│  └─ noise_protocol.md           # Noise simulation guide
│
├─ examples/
│  ├─ regression_sine.py
│  ├─ regression_sine_noisy.py
│  ├─ regression_zscore_scaled.py
│  ├─ classification_threshold.py
│  ├─ classification_ce_fast.py
│  ├─ verify_p001_g001_multi_seed.py
│  ├─ regression_sine_qbraid.py   # NEW: qBraid/IonQ pipeline
│
├─ tests/
│  ├─ test_shapes.py
│  └─ test_training_step.py
│
├─ results/
│  ├─ noise_qru.csv
│  └─ noise_qru_p001_g001_seeds.csv
│
├─ README.md
├─ LICENSE
└─ requirements.txt
```

---

# ⚙️ Installation

### Core environment

```bash
pip install -r requirements.txt
```

### qBraid + IonQ support

```bash
pip install qbraid "qbraid[ionq]"
```

Set API keys:

```bash
set QBRAID_API_KEY=...
set IONQ_API_KEY=...
```

Edit your local `pyproject.toml` or install the repo as editable:

```bash
pip install -e .
```

---

# 🚀 Quick Start

### Verify installation

```bash
python - <<EOF
from qru import make_qru_torchlayer
print("QRU template OK:", callable(make_qru_torchlayer))
EOF
```

### Run tests

```bash
pytest -q    # expected: 2 passed
```

### CPU-only examples

```bash
python examples/regression_sine.py
python examples/classification_ce_fast.py
```

---

# 🧪 Hardware-Ready Example (qBraid + IonQ)

### Train, export QASM, and run inference on IonQ simulators

```bash
python examples/regression_sine_qbraid.py --mode Q --device simulator --shots 200
```

Available devices on qBraid:

| ID                  | Description                       |
| ------------------- | --------------------------------- |
| `simulator`         | Ideal 29q simulator               |
| `simulator_aria1`   | Noisy sim (IonQ Aria-1 hardware)  |
| `simulator_harmony` | Noisy sim (IonQ Harmony hardware) |

This script:

1. trains a QRU(1q) (Torch + PL)
2. previews CPU predictions
3. exports circuit → **OpenQASM 2.0**
4. submits QASM to qBraid IonQ runtime
5. displays backend counts

This is the recommended workflow for hardware-aligned experiments.

---

# 📊 Reference CPU Results (PennyLane 0.36)

| Example                       | Epochs | Metric | Result          |
| ----------------------------- | ------ | ------ | --------------- |
| `regression_sine.py`          | 100    | MSE ↓  | 0.45 → 0.014    |
| `regression_zscore_scaled.py` | 100    | MSE ↓  | 0.022 → 0.00023 |
| `classification_ce_fast.py`   | ~150   | Acc ↑  | ~0.85–0.90      |

**Notes:**

* Normalizing inputs greatly improves stability.
* QRU(1q) can approximate nontrivial functions even at low depth.

---

# 🔬 Noise Experiments (NISQ-like)

Noise applied after each QRU block:

* `DepolarizingChannel(p)`
* `AmplitudeDamping(γ)`
* `PhaseDamping(γ)`

### Summary (single seed, p=γ grid)

| p \ γ     | 0       | 0.001   | 0.01        |
| --------- | ------- | ------- | ----------- |
| **0**     | 0.02569 | 0.03254 | 0.09397     |
| **0.001** | 0.06018 | 0.01037 | 0.10646     |
| **0.01**  | 0.16276 | 0.03392 | **0.05776** |

### Multi-seed stability (p=γ=0.01, 5 seeds)

MSEs = `0.03421, 0.01282, 0.03409, 0.12565, 0.08201`
→ mean ± std = **0.0578 ± 0.0408**

---

# 🧱 API Summary

```python
from qru import make_qru_torchlayer

model = make_qru_torchlayer(
    L=6,
    input_norm="zscore",
    input_stats={"mean": m, "std": s},
    input_angle_scale="pi",
    output_range=(0,1),
    ry_scale_max=1.0,
)

# IMPORTANT: call after opt.step()
model.constrain_()
```

---

# 🧩 Notes on Stability

* Normalize inputs when they vary strongly.
* Enforce periodicity for RX/RZ.
* Clamp RY scales to avoid exploding gradients.
* Batch loop is explicit (for PL ≤ 0.36).

---

# 📜 License

Apache 2.0 – see `LICENSE`.

---
