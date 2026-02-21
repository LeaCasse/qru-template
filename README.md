# 📘 QRU Template (1-Qubit Quantum Re-Uploading Unit)

A minimal, transparent, **hardware-ready QRU template** for PennyLane, with clean PyTorch integration, noise-simulation tools, and **qBraid + IonQ backend support** (OpenQASM 2.0 export + execution on IonQ simulators).

This repository provides:

* a reusable 1-qubit **QRU block** implemented in PennyLane
* a configurable **TorchLayer wrapper** (normalization, scaling, clamping, batching)
* clean **regression and classification examples**
* NISQ-style **noise protocols**
* a **hardware-ready pipeline**: train → export QASM → run via qBraid/IonQ
* a fully pedagogical **Jupyter Community Demo notebook**

---

# 🆕 PennyLane Community Demo

This repository includes a dedicated pedagogical notebook:

```
demos/QRU_PennyLane_Community_Demo.ipynb
```

The notebook provides:

* a conceptual introduction to quantum re-uploading
* a minimal 1-qubit QRU implementation
* regression training on a sine function
* learning curve visualization
* parameter inspection
* qualitative frequency comparison

The goal of the notebook is **clarity and reproducibility**, making it suitable for educational use and PennyLane Community Demo submission.

---

# 🧠 QRU Overview

Each QRU layer applies:

```
RX(w[l, 0])
RY(w[l, 1] * x)   ← data re-uploading
RZ(w[l, 2])
```

For depth **L**, the circuit returns a single quantum feature:

⟨Z⟩ ∈ [-1, 1]

The PyTorch wrapper adds:

* input normalization (`identity`, `zscore`, `minmax`)
* angle rescaling (`none`, `pi`, `2pi`)
* output mapping (`[-1,1] → [a,b]`)
* periodic parameter constraints (wrap RX/RZ, clamp RY scale)
* explicit batch loop (required for PennyLane ≤ 0.36)

---

# 📂 Repository Structure

```
qru-template/
├─ qru/
│  ├─ __init__.py
│  ├─ qru_pennylane.py
│  └─ noise_protocol.md
│
├─ examples/
│  ├─ regression_sine.py
│  ├─ regression_sine_noisy.py
│  ├─ regression_zscore_scaled.py
│  ├─ classification_threshold.py
│  ├─ classification_ce_fast.py
│  ├─ verify_p001_g001_multi_seed.py
│  └─ regression_sine_qbraid.py
│
├─ demos/
│  └─ QRU_PennyLane_Community_Demo.ipynb
│
├─ tests/
│  ├─ test_shapes.py
│  └─ test_training_step.py
│
├─ README.md
├─ LICENSE
└─ requirements.txt
```

⚠️ Note: the `results/` directory is generated dynamically by noise experiments and is not required for running the core examples.

---

# ⚙️ Installation

### Core environment

```bash
pip install -r requirements.txt
```

Install locally:

```bash
pip install -e .
```

### Optional: qBraid + IonQ support

```bash
pip install qbraid "qbraid[ionq]"
```

Set API keys:

```bash
set QBRAID_API_KEY=...
set IONQ_API_KEY=...
```

---

# 🚀 Quick Start

### Run the Community Demo Notebook

```bash
jupyter notebook demos/QRU_PennyLane_Community_Demo.ipynb
```

The notebook runs entirely on CPU and requires no hardware access.

---

### Run tests

```bash
pytest -q
```

Expected: 2 tests passing.

---

### Run CPU examples

```bash
python examples/regression_sine.py
python examples/classification_ce_fast.py
```

---

# 🧪 Hardware-Ready Example (qBraid + IonQ)

Train, export QASM, and run inference:

```bash
python examples/regression_sine_qbraid.py --mode Q --device simulator --shots 200
```

Available devices:

| ID                  | Description         |
| ------------------- | ------------------- |
| `simulator`         | Ideal 29q simulator |
| `simulator_aria1`   | Aria-1 noise model  |
| `simulator_harmony` | Harmony noise model |

Workflow:

1. Train QRU (Torch + PennyLane)
2. Preview CPU predictions
3. Export to OpenQASM 2.0
4. Submit via qBraid runtime
5. Retrieve backend counts

---

# 🔬 Noise Experiments

Noise applied after each QRU layer:

* DepolarizingChannel(p)
* AmplitudeDamping(γ)
* PhaseDamping(γ)

See:

```
examples/regression_sine_noisy.py
qru/noise_protocol.md
```

---

# 🧩 API Example

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

model.constrain_()  # call after optimizer step
```

---

# 🧠 Stability Notes

* Normalize inputs when scale varies.
* Wrap RX/RZ angles to preserve periodicity.
* Clamp RY scale to prevent gradient explosion.
* Explicit batch loop ensures compatibility with PL ≤ 0.36.

---

# 📜 License

Apache 2.0 – see `LICENSE`.
