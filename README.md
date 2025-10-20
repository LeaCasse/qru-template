# QRU Template (Quantum Re-Uploading Unit, 1-qubit)

Minimal, transparent QRU (Quantum Re-Uploading Unit) block implemented in PennyLane, with a clean PyTorch interface and reproducible examples/tests.  

---

## 🧠 Overview

Each QRU layer applies:
- `RX(weights[l, 0])`
- `RY(weights[l, 1] * x)` ← data re-uploading
- `RZ(weights[l, 2])`

The QNode returns `⟨Z⟩` as a scalar quantum feature.  
The Torch wrapper adds input/output scaling, parameter constraints, and batch handling.

---

## 📂 Repository Structure

```
qru-template/
├─ qru/
│  ├─ __init__.py
│  ├─ qru_pennylane.py       # Core QRU + TorchLayer wrapper
│  └─ noise_protocol.md       # NISQ noise simulation guide
├─ examples/
│  ├─ regression_sine.py
│  ├─ regression_zscore_scaled.py
│  ├─ regression_sine_noisy.py
│  ├─ classification_ce_fast.py
│  ├─ classification_threshold.py
│  └─ classification_threshold_fast.py
├─ tests/
│  ├─ test_shapes.py
│  └─ test_training_step.py
├─ requirements.txt
└─ README.md
```

---

## ⚙️ Installation

```bash
# Python ≥3.10 recommended
pip install -r requirements.txt
```

On Windows / Anaconda:
```bash
set PYTHONPATH=.
```

---

## 🚀 Quick Start

### Sanity check
```
python -c "from qru import make_qru_torchlayer; print('OK:', callable(make_qru_torchlayer))"
```
### Run tests
```
python -m pytest -q    # Expected: 2 passed
```
### Run examples
```
python examples\regression_sine.py
python examples\regression_zscore_scaled.py
python examples\classification_ce_fast.py
```

Tip (Windows/CPU): limiting threads can reduce overhead
```
import torch; torch.set_num_threads(1)
```
---

## 📊 Reference Results

All runs performed on CPU (Windows/Anaconda, PL 0.36).

| Example | Epochs | Metric | Result |
|----------|---------|--------|---------|
| `regression_sine.py` | 100 | MSE ↓ | 0.452 → 0.014 |
| `regression_zscore_scaled.py` | 100 | Loss ↓ | 0.022 → 0.00023 |
| `classification_ce_fast.py` | ~150 | Accuracy ↑ | ~0.85–0.90 |

**Observation:**  
Input normalization (`zscore`) and output scaling ([0, 1]) yield faster and more stable convergence.  
Unscaled sine still converges to low MSE, validating QRU expressivity.

---

## 🧩 NISQ / Noise Protocol

Provided via `qru/noise_protocol.md` and `examples/regression_sine_noisy.py`.

**Setup:**
- Device: `default.mixed` with finite shots (e.g. 1000)
- Noise: `DepolarizingChannel(p)`, `AmplitudeDamping(γ)`, `PhaseDamping(γ)`
- Sweep: `p, γ ∈ {0, 0.001, 0.01, 0.05}`
- Metrics: MSE / Accuracy
- Output: `results/noise_qru.csv`

**Run:**
```bash
python examples\regression_sine_noisy.py --p 0.001 --g 0.001
python examples\regression_sine_noisy.py --sweep quick
```

---

## 🧱 API Example

```python
from qru import make_qru_torchlayer

model = make_qru_torchlayer(
    L=6,
    input_norm="zscore",                 # "identity" | "zscore" | "minmax"
    input_stats={"mean": m, "std": s},
    input_angle_scale="pi",              # "pi" | "2pi" | "none"
    output_range=(0.0, 1.0),             # map ⟨Z⟩∈[-1,1] → [a,b]
    ry_scale_max=1.0
)

# After each optimizer step
model.constrain_()
```

---

## 🧩 Stability Notes

- Normalize inputs to avoid meaningless large angles.  
- For probabilities: map `(z+1)/2 → [0, 1]` (or use `output_range=(0, 1)`).  
- Periodic parameters (RX, RZ) benefit from wrapping in `(-π, π]`.  
- On PennyLane ≤ 0.36, batch evaluation is looped per-sample; keep N, L modest.

---

## 📜 License

Licensed under the **Apache 2.0 License** — see [`LICENSE`](LICENSE) for details.
