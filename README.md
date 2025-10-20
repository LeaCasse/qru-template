# QRU Template (Quantum Re-Uploading Unit, 1-qubit)

Minimal, transparent QRU (Quantum Re-Uploading Unit) block implemented in PennyLane, with a clean PyTorch interface and reproducible examples/tests.  
Focus: **clarity**, **stability**, and **NISQ-readiness** (noise protocol included).

---

## Contents

qru-template/
├─ qru/
│ ├─ init.py
│ ├─ qru_pennylane.py # QRU template + TorchLayer wrapper + input/output scaling + param constraints
│ └─ noise_protocol.md # NISQ simulation protocol
├─ examples/
│ ├─ regression_sine.py
│ ├─ regression_zscore_scaled.py
│ ├─ regression_sine_noisy.py # NISQ: default.mixed + shots + CSV logging
│ ├─ classification_ce_fast.py # QRU feature + 1→3 linear head + CrossEntropy
│ ├─ classification_threshold.py
│ └─ classification_threshold_fast.py
├─ tests/
│ ├─ test_shapes.py
│ └─ test_training_step.py
└─ requirements.txt


---

## Installation

```
# Python ≥3.10 recommended
pip install -r requirements.txt
```
On Windows/Anaconda, run commands from the activated environment. From the repo root, you may also set:

```
set PYTHONPATH=.
```
Quick Start
1) Sanity check
```
python -c "from qru import make_qru_torchlayer; print('OK:', callable(make_qru_torchlayer))"
```
# Expected: OK: True

2) Run tests
```
python -m pytest -q
```
# Expected: 2 passed

3) Run the basic examples
```
python examples\regression_sine.py
python examples\regression_zscore_scaled.py
python examples\classification_ce_fast.py
```
Tip (Windows/CPU): limiting threads can reduce overhead
```
import torch; torch.set_num_threads(1)
```
What the QRU block does

A single-qubit template with L sequential layers. Each layer applies:
```
    RX(weights[l, 0])

    RY(weights[l, 1] * x) ← data re-uploading

    RZ(weights[l, 2])
```
The QNode returns expval(PauliZ(0)) as the quantum feature / output.
The PyTorch wrapper (make_qru_torchlayer) adds:

   Input normalization (identity, zscore, minmax)
   Angle scaling (×π, ×2π, or none)
   Output mapping from [-1, 1] to arbitrary [a, b] (e.g., [0, 1])
   Post-update constraints: wrap RX/RZ into (-π, π], clamp RY scale to [0, s_max]
   Batch handling (PennyLane ≤0.36): the forward pass iterates over batch items internally to keep shapes consistent and avoid broadcast issues.

Reproducible Results (reference runs)

All numbers below were obtained on CPU (Windows/Anaconda), using the default settings in the provided scripts.
Regression (sine on [-π, π])

examples/regression_sine.py
MSE progression (every 20 epochs, out of 100):

[020] 0.452
[040] 0.141
[060] 0.0628
[080] 0.0200
[100] 0.0140

Regression with z-score + output in [0,1]

examples/regression_zscore_scaled.py
Loss progression (every 20 epochs, out of 100):

[020] 0.0219
[040] 0.00477
[060] 0.00110
[080] 0.00042
[100] 0.00023

Interpretation: The [0,1] target scaling and input normalization yield a numerically smaller loss and faster convergence (well-conditioned objective); the unscaled sine still converges to low MSE, validating the expressivity of the 1-qubit QRU with re-uploading.
Classification (3 thresholds, fast mode)

examples/classification_ce_fast.py (QRU feature → Linear(1→3) → CrossEntropy, mini-batches)
Accuracy increases steadily; typical end-of-run:

acc ≈ 0.85–0.90 (N=300, L=4, ~150 steps)

  A threshold-based MSE variant is also included (classification_threshold_fast.py) for pedagogical purposes; CrossEntropy is the canonical objective for multi-class classification.

NISQ / Noise Protocol

A reproducible protocol for noisy simulations is provided:

  Device: default.mixed with finite shots (e.g., 1000)
  Noise per layer: DepolarizingChannel(p), AmplitudeDamping(γ), PhaseDamping(γ)
  Grids: small values such as p, γ ∈ {0.0, 0.001, 0.01, 0.05}
  Metric: MSE for regression / Accuracy for classification
  Output: CSV appended at results/noise_qru.csv

See qru/noise_protocol.md and run:
```
python examples\regression_sine_noisy.py --p 0.001 --g 0.001
python examples\regression_sine_noisy.py --sweep quick
```
This enables quantitative robustness studies (e.g., MSE degradation vs noise intensity) and makes the project NISQ-relevant.
API

from qru import make_qru_torchlayer, make_qru_qnode, qru_template

# Torch layer for end-to-end training
```
model = make_qru_torchlayer(
    L=6,
    input_norm="zscore",                 # "identity" | "zscore" | "minmax"
    input_stats={"mean": m, "std": s},   # or {"min": a, "max": b} for minmax
    input_angle_scale="pi",              # "pi" | "2pi" | "none"
    output_range=(0.0, 1.0),             # map ⟨Z⟩∈[-1,1] -> [a,b]; or None
    ry_scale_max=1.0
)
```
# After each optimizer step:
```
model.constrain_()
```
Notes on Stability

  Inputs: Always normalize to avoid meaningless huge angles; map to ×π/×2π when appropriate.
  Outputs: For probabilities, map (z+1)/2 to [0,1] (or use output_range=(0,1)).
  Parameters: Periodic parameters benefit from wrapping/clamping to avoid plateaus.
  Performance: On PL ≤0.36, batch evaluation is looped; keep N and L modest for quick demos.

Testing
```
python -m pytest -q
```
# Expected: 2 passed

Roadmap (short)

  Add vectorized batching when targeting newer PennyLane versions.
  Provide mirrored implementation in Qiskit ML (identical API).
  Extend examples with train/test split and classical baselines (for benchmarking).
  Optional: multi-qubit QRU variants and residual/reuploading hybrids.

License

TBD.


---
