# Quantum Noise Protocol (NISQ Simulation) — QRU(1q)

## Objective
Evaluate the robustness of the **Quantum Re-Uploading Unit (QRU)** under realistic noisy conditions, simulating a **Noisy Intermediate-Scale Quantum (NISQ)** device.

This protocol defines:
- **Device setup:** mixed-state simulator + finite measurement shots  
- **Noise model:** depolarizing, amplitude damping, and phase damping channels  
- **Noise grid:** small probability values (0 → 0.05)  
- **Metrics:** Mean Squared Error (MSE) for regression or accuracy for classification  
- **Output:** reproducible CSV file for quantitative plots

---

## Device configuration

```python
dev = qml.device("default.mixed", wires=1, shots=1000)

    default.mixed: mixed-state simulator supporting noise channels

    shots: number of circuit executions (finite sampling noise)

Noise channels per layer

qml.DepolarizingChannel(p, wires=0)
qml.AmplitudeDamping(gamma, wires=0)
qml.PhaseDamping(gamma, wires=0)

Physical interpretation
Channel	Parameter	Description
DepolarizingChannel(p)	p	Probability of global randomization (bit-flip or phase-flip)
AmplitudeDamping(γ)	γ	Relaxation towards |0⟩ (T₁ decay)
PhaseDamping(γ)	γ	Dephasing between |0⟩ and |1⟩ (T₂ decay)
Suggested noise grid
Parameter	Values
p	{0.0, 0.001, 0.01, 0.05}
γ	{0.0, 0.001, 0.01, 0.05}

Each combination defines one experimental condition.
Reference task

Regression:
y=sin⁡(x)y=sin(x) on x∈[−π,π]x∈[−π,π]
Training for ~60 epochs, depth L=4L=4, batch size N=128N=128.
Evaluation metrics
Metric	Use case	Description
Mean Squared Error (MSE)	Regression	Compare predicted vs true signal
Accuracy	Classification	Fraction of correct predictions
Output format

All results are appended to results/noise_qru.csv with the following structure:

p,gamma,shots,L,N,epochs,lr,seed,train_mse
0.001,0.001,1000,4,128,60,0.01,0,0.0231
0.010,0.010,1000,4,128,60,0.01,0,0.0452
...

This file can be used to plot degradation curves such as:

    MSE vs (p, γ)

    Accuracy vs noise intensity

Example usage

# Single run (specific noise levels)
python examples/regression_sine_noisy.py --p 0.001 --g 0.001

# Quick 3×3 sweep
python examples/regression_sine_noisy.py --sweep quick

# Full 4×4 sweep (longer)
python examples/regression_sine_noisy.py --sweep full

Notes

    Keep LL and dataset size modest to ensure reasonable runtime on CPU.

    Always compare to a noise-free baseline (p=0, γ=0) for reference.

    Document any degradation pattern and comment on robustness vs model depth or normalization choices.