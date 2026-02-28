# Flare-Aware Inverse Lithography: How We Optimize Flare

This document explains how the flare-aware ILT pipeline optimizes masks to reduce EUV flare while maintaining imaging fidelity.

## Overview

The goal is to find a mask \( M \) that (1) prints the target pattern accurately and (2) minimizes flare-dominated distortion. The key insight: **not all image locations respond equally to mask changes**. In flare-dominated regions, mask edits barely improve the printed image, so we down-weight those pixels in the loss and up-weight regions where the mask still has control.

---

## 1. Forward Imaging Model with Flare

The flare-degraded aerial image at wafer position \( \mathbf{r} \) is:

```
I(r) = I_diff(r) + α(r) · [PSF_SC * I_diff](r)
```

- **I_diff**: Diffraction-limited aerial image (from Abbe/Hopkins or a differentiable proxy such as Gaussian blur)
- **PSF_SC**: Shiraishi scatter PSF (flare kernel)
- **α(r)**: Spatially varying flare intensity (higher in dense regions)
- **I_flare = PSF_SC * I_diff**: Flare contribution

A simple resist model maps the aerial image to printed contours:

```
printed = sigmoid(θ_Z · (I - I_th))
```

The full forward chain is differentiable, so gradients flow from the loss back to the mask parameters.

---

## 2. Sensitivity Map S

We quantify how much each pixel responds to mask changes:

```
S(r) = ||∇_M I_diff|| / (||∇_M I_diff|| + ||∇_M I_flare|| + ε)
```

- **S ≈ 1**: Mask-controllable (diffraction-dominated)
- **S ≈ 0**: Flare-dominated (mask edits have little effect)

\( \nabla_M \) denotes the gradient with respect to the mask; we obtain it via autograd through the forward model. If the lithography model is not differentiable (e.g., TorchLitho with detached outputs), we fall back to **S = 1** everywhere.

---

## 3. Adaptive Weights w

The fidelity loss is spatially weighted:

```
w(r) = S(r) / (1 + β · α(r))
```

- **High α** (dense region): lower w → less emphasis on matching the target there (flare-dominated)
- **Low α** (sparse region): higher w → more emphasis on fidelity where the mask has control

The parameter β (default 2) controls how strongly α reduces w in flare-heavy regions.

---

## 4. Loss Function

We minimize:

```
L = L_fidelity + μ · L_flare + λ_bin · L_bin
```

### L_fidelity (weighted L2)

```
L_fidelity = mean( w · (printed - target)² )
```

This penalizes deviation from the target, but only where w is high (mask-controllable regions).

### L_flare (flare regularization)

```
L_flare = mean( ||∇[PSF_SC * I_diff]||² )
```

This discourages mask patterns that create strong, spatially varying flare (e.g., large uniform reflective regions).

### L_bin (binarization)

```
L_bin = mean( M · (1 - M) )
```

This pushes mask values toward 0 or 1 and reduces stippling.

---

## 5. Optimization Loop

1. **Parameters**: Mask is parameterized as \( M = \sigma(\theta_M \cdot P) \), with unconstrained parameters \( P \).
2. **Forward**: Compute I_diff → I_flare → I_total → printed.
3. **Sensitivity**: Recompute S every few iterations (default 5) to save compute.
4. **Weights**: \( w = S / (1 + \beta \alpha) \).
5. **Loss**: \( L = L_fidelity + \mu L_flare + \lambda_{bin} L_bin \).
6. **Backward**: Compute \( \nabla_P L \) via autograd.
7. **Update**: Adam step on P.

Parameters (typical defaults): \( \theta_M = 10 \), \( \mu = 0.3 \), \( \beta = 2 \), \( \lambda_{bin} = 0.05 \).

---

## 6. Lithography Models

| Model | Differentiable | S varies | Use case |
|-------|----------------|----------|----------|
| **SimpleDiffractionLitho** | Yes (Gaussian blur) | Limited (S ≈ 0.5) | Proof-of-concept, fast runs |
| **TorchLithoAbbe** | No (detached) | No (S = 1 fallback) | Real I_diff for comparison only |

For gradient-based optimization, use SimpleDiffractionLitho (or any differentiable lithography model). TorchLitho gives realistic I_diff but no gradients, so the optimizer cannot improve the mask from the fidelity or flare terms.

---

## 7. Running the Optimization

```bash
# Benchmark pattern
python examples/run_flare_aware_ilt.py

# Synthetic dense/sparse pattern (strong flare contrast)
python examples/run_flare_aware_ilt.py --synthetic

# TorchLitho for I_diff (no gradients; S uniform)
python examples/run_flare_aware_ilt.py --synthetic --torchlitho

# Diagnose results
python examples/diagnose_flare_results.py
```

Results are saved under `results/flare_aware_YYYYMMDD_HHMMSS/`:

- `flare_aware_results.pt`: mask, target, history
- `mask_vs_target.png`: optimized mask vs target
- `loss_curves.png`: loss components and gradient norm
- `flare_diagnostics.png`: S, w, and α maps

### Which run produced my plots?

`diagnose_flare_results.py` uses the **latest** results folder (highest timestamp) by default. If you run both commands in order:

| Order | Command | I_diff model | Results folder | S in plots |
|-------|---------|--------------|----------------|------------|
| 1st | `run_flare_aware_ilt.py --synthetic` | **Gaussian** (SimpleDiffractionLitho) | e.g. `flare_aware_20260228_042139` | S can vary (≈0.5) |
| 2nd | `run_flare_aware_ilt.py --synthetic --torchlitho` | **Abbe** (TorchLitho) | e.g. `flare_aware_20260228_042629` | S = 1 (uniform) |

So if you then run `diagnose_flare_results.py` with no argument, the plots (mask_vs_target, flare_diagnostics, loss_curves, diagnostic_plot) correspond to the **second (TorchLitho/Abbe)** run. To diagnose the **non-TorchLitho (Gaussian)** run, pass that folder explicitly:

```bash
# Diagnose the Gaussian (SimpleDiffractionLitho) run, e.g.:
python examples/diagnose_flare_results.py results/flare_aware_20260228_042139
```

Diagnosis will print mask stats, loss history, S/w maps, and write `diagnostic_plot.png` and `flare_diagnostics.png` into that folder.

**Interpretation:**

- **Gaussian run:** Full gradient flow; loss drops; S has limited variation (I_diff and I_flare gradients similar); mask is optimized by fidelity + flare + binarization.
- **TorchLitho run:** No gradients from litho; S is forced to 1; only binarization term updates the mask; w still varies with α; total loss barely changes; binary error can be low because the initial pattern is already close to the target.

---

## 8. Config Options

In `run_flare_aware_ilt.py`, key config keys:

| Key | Default | Description |
|-----|---------|-------------|
| `max_iters` | 40 | Optimization iterations |
| `lr` | 0.1 | Adam learning rate |
| `flare_weight_beta` | 2 | Down-weighting in flare regions |
| `flare_reg_weight` | 0.3 | Flare regularization strength |
| `bin_weight` | 0.05 | Binarization penalty |
| `sensitivity_interval` | 5 | How often to recompute S |

---

## 9. Observations

### Gaussian (SimpleDiffractionLitho) runs

- **Loss**: Total loss decreases (e.g., ~0.02 → ~0.008) over iterations; optimizer updates the mask.
- **Sensitivity S**: Often near 0.5 (range ~0.50–0.51). I_diff and I_flare share similar gradient structure, so S does not vary much spatially.
- **Weights w**: Vary spatially (e.g., range 0.33–0.48). Driven mainly by α since S ≈ 0.5; w is lower in dense regions, higher in sparse regions.
- **Alpha α**: Clear spatial structure (e.g., high in dense left half, low in sparse right half) with synthetic pattern.
- **Flare reg**: Δ reg becomes non-zero when `flare_reg_weight` ≥ 0.1; higher μ (e.g., 0.3) makes flare regularization effective.
- **Binarization**: `bin_weight` 0.05 reduces stippling and pushes pixels toward 0/1.
- **β tuning**: Smaller β (e.g., 2 vs 5) increases the effect of α on w, giving stronger spatial variation.

### TorchLitho (Abbe) runs

- **Sensitivity S**: Always 1.0 (uniform). I_diff is detached, so S falls back to 1 everywhere.
- **Weights w**: Still vary spatially (e.g., 0.83–0.98), via w = 1/(1 + β α).
- **Loss**: Total loss barely changes (e.g., ~0.237 → ~0.231). No gradients from litho; fidelity and flare reg do not affect the mask.
- **Update mechanism**: Only the binarization term provides gradients; optimization mainly binarizes, not ILT.
- **Binary error**: Can be low (e.g., 0.23%) because the initial synthetic pattern is close to the target and the bin term cleans it up.

### Synthetic vs benchmark pattern

- **Synthetic** (dense left, sparse right): Produces strong α contrast; good for demonstrating flare-aware behavior.
- **Benchmark** (ICCAD13): Typically lower binary error (~1.5%); less dense/sparse contrast, so flare effects are subtler.

---

## 10. Representative Results

Typical numbers from 40-iteration runs on 256×256 patterns (synthetic dense/sparse unless noted).

### Gaussian (SimpleDiffractionLitho) run

```
Command: python examples/run_flare_aware_ilt.py --synthetic

Loss
  total:     [0.0082, 0.0197]  final: 0.0082
  Δ total:   -0.0115
  Δ L2:      -0.0115
  Δ reg:     -0.0019

Metrics
  L2 Error:    15619
  PV Band:     294
  binary error: ~7% (4597 / 65536)  [synthetic pattern]

Maps
  S:  range [0.50, 0.50]  mean 0.50  std 0.00
  w:  range [0.33, 0.48]  mean 0.39  std 0.04
  α:  varies spatially (high in dense region)
```

### TorchLitho (Abbe) run

```
Command: python examples/run_flare_aware_ilt.py --synthetic --torchlitho

Loss
  total:     [0.2308, 0.2367]  final: 0.2308
  Δ total:   -0.0059
  Δ L2:      -0.0059
  Δ reg:     -0.0000
  Δ bin:     +0.0004

Metrics
  L2 Error:    14762
  PV Band:     10351
  binary error: 0.23% (150 / 65536)

Maps
  S:  range [1.0, 1.0]  mean 1.0  std 0.0  (uniform fallback)
  w:  range [0.83, 0.98]  mean 0.88  std 0.04
  α:  varies spatially
```

### Benchmark (ICCAD13) run with Gaussian

```
Command: python examples/run_flare_aware_ilt.py

Metrics
  L2 Error:    ~1168
  PV Band:     ~590
  binary error: ~1.5% (982 / 65536)
```

### Comparison: flare-aware vs non-flare-aware (synthetic, 40 iters)

Same pattern and initial params; only μ (flare_reg_weight) differs. Run:

```bash
python examples/compare_flare_aware.py --synthetic
```

| Metric | Flare-aware (μ=0.3) | Non-flare-aware (μ=0) |
|--------|---------------------|------------------------|
| L2 Error | 15603 | 15637 |
| PV Band | 285 | 292 |
| Binary error (%) | 7.15 | 6.98 |
| Final loss | 8.23e-03 | 6.13e-03 |

**Observation:** With flare regularization (μ=0.3), L2 and PV band improve (lower is better); binary error is slightly higher. The flare-aware run trades a small amount of pixel-level fidelity for better printed-image metrics (L2, PVB), consistent with the regularizer biasing the mask away from flare-amplifying patterns.

### Large run: 2048×2048, 300 iters (synthetic, flare-aware)

Paper-style run with default size and iterations:

```bash
python examples/run_flare_aware_ilt.py --synthetic
# Uses --size 2048, max_iters 300 by default
```

```
Mask size: 2048x2048  |  iters: 300  |  device: cuda

Loss
  total:     [0.0108, 0.0232]  final: 0.0108
  Δ total:   -0.0124
  Δ L2:      -0.0120
  Δ reg:     -0.0011
  Δ bin:     +0.0002

Metrics
  L2 Error:    1061314
  PV Band:     51815
  binary error: 4.66% (195356 / 4194304)

Maps
  S:  range [0.50, 0.50]  mean 0.50  std 0.00
  w:  range [0.42, 0.49]  mean 0.44  std 0.02
```

### Comparison: size and iterations

All runs use synthetic dense/sparse pattern and SimpleDiffractionLitho (Gaussian). Flare-aware (μ=0.3) unless noted.

| Size    | Iters | Δ total   | Δ reg    | Binary error | L2 (eval) | PVB (eval) |
|---------|-------|----------|----------|--------------|-----------|------------|
| 256×256 | 40    | −0.0115  | −0.0019  | ~7%          | ~15600    | ~285       |
| 256×256 | 40 (μ=0) | −0.006  | 0        | ~7%          | ~15637    | ~292       |
| 2048×2048 | 300  | −0.0124  | −0.0011  | 4.66%        | 1061314   | 51815      |

- **256, 40 iters:** Quick comparison; flare-aware improves L2/PVB vs μ=0.
- **2048, 300 iters:** Paper-style; more iterations and resolution give lower binary error (4.66%) and clear loss decrease; L2/PVB scale with the number of pixels.
