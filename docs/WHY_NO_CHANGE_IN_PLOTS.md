# Why Do the Loss Curves Show No Change?

If you run flare-aware ILT and then `diagnose_flare_results.py`, you may see:

- **Flat loss curves**: `total`, `weighted_l2`, and `flare_reg` barely move over iterations
- **Δ total ≈ 0, Δ L2 ≈ 0, Δ reg ≈ 0**
- **Gradient norm** very small (e.g. ~10⁻⁸)
- **Binary error 0%**: optimized mask already matches the target

This README explains why that happens and when to expect it.

---

## 1. The Optimizer Has Nothing Left to Do

The most common cause is that the **starting point is already very good**.

- **Initialization**: OpenILT’s `PixelInit` (and similar) often sets initial mask parameters from the target (e.g. `params = 2*target - 1` plus small noise). So the initial mask is already close to the desired pattern.
- **First iteration**: The loss is already low. After a few steps the optimizer reaches a minimum (or a very flat region).
- **Later iterations**: Gradients become tiny (~10⁻⁸). Parameter updates are negligible, so the loss curves stay flat.

So “no change” in the **plots** (flat loss, tiny gradient) can simply mean: **convergence**, often very fast.

---

## 2. When You’ll See Flat Curves

| Situation | What you see |
|-----------|----------------|
| **Benchmark pattern, 512×512** | Initial mask ≈ target → loss low from iter 0 → flat curves, 0% binary error |
| **Small learning rate** | Updates too small to move the loss visibly in the number of iterations you plot |
| **Already converged** | If you re-run diagnosis on a run that had already converged, you again see flat curves |

So flat curves are normal when:

- The initial mask is already close to the target, or  
- The run has already converged.

---

## 3. When You *Will* See Change

You’ll see loss and gradient activity when the starting mask is **not** already near the target:

- **Synthetic pattern** (`--synthetic`): Dense/sparse pattern; initial params are perturbed. You should see loss decrease and non-zero Δ total / Δ L2.
- **Flare-aware vs non-flare-aware**: Use `compare_flare_aware.py --synthetic`; the two runs differ in μ and show different metrics.
- **Larger flare weight (μ)** or **harder target**: More room for the optimizer to improve, so curves move more.

---

## 4. What Your Run Tells You

From your diagnosis:

- **Binary error: 0 / 262144 (0.00%)** → The mask (after binarization) **exactly** matches the target. So in terms of the binary pattern, optimization has “succeeded.”
- **Δ total ≈ 0, flat loss** → The scalar loss did not improve over 60 iters because it was already at (or very near) a minimum.
- **Gradient norm ~10⁻⁸** → Updates are negligible; the solver is effectively at a stationary point.

So in this run:

1. The **code and plotting are working** (they are correctly showing flat loss and tiny gradients).
2. The **lack of change** is because the problem is already solved for this pattern/initialization: the mask already matches the target and the loss is already minimal.

---

## 5. If You Want to See Changing Curves

To get runs where loss and plots clearly change:

1. **Use the synthetic pattern**  
   ```bash
   python examples/run_flare_aware_ilt.py --synthetic
   ```
2. **Compare with and without flare**  
   ```bash
   python examples/compare_flare_aware.py --synthetic
   ```
3. **Diagnose the run that had activity**  
   Use the results folder from the synthetic run (or the comparison), not the 512×512 benchmark run that started near-optimal.

---

## Summary

- **No change in the plots** (flat loss, Δ ≈ 0, tiny gradient) usually means the run **converged**, often because the initial mask was already close to the target.
- **0% binary error** means the optimized mask matches the target; the optimizer had nothing left to improve.
- To see **visible change** in the curves, use the **synthetic** pattern or compare **flare-aware vs non–flare-aware** and diagnose those runs.
