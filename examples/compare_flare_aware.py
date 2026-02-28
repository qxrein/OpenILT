#!/usr/bin/env python
"""
Compare flare-aware (μ > 0) vs non-flare-aware (μ = 0) on the same pattern.
Uses the same target and initial params for both runs; reports L2, PV band,
binary error, and final loss.
"""

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

import pycommon.glp as glp
import pyilt.initializer as initializer
import pyilt.evaluation as evaluation
import pyilt.simpleilt as simpleilt
from pyilt.flare_psf import FlarePSF
from pyilt.flare_aware_solver import FlareAwareILT
from pyilt.simple_diffraction import SimpleDiffractionLitho


def create_synthetic_pattern(h, w):
    target = torch.zeros(h, w, dtype=torch.float32)
    target[:, : w // 2] = 1.0
    for c in range(w // 2, w, 16):
        target[:, c : min(c + 8, w)] = 1.0
    params = 2 * target - 1 + 0.05 * torch.randn(h, w)
    return target, params.to(torch.float32)


def get_benchmark_target_params():
    design = glp.Design("./benchmark/ICCAD2013/M1_test1.glp", down=1)
    cfg_simple = simpleilt.SimpleCfg("./config/multilevel256.txt")
    design.center(
        cfg_simple["TileSizeX"], cfg_simple["TileSizeY"],
        cfg_simple["OffsetX"], cfg_simple["OffsetY"],
    )
    init = initializer.PixelInit()
    target, params = init.run(
        design,
        cfg_simple["TileSizeX"], cfg_simple["TileSizeY"],
        cfg_simple["OffsetX"], cfg_simple["OffsetY"],
    )
    target = torch.tensor(target, dtype=torch.float32) if not isinstance(target, torch.Tensor) else target.detach().clone().to(torch.float32)
    params = torch.tensor(params, dtype=torch.float32) if not isinstance(params, torch.Tensor) else params.detach().clone().to(torch.float32)
    return target, params


def run_one(cfg, target, params, flare_psf, litho, name):
    solver = FlareAwareILT(cfg, litho, flare_psf)
    _, best_mask, info = solver.solve(target, params, return_history=True)
    best_mask_np = best_mask.detach().cpu().numpy()
    target_np = target.detach().cpu().numpy()
    basic_eval = evaluation.Basic()
    l2, pvb = basic_eval.run(best_mask_np, target_np)
    mask_bin = (best_mask.detach().cpu() > 0.5).float()
    target_t = target.cpu().clamp(0, 1)
    binary_err = (mask_bin != target_t).float().sum().item()
    total_px = target.numel()
    binary_error_pct = 100.0 * binary_err / total_px
    final_loss = info["history"]["loss"][-1] if info["history"]["loss"] else float("nan")
    return {
        "name": name,
        "l2": l2,
        "pvb": pvb,
        "binary_error_px": int(binary_err),
        "binary_error_pct": binary_error_pct,
        "final_loss": final_loss,
        "mask": best_mask,
        "info": info,
    }


def main():
    parser = argparse.ArgumentParser(description="Compare flare-aware vs non-flare-aware ILT")
    parser.add_argument("--synthetic", action="store_true", help="Use synthetic pattern (default: benchmark)")
    args = parser.parse_args()

    base_cfg = {
        "max_iters": 40,
        "lr": 0.1,
        "flare_weight_beta": 2.0,
        "lambda1": 0.0,
        "theta_M": 10.0,
        "bin_weight": 0.05,
        "sensitivity_interval": 5,
        "use_amp": True,
    }

    if args.synthetic:
        target, params = create_synthetic_pattern(256, 256)
        pattern_name = "synthetic"
    else:
        target, params = get_benchmark_target_params()
        pattern_name = "benchmark"

    flare_psf = FlarePSF(tis=0.08, kernel_size=33, pixel_size=1.0)
    litho = SimpleDiffractionLitho()

    cfg_aware = {**base_cfg, "flare_reg_weight": 0.3}
    cfg_no_flare = {**base_cfg, "flare_reg_weight": 0.0}

    print(f"Pattern: {pattern_name}  size: {target.shape}")
    print("Running flare-aware (μ=0.3)...")
    result_aware = run_one(cfg_aware, target, params, flare_psf, litho, "flare-aware (μ=0.3)")
    print("Running non-flare-aware (μ=0)...")
    result_no = run_one(cfg_no_flare, target, params, flare_psf, litho, "non-flare-aware (μ=0)")

    # Table
    print("\n" + "=" * 70)
    print("COMPARISON: Flare-aware vs non-flare-aware (same pattern, 40 iters)")
    print("=" * 70)
    print(f"{'Metric':<22} {'Flare-aware (μ=0.3)':<24} {'Non-flare-aware (μ=0)':<24}")
    print("-" * 70)
    print(f"{'L2 Error':<22} {result_aware['l2']:<24.2f} {result_no['l2']:<24.2f}")
    print(f"{'PV Band':<22} {result_aware['pvb']:<24.2f} {result_no['pvb']:<24.2f}")
    print(f"{'Binary error (px)':<22} {result_aware['binary_error_px']:<24} {result_no['binary_error_px']:<24}")
    print(f"{'Binary error (%)':<22} {result_aware['binary_error_pct']:<24.2f} {result_no['binary_error_pct']:<24.2f}")
    print(f"{'Final loss':<22} {result_aware['final_loss']:<24.4e} {result_no['final_loss']:<24.4e}")
    print("=" * 70)

    # Save
    outdir = os.path.join("results", f"flare_comparison_{pattern_name}_{time.strftime('%Y%m%d_%H%M%S')}")
    os.makedirs(outdir, exist_ok=True)
    with open(os.path.join(outdir, "comparison.txt"), "w") as f:
        f.write(f"Pattern: {pattern_name}\n")
        f.write(f"Flare-aware:      L2={result_aware['l2']:.2f}  PVB={result_aware['pvb']:.2f}  err%={result_aware['binary_error_pct']:.2f}  loss={result_aware['final_loss']:.4e}\n")
        f.write(f"Non-flare-aware:  L2={result_no['l2']:.2f}  PVB={result_no['pvb']:.2f}  err%={result_no['binary_error_pct']:.2f}  loss={result_no['final_loss']:.4e}\n")
    torch.save(
        {
            "pattern": pattern_name,
            "target": target,
            "flare_aware": {k: v for k, v in result_aware.items() if k not in ("mask", "info")},
            "non_flare_aware": {k: v for k, v in result_no.items() if k not in ("mask", "info")},
        },
        os.path.join(outdir, "comparison.pt"),
    )
    print(f"\nResults saved to {outdir}/")


if __name__ == "__main__":
    main()
