#!/usr/bin/env python
"""
Example script for flare-aware ILT optimization using OpenILT + Torch-based
flare modeling.

This script is intentionally simple and uses the existing OpenILT kernel-based
simulator wrapped by `pyilt.lithosim.LithoSim`. For true Abbe/Hopkins
behavior, replace that wrapper with a TorchLitho-based implementation.
"""

import argparse
import os
import sys
import time

sys.path.append(".")

import torch
import matplotlib

# Avoid permission issues on shared systems
os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplconfig")

if not os.environ.get("DISPLAY"):
    matplotlib.use("Agg")
import matplotlib.pyplot as plt

import pycommon.glp as glp
import pyilt.initializer as initializer
import pyilt.evaluation as evaluation
import pyilt.simpleilt as simpleilt

from pyilt.flare_psf import FlarePSF
from pyilt.flare_aware_solver import FlareAwareILT
from pyilt.lithosim import LithoSim


def create_synthetic_pattern(h: int, w: int) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Create dense (left) vs sparse (right) pattern for flare contrast.
    Left half: dense (all 1s) → high alpha (flare). Right half: sparse lines → low alpha.
    """
    target = torch.zeros(h, w, dtype=torch.float32)
    # Left 50%: fully dense
    target[:, : w // 2] = 1.0
    # Right 50%: sparse vertical lines (every 16 px, 8 px wide)
    for c in range(w // 2, w, 16):
        target[:, c : min(c + 8, w)] = 1.0
    # Initial params: target + small noise
    params = 2 * target - 1 + 0.05 * torch.randn(h, w)
    return target, params.to(torch.float32)


def main():
    parser = argparse.ArgumentParser(description="Flare-aware ILT optimization")
    parser.add_argument(
        "--synthetic",
        action="store_true",
        help="Use synthetic dense/sparse pattern instead of benchmark (strong flare contrast)",
    )
    parser.add_argument(
        "--torchlitho",
        action="store_true",
        help="Use TorchLitho for I_diff (no gradients; S will be uniform). For S verification, TorchLitho would need to be differentiable.",
    )
    parser.add_argument(
        "--no-flare",
        action="store_true",
        help="Disable flare regularization (μ=0) for comparison with flare-aware run.",
    )
    parser.add_argument(
        "--size",
        type=int,
        default=2048,
        help="Mask size (H=W). Default 2048 for paper-quality. Use 256 for quick test (less memory).",
    )
    args = parser.parse_args()

    cfg = {
        # Paper-quality: 2048x2048, 300 iters (reduce --size 256 for quick test)
        "max_iters": 300,
        "lr": 0.1,
        "flare_weight_beta": 2.0,  # Smaller β so w varies more with α
        "flare_reg_weight": 0.0 if args.no_flare else 0.3,  # μ=0 for non-flare-aware comparison
        "lambda1": 0.0,
        "theta_M": 10.0,
        "bin_weight": 0.05,  # Push mask toward 0/1 (reduces stippling)
        "sensitivity_interval": 5,
        "use_amp": True,
    }

    # Small flare kernel for speed (33 vs 101: ~10x faster conv)
    flare_psf = FlarePSF(tis=0.08, kernel_size=33, pixel_size=1.0)

    # Litho model: SimpleDiffractionLitho (differentiable) or TorchLitho (no gradients)
    if args.torchlitho:
        from pyilt.torchlitho_adapter import TorchLithoAbbe

        litho = TorchLithoAbbe(image_method="abbe")
        print("Using TorchLitho (S will be uniform: no gradients from litho)")
    else:
        from pyilt.simple_diffraction import SimpleDiffractionLitho

        litho = SimpleDiffractionLitho()
        print("Using SimpleDiffractionLitho (differentiable Gaussian blur)")

    solver = FlareAwareILT(cfg, litho, flare_psf)

    if args.synthetic:
        # Synthetic: dense (left) vs sparse (right) for strong flare contrast
        size = args.size
        target, params = create_synthetic_pattern(size, size)
        print(f"Using synthetic dense/sparse pattern {size}x{size}")
    else:
        # Load benchmark testcase (config by --size)
        design = glp.Design("./benchmark/ICCAD2013/M1_test1.glp", down=1)
        config_path = "./config/simpleilt2048.txt" if args.size >= 2048 else "./config/multilevel256.txt"
        cfg_simple = simpleilt.SimpleCfg(config_path)
        design.center(
            cfg_simple["TileSizeX"],
            cfg_simple["TileSizeY"],
            cfg_simple["OffsetX"],
            cfg_simple["OffsetY"],
        )
        init = initializer.PixelInit()
        target, params = init.run(
            design,
            cfg_simple["TileSizeX"],
            cfg_simple["TileSizeY"],
            cfg_simple["OffsetX"],
            cfg_simple["OffsetY"],
        )
        if isinstance(target, torch.Tensor):
            target = target.detach().clone().to(dtype=torch.float32)
        else:
            target = torch.tensor(target, dtype=torch.float32)
        if isinstance(params, torch.Tensor):
            params = params.detach().clone().to(dtype=torch.float32)
        else:
            params = torch.tensor(params, dtype=torch.float32)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    h, w = target.shape[-2], target.shape[-1]
    print(f"Mask size: {h}x{w}  |  iters: {cfg['max_iters']}  |  device: {device}")
    print("Starting flare-aware ILT optimization...")
    best_params, best_mask, info = solver.solve(target, params, return_history=True)

    # Convert to numpy for basic evaluation + preview outputs
    best_mask_np = best_mask.detach().cpu().numpy()
    target_np = target.detach().cpu().numpy()

    # Use the original OpenILT lithography model for evaluation, so that
    # the printed images match the target resolution.
    basic_eval = evaluation.Basic()
    l2, pvb = basic_eval.run(best_mask_np, target_np)

    print("\n=== Final Results (quick test) ===")
    print(f"L2 Error: {l2:.2f}")
    print(f"PV Band: {pvb:.2f}")

    # Results bundle for paper-friendly proof-of-concept
    outdir = os.path.join("results", f"flare_aware_{time.strftime('%Y%m%d_%H%M%S')}")
    os.makedirs(outdir, exist_ok=True)

    torch.save(
        {
            "params": best_params,
            "mask": best_mask,
            "target": target,
            "metrics": {"l2": l2, "pvb": pvb},
            "history": info["history"],
            "best_snapshot": {k: v.detach().cpu() for k, v in info["best_snapshot"].items()},
        },
        os.path.join(outdir, "flare_aware_results.pt"),
    )

    print(f"\nResults saved to {outdir}/flare_aware_results.pt")

    # Quick preview of target vs optimized mask (grayscale, print-friendly)
    plt.figure(figsize=(6, 3))
    plt.subplot(1, 2, 1)
    plt.title("Optimized Mask")
    plt.imshow(best_mask_np, cmap="gray")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.title("Target")
    plt.imshow(target_np, cmap="gray")
    plt.axis("off")

    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "mask_vs_target.png"), dpi=200)
    if os.environ.get("DISPLAY"):
        plt.show()
    plt.close()

    # Save loss curves (monochrome, dotted/marker styles for gray printouts)
    # Add grad_norm subplot if available (to verify gradient flow)
    has_grad = "grad_norm" in info["history"] and len(info["history"]["grad_norm"]) > 0
    ncols = 2 if has_grad else 1
    plt.figure(figsize=(6 * ncols, 3))
    iters = range(len(info["history"]["loss"]))

    plt.subplot(1, ncols, 1)
    plt.plot(
        iters,
        info["history"]["loss"],
        linestyle="--",
        marker="o",
        color="black",
        label="total",
        linewidth=1.0,
        markersize=3,
    )
    plt.plot(
        iters,
        info["history"]["weighted_l2"],
        linestyle=":",
        marker="s",
        color="dimgray",
        label="weighted_l2",
        linewidth=1.0,
        markersize=3,
    )
    plt.plot(
        iters,
        info["history"]["flare_reg"],
        linestyle="-.",
        marker="^",
        color="gray",
        label="flare_reg",
        linewidth=1.0,
        markersize=3,
    )
    if "bin_loss" in info["history"]:
        plt.plot(
            iters,
            info["history"]["bin_loss"],
            linestyle=":",
            marker="d",
            color="silver",
            label="bin_loss",
            linewidth=1.0,
            markersize=3,
        )
    plt.yscale("log")
    plt.xlabel("iteration")
    plt.ylabel("loss")
    plt.legend()
    plt.title("Loss curves")

    if has_grad:
        plt.subplot(1, ncols, 2)
        plt.plot(iters, info["history"]["grad_norm"], color="black", linewidth=1.0)
        plt.xlabel("iteration")
        plt.ylabel("||∇L||")
        plt.yscale("log")
        plt.title("Gradient norm")

    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "loss_curves.png"), dpi=200)
    plt.close()


if __name__ == "__main__":
    main()

