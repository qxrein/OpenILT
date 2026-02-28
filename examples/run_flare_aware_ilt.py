#!/usr/bin/env python
"""
Example script for flare-aware ILT optimization using OpenILT + Torch-based
flare modeling.

This script is intentionally simple and uses the existing OpenILT kernel-based
simulator wrapped by `pyilt.lithosim.LithoSim`. For true Abbe/Hopkins
behavior, replace that wrapper with a TorchLitho-based implementation.
"""

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


def main():
    cfg = {
        # ILT config tuned for gradient flow + clear convergence
        "max_iters": 60,
        "lr": 0.1,
        "flare_weight_beta": 5.0,
        "flare_reg_weight": 0.01,
        "lambda1": 0.0,
        "theta_M": 10.0,
    }

    # Smaller flare kernel for quick tests
    flare_psf = FlarePSF(tis=0.08, kernel_size=101, pixel_size=1.0)

    # Use SimpleDiffractionLitho for optimization so gradients flow.
    # TorchLithoAbbe returns detached tensors (no gradients), so the
    # optimizer cannot update the mask. SimpleDiffractionLitho uses
    # I_diff = mask (identity), which is fully differentiable.
    from pyilt.simple_diffraction import SimpleDiffractionLitho

    litho = SimpleDiffractionLitho()

    solver = FlareAwareILT(cfg, litho, flare_psf)

    # Load a benchmark testcase
    design = glp.Design("./benchmark/ICCAD2013/M1_test1.glp", down=1)

    # Use the same tiling/centering as SimpleILT so the tile actually
    # contains geometry.
    cfg_simple = simpleilt.SimpleCfg("./config/simpleilt512.txt")
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

    # Ensure float32 tensors without triggering copy-construction warnings
    if isinstance(target, torch.Tensor):
        target = target.detach().clone().to(dtype=torch.float32)
    else:
        target = torch.tensor(target, dtype=torch.float32)

    if isinstance(params, torch.Tensor):
        params = params.detach().clone().to(dtype=torch.float32)
    else:
        params = torch.tensor(params, dtype=torch.float32)

    # Run optimization
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

