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
        # Very small iteration count for fast testing
        "max_iters": 20,
        "lr": 0.01,
        "flare_weight_beta": 5.0,
        "flare_reg_weight": 0.1,
        "lambda1": 1.0,
        "theta_M": 10.0,
    }

    # Smaller flare kernel for quick tests
    flare_psf = FlarePSF(tis=0.08, kernel_size=101, pixel_size=1.0)

    from pyilt.torchlitho_adapter import TorchLithoAbbe

    litho = TorchLithoAbbe(image_method="abbe")

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

    # Quick preview of target vs optimized mask
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

    # Save loss curves
    plt.figure(figsize=(6, 3))
    plt.plot(info["history"]["loss"], label="total")
    plt.plot(info["history"]["weighted_l2"], label="weighted_l2")
    plt.plot(info["history"]["flare_reg"], label="flare_reg")
    plt.yscale("log")
    plt.xlabel("iteration")
    plt.ylabel("loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "loss_curves.png"), dpi=200)
    plt.close()


if __name__ == "__main__":
    main()

