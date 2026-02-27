#!/usr/bin/env python
"""
Example script for flare-aware ILT optimization using OpenILT + Torch-based
flare modeling.

This script is intentionally simple and uses the existing OpenILT kernel-based
simulator wrapped by `pyilt.lithosim.LithoSim`. For true Abbe/Hopkins
behavior, replace that wrapper with a TorchLitho-based implementation.
"""

import sys

sys.path.append(".")

import torch
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
    best_params, best_mask = solver.solve(target, params)

    # Convert to numpy for basic evaluation
    best_mask_np = best_mask.detach().cpu().numpy()
    target_np = target.detach().cpu().numpy()

    # Use the original OpenILT lithography model for evaluation, so that
    # the printed images match the target resolution.
    basic_eval = evaluation.Basic()
    l2, pvb = basic_eval.run(best_mask_np, target_np)

    print("\n=== Final Results (quick test) ===")
    print(f"L2 Error: {l2:.2f}")
    print(f"PV Band: {pvb:.2f}")

    torch.save(
        {
            "params": best_params,
            "mask": best_mask,
            "target": target,
            "metrics": {"l2": l2, "pvb": pvb},
        },
        "flare_aware_results.pt",
    )

    print("\nResults saved to flare_aware_results.pt")

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
    plt.show()


if __name__ == "__main__":
    main()

