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

import pycommon.glp as glp
import pyilt.initializer as initializer
import pyilt.evaluation as evaluation

from pyilt.flare_psf import FlarePSF
from pyilt.flare_aware_solver import FlareAwareILT
from pyilt.lithosim import LithoSim


def main():
    cfg = {
        # Reduced for ~2 minute runs on mid-range GPUs
        "max_iters": 60,
        "lr": 0.01,
        "flare_weight_beta": 5.0,
        "flare_reg_weight": 0.1,
        "lambda1": 1.0,
        "theta_M": 10.0,
    }

    flare_psf = FlarePSF(tis=0.08, kernel_size=401, pixel_size=1.0)

    from pyilt.torchlitho_adapter import TorchLithoAbbe

    litho = TorchLithoAbbe(image_method="abbe")

    solver = FlareAwareILT(cfg, litho, flare_psf)

    # Load a benchmark testcase
    design = glp.Design("./benchmark/ICCAD2013/M1_test1.glp")

    # Use existing pixel initializer from OpenILT
    init = initializer.PixelInit()
    target, params = init.run(design, 2048, 2048, 0, 0)

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

    # Convert to numpy for evaluation
    best_mask_np = best_mask.detach().cpu().numpy()
    target_np = target.detach().cpu().numpy()

    basic_eval = evaluation.Basic(litho, 0.5)
    l2, pvb = basic_eval.run(best_mask_np, target_np)

    epe_checker = evaluation.EPEChecker(litho, 0.5)
    epe_in, epe_out = epe_checker.run(best_mask_np, target_np)
    epe = epe_in + epe_out

    shot_counter = evaluation.ShotCounter(litho, 0.5)
    shots = shot_counter.run(best_mask_np)

    print("\n=== Final Results ===")
    print(f"L2 Error: {l2:.2f}")
    print(f"PV Band: {pvb:.2f}")
    print(f"EPE Violations: {epe}")
    print(f"Shot Count: {shots}")

    torch.save(
        {
            "params": best_params,
            "mask": best_mask,
            "target": target,
            "metrics": {"l2": l2, "pvb": pvb, "epe": epe, "shots": shots},
        },
        "flare_aware_results.pt",
    )

    print("\nResults saved to flare_aware_results.pt")


if __name__ == "__main__":
    main()

