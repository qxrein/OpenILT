#!/usr/bin/env python
"""
Diagnose flare-aware ILT results from a results folder or .pt file.

Usage:
  python examples/diagnose_flare_results.py results/flare_aware_20260227_223659
  python examples/diagnose_flare_results.py results/flare_aware_20260227_223659/flare_aware_results.pt
  python examples/diagnose_flare_results.py   # uses latest results/flare_aware_*/
"""

import argparse
import glob
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

# Headless-friendly matplotlib
os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplconfig")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def find_results_path(path_arg):
    """Resolve to (outdir, pt_path)."""
    if path_arg:
        if path_arg.endswith(".pt"):
            pt_path = path_arg
            outdir = os.path.dirname(pt_path)
        else:
            outdir = path_arg.rstrip("/")
            pt_path = os.path.join(outdir, "flare_aware_results.pt")
        if not os.path.isfile(pt_path):
            raise FileNotFoundError(f"No such file: {pt_path}")
        return outdir, pt_path
    # Default: latest results folder
    dirs = sorted(glob.glob(os.path.join("results", "flare_aware_*")))
    if not dirs:
        raise FileNotFoundError("No results found under results/flare_aware_*/")
    outdir = dirs[-1]
    pt_path = os.path.join(outdir, "flare_aware_results.pt")
    if not os.path.isfile(pt_path):
        raise FileNotFoundError(f"No flare_aware_results.pt in {outdir}")
    return outdir, pt_path


def diagnose(outdir, pt_path):
    data = torch.load(pt_path, map_location="cpu")
    mask = data["mask"]
    target = data["target"]

    print("=" * 55)
    print("FLARE-AWARE ILT DIAGNOSIS")
    print("=" * 55)
    print(f"Results: {pt_path}\n")

    # Mask stats
    print("MASK")
    print("-" * 40)
    print(f"  shape: {tuple(mask.shape)}")
    print(f"  range: [{mask.min().item():.4f}, {mask.max().item():.4f}]")
    print(f"  mean:  {mask.mean().item():.4f}")
    print(f"  std:   {mask.std().item():.4f}")

    mask_bin = (mask > 0.5).float()
    diff = (mask_bin != target).float()
    err_px = diff.sum().item()
    total_px = mask.numel()
    err_pct = 100.0 * err_px / total_px
    print(f"  binary error: {int(err_px)} / {int(total_px)} ({err_pct:.2f}%)")

    # Loss history
    hist = data.get("history", {})
    if hist:
        loss = np.array(hist["loss"], dtype=float)
        wl2 = np.array(hist["weighted_l2"], dtype=float)
        reg = np.array(hist["flare_reg"], dtype=float)
        n = len(loss)
        print("\nLOSS HISTORY")
        print("-" * 40)
        print(f"  iterations: {n}")
        print(f"  total loss: [{loss.min():.4f}, {loss.max():.4f}]  final: {loss[-1]:.4f}")
        print(f"  Δ total:   {loss[-1] - loss[0]:+.4f}")
        print(f"  Δ L2:      {wl2[-1] - wl2[0]:+.4f}")
        print(f"  Δ reg:     {reg[-1] - reg[0]:+.4f}")
    else:
        loss = wl2 = reg = np.array([])
        n = 0

    # Metrics if saved
    metrics = data.get("metrics", {})
    if metrics:
        print("\nMETRICS")
        print("-" * 40)
        for k, v in metrics.items():
            print(f"  {k}: {v}")

    # Flare maps (S, w) from best_snapshot if available
    snapshot = data.get("best_snapshot", {})
    S = snapshot.get("S")
    w = snapshot.get("w")
    alpha = snapshot.get("alpha_mask")
    if S is not None:
        S_np = S.cpu().numpy()
        if S_np.ndim == 4:
            S_np = S_np.squeeze()
        print("\nSENSITIVITY MAP S")
        print("-" * 40)
        print(f"  shape: {S_np.shape}  range: [{S_np.min():.4f}, {S_np.max():.4f}]")
        print(f"  mean: {S_np.mean():.4f}  std: {S_np.std():.4f}")
    if w is not None:
        w_np = w.cpu().numpy()
        if w_np.ndim == 4:
            w_np = w_np.squeeze()
        print("\nADAPTIVE WEIGHTS w")
        print("-" * 40)
        print(f"  shape: {w_np.shape}  range: [{w_np.min():.4f}, {w_np.max():.4f}]")
        print(f"  mean: {w_np.mean():.4f}  std: {w_np.std():.4f}")

    # Diagnostic figure
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))

    t = target.cpu().numpy()
    m = mask.cpu().numpy()

    axes[0, 0].imshow(t, cmap="gray")
    axes[0, 0].set_title("Target")
    axes[0, 0].axis("off")

    im1 = axes[0, 1].imshow(m, cmap="gray")
    axes[0, 1].set_title("Optimized Mask")
    axes[0, 1].axis("off")
    plt.colorbar(im1, ax=axes[0, 1])

    diff_img = np.abs(m - t)
    im2 = axes[0, 2].imshow(diff_img, cmap="hot")
    axes[0, 2].set_title("|Mask - Target|")
    axes[0, 2].axis("off")
    plt.colorbar(im2, ax=axes[0, 2])

    if n > 0:
        iters = np.arange(n)
        axes[1, 0].plot(iters, loss, "--o", color="black", label="total", markersize=2)
        axes[1, 0].plot(iters, wl2, ":s", color="dimgray", label="weighted_l2", markersize=2)
        axes[1, 0].plot(iters, reg, "-.^", color="gray", label="flare_reg", markersize=2)
        axes[1, 0].set_yscale("log")
        axes[1, 0].set_xlabel("iteration")
        axes[1, 0].set_ylabel("loss")
        axes[1, 0].set_title("Loss Curves")
        axes[1, 0].legend()

    axes[1, 1].hist(m.ravel(), bins=50, color="gray", alpha=0.8)
    axes[1, 1].axvline(0.5, color="black", linestyle="--", label="threshold")
    axes[1, 1].set_title("Mask Value Distribution")
    axes[1, 1].set_xlabel("pixel value")
    axes[1, 1].legend()

    err_map = (mask_bin != target).float().cpu().numpy()
    axes[1, 2].imshow(err_map, cmap="Reds")
    axes[1, 2].set_title(f"Error Map ({int(err_px)} px)")
    axes[1, 2].axis("off")

    plt.tight_layout()
    diag_path = os.path.join(outdir, "diagnostic_plot.png")
    plt.savefig(diag_path, dpi=150)
    plt.close()
    print(f"\nDiagnostic plot saved: {diag_path}")

    # Flare diagnostics: S, w, alpha
    if S is not None or w is not None or alpha is not None:
        fig2, ax2 = plt.subplots(1, 3, figsize=(12, 4))
        idx = 0
        for name, arr in [("Sensitivity S", S), ("Weights w", w), ("Alpha α", alpha)]:
            if arr is not None:
                a = arr.cpu().numpy()
                if a.ndim == 4:
                    a = a.squeeze()
                im = ax2[idx].imshow(a, cmap="viridis")
                ax2[idx].set_title(name)
                ax2[idx].axis("off")
                plt.colorbar(im, ax=ax2[idx])
                idx += 1
        for j in range(idx, 3):
            ax2[j].axis("off")
        plt.tight_layout()
        flare_path = os.path.join(outdir, "flare_diagnostics.png")
        plt.savefig(flare_path, dpi=150)
        plt.close()
        print(f"Flare diagnostics saved: {flare_path}")
    print("=" * 55)


def main():
    parser = argparse.ArgumentParser(description="Diagnose flare-aware ILT results")
    parser.add_argument(
        "path",
        nargs="?",
        default=None,
        help="Results folder (e.g. results/flare_aware_20260227_223659) or path to .pt file",
    )
    args = parser.parse_args()
    outdir, pt_path = find_results_path(args.path)
    diagnose(outdir, pt_path)


if __name__ == "__main__":
    main()
