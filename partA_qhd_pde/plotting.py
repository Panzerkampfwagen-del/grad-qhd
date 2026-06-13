"""Plotting utilities for Part A QHD results."""
from __future__ import annotations
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from typing import Optional


def plot_metrics(
    results: dict,
    title: str = "",
    output_path: Optional[str] = None,
) -> plt.Figure:
    """
    Three-panel figure: E[f], E[||grad f||^2], P[f-f*<=delta] vs step.

    results : dict mapping label -> object with .steps, .ef, .eg2, .prob
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    linestyles = ["-", "--", "-.", ":"]

    for i, (name, res) in enumerate(results.items()):
        c = colors[i % len(colors)]
        ls = linestyles[i % len(linestyles)]
        axes[0].plot(res.steps, res.ef,   label=name, color=c, ls=ls, lw=1.5)
        axes[1].plot(res.steps, res.eg2,  label=name, color=c, ls=ls, lw=1.5)
        axes[2].plot(res.steps, res.prob, label=name, color=c, ls=ls, lw=1.5)

    axes[0].set_ylabel(r"$\mathbb{E}[f(X)]$")
    axes[1].set_ylabel(r"$\mathbb{E}[\|\nabla f(X)\|^2]$")
    axes[2].set_ylabel(r"$P[f(X) - f^* \leq \delta]$")
    for ax in axes:
        ax.set_xlabel("Step $k$")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    if title:
        fig.suptitle(title, fontsize=12, y=1.02)
    plt.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {output_path}")
    return fig


def plot_density_snapshots(
    snapshots: list,
    X: np.ndarray,
    Y: np.ndarray,
    f_grid: np.ndarray,
    title: str = "",
    output_path: Optional[str] = None,
) -> plt.Figure:
    """
    Plot |psi|^2 density at several time points overlaid on f contours.
    snapshots : list of (step_idx, density_2d) — density shape (N, N), indexing (ix, iy)
    """
    n = len(snapshots)
    if n == 0:
        return plt.figure()

    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4), squeeze=False)
    axes = axes[0]

    f_lo = float(f_grid.min())
    f_hi = float(np.percentile(f_grid, 90))
    levels = np.linspace(f_lo, f_hi, 20)

    xmin, xmax = float(X.min()), float(X.max())
    ymin, ymax = float(Y.min()), float(Y.max())

    for ax, (step, density) in zip(axes, snapshots):
        ax.contour(X, Y, f_grid, levels=levels, colors="gray",
                   alpha=0.5, linewidths=0.5)
        # density[ix, iy]: first index = x → transpose for imshow(rows=y, cols=x)
        im = ax.imshow(
            density.T, origin="lower",
            extent=[xmin, xmax, ymin, ymax],
            cmap="hot", aspect="auto", interpolation="bilinear",
        )
        ax.set_title(f"step {step}")
        ax.set_xlabel("$x$")
        ax.set_ylabel("$y$")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    if title:
        fig.suptitle(title, fontsize=12, y=1.02)
    plt.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {output_path}")
    return fig
