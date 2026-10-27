#!/usr/bin/env python3
"""Pseudospectra of diag(J_2(2), J_3(3)) before/after complex Gaussian noise.

Run: python shattering_experiment.py --outdir figures --grid 1401
Dependencies: numpy, scipy, matplotlib. No EigTool is required.

Gaussian scale and pseudospectral level are different parameters:
    B = A + gamma*G,  E|G_ij|^2 = 1/n,  gamma = 1e-4;
    Lambda_epsilon(B) = {z: sigma_min(z I - B) < epsilon}.
The Gaussian draw is NOT normalized by its realized norm.
Contours use direct SVDs, never eigenvalues of (zI-B)^*(zI-B).
Each zoom evaluates the full 5-by-5 matrix, including cross-block noise.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import scipy
import matplotlib
import matplotlib.pyplot as plt
from scipy.linalg import block_diag, eig, expm, norm
from scipy.ndimage import label


def jordan(m: int, lam: float) -> np.ndarray:
    return lam*np.eye(m) + np.diag(np.ones(m-1), 1)


def smallest_singular_values(M: np.ndarray, x: np.ndarray,
                             y: np.ndarray, batch: int = 32) -> np.ndarray:
    """Evaluate sigma_min(z I-M) in row batches to bound memory use."""
    n = M.shape[0]
    if M.shape != (n, n) or batch < 1:
        raise ValueError("M must be square and batch must be positive.")
    I = np.eye(n, dtype=complex)
    values = np.empty((len(y), len(x)))
    for j in range(0, len(y), batch):
        z = x[None, :] + 1j*y[j:j+batch, None]
        pencils = z[..., None, None]*I - M
        values[j:j+batch, :] = np.linalg.svd(
            pencils, compute_uv=False)[..., -1]
    return values


def make_example(seed: int = 143, gamma: float = 1e-4):
    A = block_diag(jordan(2, 2.), jordan(3, 3.)).astype(complex)
    n = len(A)
    rng = np.random.default_rng(seed)
    G = (rng.standard_normal((n, n)) +
         1j*rng.standard_normal((n, n)))/np.sqrt(2*n)
    E = gamma*G
    B = A+E
    lam, V = eig(B)
    order = np.lexsort((lam.imag, lam.real))
    lam, V = lam[order], V[:, order]
    V /= np.linalg.norm(V, axis=0)  # Explicit unit-norm columns.
    Vinv = np.linalg.solve(V, np.eye(n))
    gap = np.min(np.abs(lam[:, None]-lam[None, :])[~np.eye(n, dtype=bool)])
    kappa = norm(V, 2)*norm(Vinv, 2)  # An upper bound on infimum kappa_V(B).
    # Right solve, without explicitly forming the inverse in reconstruction.
    reconstructed = np.linalg.solve(V.T, (V*lam).T).T
    expB_eig = np.linalg.solve(V.T, (V*np.exp(lam)).T).T
    omega, x0 = 0.02, 0.006
    dx = np.abs((lam.real-x0+omega/2) % omega - omega/2)
    dy = np.abs((lam.imag+omega/2) % omega - omega/2)
    distance_to_grid = float(np.minimum(dx, dy).min())
    cells = np.column_stack((np.floor((lam.real-x0)/omega),
                             np.floor(lam.imag/omega))).astype(int)
    results = {
        'seed': seed, 'gamma': gamma,
        'illustrative_grid_mesh': omega, 'illustrative_grid_real_offset': x0,
        'illustrative_grid_cells': cells.tolist(),
        'minimum_eigenvalue_distance_to_grid': distance_to_grid,
        'Bauer_Fike_lower_bound_for_sigma_on_grid': distance_to_grid/kappa,
        'numpy_version': np.__version__, 'scipy_version': scipy.__version__,
        'matplotlib_version': matplotlib.__version__,
        'norm_A': float(norm(A, 2)), 'norm_E': float(norm(E, 2)),
        'relative_norm_E': float(norm(E, 2)/norm(A, 2)),
        'eigenvalues_real_imag': np.column_stack((lam.real, lam.imag)).tolist(),
        'unit_column_eigenvector_condition_number': float(kappa),
        'individual_eigenvalue_condition_numbers': norm(Vinv, axis=1).tolist(),
        'minimum_eigenvalue_gap': float(gap),
        'disjoint_Bauer_Fike_disks_if_epsilon_less_than': float(gap/(2*kappa)),
        'eigenpair_residual_for_B': float(norm(B@V-V*lam, 2)),
        'reconstruction_error_for_B': float(norm(reconstructed-B, 2)),
        'reconstruction_error_for_A': float(norm(reconstructed-A, 2)),
        'relative_exponential_error_for_B': float(norm(expB_eig-expm(B), 2)/norm(expm(B), 2)),
        'relative_exponential_error_for_A': float(norm(expB_eig-expm(A), 2)/norm(expm(A), 2)),
    }
    return A, B, G, E, lam, V, results


def _plot_zoom_on_ax(ax, M: np.ndarray, eigenvalues: np.ndarray,
                     center: float, halfwidth: float, title: str,
                     grid: int = 1401, draw_grid: bool = False):
    """Plot one pseudospectral zoom on an existing Matplotlib axis."""
    offsets = np.linspace(-halfwidth, halfwidth, grid)
    sigma = smallest_singular_values(M, center+offsets, offsets)
    log_sigma = np.log10(np.maximum(sigma, np.finfo(float).tiny))
    levels = [-6., -5., -4.]

    if draw_grid:
        # A hand-chosen illustrative grid, not the theorem's random grid.
        # Since 2 and 3 are multiples of .02, these offsets serve both zooms.
        for t in .006 + .02*np.arange(-10, 11):
            if -halfwidth < t < halfwidth:
                ax.axvline(t, linestyle=':', linewidth=.8, alpha=.5)
        for t in .02*np.arange(-10, 11):
            if -halfwidth < t < halfwidth:
                ax.axhline(t, linestyle=':', linewidth=.8, alpha=.5)

    contours = ax.contour(offsets, offsets, log_sigma, levels=levels,
                          linewidths=1.45)
    inside = ((abs(eigenvalues.real-center) < halfwidth) &
              (abs(eigenvalues.imag) < halfwidth))
    ax.plot(eigenvalues.real[inside]-center, eigenvalues.imag[inside],
            marker='x', linestyle='none', markersize=4.2, markeredgewidth=1.15)
    ax.set(xlim=(-halfwidth, halfwidth), ylim=(-halfwidth, halfwidth),
           xlabel=rf'$\operatorname{{Re}}z-{center:g}$',
           ylabel=r'$\operatorname{Im}z$', title=title)
    ax.set_aspect('equal', adjustable='box')

    # These are sampled-grid diagnostics, not an interval-arithmetic proof.
    diagnostics = {f'{10**level:.0e}': int(label(sigma < 10**level)[1])
                   for level in levels}
    return contours, diagnostics


def plot_shattering_figure(plots, stem: Path, grid: int = 1401,
                           show: bool = True):
    """Create the 2-by-2 shattering figure with one centered legend at the bottom."""
    # Slightly larger type than the original four-panel construction.  Because
    # the complete figure is now saved as one full-width graphic, these sizes
    # are also preserved much better when the figure is included in LaTeX.
    with plt.rc_context({
        'font.size': 11.0,
        'axes.titlesize': 11.5,
        'axes.labelsize': 11.5,
        'xtick.labelsize': 10.5,
        'ytick.labelsize': 10.5,
        'legend.fontsize': 10.5,
    }):
        fig, axes = plt.subplots(2, 2, figsize=(7.2, 6.6))
        diagnostics = {}
        legend_handles = None
        levels = [-6., -5., -4.]

        for ax, (M, eigenvalues, center, width, title, name) in zip(
                axes.flat, plots):
            contours, diag = _plot_zoom_on_ax(
                ax, M, eigenvalues, center, width, title, grid,
                draw_grid=(name in {'shattering_after2', 'shattering_after3'}))
            diagnostics[name] = diag
            if legend_handles is None:
                legend_handles, _ = contours.legend_elements()

        fig.legend(
            legend_handles,
            [rf'$\epsilon=10^{{{int(level)}}}$' for level in levels],
            loc='lower center', bbox_to_anchor=(0.5, 0.018),
            ncol=3, frameon=False, columnspacing=1.5,
            handletextpad=0.6
        )
        fig.subplots_adjust(left=0.105, right=0.985, top=0.955,
                            bottom=0.135, wspace=0.27, hspace=0.33)

        for extension in ('pdf', 'png'):
            fig.savefig(stem.with_suffix('.'+extension), dpi=240)
        if show:
            plt.show()
        plt.close(fig)

    return diagnostics

def main(outdir: str | Path = 'figures', grid: int = 1401,
         show: bool = True) -> dict:
    if grid < 101:
        raise ValueError('Use at least 101 grid points per coordinate.')
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    A, B, G, E, lam, V, results = make_example()
    np.savez(outdir/'shattering_data.npz', A=A, B=B, G=G, E=E,
             eigenvalues=lam, eigenvectors=V, epsilon_levels=[1e-6, 1e-5, 1e-4])
    original_eigenvalues = np.array([2., 2., 3., 3., 3.], dtype=complex)
    results['sampled_components_by_zoom'] = {}
    plots = [
        (A, original_eigenvalues, 2., .014, '(a) Before perturbation: near 2', 'shattering_before2'),
        (B, lam,                  2., .014, '(b) After perturbation: near 2', 'shattering_after2'),
        (A, original_eigenvalues, 3., .060, '(c) Before perturbation: near 3', 'shattering_before3'),
        (B, lam,                  3., .060, '(d) After perturbation: near 3', 'shattering_after3'),
    ]
    results['sampled_components_by_zoom'] = plot_shattering_figure(
        plots, outdir/'shattering_figure', grid=grid, show=show)
    (outdir/'shattering_diagnostics.json').write_text(json.dumps(results, indent=2)+'\n')
    print(json.dumps(results, indent=2))
    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--outdir', type=Path, default=Path('figures'))
    parser.add_argument('--grid', type=int, default=1401)
    parser.add_argument('--no-show', action='store_true')
    args = parser.parse_args()
    main(args.outdir, args.grid, show=not args.no_show)
