"""Monte Carlo condition numbers for projection regularization of the
singular pencil A0 - lambda*A1 defined below.

Requires numpy and scipy.  Uses independent random complex subspaces,
with fixed reproducible seed and 95% distribution-free quantile CIs.
"""
import numpy as np
from scipy.linalg import eig
from scipy.optimize import brentq
from scipy.stats import binom

N = 100_000
BATCH = 2_500
SEED = 20261011
rng = np.random.default_rng(SEED)

A0 = np.array([[0, 1, 0], [0, 0, 0], [0, 0, 2]], dtype=complex)
A1 = np.diag([1., 0., 1.]).astype(complex)
REFERENCE = np.sqrt(5.)


def random_complements(size):
    """Orthonormal bases for orthogonal complements of independent Haar lines."""
    g = (rng.standard_normal((size, 3, 3))
         + 1j * rng.standard_normal((size, 3, 3))) / np.sqrt(2)
    q, _ = np.linalg.qr(g)
    return q[:, :, 1:]  # shape (size, 3, 2)


def quantile_ci(values, p, alpha=0.05):
    """Distribution-free 1-alpha CI for the population p-quantile."""
    x = np.sort(values)
    n = len(x)
    # For continuous F, #{Xi <= population p-quantile} ~ Binomial(n,p).
    lo = int(binom.ppf(alpha / 2, n, p))
    hi = int(binom.isf(alpha / 2, n, p))
    # endpoints X_(lo), X_(hi+1) with order statistics starting at 1
    return x[max(0, lo - 1)], x[min(n - 1, hi)]


conds, eigen_errors = [], []
for start in range(0, N, BATCH):
    size = min(BATCH, N - start)
    omega_perp = random_complements(size)
    psi_perp = random_complements(size)
    for R, L in zip(omega_perp, psi_perp):
        Ap = L.conj().T @ A0 @ R
        Bp = L.conj().T @ A1 @ R
        eigenvalues, vl, vr = eig(Ap, Bp, left=True, right=True)
        j = np.argmin(np.abs(eigenvalues - 2))  # preserved eigenvalue
        lam = eigenvalues[j]
        x, y = vr[:, j], vl[:, j]
        # Absolute eigenvalue conditioning for paired Frobenius-norm perturbations:
        cond = np.sqrt(1 + np.abs(lam)**2) * (np.linalg.norm(x) * np.linalg.norm(y)) \
               / np.abs(np.vdot(y, Bp @ x))
        conds.append(cond)
        eigen_errors.append(abs(lam - 2))
conds = np.array(conds)


def exact_quantile(p):
    # kappa/sqrt(5) = 1/sqrt(UV) for iid U,V~Uniform(0,1).
    # P{kappa/sqrt(5)>t} = (1+2log t)/t^2 for t>=1.
    t = brentq(lambda t: (1 + 2*np.log(t))/t**2 - (1-p), 1.00000001, 1e5)
    return REFERENCE * t

print(f"N = {N:,}, seed = {SEED}, reference = sqrt(5) = {REFERENCE:.6f}")
print(f"max eigenvalue preservation error: {max(eigen_errors):.3e}")
for p in (0.25, 0.5, 0.75, 0.9, 0.95, 0.99):
    q = np.quantile(conds, p)
    lo, hi = quantile_ci(conds, p)
    theory = exact_quantile(p)
    print(f"p={p:.2f}: observed={q:.5f} = {q/REFERENCE:.3f} sqrt(5), "
          f"95% CI=({lo:.5f}, {hi:.5f}), theory={theory:.5f}")
print(f"Mean = {conds.mean():.5f} (theory {4*REFERENCE:.5f}; not preferred due to infinite variance)")
print(f"Fraction <= 4sqrt(5): {np.mean(conds <= 4*REFERENCE):.5f} "
      f"(theory {1-(1+2*np.log(4))/16:.5f})")
