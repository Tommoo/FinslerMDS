"""Corrected metric Finsler SMACOF, adapted from scikit-learn's _mds.py.

See Appendix C of Dagès, Weber, Cremers and Kimmel, *Harnessing Data
Asymmetry in Manifold Learning* (NeurIPS 2026).

The public ``smacof`` function runs ``_smacof_single`` once per initialization,
selects the minimum-stress result, and returns (embedding, stress) by default,
or (embedding, stress, n_iter) when ``return_n_iter=True``.

For symmetric pair weights W, the update is

    Y_new = V^+ @ (B(Y) @ Y - C/2) @ (I + omega omega.T)^(-1),

where V = diag(W @ 1) - W, omega = (0, ..., 0, alpha), B is built from
the *symmetric part* of the directed dissimilarities and
C = (W * (D - D.T)) @ 1 @ omega.T.

For uniform off-diagonal weights w, the left multiplication by V^+ reduces
exactly to division by n*w, without forming a pseudoinverse. For other
weights the n-by-n Laplacian pseudoinverse is computed once per smacof call,
and reused for all restarts and iterations. There is no Kronecker product,
GMRES, CG, or matrix vectorization.
"""

from __future__ import annotations

import warnings

import numpy as np
from joblib import Parallel, delayed, effective_n_jobs
from sklearn.metrics import euclidean_distances
from sklearn.utils import check_random_state


def _validate_dissimilarities(dissimilarities):
    D = np.asarray(dissimilarities, dtype=float)
    if D.ndim != 2 or D.shape[0] == 0 or D.shape[0] != D.shape[1]:
        raise ValueError("dissimilarities must be a nonempty square matrix")
    if not np.all(np.isfinite(D)) or np.any(D < 0):
        raise ValueError("dissimilarities must be finite and nonnegative")
    if not np.allclose(np.diag(D), 0.0, rtol=0, atol=1e-12):
        raise ValueError("dissimilarities must have a zero diagonal")
    D = D.copy()
    np.fill_diagonal(D, 0.0)
    return D


def _validate_weights(weight, n_samples):
    if weight is None:
        W = np.ones((n_samples, n_samples), dtype=float)
    else:
        W = np.asarray(weight, dtype=float)
        if W.shape != (n_samples, n_samples):
            raise ValueError("weight must have shape (n_samples, n_samples)")
        if not np.all(np.isfinite(W)) or np.any(W < 0):
            raise ValueError("weight must be finite and nonnegative")
        if not np.allclose(W, W.T, rtol=1e-12, atol=1e-14):
            raise ValueError("weight must be symmetric for the corrected update")
        W = 0.5 * (W + W.T)
    W = W.copy()
    # Several original demos pass np.ones_like(D), including its diagonal.
    # Self-pairs have zero stress and must have zero Laplacian contribution.
    np.fill_diagonal(W, 0.0)
    return W


def _validate_options(randers_w_alpha, metric, normalized_stress,
                      n_components, max_iter, eps):
    if not np.isscalar(randers_w_alpha) or not np.isfinite(randers_w_alpha):
        raise ValueError("randers_w_alpha must be a finite scalar")
    if not 0 <= randers_w_alpha < 1:
        raise ValueError("randers_w_alpha must satisfy 0 <= alpha < 1")
    if not metric:
        raise ValueError("Non-metric Finsler MDS is not implemented")
    if isinstance(normalized_stress, str):
        if normalized_stress not in ("warn", "auto"):
            raise ValueError("unknown normalized_stress option")
        normalized_stress = False  # Historic values for metric MDS.
    if not isinstance(normalized_stress, (bool, np.bool_)) or normalized_stress:
        raise ValueError("normalized_stress=True is unsupported for metric Finsler MDS")
    if not isinstance(n_components, (int, np.integer)) or n_components < 1:
        raise ValueError("n_components must be a positive integer")
    if not isinstance(max_iter, (int, np.integer)) or max_iter < 1:
        raise ValueError("max_iter must be a positive integer")
    if not np.isscalar(eps) or not np.isfinite(eps) or eps < 0:
        raise ValueError("eps must be finite and nonnegative")


def _warn_obsolete_options(pseudo_inv_solver, project_on_V):
    if pseudo_inv_solver is not None:
        warnings.warn(
            "pseudo_inv_solver is obsolete and ignored: the corrected method "
            "needs neither GMRES nor CG; only nonuniform weights use a cached "
            "Laplacian pseudoinverse.",
            DeprecationWarning, stacklevel=3,
        )
    if project_on_V is not None:
        warnings.warn(
            "project_on_V is obsolete and ignored: the corrected update "
            "already yields the centered pseudoinverse solution.",
            DeprecationWarning, stacklevel=3,
        )


def _prepare_smacof(D, W, alpha, n_components):
    """Cache quantities common to all iterations and random restarts.

    Private implementation detail: ``_smacof_single`` can also prepare these
    quantities itself when called directly, just like scikit-learn's helper.
    """
    n_samples = len(D)
    weighted_symmetric_D = W * (D + D.T) / 2
    C = np.zeros((n_samples, n_components), dtype=float)
    C[:, -1] = alpha * (W * (D - D.T)).sum(axis=1)

    # Handle n=1 and the entirely zero-weight graph without a pseudoinverse.
    uniform_weight = None
    V_pinv = None
    if np.any(W):
        off_diagonal = W[~np.eye(n_samples, dtype=bool)]
        if np.all(off_diagonal == off_diagonal[0]):
            uniform_weight = float(off_diagonal[0])
        else:
            V = np.diag(W.sum(axis=1)) - W
            V_pinv = np.linalg.pinv(V, hermitian=True)

    return D, W, weighted_symmetric_D, C, uniform_weight, V_pinv


def _directed_stress(X, distances, D, W, alpha):
    """Sum over *both* orientations (i,j) with the Randers distance."""
    drift = alpha * X[:, -1]
    finsler_distances = distances + drift[None, :] - drift[:, None]
    return float(np.sum(W * (finsler_distances - D)**2))


def _smacof_single(
    dissimilarities,
    randers_w_alpha=0.0,
    metric=True,
    n_components=2,
    init=None,
    max_iter=300,
    verbose=0,
    eps=1e-3,
    random_state=None,
    normalized_stress=False,
    weight=None,
    pseudo_inv_solver=None,
    project_on_V=None,
    check_monotony=True,
    *,
    _prepared=None,
):
    """Run a *single* corrected Finsler SMACOF initialization.

    Returns (embedding, raw directed stress, number of completed updates).
    ``_prepared`` is private, supplied by ``smacof`` to reuse V^+ across
    random starts. Direct calls to this function require no such argument.
    """
    _warn_obsolete_options(pseudo_inv_solver, project_on_V)
    if init is not None:
        init = np.asarray(init, dtype=float)
        if init.ndim != 2:
            raise ValueError("init must be a two-dimensional array")
        n_components = init.shape[1]  # Same dimension override as sklearn.
    _validate_options(randers_w_alpha, metric, normalized_stress,
                      n_components, max_iter, eps)

    if _prepared is None:
        D = _validate_dissimilarities(dissimilarities)
        W = _validate_weights(weight, len(D))
        prepared = _prepare_smacof(D, W, randers_w_alpha, n_components)
    else:
        prepared = _prepared
    D, W, weighted_symmetric_D, C, uniform_weight, V_pinv = prepared
    n_samples = len(D)

    random_state = check_random_state(random_state)
    if init is None:
        X = random_state.uniform(size=(n_samples, n_components))
    else:
        if init.shape != (n_samples, n_components) or not np.all(np.isfinite(init)):
            raise ValueError("init must have shape (n_samples, n_components) and be finite")
        X = init.copy()  # Do not modify a caller-owned initial configuration.

    # As in sklearn, the single-run helper owns the entire iteration loop.
    distances = euclidean_distances(X)
    stress = _directed_stress(X, distances, D, W, randers_w_alpha)
    n_iter = 0
    for it in range(max_iter):
        # The B matrix uses symmetric data D_S and *Euclidean* distances,
        # not asymmetric Finsler distances. Exact coincidence implies B_ij=0.
        B = np.zeros_like(D)
        np.divide(-weighted_symmetric_D, distances, out=B,
                  where=distances > 0)
        np.fill_diagonal(B, 0.0)
        B[np.arange(n_samples), np.arange(n_samples)] = -B.sum(axis=1)

        rhs = B @ X - 0.5 * C
        if uniform_weight is not None:
            X_new = rhs / (n_samples * uniform_weight)
        elif V_pinv is not None:
            X_new = V_pinv @ rhs
        else:
            # No constrained pairs: the componentwise-centered solution is 0.
            X_new = np.zeros_like(X)

        # (I + omega omega.T)^(-1): omega=(0,...,0,alpha), so only the
        # last embedding coordinate is rescaled.
        X_new[:, -1] /= 1 + randers_w_alpha**2

        distances_new = euclidean_distances(X_new)
        stress_new = _directed_stress(
            X_new, distances_new, D, W, randers_w_alpha
        )
        if not np.isfinite(stress_new):
            raise FloatingPointError("non-finite stress in Finsler SMACOF")

        if check_monotony and stress_new > stress + 1e-10 * max(1.0, stress):
            warnings.warn(
                "Finsler stress increased beyond numerical tolerance; "
                "retaining the previous iterate.", RuntimeWarning, stacklevel=2
            )
            break

        improvement = stress - stress_new
        X, distances, stress = X_new, distances_new, stress_new
        n_iter = it + 1
        if verbose >= 2:
            print(f"Finsler SMACOF iteration {n_iter}: stress={stress:.12g}")

        # Legacy check_monotony=False explicitly requested fixed iterations.
        # Keep that behavior, rather than stopping early on eps.
        if check_monotony and improvement <= eps * max(
            stress + improvement, 1e-12
        ):
            if verbose:
                print(f"Finsler SMACOF converged at iteration {n_iter}")
            break

    return X, stress, n_iter


def smacof(
    dissimilarities,
    *,
    randers_w_alpha=0.0,
    metric=True,
    n_components=2,
    init=None,
    n_init=8,
    n_jobs=None,
    max_iter=300,
    verbose=0,
    eps=1e-3,
    random_state=None,
    return_n_iter=False,
    normalized_stress="warn",
    weight=None,
    pseudo_inv_solver=None,
    project_on_V=None,
    check_monotony=True,
):
    """Minimize canonical-Randers stress with corrected metric SMACOF.

    Matches the original FinslerMDS ``smacof`` call signature: the default
    return is **two** values ``(embedding, stress)``. Setting
    ``return_n_iter=True`` adds the third value ``n_iter``. Supplied ``init``
    overrides ``n_components`` and causes a single optimization run.

    ``weight`` is a symmetric nonnegative NxN pair-weight matrix (diagonal
    ignored); ``None`` means uniform unit weights. ``metric=False`` and
    ``normalized_stress=True`` are unsupported, as in the original metric
    Finsler implementation. Deprecated solver options are accepted but ignored.

    For ``check_monotony=True``, iteration stops upon relative stress
    convergence or an unexpected numerical stress increase. When False, it
    executes exactly ``max_iter`` iterations (legacy fixed-iteration behavior).

    See Appendix C of *Harnessing Data Asymmetry in Manifold Learning*
    (NeurIPS 2026) for the corrected update and its monotonicity proof.
    """
    _warn_obsolete_options(pseudo_inv_solver, project_on_V)
    if not isinstance(n_init, (int, np.integer)) or n_init < 1:
        raise ValueError("n_init must be a positive integer")
    if init is not None:
        init = np.asarray(init, dtype=float)
        if init.ndim != 2:
            raise ValueError("init must be a two-dimensional array")
        n_components = init.shape[1]
        if n_init != 1:
            warnings.warn(
                "Explicit initial positions passed: performing only one "
                f"initialization instead of {n_init}.", UserWarning,
                stacklevel=2,
            )
        n_init = 1

    _validate_options(randers_w_alpha, metric, normalized_stress,
                      n_components, max_iter, eps)
    D = _validate_dissimilarities(dissimilarities)
    W = _validate_weights(weight, len(D))
    prepared = _prepare_smacof(D, W, randers_w_alpha, n_components)
    random_state = check_random_state(random_state)

    # Same two-level structure as sklearn: smacof selects the best of
    # n_init calls to _smacof_single, which owns the iteration itself.
    best_pos, best_stress, best_iter = None, None, None
    if effective_n_jobs(n_jobs) == 1:
        for _ in range(n_init):
            pos, stress, n_iter_ = _smacof_single(
                D, randers_w_alpha=randers_w_alpha, metric=metric,
                n_components=n_components, init=init, max_iter=max_iter,
                verbose=verbose, eps=eps, random_state=random_state,
                normalized_stress=normalized_stress, weight=W,
                check_monotony=check_monotony, _prepared=prepared,
            )
            if best_stress is None or stress < best_stress:
                best_pos, best_stress, best_iter = pos.copy(), stress, n_iter_
    else:
        seeds = random_state.randint(np.iinfo(np.int32).max, size=n_init)
        results = Parallel(n_jobs=n_jobs, verbose=max(verbose - 1, 0))(
            delayed(_smacof_single)(
                D, randers_w_alpha=randers_w_alpha, metric=metric,
                n_components=n_components, init=init, max_iter=max_iter,
                verbose=verbose, eps=eps, random_state=seed,
                normalized_stress=normalized_stress, weight=W,
                check_monotony=check_monotony, _prepared=prepared,
            )
            for seed in seeds
        )
        best_pos, best_stress, best_iter = min(results, key=lambda r: r[1])

    if return_n_iter:
        return best_pos, best_stress, best_iter
    return best_pos, best_stress
