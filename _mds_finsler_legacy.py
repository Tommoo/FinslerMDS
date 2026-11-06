
import warnings
from numbers import Integral, Real

import numpy as np
from joblib import effective_n_jobs

from sklearn.base import BaseEstimator, _fit_context
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import euclidean_distances
from sklearn.utils import check_array, check_random_state, check_symmetric
from sklearn.utils._param_validation import Hidden, Interval, StrOptions, validate_params
from sklearn.utils.parallel import Parallel, delayed

from utils import canonical_randers_dissimilarity
import time
import scipy


def _smacof_single(
    dissimilarities,
    randers_w_alpha=0.,
    metric=True,
    n_components=2,
    init=None,
    max_iter=300,
    verbose=0,
    eps=1e-3,
    random_state=None,
    normalized_stress=False,
    weight=None,
    pseudo_inv_solver="gmres",
    project_on_V=False,
    check_monotony=True,
):
    """Computes Finsler multidimensional scaling using SMACOF algorithm.

    Parameters
    ----------
    dissimilarities : ndarray of shape (n_samples, n_samples)
        Pairwise dissimilarities between the points. No longer symmetric.

    randers_w_alpha : float, default=0.
        Weighting factor for the drift component of the Randers metric.
        If 0., the metric is the Euclidean metric.
        Must be in the range [0, 1).

    metric : bool, default=True
        Compute metric or nonmetric SMACOF algorithm.
        When ``False`` (i.e. non-metric MDS), dissimilarities with 0 are considered as
        missing values. Non-metric weighted MDS is not implemented.

    n_components : int, default=2
        Number of dimensions in which to immerse the dissimilarities. If an
        ``init`` array is provided, this option is overridden and the shape of
        ``init`` is used to determine the dimensionality of the embedding
        space.

    init : ndarray of shape (n_samples, n_components), default=None
        Starting configuration of the embedding to initialize the algorithm. By
        default, the algorithm is initialized with a randomly chosen array.

    max_iter : int, default=300
        Maximum number of iterations of the SMACOF algorithm for a single run.

    verbose : int, default=0
        Level of verbosity.

    eps : float, default=1e-3
        Relative tolerance with respect to stress at which to declare
        convergence. The value of `eps` should be tuned separately depending
        on whether or not `normalized_stress` is being used.

    random_state : int, RandomState instance or None, default=None
        Determines the random number generator used to initialize the centers.
        Pass an int for reproducible results across multiple function calls.
        See :term:`Glossary <random_state>`.

    normalized_stress : bool, default=False
        Whether use and return normed stress value (Stress-1) instead of raw
        stress calculated by default. Only supported in non-metric MDS. The
        caller must ensure that if `normalized_stress=True` then `metric=False`

    weight : ndarray of shape (n_samples, n_samples), default=None
        symmetric weighting matrix of similarities.
        In default, weight is set to None, suggesting all weights are 1.

    pseudo_inv_solver : str, default="gmres"
        Can be any of ["gmres", "cg", "pinv"]. Defines how to compute
        the pseudo-inverse.

    project_on_V : bool, default=True
        Whether to project the points on the V space or not.

    check_monotony: bool, default=True
        Whether to check the monotony of the stress value and break if not.
        The stress should decrease but due to approximations it can increase.
        Set check_monotony=False if the stress increases in the first iterations
        and instead run for a fixed amount of iterations.

    Returns
    -------
    X : ndarray of shape (n_samples, n_components)
        Coordinates of the points in a ``n_components``-space.

    stress : float
        The final value of the stress (sum of squared distance of the
        disparities and the distances for all constrained points).

    n_iter : int
        The number of iterations corresponding to the best stress.
    """

    assert 0 <= randers_w_alpha < 1

    if weight is None:
        weight = np.ones_like(dissimilarities)  # Quick fix to have code flexibility compared to dirty source code

    n_samples = dissimilarities.shape[0]
    random_state = check_random_state(random_state)

    if init is None:
        # Randomly choose initial configuration
        X = random_state.uniform(size=n_samples * n_components)
        X = X.reshape((n_samples, n_components))
    else:
        # overrides the parameter p
        n_components = init.shape[1]
        if n_samples != init.shape[0]:
            raise ValueError(
                "init matrix should be of shape (%d, %d)" % (n_samples, n_components)
            )
        X = init

    if weight is not None:
        V = -weight.copy()
        V[np.arange(len(V)), np.arange(len(V))] = 0
        V[np.arange(len(V)), np.arange(len(V))] += np.abs(V.sum(axis=1))
        # V_pinv = np.linalg.pinv(V)  # For Vanilla non-Finsler smacof

        diag_one_end = np.zeros((n_components, n_components))
        diag_one_end[-1, -1] = 1
        diag_sum_weights = np.diag(weight.sum(axis=1))
        A = (-randers_w_alpha * weight + randers_w_alpha * diag_sum_weights)

        if metric:
            disparities = dissimilarities
        else:
            raise ValueError("Non-metric MDS is not implemented yet.")
        mat_one_last_col = np.zeros((X.shape[0], X.shape[1]))
        mat_one_last_col[:, -1] = 1
        C = randers_w_alpha * ((weight * disparities - weight.T * disparities.T)) @ mat_one_last_col

    old_X = X.copy()
    old_stress = None
    ir = IsotonicRegression()
    print()
    for it in range(max_iter):
        # print('iteration', it)
        print('\r', it, end='')
        # Compute distance and monotonic regression
        if randers_w_alpha > 0:
            embedded_dissimilarity_func = canonical_randers_dissimilarity(randers_w_alpha)
        else:
            embedded_dissimilarity_func = euclidean_distances
        dis = embedded_dissimilarity_func(X)

        if metric:
            disparities = dissimilarities
        else:
            # dis_flat = dis.ravel()
            # # dissimilarities with 0 are considered as missing values
            # dis_flat_w = dis_flat[sim_flat != 0]
            #
            # # Compute the disparities using a monotonic regression
            # disparities_flat = ir.fit_transform(sim_flat_w, dis_flat_w)
            # disparities = dis_flat.copy()
            # disparities[sim_flat != 0] = disparities_flat
            # disparities = disparities.reshape((n_samples, n_samples))
            # disparities *= np.sqrt(
            #     (n_samples * (n_samples - 1) / 2) / (disparities**2).sum()
            # )
            raise ValueError("Non-metric MDS is not implemented yet.")

        # Compute stress
        stress = (weight.ravel() * (dis.ravel() - disparities.ravel()) ** 2).sum()  # / 2
        if normalized_stress:
            stress = np.sqrt(
                stress / ((weight.ravel() * disparities.ravel() ** 2).sum())  # / 2)
            )
        # Update X using the finsler-smacof algorithm update
        dis[dis == 0] = 1e-5
        ratio = disparities / dis
        if weight is None:
            B = -ratio
        else:
            B = -ratio * weight
        B[np.arange(len(B)), np.arange(len(B))] += -B.sum(axis=1)
        if weight is None:
            X = 1.0 / n_samples * np.dot(B, X)
            # TODO: deal with this case
        else:
            if randers_w_alpha == None:
                X = np.linalg.pinv(V) @ B @ X
            else:
                # Vanilla Finsler smacof: tends to struggle to compute the pseudo inverse
                # left_mat @ X + left_mat_2 @ X @ right_mat = total_right_mat

                left_mat = V
                left_mat_2 = A
                right_mat = diag_one_end
                total_right_mat = B @ X - C
                if project_on_V:
                    left_mat = V.T @ left_mat
                    left_mat_2 = V.T @ left_mat_2
                    total_right_mat = V.T @ total_right_mat
                I_kron_left_mat = scipy.sparse.csr_array(np.kron(np.eye(*right_mat.shape), left_mat))
                right_mat_T_kron_I = scipy.sparse.csr_array(np.kron(right_mat.T, left_mat_2))
                if pseudo_inv_solver == "pinv":
                    X_flat = scipy.linalg.pinv((I_kron_left_mat + right_mat_T_kron_I).todense()) @ (total_right_mat.flatten(order='F'))
                elif pseudo_inv_solver == "cg":  # similar to gmres
                    X_flat, _ = scipy.sparse.linalg.cg((I_kron_left_mat + right_mat_T_kron_I), total_right_mat.flatten(order='F'))
                elif pseudo_inv_solver == "gmres":
                    X_flat, _ = scipy.sparse.linalg.gmres((I_kron_left_mat + right_mat_T_kron_I), total_right_mat.flatten(order='F'))
                else:
                    raise ValueError("Unknown pseudo-inverse solver")
                X = X_flat.reshape((n_samples, n_components), order='F')

        if weight is None:
            weight = np.ones(disparities.shape)

        if not metric:
            dis = np.sqrt((X**2).sum(axis=1)).sum()
        else:
            dis = 1
        if verbose >= 2:
            print("it: %d, stress %s" % (it, stress))
        if old_stress is not None:
            print('', (old_stress - stress / dis), end='')
            # if (old_stress - stress / dis) < eps:
            if check_monotony:
                if stress > old_stress + 0.1:  # Instability occurred making the stress increase, the algorithm will then diverge
                    X = old_X
                    stress = old_stress
                    it -= 1
                    if verbose:
                        print("\rbreaking at iteration %d with stress %s due to stress increase" % (it, stress))
                    break
                if np.abs(1 - stress / old_stress) < eps:
                    if verbose:
                        print("\rbreaking at iteration %d with stress %s" % (it, stress))
                    break
        if not metric:
            old_stress = stress / dis
        else:
            old_stress = stress
        old_X = X.copy()
    print()

    return X, stress, it + 1


@validate_params(
    {
        "dissimilarities": ["array-like"],
        "randers_w_alpha": [Real],
        "n_components": [Interval(Integral, 1, None, closed="left")],
        "init": ["array-like", None],
        "n_init": [Interval(Integral, 1, None, closed="left")],
        "n_jobs": [Integral, None],
        "max_iter": [Interval(Integral, 1, None, closed="left")],
        "verbose": ["verbose"],
        "eps": [Interval(Real, 0, None, closed="left")],
        "random_state": ["random_state"],
        "return_n_iter": ["boolean"],
        "normalized_stress": [
            "boolean",
            StrOptions({"auto"}),
            Hidden(StrOptions({"warn"})),
        ],
    },
    prefer_skip_nested_validation=True,
)
def smacof(
    dissimilarities,
    *,
    randers_w_alpha=0.,
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
    pseudo_inv_solver="gmres",
    project_on_V=False,
    check_monotony=True,
):
    """Compute Finsler multidimensional scaling using the SMACOF algorithm.

    The SMACOF (Scaling by MAjorizing a COmplicated Function) algorithm is a
    multidimensional scaling algorithm which minimizes an objective function
    (the *stress*) using a majorization technique. Stress majorization, also
    known as the Guttman Transform, guarantees a monotone convergence of
    stress, and is more powerful than traditional techniques such as gradient
    descent.

    The SMACOF algorithm for metric MDS can be summarized by the following
    steps:

    1. Set an initial start configuration, randomly or not.
    2. Compute the stress
    3. Compute the Guttman Transform
    4. Iterate 2 and 3 until convergence.

    The nonmetric algorithm adds a monotonic regression step before computing
    the stress.

    Parameters
    ----------
    dissimilarities : array-like of shape (n_samples, n_samples)
        Pairwise dissimilarities between the points. Must be symmetric.

    randers_w_alpha : float, default=0.
        Weighting factor for the drift component of the Randers metric.
        If 0., the metric is the Euclidean metric.
        Must be in the range [0, 1).

    metric : bool, default=True
        Compute metric or nonmetric SMACOF algorithm.
        When ``False`` (i.e. non-metric MDS), dissimilarities with 0 are considered as
        missing values.

    n_components : int, default=2
        Number of dimensions in which to immerse the dissimilarities. If an
        ``init`` array is provided, this option is overridden and the shape of
        ``init`` is used to determine the dimensionality of the embedding
        space.

    init : array-like of shape (n_samples, n_components), default=None
        Starting configuration of the embedding to initialize the algorithm. By
        default, the algorithm is initialized with a randomly chosen array.

    n_init : int, default=8
        Number of times the SMACOF algorithm will be run with different
        initializations. The final results will be the best output of the runs,
        determined by the run with the smallest final stress. If ``init`` is
        provided, this option is overridden and a single run is performed.

    n_jobs : int, default=None
        The number of jobs to use for the computation. If multiple
        initializations are used (``n_init``), each run of the algorithm is
        computed in parallel.

        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
        for more details.

    max_iter : int, default=300
        Maximum number of iterations of the SMACOF algorithm for a single run.

    verbose : int, default=0
        Level of verbosity.

    eps : float, default=1e-3
        Relative tolerance with respect to stress at which to declare
        convergence. The value of `eps` should be tuned separately depending
        on whether or not `normalized_stress` is being used.

    random_state : int, RandomState instance or None, default=None
        Determines the random number generator used to initialize the centers.
        Pass an int for reproducible results across multiple function calls.
        See :term:`Glossary <random_state>`.

    return_n_iter : bool, default=False
        Whether or not to return the number of iterations.

    normalized_stress : bool or "auto" default=False
        Whether use and return normed stress value (Stress-1) instead of raw
        stress calculated by default. Only supported in non-metric MDS.

    weight : ndarray of shape (n_samples, n_samples), default=None
        symmetric weighting matrix of similarities.
        In default, weight is set to None, suggesting all weights are 1.

    pseudo_inv_solver : str, default="gmres"
        Can be any of ["gmres", "cg", "pinv"]. Defines how to compute
        the pseudo-inverse.

    project_on_V : bool, default=False
        Whether to project the points on the V space or not.

    check_monotony: bool, default=True
        Whether to check the monotony of the stress value and break if not.
        The stress should decrease but due to approximations it can increase.
        Set check_monotony=False if the stress increases in the first iterations
        and instead run for a fixed amount of iterations.

    Returns
    -------
    X : ndarray of shape (n_samples, n_components)
        Coordinates of the points in a ``n_components``-space.

    stress : float
        The final value of the stress (sum of squared distance of the
        disparities and the distances for all constrained points).
        If `normalized_stress=True`, and `metric=False` returns Stress-1.
        A value of 0 indicates "perfect" fit, 0.025 excellent, 0.05 good,
        0.1 fair, and 0.2 poor [1]_.

    n_iter : int
        The number of iterations corresponding to the best stress. Returned
        only if ``return_n_iter`` is set to ``True``.

    References
    ----------
    .. [1] "Nonmetric multidimensional scaling: a numerical method" Kruskal, J.
           Psychometrika, 29 (1964)

    .. [2] "Multidimensional scaling by optimizing goodness of fit to a nonmetric
           hypothesis" Kruskal, J. Psychometrika, 29, (1964)

    .. [3] "Modern Multidimensional Scaling - Theory and Applications" Borg, I.;
           Groenen P. Springer Series in Statistics (1997)
    """

    dissimilarities = check_array(dissimilarities)
    random_state = check_random_state(random_state)

    # TODO(1.4): Remove
    if normalized_stress == "warn":
        warnings.warn(
            (
                "The default value of `normalized_stress` will change to `'auto'` in"
                " version 1.4. To suppress this warning, manually set the value of"
                " `normalized_stress`."
            ),
            FutureWarning,
        )
        normalized_stress = False

    if normalized_stress == "auto":
        normalized_stress = not metric

    if normalized_stress and metric:
        raise ValueError(
            "Normalized stress is not supported for metric MDS. Either set"
            " `normalized_stress=False` or use `metric=False`."
        )
    if hasattr(init, "__array__"):
        init = np.asarray(init).copy()
        if not n_init == 1:
            warnings.warn(
                "Explicit initial positions passed: "
                "performing only one init of the MDS instead of %d" % n_init
            )
            n_init = 1

    best_pos, best_stress = None, None

    if effective_n_jobs(n_jobs) == 1:
        for it in range(n_init):
            pos, stress, n_iter_ = _smacof_single(dissimilarities,
                                                  randers_w_alpha=randers_w_alpha,
                                                  metric=metric, n_components=n_components, init=init,
                                                  max_iter=max_iter, verbose=verbose, eps=eps,
                                                  random_state=random_state, normalized_stress=normalized_stress,
                                                  weight=weight,
                                                  pseudo_inv_solver=pseudo_inv_solver, project_on_V=project_on_V,
                                                  check_monotony=check_monotony)
            if best_stress is None or stress < best_stress:
                best_stress = stress
                best_pos = pos.copy()
                best_iter = n_iter_
    else:
        seeds = random_state.randint(np.iinfo(np.int32).max, size=n_init)
        results = Parallel(n_jobs=n_jobs, verbose=max(verbose - 1, 0))(
            delayed(_smacof_single)(
                dissimilarities,
                randers_w_alpha=randers_w_alpha,
                metric=metric,
                n_components=n_components,
                init=init,
                max_iter=max_iter,
                verbose=verbose,
                eps=eps,
                random_state=seed,
                normalized_stress=normalized_stress,
                weight=weight,
                pseudo_inv_solver=pseudo_inv_solver,
                project_on_V=project_on_V,
                check_monotony=check_monotony,
            )
            for seed in seeds
        )
        positions, stress, n_iters = zip(*results)
        best = np.argmin(stress)
        best_stress = stress[best]
        best_pos = positions[best]
        best_iter = n_iters[best]

    if return_n_iter:
        return best_pos, best_stress, best_iter
    else:
        return best_pos, best_stress
