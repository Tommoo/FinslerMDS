
import warnings
import numpy as np
# import sklearn.manifold
from sklearn.neighbors import NearestNeighbors, radius_neighbors_graph
from sklearn.decomposition import KernelPCA
from scipy.sparse.csgraph import connected_components, shortest_path
from scipy.sparse import issparse
from sklearn.utils.graph import _fix_connected_components
from sklearn.metrics import pairwise_distances
from sklearn.manifold import Isomap
from sklearn.neighbors import kneighbors_graph
# from _mds import smacof
# import scipy
import scipy.sparse.linalg
import scipy.linalg
import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial import cKDTree

# function that outputs N x k matrix with k nearest neighbors for each observation in X
def nearest_neighbors(X, k):
    # we use k+1 here since Xi will have the shortest distance to itself
    knn_matrix = np.zeros((len(X), k))
    # compute pairwise distances
    dist_matrix = pairwise_distances(X)
    # for each row find indices of k nearest neighbors
    for i in range(len(X)):
        knn_matrix[i] = dist_matrix[i,:].argsort()[1:k+1]
    return knn_matrix


def set_axes_equal(ax: plt.Axes):
    """Set 3D plot axes to equal scale.

    Make axes of 3D plot have equal scale so that spheres appear as
    spheres and cubes as cubes.  Required since `ax.axis('equal')`
    and `ax.set_aspect('equal')` don't work on 3D.
    """
    limits = np.array([
        ax.get_xlim3d(),
        ax.get_ylim3d(),
        ax.get_zlim3d(),
    ])
    origin = np.mean(limits, axis=1)
    radius = 0.5 * np.max(np.abs(limits[:, 1] - limits[:, 0]))
    _set_axes_radius(ax, origin, radius)


def _set_axes_radius(ax, origin, radius):
    x, y, z = origin
    ax.set_xlim3d([x - radius, x + radius])
    ax.set_ylim3d([y - radius, y + radius])
    ax.set_zlim3d([z - radius, z + radius])



def plot_points(X, X_noiseless=None, shape_type=None, quiver_field=None, step_quiver=None):
    assert X.shape[1] == 3

    X_ctr = X - np.mean(X, axis=0)
    X_color = X_noiseless if X_noiseless is not None else X

    fig = plt.figure(figsize=(10,10))
    ax = fig.add_subplot(111, projection='3d')

    scale_dot_plt = 200
    if shape_type is None:
        ax.scatter(X_ctr[:, 0], X_ctr[:, 1], X_ctr[:, 2], s=scale_dot_plt, lw=0, alpha=1)
    elif shape_type in ['swiss_roll']:
        factor = 200
        ax.scatter(X_ctr[:, 0], X_ctr[:, 1], X_ctr[:, 2],
                   c=plt.cm.jet((X_color[:, 0] ** 2 + X_color[:, 2] ** 2) / factor),
                   s=scale_dot_plt, lw=0, alpha=1)
    else:
        raise ValueError('shape_type not implemented.')

    if quiver_field is not None:
        if step_quiver is None:
            step = 1  # 1, 50
        else:
            step = step_quiver
        ax.quiver(X_ctr[:, 0][::step], X_ctr[:, 1][::step], X_ctr[:, 2][::step],
                  quiver_field[:, 0][::step], quiver_field[:, 1][::step], quiver_field[:, 2][::step],
                  color='k', length=2, normalize=True)

        # if quiver_field.max() > 0:
        #     # Hack to plot only quiver field in a separate figure
        #     fig_quiver = plt.figure(figsize=(10, 10))
        #     ax_quiver = fig_quiver.add_subplot(111, projection='3d')
        #     # step_plot = 10 if X.shape[0] == 3000 else 2
        #     step_plot = 2 if X.shape[0] == 3000 else 2
        #     ax_quiver.scatter(X_ctr[:, 0], X_ctr[:, 1], X_ctr[:, 2],
        #                c=plt.cm.jet((X_color[:, 0] ** 2 + X_color[:, 2] ** 2) / factor),
        #                s=scale_dot_plt, lw=0, alpha=0)
        #     ax_quiver.quiver(X_ctr[:, 0][::step_plot], X_ctr[:, 1][::step_plot], X_ctr[:, 2][::step_plot],
        #                      quiver_field[:, 0][::step_plot], quiver_field[:, 1][::step_plot], quiver_field[:, 2][::step_plot],
        #                      color=plt.cm.jet((X_color[:, 0][::step_plot] ** 2 + X_color[:, 2][::step_plot] ** 2) / factor),
        #                      length=5, normalize=True)
        #     ax_quiver.set_box_aspect([1, 1, 1])  # IMPORTANT - this is the new, key line
        #     # ax.set_proj_type('ortho') # OPTIONAL - default is perspective (shown in image above)
        #     set_axes_equal(ax_quiver)  # IMPORTANT - this is also required
        #     ax_quiver.axis("off");
        #     if X.shape[0] == 3000:
        #         str_full_or_hole = 'full'
        #     elif X.shape[0] == 2000:
        #         str_full_or_hole = 'partial_full'
        #     else:
        #         str_full_or_hole = 'hole'
        #     fig_quiver.savefig('res/swiss_roll_'+str_full_or_hole+'_quiver_field_only.pdf', format='pdf')
        #     # raise ValueError('Stop')


    # Equal aspect ratio
    ax.set_box_aspect([1, 1, 1])
    set_axes_equal(ax)
    ax.axis("off");

    return fig, ax

def plot_proj_points(proj, X=None, knn=None, X_noiseless=None, shape_type=None, edge_alpha=1., extrema=None, fig_ax=None):
    assert (proj.shape[1] == 2 or proj.shape[1] == 3)

    X_color = X_noiseless if X_noiseless is not None else X

    # plot the mds projection
    if fig_ax is None:
        fig = plt.figure(figsize=(10, 10))
        if proj.shape[1] == 2:
            ax = fig.add_subplot(111)
        elif proj.shape[1] == 3:
            ax = fig.add_subplot(111, projection='3d')
    else:
        fig, ax = fig_ax

    proj_scatter = (proj[:, 0], proj[:, 1]) if proj.shape[1] == 2 else (proj[:, 0], proj[:, 1], proj[:, 2])

    if shape_type is None:
        ax.scatter(*proj_scatter, s=200, lw=0, alpha=1)
    elif shape_type in ['swiss_roll']:
        factor = 200
        c = plt.cm.jet((X_color[:, 0] ** 2 + X_color[:, 2] ** 2) / factor) if X_color is not None else 'blue'
        ax.scatter(*proj_scatter, c=c, s=200, lw=0, alpha=1)
    elif shape_type in ['river', 'sea']:
        c = plt.cm.jet(X_color[:, 0] / (X_color[:, 0].max() - X_color[:, 0].min())) if X_color is not None else 'blue'
        ax.scatter(*proj_scatter, c=c, s=100, lw=0, alpha=1, zorder=-1)
    else:
        raise ValueError('shape_type not implemented.')

    if extrema is not None and shape_type in ['sea'] and proj.shape[1] == 3:
        # if extrema is None:
        #     extrema = get_extrema(proj)

        maxima = extrema['maxima']['values']
        minima = extrema['minima']['values']

        # Plot local maxima and minima
        ax.scatter(maxima[:, 0], maxima[:, 1], maxima[:, 2], color='black', s=200, alpha=1, marker='o', zorder=1e6)
        ax.scatter(minima[:, 0], minima[:, 1], minima[:, 2], color='gray', s=200, alpha=1, marker='o', zorder=1e6)
        # ax.set_aspect('equal', adjustable='box')


        ###################
        # Hack to overlay extrema on the plot, problem is matplotlib not a real 3D plotter and cannot do mutual overlap

        s_points = 100

        # Plot 1: Scatter plot of the surface data
        fig_base = plt.figure()
        ax_base = fig_base.add_subplot(111, projection='3d')
        ax_base.scatter(*proj_scatter, c=c, s=s_points, lw=0, alpha=1, zorder=-1)
        ax_base.scatter(maxima[:, 0], maxima[:, 1], maxima[:, 2], color='black', marker='o', s=100, alpha=0)
        ax_base.scatter(minima[:, 0], minima[:, 1], minima[:, 2], color='gray', marker='o', s=100, alpha=0)
        # ax_base.set_axis_off()  # Remove axes for transparency

        # ax_base.set_aspect('equal', adjustable='box')
        ax_base.azim = -110
        ax_base.elev = 40

        # Save the base plot as a separate layer
        fig_base.savefig('res/'+shape_type+'_base_plot.pdf', format='pdf')

        # Plot 2: Overlay with only local extrema
        fig_extrema = plt.figure()
        ax_extrema = fig_extrema.add_subplot(111, projection='3d')
        ax_extrema.scatter(*proj_scatter, c=c, s=s_points, lw=0, alpha=0, zorder=-1)
        ax_extrema.scatter(maxima[:, 0], maxima[:, 1], maxima[:, 2], color='black', marker='o', s=100, alpha=1)
        ax_extrema.scatter(minima[:, 0], minima[:, 1], minima[:, 2], color='gray', marker='o', s=100, alpha=1)
        # Apply equal limits to make the aspect ratio appear equal
        # ax_extrema.set_aspect('equal', adjustable='box')
        ax_extrema.azim = -110
        ax_extrema.elev = 40
        ax_extrema.set_axis_off()  # Remove axes for transparency
        fig_extrema.patch.set_alpha(0)  # Make background transparent

        # Save the extrema plot with transparency
        fig_extrema.savefig('res/'+shape_type+'_extrema_overlay.pdf', format='pdf', transparent=True)

    if knn is not None:
        # plot lines connecting the same neighboring points from our original data
        for i in range(len(X)):
            neighbors = knn[i]
            for j in range(len(neighbors)):
                if proj.shape[1] == 2:
                    ax.plot(proj[[i, neighbors.astype('int')[j]], 0],
                             proj[[i, neighbors.astype('int')[j]], 1], color='black', alpha=edge_alpha, zorder=-2);
                elif proj.shape[1] == 3:
                    ax.plot(proj[[i, neighbors.astype('int')[j]], 0],
                             proj[[i, neighbors.astype('int')[j]], 1],
                             proj[[i, neighbors.astype('int')[j]], 2], color='black', alpha=edge_alpha, zorder=-2);


    ax.set_aspect('equal', adjustable='box')
    # plt.axis("off")
    return fig, ax


class IsomapWithPreds(Isomap):
    # Changes fit_transform so that the dist calc returns also the predecessors

    def __init__(
            self,
            *,
            n_neighbors=5,
            radius=None,
            n_components=2,
            eigen_solver="auto",
            tol=0,
            max_iter=None,
            path_method="auto",
            neighbors_algorithm="auto",
            n_jobs=None,
            metric="minkowski",
            p=2,
            metric_params=None,
    ):
        self.n_neighbors = n_neighbors
        self.radius = radius
        self.n_components = n_components
        self.eigen_solver = eigen_solver
        self.tol = tol
        self.max_iter = max_iter
        self.path_method = path_method
        self.neighbors_algorithm = neighbors_algorithm
        self.n_jobs = n_jobs
        self.metric = metric
        self.p = p
        self.metric_params = metric_params

    def _fit_transform(self, X):
        if self.n_neighbors is not None and self.radius is not None:
            raise ValueError(
                "Both n_neighbors and radius are provided. Use"
                f" Isomap(radius={self.radius}, n_neighbors=None) if intended to use"
                " radius-based neighbors"
            )

        self.nbrs_ = NearestNeighbors(
            n_neighbors=self.n_neighbors,
            radius=self.radius,
            algorithm=self.neighbors_algorithm,
            metric=self.metric,
            p=self.p,
            metric_params=self.metric_params,
            n_jobs=self.n_jobs,
        )
        self.nbrs_.fit(X)
        self.n_features_in_ = self.nbrs_.n_features_in_
        if hasattr(self.nbrs_, "feature_names_in_"):
            self.feature_names_in_ = self.nbrs_.feature_names_in_

        self.kernel_pca_ = KernelPCA(
            n_components=self.n_components,
            kernel="precomputed",
            eigen_solver=self.eigen_solver,
            tol=self.tol,
            max_iter=self.max_iter,
            n_jobs=self.n_jobs,
        ).set_output(transform="default")

        if self.n_neighbors is not None:
            nbg = kneighbors_graph(
                self.nbrs_,
                self.n_neighbors,
                metric=self.metric,
                p=self.p,
                metric_params=self.metric_params,
                mode="distance",
                n_jobs=self.n_jobs,
            )
        else:
            nbg = radius_neighbors_graph(
                self.nbrs_,
                radius=self.radius,
                metric=self.metric,
                p=self.p,
                metric_params=self.metric_params,
                mode="distance",
                n_jobs=self.n_jobs,
            )

        # Compute the number of connected components, and connect the different
        # components to be able to compute a shortest path between all pairs
        # of samples in the graph.
        # Similar fix to cluster._agglomerative._fix_connectivity.
        n_connected_components, labels = connected_components(nbg)
        if n_connected_components > 1:
            if self.metric == "precomputed" and issparse(X):
                raise RuntimeError(
                    "The number of connected components of the neighbors graph"
                    f" is {n_connected_components} > 1. The graph cannot be "
                    "completed with metric='precomputed', and Isomap cannot be"
                    "fitted. Increase the number of neighbors to avoid this "
                    "issue, or precompute the full distance matrix instead "
                    "of passing a sparse neighbors graph."
                )
            warnings.warn(
                (
                    "The number of connected components of the neighbors graph "
                    f"is {n_connected_components} > 1. Completing the graph to fit"
                    " Isomap might be slow. Increase the number of neighbors to "
                    "avoid this issue."
                ),
                stacklevel=2,
            )

            # use array validated by NearestNeighbors
            nbg = _fix_connected_components(
                X=self.nbrs_._fit_X,
                graph=nbg,
                n_connected_components=n_connected_components,
                component_labels=labels,
                mode="distance",
                metric=self.nbrs_.effective_metric_,
                **self.nbrs_.effective_metric_params_,
            )

        self.dist_matrix_, self.preds_ = shortest_path(nbg, method=self.path_method, directed=False, return_predecessors=True)

        if self.nbrs_._fit_X.dtype == np.float32:
            self.dist_matrix_ = self.dist_matrix_.astype(
                self.nbrs_._fit_X.dtype, copy=False
            )

        G = self.dist_matrix_ ** 2
        G *= -0.5

        self.embedding_ = self.kernel_pca_.fit_transform(G)
        self._n_features_out = self.embedding_.shape[1]


def compute_dist_matrix(
        X,
        n_neighbors=5,
        radius=None,
        path_method="auto",
        neighbors_algorithm="auto",
        n_jobs=None,
        metric="minkowski",
        p=2,
        metric_params=None,
        randers_field=None,
        nn_riemannian_precomputed=None,  # if we have a precomputed nearest neighbour graph without Randers field
):
    nbrs_ = None
    if nn_riemannian_precomputed is None:
        # Compute in the same way as in Isomap and change at the end the edge distances
        nbrs_ = NearestNeighbors(
            n_neighbors=n_neighbors,
            radius=radius,
            algorithm=neighbors_algorithm,
            metric=metric,
            p=p,
            metric_params=metric_params,
            n_jobs=n_jobs,
        )
        nbrs_.fit(X)
        n_features_in_ = nbrs_.n_features_in_
        if hasattr(nbrs_, "feature_names_in_"):
            feature_names_in_ = nbrs_.feature_names_in_

        if n_neighbors is not None:
            nbg = kneighbors_graph(
                nbrs_,
                n_neighbors,
                metric=metric,
                p=p,
                metric_params=metric_params,
                mode="distance",
                n_jobs=n_jobs,
            )
        else:
            nbg = radius_neighbors_graph(
                nbrs_,
                radius=radius,
                metric=metric,
                p=p,
                metric_params=metric_params,
                mode="distance",
                n_jobs=n_jobs,
            )
    else:
        assert type(nn_riemannian_precomputed) == scipy.sparse.csr.csr_matrix

        nbg = nn_riemannian_precomputed

    # Compute the number of connected components, and connect the different
    # components to be able to compute a shortest path between all pairs
    # of samples in the graph.
    # Similar fix to cluster._agglomerative._fix_connectivity.
    n_connected_components, labels = connected_components(nbg)
    if n_connected_components > 1:
        if metric == "precomputed" and issparse(X):
            raise RuntimeError(
                "The number of connected components of the neighbors graph"
                f" is {n_connected_components} > 1. The graph cannot be "
                "completed with metric='precomputed', and Isomap cannot be"
                "fitted. Increase the number of neighbors to avoid this "
                "issue, or precompute the full distance matrix instead "
                "of passing a sparse neighbors graph."
            )
        warnings.warn(
            (
                "The number of connected components of the neighbors graph "
                f"is {n_connected_components} > 1. Completing the graph to fit"
                " Isomap might be slow. Increase the number of neighbors to "
                "avoid this issue."
            ),
            stacklevel=2,
        )

        # use array validated by NearestNeighbors
        nbg = _fix_connected_components(
            X=nbrs_._fit_X,
            graph=nbg,
            n_connected_components=n_connected_components,
            component_labels=labels,
            mode="distance",
            metric=nbrs_.effective_metric_,
            **nbrs_.effective_metric_params_,
        )

    # Update the nbg graph with the Randers field
    # Modification formula is:
    # d(x, y) = d(x, y) + <randers_field, y - x>
    if randers_field is not None:
        # Get binary mask of the graph
        edges_mask = nbg.toarray() != 0
        # for i in range(len(X)):
        #     for j in range(len(X)):
        #         nbg[i, j] = nbg[i, j] + np.dot(randers_field[i], X[j] - X[i])
        # Vectorize the operation
        # for i in range(len(X)):
        #     nbg[i] = nbg[i] + np.dot(X - X[i], randers_field[i]) * edges_mask[i]
        randers_update_all = np.zeros((len(X), len(X)))
        nbg_old = nbg.todense().copy()
        for i in range(len(X)):
            randers_update = np.dot(X - X[i], randers_field[i]) * edges_mask[i]
            randers_update_all[i] = randers_update
            nbg[i, edges_mask[i]] = nbg[i, edges_mask[i]] + randers_update[edges_mask[i]]
        # make nbg sparse
        nbg = nbg.tocsr()

        directed = True
    else:
        directed = False

    dist_matrix_, preds_ = shortest_path(nbg, method=path_method, directed=directed, return_predecessors=True)

    if nbrs_ is not None:
        if nbrs_._fit_X.dtype == np.float32:
            dist_matrix_ = dist_matrix_.astype(
                nbrs_._fit_X.dtype, copy=False
            )

    return dist_matrix_, preds_


def canonical_randers_metric(alpha):
    def randers_metric(X, Y):
        randers_field = np.zeros(X.shape[1])
        randers_field[-1] = 1
        return np.linalg.norm(X - Y) + alpha * np.dot(randers_field, Y - X)
    return randers_metric

def canonical_randers_dissimilarity(alpha):
    def randers_dissimilarity(X):
        # Return distance between each entry of X according to the canonical Randers metric
        n = X.shape[0]
        randers_field = np.zeros(X.shape[1])
        randers_field[-1] = 1
        dissimilarity = np.zeros((n, n))
        # for i in range(n):
        #     for j in range(n):
        #         dissimilarity[i, j] = np.linalg.norm(X[i] - X[j]) + alpha * np.dot(randers_field, X[j] - X[i])
        # Vectorize the operation
        for i in range(n):
            dissimilarity[i] = np.linalg.norm(X - X[i], axis=1) + alpha * np.dot(X - X[i], randers_field)
        return dissimilarity
    return randers_dissimilarity

def get_extrema(X, radius=1):
    assert X.shape[1] == 3

    # Find local maxima and minima
    tree = cKDTree(X)
    maxima = {'indices': [], 'values': []}
    minima = {'indices': [], 'values': []}
    for i, point in enumerate(X):
        # Find neighbors within the specified radius
        neighbors_idx = tree.query_ball_point(point, radius)

        # Check if the current point is a local maximum and/or minimum among neighbors
        is_max = all(X[i, 2] >= X[j, 2] for j in neighbors_idx if i != j)  # Local maximum in z-dimension
        is_min = all(X[i, 2] <= X[j, 2] for j in neighbors_idx if i != j)  # Local minimum in z-dimension

        if is_max:
            maxima['values'].append(point)
            maxima['indices'].append(i)
        if is_min:
            minima['values'].append(point)
            minima['indices'].append(i)

    # Convert lists to numpy arrays for plotting
    maxima['values'] = np.array(maxima['values'])
    maxima['indices'] = np.array(maxima['indices'])
    minima['values'] = np.array(minima['values'])
    minima['indices'] = np.array(minima['indices'])

    extrema = {'minima': minima, 'maxima': maxima}

    return extrema


def wormhole_mask(dists, border, X, randers_drift_upper_bound=0., small_dists_threshold=1e-6):
    """
    :param dists:
        Pairwise distances between points. Should be symmetric if randers_field is 0.

    :param border:
        Boolean array indicating which points are on the boundary.

    :param X:
        Data points.

    :param randers_drift_upper_bound:
        Upper bound on the Randers drift component w. If 0, then we are in the Euclidean case as in the original wormhole paper NeurIPS 2024.
        If not 0, then we are in the Finsler case with a Randers drift component w.
        The constraint is ||w||_2 < rander_drift_upper_bound < 1 everywhere on the full manifold.

    :param small_dists_threshold:
        Small distances threshold. If the distance between two points is smaller than this threshold, we set the mask to 1.

    :return:
        mask_euclidean:
            Boolean array indicating wormhole guaranteed pairs.

        mask_criterion_sum_dists_boundary:
            Boolean array indicating wormhole guaranteed pairs, without the small_dists_threshold.

        mask_small_dists:
            Boolean array indicating the pairs within the small_dists_threshold.
    """

    # This is a bad quadratic implementation, can be made much faster by running another Djikstra with connections
    # between all boundary points

    import warnings
    warnings.warn(
        "Running a poor implementation of the wormhole mask. Currently quadratic but can be made much faster.")

    if randers_drift_upper_bound == 0:
        assert np.allclose(dists, dists.T)  # "Euclidean distances should be symmetric"

    # If rander_drift_upper_bound == 0, then we are in the Euclidean case as in the original wormhole paper NeurIPS 2024
    # If rander_drift_upper_bound != 0, then we are in the Finsler case with a Randers drift component w
    # The constraint is ||w||_2 < rander_drift_upper_bound < 1 everywhere on the full manifold

    #

    dists_boundary = dists[:, border]

    mask_criterion_sum_dists_boundary = np.ones(dists.shape, dtype=bool)

    print('')
    border_ids = np.nonzero(border)[0]
    for b1 in range(len(border_ids)):
        for b2 in range(len(border_ids)):
            print('\r', b1, b2, '[total:', len(border_ids), ']', end='')
            meshgrid_dists_boundary_b1_b2 = np.meshgrid(dists_boundary[:, b1], dists_boundary[:, b2], indexing='ij')
            sum_min_dists_boundary_b1_b2 = meshgrid_dists_boundary_b1_b2[0] + meshgrid_dists_boundary_b1_b2[1]
            euclidean_dist_b1_b2 = np.linalg.norm(X[border_ids[b1], :] - X[border_ids[b2], :])
            randers_bound_dist_b1_b2 = (1 - randers_drift_upper_bound) * euclidean_dist_b1_b2  # Worst case bound
            mask_criterion_sum_dists_boundary_b1_b2 = sum_min_dists_boundary_b1_b2 + randers_bound_dist_b1_b2 > dists
            mask_criterion_sum_dists_boundary = np.logical_and(mask_criterion_sum_dists_boundary,
                                                               mask_criterion_sum_dists_boundary_b1_b2)
    print('')

    mask_small_dists = dists < small_dists_threshold

    mask_euclidean = np.logical_or(mask_criterion_sum_dists_boundary, mask_small_dists)

    return mask_euclidean, mask_criterion_sum_dists_boundary, mask_small_dists


def plot_randers_w_arrow(data, ax, shape_type=None, location='top_left'):
    # Plot the randers w arrow (exaggerated size)

    assert data.shape[1] == 2, "Data should be 2D"

    # Determine dynamic limits based on data range
    x_min_proj, x_max_proj = data[:, 0].min(), data[:, 0].max()
    y_min_proj, y_max_proj = data[:, 1].min(), data[:, 1].max()

    if shape_type in [None, 'swiss_roll']:
        offset_x_arrow, offset_y_arrow = x_min_proj + 0.01 * (x_max_proj - x_min_proj), y_max_proj - 0.01 * (y_max_proj - y_min_proj)  # Offset position
        arrow_len = 0.2 * (y_max_proj - y_min_proj)
        fontsize = 50
        alpha_head = 0.05
    elif shape_type in ['binary_tree']:

        # plt.show(block=False)
        # plt.pause(0.1)

        arrow_len = 0.2 * (y_max_proj - y_min_proj)

        if location == 'top_left':
            eps_x = 0.2  # From left to right
            eps_y = -0.05  # From top to bottom
        elif location == 'top_right':
            eps_x = 1-0.2
            eps_y = -0.05
        elif location == 'bottom_left':
            eps_x = 0.2
            eps_y = -(1-0.23 - arrow_len / (y_max_proj - y_min_proj))
        elif location == 'bottom_right':
            eps_x = 1 - 0.01
            eps_y = -(1-0.01)
        offset_x_arrow, offset_y_arrow = (
            x_min_proj + eps_x * (x_max_proj - x_min_proj),
            y_max_proj + eps_y * (y_max_proj - y_min_proj))  # Offset position

        fontsize = 30
        alpha_head = 0.08


    # # Downwards arrow
    # ax.arrow(offset_x_arrow, offset_y_arrow, 0, -arrow_len,
    #                          head_width=0.1 * arrow_len, head_length=0.1 * arrow_len,
    #                          fc="black", ec="black", lw=4, length_includes_head=True)
    # ax.text(offset_x_arrow, offset_y_arrow - (0.2 + 0.05) * (y_max_proj - y_min_proj), r'$\omega$', ha='center', va='center',
    #                         fontsize=50)

    # Upwards arrow
    ax.arrow(offset_x_arrow, offset_y_arrow - arrow_len - alpha_head * (y_max_proj - y_min_proj), 0, arrow_len,
                             head_width=0.1 * arrow_len, head_length=0.1 * arrow_len,
                             fc="black", ec="black", lw=4, length_includes_head=True)
    ax.text(offset_x_arrow, offset_y_arrow, r'$\omega$', ha='center', va='center',
                            fontsize=fontsize)