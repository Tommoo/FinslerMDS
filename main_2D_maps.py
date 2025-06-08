
import matplotlib.pyplot as plt
import numpy as np
import utils
import _mds_finsler
import scipy
import os
import xarray as xr

def main_river():

    # Hyperparameters
    n = 2000                        # 2000
    randers_w_alpha_manifold = 0.2  # 0.2 # on the input manifold
    randers_w_alpha_embedding = 0.5 # 0.5 # in the flat randers embedding
    k = 10                          # 10
    proj_dim = 3                    # 3
    river_length = 10               # 10
    river_width = 1                 # 1
    dir_res = 'res'                 # 'res'

    np.random.seed(0)

    if not os.path.exists(dir_res):
        os.makedirs(dir_res)

    X = np.random.rand(n, 2)
    X[:, 0] = X[:, 0] * river_length
    X[:, 1] = X[:, 1] * river_width
    # Create differential current field along the length. Strong along the middle and linear decreasing to the borders
    randers_field = np.zeros((n, 2))
    randers_field[:, 0] = 1 - np.abs(2 * X[:, 1] / river_width - 1)
    randers_field = randers_field / np.linalg.norm(randers_field, axis=1)[:, None].max()  # Make sure norms are smaller than 1
    randers_field = randers_field * randers_w_alpha_manifold

    current_field = -randers_field  # Approximation

    # Rewrite the plot with fig_river, ax_river
    fig_river, ax_river = plt.subplots(figsize=(10, 1))
    c = plt.cm.jet(X[:, 0] / (X[:, 0].max() - X[:, 0].min()))
    ax_river.scatter(*(X[:,0], X[:,1]), c=c, s=200, lw=0, alpha=1)
    step = 1
    ax_river.quiver(X[:, 0][::step], X[:, 1][::step], randers_field[:, 0][::step], randers_field[:, 1][::step], color='black')
    # ax_river.set_title('River')
    # ax_river.set_xlabel('Length')
    # ax_river.set_ylabel('Width')
    ax_river.axis('equal')
    ax_river.set_axis_off()

    fig_river.savefig(os.path.join(dir_res, 'river_map_randers_field.pdf'))

    fig_river_current, ax_river_current = plt.subplots(figsize=(10, 1))
    c = plt.cm.jet(X[:, 0] / (X[:, 0].max() - X[:, 0].min()))
    ax_river_current.scatter(*(X[:,0], X[:,1]), c=c, s=200, lw=0, alpha=1)
    step = 1
    ax_river_current.quiver(X[:, 0][::step], X[:, 1][::step], current_field[:, 0][::step], current_field[:, 1][::step], color='black')
    ax_river_current.axis('equal')
    ax_river_current.set_axis_off()

    fig_river_current.savefig(os.path.join(dir_res, 'river_map_current_field.pdf'))

    knn = utils.nearest_neighbors(X, k)  # For visualisation purposes only

    # Compute Randers geodesic distances

    dists_f, preds_f = utils.compute_dist_matrix(
        X,
        n_neighbors=k,
        radius=None,
        path_method="auto",
        neighbors_algorithm="auto",
        n_jobs=None,
        metric="minkowski",
        p=2,
        metric_params=None,
        randers_field=randers_field
    )

    fig_dists, ax_dists = plt.subplots()
    im = ax_dists.imshow(dists_f)

    isomap = utils.IsomapWithPreds(n_components=proj_dim, n_neighbors=k)
    proj = isomap.fit_transform(X)

    print('Finsler SMACOF')
    # init = np.random.rand(n, proj_dim)
    init = proj
    unif_weights = np.ones_like(dists_f)
    proj_smacof_f_unif, _ = _mds_finsler.smacof(
        dists_f, randers_w_alpha=randers_w_alpha_embedding, metric=True,
        init=init, n_components=proj_dim, n_init=1, n_jobs=1,
        weight=unif_weights*1., max_iter=1000,
        pseudo_inv_solver="gmres", project_on_V=True,
        check_monotony=True,
    )

    fig_smacof_f_unif, ax_smacof_f_unif = \
        utils.plot_proj_points(proj_smacof_f_unif, X=X, knn=knn, X_noiseless=X, shape_type='river', edge_alpha=1.)

    fig_smacof_f_unif.savefig(os.path.join(dir_res, 'river_smacof_f_unif.pdf'))

    plt.show()


def main_sea():
    # Hyperparameters
    n = 2000                        # 2000
    randers_w_alpha_manifold = 0.5  # 0.5 # on the input manifold
    randers_w_alpha_embedding = 0.5 # 0.5 # in the flat randers embedding
    k = 10                          # 10
    proj_dim = 3                    # 3
    sea_length = 10                 # 10
    sea_width = 10                  # 10
    frequency = 2                   # 2
    shape_type = 'sea'              # 'sea'
    dir_res = 'res'                 # 'res'

    np.random.seed(0)

    if not os.path.exists(dir_res):
        os.makedirs(dir_res)

    X = np.random.rand(n, 2)
    X[:, 0] = X[:, 0] * sea_length
    X[:, 1] = X[:, 1] * sea_width
    # Create smooth structured current field
    randers_field = np.zeros((n, 2))
    randers_field[:, 0] = np.sin(frequency*X[:, 0]) + np.cos(frequency*X[:, 1])  # X component of the vector field (flow)
    randers_field[:, 1] = np.cos(frequency*X[:, 0]) - np.sin(frequency*X[:, 1])  # Y component of the vector field (flow)
    randers_field = randers_field / np.linalg.norm(randers_field, axis=1)[:, None].max()  # Make sure norms are smaller than 1
    randers_field = randers_field * randers_w_alpha_manifold

    current_field = -randers_field  # Approximation

    # Plot map
    fig, ax = plt.subplots()
    c = plt.cm.jet(X[:, 0] / (X[:, 0].max() - X[:, 0].min()))
    ax.scatter(*(X[:,0], X[:,1]), c=c, s=200, lw=0, alpha=1)
    step = 1
    ax.quiver(X[:, 0][::step], X[:, 1][::step], randers_field[:, 0][::step], randers_field[:, 1][::step], color='black')
    # ax.set_title('Sea')
    # ax.set_xlabel('Length')
    # ax.set_ylabel('Width')
    # Include colorbar
    ax.axis('equal')
    ax.set_axis_off()

    fig.savefig(os.path.join(dir_res, f'{shape_type}_map_randers_field.pdf'))

    fig_current, ax_current = plt.subplots()
    c = plt.cm.jet(X[:, 0] / (X[:, 0].max() - X[:, 0].min()))
    ax_current.scatter(*(X[:,0], X[:,1]), c=c, s=200, lw=0, alpha=1)
    step = 1
    ax_current.quiver(X[:, 0][::step], X[:, 1][::step], current_field[:, 0][::step], current_field[:, 1][::step], color='black')
    ax_current.axis('equal')
    ax_current.set_axis_off()

    fig_current.savefig(os.path.join(dir_res, f'{shape_type}_map_current_field.pdf'))


    knn = utils.nearest_neighbors(X, k)  # For visualisation purposes only

    # Compute Randers geodesic distances

    dists_f, preds_f = utils.compute_dist_matrix(
        X,
        n_neighbors=k,
        radius=None,
        path_method="auto",
        neighbors_algorithm="auto",
        n_jobs=None,
        metric="minkowski",
        p=2,
        metric_params=None,
        randers_field=randers_field
    )

    fig_dists, ax_dists = plt.subplots()
    im = ax_dists.imshow(dists_f)

    isomap = utils.IsomapWithPreds(n_components=proj_dim, n_neighbors=k)
    proj = isomap.fit_transform(X)

    print('Finsler SMACOF')
    # init = np.random.rand(n, proj_dim)
    init = proj
    unif_weights = np.ones_like(dists_f)
    proj_smacof_f_unif, _ = _mds_finsler.smacof(
        dists_f, randers_w_alpha=randers_w_alpha_embedding, metric=True,
        init=init, n_components=proj_dim, n_init=1, n_jobs=1,
        weight=unif_weights*1., max_iter=1000,
        pseudo_inv_solver="gmres", project_on_V=True,
        check_monotony=False,
    )

    extrema_proj_smacof_f_unif = utils.get_extrema(proj_smacof_f_unif) if proj_smacof_f_unif.shape[1] == 3 else None

    fig_smacof_f_unif, ax_smacof_f_unif = \
        utils.plot_proj_points(proj_smacof_f_unif, X=X, knn=knn, X_noiseless=X, shape_type=shape_type, edge_alpha=1.,
                               extrema=extrema_proj_smacof_f_unif)

    if proj_smacof_f_unif.shape[1] == 3:
        ax_smacof_f_unif.set_axis_on()

    fig_smacof_f_unif.savefig(os.path.join(dir_res, f'{shape_type}_smacof_f_unif.pdf'))

    if extrema_proj_smacof_f_unif is not None:
        # Plot X extrema on the original map
        ax.scatter(*(X[extrema_proj_smacof_f_unif['maxima']['indices'], 0], X[extrema_proj_smacof_f_unif['maxima']['indices'], 1]), c='black', s=200, lw=0, alpha=1)
        ax.scatter(*(X[extrema_proj_smacof_f_unif['minima']['indices'], 0], X[extrema_proj_smacof_f_unif['minima']['indices'], 1]), c='gray', s=200, lw=0, alpha=1)

        fig.savefig(os.path.join(dir_res, f'{shape_type}_map_randers_field_extrema.pdf'))

        ax_current.scatter(*(X[extrema_proj_smacof_f_unif['maxima']['indices'], 0], X[extrema_proj_smacof_f_unif['maxima']['indices'], 1]), c='black', s=200, lw=0, alpha=1)
        ax_current.scatter(*(X[extrema_proj_smacof_f_unif['minima']['indices'], 0], X[extrema_proj_smacof_f_unif['minima']['indices'], 1]), c='gray', s=200, lw=0, alpha=1)

        fig_current.savefig(os.path.join(dir_res, f'{shape_type}_map_current_field_extrema.pdf'))

    plt.show()



if __name__ == '__main__':
    main_river()
    main_sea()

    plt.show()