import warnings

import numpy as np
import utils
import matplotlib
import matplotlib.pyplot as plt
import _mds
import _mds_finsler
import os


def main_swiss_roll_partial():

    # Hyperparameters
    n = 2000                            # 2000
    noise_level = 0.                    # 0.
    randers_field_dir = 'length'        # 'length'
    randers_w_alpha_manifold = 0.5      # 0.5 # on the input manifold
    randers_w_alpha_embedding = 0.5     # 0.5 # in the flat randers embedding
    init_strat = 'isomap'               # 'isomap' | 'isomap', 'rand'
    k = 15                              # 15
    proj_dim = 2                        # 2
    dir_res = 'res/'                    # '/res/'
    folder_res_raw = 'raw/'             # 'raw/'

    seed = 3                            # 3

    np.random.seed(seed)  # Will reseed for each application invoking random number generation

    dir_res_raw = os.path.join(dir_res, folder_res_raw)

    if not os.path.exists(dir_res):
        os.makedirs(dir_res)
    if not os.path.exists(dir_res_raw):
        os.makedirs(dir_res_raw)

    ################### Swiss roll generation #################

    x_ = np.random.rand(n, 2)
    x = np.zeros((n, 2))
    x[:, 0] = x_[:, 0] * 3 * np.pi + 1.5 * np.pi
    x[:, 1] = x_[:, 1] * 20

    X_noiseless = np.zeros((n, 3))
    X_noiseless[:, 0] = x[:, 0] * np.cos(x[:, 0])
    X_noiseless[:, 1] = x[:, 1]
    X_noiseless[:, 2] = x[:, 0] * np.sin(x[:, 0])

    noise = noise_level * np.random.randn(n, 3)

    X = X_noiseless + noise

    ################### Plotting the Swiss roll #################

    # Plotting the Swiss roll
    fig, ax = utils.plot_points(X, X_noiseless=X_noiseless, shape_type='swiss_roll')

    fig.savefig(os.path.join(dir_res, 'swiss_roll_partial_full.pdf'))

    knn = utils.nearest_neighbors(X, k)  # For visualisation purposes only

    ################### Euclidean embeddings (Isomap/SMACOF) #################

    print('Isomap')

    isomap = utils.IsomapWithPreds(n_components=proj_dim, n_neighbors=k)
    proj = isomap.fit_transform(X)
    dists, preds = isomap.dist_matrix_, isomap.preds_

    fig_isomap, ax_isomap = utils.plot_proj_points(proj, X=X, knn=knn, X_noiseless=X_noiseless, shape_type='swiss_roll',
                                                   edge_alpha=1.)
    ax_isomap.axis('off')

    fig_isomap.savefig(os.path.join(dir_res, 'swiss_roll_partial_isomap_full.pdf'))

    ################### Randers field generation #################

    # Tangent field along the length of the swiss roll:
    # We calculate the tangent vectors at different points
    if randers_field_dir == 'length':
        tangent_x = np.cos(x[:, 0]) - x[:, 0] * np.sin(x[:, 0])
        tangent_z = np.sin(x[:, 0]) + x[:, 0] * np.cos(x[:, 0])
        tangent_y = np.zeros(n)
    else:
        raise ValueError('Unknown Randers field direction')

    param_str = (str(seed) + '_' + str(n) + '_' + str(proj_dim) + '_' + str(k) + '_' + str(
        noise_level) + '_' + randers_field_dir +
                 '_' + str(randers_w_alpha_manifold) + '_' + str(randers_w_alpha_embedding)) + '_' + init_strat

    randers_field = np.stack([tangent_x, tangent_y, tangent_z], axis=1)
    randers_field = randers_field / np.linalg.norm(randers_field, axis=1)[:, None]
    randers_field = randers_w_alpha_manifold * randers_field

    # Plotting the Swiss roll with Randers field
    fig_randers, ax_randers = utils.plot_points(X, X_noiseless=X_noiseless, shape_type='swiss_roll',
                                                quiver_field=randers_field, step_quiver=1)

    fig_randers.savefig(os.path.join(dir_res, 'swiss_roll_partial_randers_full.pdf'))


    print('Finsler SMACOF on full')

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

    np.random.seed(seed + 1)
    if init_strat == 'isomap':
        init = proj.copy()
        init[:, 0] = -proj[:, 1]
        init[:, 1] = proj[:, 0]
    elif init_strat == 'rand':
        init = np.random.rand(X.shape[0], proj_dim)
    else:
        raise ValueError('Unknown initialisation strategy')
    unif_weights = np.ones_like(dists)
    if not os.path.exists(os.path.join(dir_res_raw, 'proj_smacof_f_unif_' + param_str + '.npy')):
        proj_smacof_f_unif, _ = _mds_finsler.smacof(
            dists_f, randers_w_alpha=randers_w_alpha_embedding, metric=True,
            init=init, n_components=proj_dim, n_init=1, n_jobs=1,
            weight=unif_weights * 1., max_iter=150,
            pseudo_inv_solver="gmres", project_on_V=True,
            check_monotony=False,
        )
        np.save(os.path.join(dir_res_raw, 'proj_smacof_f_unif_' + param_str + '.npy'), proj_smacof_f_unif)
    else:
        proj_smacof_f_unif = np.load(os.path.join(dir_res_raw, 'proj_smacof_f_unif_' + param_str + '.npy'))

    fig_smacof_f_unif, ax_smacof_f_unif = utils.plot_proj_points(proj_smacof_f_unif, X=X, knn=knn, X_noiseless=X_noiseless, shape_type='swiss_roll', edge_alpha=1.)
    # matplotlib.rcParams['text.usetex'] = True
    # utils.plot_randers_w_arrow(proj_smacof_f_unif, ax_smacof_f_unif, shape_type='swiss_roll')

    ax_smacof_f_unif.axis('off')

    fig_smacof_f_unif.savefig(os.path.join(dir_res, 'swiss_roll_partial_smacof_f_unif_full.pdf'))

    #



    ################### Hole Swiss roll #################

    # Find boundary points
    knn_full = utils.nearest_neighbors(X, 6)  # don't put too many neighbours to find boundaries

    # Hole
    hole_mask = np.logical_and(np.logical_and(x_[:, 1] > 0.4, x_[:, 1] < 0.6),
                               np.logical_and(x_[:, 0] > 0.1, x_[:, 0] < 0.9))
    hole = np.where(hole_mask)[0]
    # Find rows of knn_full having hole points
    border_hole = np.any(np.isin(knn_full, hole), axis=1)
    border_hole = border_hole[~hole_mask]

    x_hole_ = x_[~hole_mask, :]
    x_hole = np.zeros((x_hole_.shape[0], 2))
    x_hole[:, 0] = x_hole_[:, 0] * 3 * np.pi + 1.5 * np.pi
    x_hole[:, 1] = x_hole_[:, 1] * 20

    X_noiseless_hole = np.zeros((x_hole.shape[0], 3))
    X_noiseless_hole[:, 0] = x_hole[:, 0] * np.cos(x_hole[:, 0])
    X_noiseless_hole[:, 1] = x_hole[:, 1]
    X_noiseless_hole[:, 2] = x_hole[:, 0] * np.sin(x_hole[:, 0])

    X_hole = X_noiseless_hole + noise[~hole_mask, :]

    knn_hole = utils.nearest_neighbors(X_hole, k)  # For visualisation purposes only

    fig_hole, ax_hole = utils.plot_points(X_hole, X_noiseless=X_noiseless_hole, shape_type='swiss_roll')

    fig_hole.savefig(os.path.join(dir_res, 'swiss_roll_partial_hole.pdf'))

    fig_hole_randers, ax_hole_randers = utils.plot_points(X_hole, X_noiseless=X_noiseless_hole, shape_type='swiss_roll',
                                                          quiver_field=randers_field[~hole_mask, :], step_quiver=1)

    fig_hole_randers.savefig(os.path.join(dir_res, 'swiss_roll_partial_randers_hole.pdf'))

    print('Isomap hole')
    proj_hole = isomap.fit_transform(X_hole)
    dists_hole, preds_hole = isomap.dist_matrix_, isomap.preds_  # Do this before another isomap.fit_transform

    fig_isomap_hole, ax_isomap_hole = utils.plot_proj_points(proj_hole, X=X_hole, knn=knn_hole,
                                                             X_noiseless=X_noiseless_hole, shape_type='swiss_roll',
                                                             edge_alpha=1.)
    ax_isomap_hole.axis('off')

    fig_isomap_hole.savefig(os.path.join(dir_res, 'swiss_roll_partial_isomap_hole.pdf'))


    small_dists_threshold = 3  # 3

    # Symmetric distances on hole
    print('Vanilla Wormhole Smacof on hole')

    if not os.path.exists(os.path.join(dir_res_raw, 'mask_wormhole_' + param_str + '.npy')):
        mask_wormhole, _, _ = \
            utils.wormhole_mask(dists_hole, border_hole, X_hole,
                                randers_drift_upper_bound=0., small_dists_threshold=small_dists_threshold)
        np.save(os.path.join(dir_res_raw, 'mask_wormhole_' + param_str + '.npy'), mask_wormhole)
    else:
        mask_wormhole = np.load(os.path.join(dir_res_raw, 'mask_wormhole_' + param_str + '.npy'))

    np.random.seed(seed + 10)
    if init_strat == 'isomap':
        if proj_dim in [2, 3]:
            # init = proj_hole
            # Rotated isomap to be along the better direction
            init = proj_hole.copy()
            init[:, 0] = -proj_hole[:, 1]
            init[:, 1] = proj_hole[:, 0]
        else:
            init = proj_hole
    elif init_strat == 'rand':
        init = np.random.rand(X_hole.shape[0], proj_dim)
    else:
        raise ValueError('Unknown initialisation strategy')

    if not os.path.exists(os.path.join(dir_res_raw, 'proj_smacof_unif_hole_' + param_str + '.npy')):
        proj_smacof_unif_hole, _ = _mds.smacof(
            dists_hole, metric=True,
            init=init, n_components=proj_dim, n_init=1, n_jobs=1,
            weight=mask_wormhole * 1., max_iter=1000
        )
        np.save(os.path.join(dir_res_raw, 'proj_smacof_unif_hole_' + param_str + '.npy'), proj_smacof_unif_hole)
    else:
        proj_smacof_unif_hole = np.load(os.path.join(dir_res_raw, 'proj_smacof_unif_hole_' + param_str + '.npy'))

    fig_smacof_unif_hole, ax_smacof_unif_hole = utils.plot_proj_points(proj_smacof_unif_hole, X=X_hole, knn=knn_hole,
                                                                        X_noiseless=X_noiseless_hole, shape_type='swiss_roll',
                                                                        edge_alpha=1.)
    ax_smacof_unif_hole.axis('off')

    fig_smacof_unif_hole.savefig(os.path.join(dir_res, 'swiss_roll_partial_smacof_unif_hole.pdf'))

    # Asymmetric distances on hole
    randers_field_hole = randers_field[~hole_mask, :]


    print('Wormhole mask hole')
    dists_f_hole, preds_f_hole = utils.compute_dist_matrix(
        X_hole,
        n_neighbors=k,
        radius=None,
        path_method="auto",
        neighbors_algorithm="auto",
        n_jobs=None,
        metric="minkowski",
        p=2,
        metric_params=None,
        randers_field=randers_field_hole
    )


    fig_dists_hole, ax_dists_hole = plt.subplots()
    im = ax_dists_hole.imshow(dists_f_hole)

    print('Wormhole mask hole Finsler')
    if not os.path.exists(os.path.join(dir_res_raw, 'mask_wormhole_randers_hole_' + param_str + '.npy')):
        mask_wormhole_randers_hole, _, _ = \
            utils.wormhole_mask(dists_hole, border_hole, X_hole,
                                randers_drift_upper_bound=randers_w_alpha_manifold, small_dists_threshold=small_dists_threshold)
        np.save(os.path.join(dir_res_raw, 'mask_wormhole_randers_hole_' + param_str + '.npy'), mask_wormhole_randers_hole)
    else:
        mask_wormhole_randers_hole = np.load(os.path.join(dir_res_raw, 'mask_wormhole_randers_hole_' + param_str + '.npy'))

    np.random.seed(seed + 3)

    if init_strat == 'isomap':
        if proj_dim in [2, 3]:
            # init = proj_hole
            # Rotated isomap to be along the better direction
            init = proj_hole.copy()
            init[:, 0] = -proj_hole[:, 1]
            init[:, 1] = proj_hole[:, 0]
        else:
            init = proj_hole
    elif init_strat == 'rand':
        init = np.random.rand(X_hole.shape[0], proj_dim)
    else:
        raise ValueError('Unknown initialisation strategy')

    print('Vanilla Finsler Smacof embedding')
    np.random.seed(seed + 4)
    unif_weights = np.ones_like(dists_f_hole)
    if not os.path.exists(os.path.join(dir_res_raw, 'proj_smacof_f_unif_hole_' + param_str + '.npy')):
        proj_smacof_f_unif_hole, _ = _mds_finsler.smacof(
            dists_f_hole, randers_w_alpha=randers_w_alpha_embedding, metric=True,
            init=init, n_components=proj_dim, n_init=1, n_jobs=1,
            weight=unif_weights * 1., max_iter=150,
            pseudo_inv_solver="gmres", project_on_V=True,
            check_monotony=False,
        )
        np.save(os.path.join(dir_res_raw, 'proj_smacof_f_unif_hole_' + param_str + '.npy'), proj_smacof_f_unif_hole)
    else:
        proj_smacof_f_unif_hole = np.load(os.path.join(dir_res_raw, 'proj_smacof_f_unif_hole_' + param_str + '.npy'))

    fig_f_unif_hole, ax_f_unif_hole = utils.plot_proj_points(proj_smacof_f_unif_hole, X=X_hole, knn=knn_hole,
                                                             X_noiseless=X_noiseless_hole, shape_type='swiss_roll',
                                                             edge_alpha=1.)
    # matplotlib.rcParams['text.usetex'] = True
    # utils.plot_randers_w_arrow(proj_smacof_f_unif_hole, ax_f_unif_hole, shape_type='swiss_roll')
    ax_f_unif_hole.axis('off')

    fig_f_unif_hole.savefig(os.path.join(dir_res, 'swiss_roll_partial_finsler_unif_hole.pdf'))


    print('Wormhole Finsler Smacof embedding')
    # For our finsler implementation, we need symmetric weights
    mask_wormhole_randers_hole_sym = mask_wormhole_randers_hole & mask_wormhole_randers_hole.T
    warnings.warn(
        "Finsler smacof uses symmetric weights assumption, we symmetrised the mask by taking the intersection")

    if not os.path.exists(os.path.join(dir_res_raw, 'proj_smacof_f_wormhole_hole_' + param_str + '.npy')):
        proj_smacof_f_wormhole_hole, _ = _mds_finsler.smacof(
            dists_f_hole, randers_w_alpha=randers_w_alpha_embedding, metric=True,
            init=init, n_components=proj_dim, n_init=1, n_jobs=1,
            weight=mask_wormhole_randers_hole_sym * 1., max_iter=150,
            eps=1e-3,
            pseudo_inv_solver="gmres", project_on_V=True,
            check_monotony=False,
        )
        np.save(os.path.join(dir_res_raw, 'proj_smacof_f_wormhole_hole_' + param_str + '.npy'), proj_smacof_f_wormhole_hole)
    else:
        proj_smacof_f_wormhole_hole = np.load(os.path.join(dir_res_raw, 'proj_smacof_f_wormhole_hole_' + param_str + '.npy'))

    fig_f_wormhole_hole, ax_f_wormhole_hole = utils.plot_proj_points(proj_smacof_f_wormhole_hole, X=X_hole,
                                                                     knn=knn_hole, X_noiseless=X_noiseless_hole,
                                                                     shape_type='swiss_roll', edge_alpha=1.)
    # matplotlib.rcParams['text.usetex'] = True
    # utils.plot_randers_w_arrow(proj_smacof_f_wormhole_hole, ax_f_wormhole_hole, shape_type='swiss_roll')
    ax_f_wormhole_hole.axis('off')

    fig_f_wormhole_hole.savefig(os.path.join(dir_res, 'swiss_roll_partial_finsler_wormhole_hole.pdf'))

    plt.show()


if __name__ == '__main__':

    # import matplotlib
    # matplotlib.use('TkAgg')

    main_swiss_roll_partial()

    plt.show()