
import numpy as np
import utils
import matplotlib
import matplotlib.pyplot as plt
import _mds_finsler
import os


def main_swiss_roll_full():

    # Hyperparameters
    n = 3000                            # 3000
    noise_level = 0.                    # 0.
    randers_field_dir = 'length'        # 'length'
    randers_w_alpha_manifold_list = [0., 0.1, 0.3, 0.5, 0.6]  # [0., 0.1, 0.3, 0.5, 0.6]  # On the input manifold
    # randers_w_alpha_manifold = 0.5    # 0.5 # on the input manifold
    randers_w_alpha_embedding = 0.5     # 0.5 # in the flat randers embedding
    init_strat = 'isomap'               # 'isomap' | 'isomap', 'rand'
    k = 10                              # 10
    proj_dim = 3                        # 3
    dir_res = 'res/'                 # '../res/'
    folder_res_raw = 'raw/'             # 'raw/'

    seed = 1                            # 1

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

    fig.savefig(os.path.join(dir_res, 'swiss_roll_full.pdf'))

    knn = utils.nearest_neighbors(X, k)  # For visualisation purposes only

    ################### Euclidean embeddings (Isomap/SMACOF) #################

    print('Isomap')

    isomap = utils.IsomapWithPreds(n_components=proj_dim, n_neighbors=k)
    proj = isomap.fit_transform(X)
    dists, preds = isomap.dist_matrix_, isomap.preds_

    fig_isomap, ax_isomap = utils.plot_proj_points(proj, X=X, knn=knn, X_noiseless=X_noiseless, shape_type='swiss_roll',
                                                   edge_alpha=1.)

    fig_isomap.savefig(os.path.join(dir_res, 'swiss_roll_full_isomap.pdf'))



    ################### Randers field generation #################

    # Tangent field along the length of the swiss roll:
    # We calculate the tangent vectors at different points
    if randers_field_dir == 'length':
        tangent_x = np.cos(x[:, 0]) - x[:, 0] * np.sin(x[:, 0])
        tangent_z = np.sin(x[:, 0]) + x[:, 0] * np.cos(x[:, 0])
        tangent_y = np.zeros(n)
    else:
        raise ValueError('Unknown Randers field direction')

    fig_smacof_f_unif_all = plt.figure(figsize=(10, 10))
    ax_smacof_f_unif_all = fig_smacof_f_unif_all.add_subplot(111, projection='3d')

    #
    for randers_w_alpha_manifold in randers_w_alpha_manifold_list:
        print('Randers field manifold weight: ', randers_w_alpha_manifold)

        param_str = (str(seed) + '_' + str(n) + '_' + str(proj_dim) + '_' + str(k) + '_' + str(
            noise_level) + '_' + randers_field_dir +
                     '_' + str(randers_w_alpha_manifold) + '_' + str(randers_w_alpha_embedding)) + '_' + init_strat

        randers_field = np.stack([tangent_x, tangent_y, tangent_z], axis=1)
        randers_field = randers_field / np.linalg.norm(randers_field, axis=1)[:, None]
        randers_field = randers_w_alpha_manifold * randers_field

        # Plotting the Swiss roll with Randers field
        fig_randers, ax_randers = utils.plot_points(X, X_noiseless=X_noiseless, shape_type='swiss_roll',
                                                    quiver_field=randers_field,
                                                    step_quiver=1)

        fig_randers.savefig(os.path.join(dir_res, 'swiss_roll_full_randers_' + param_str + '.pdf'))

        # ################### Finsler embeddings #################

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

        print('Finsler SMACOF')
        np.random.seed(seed+2)
        if init_strat == 'isomap':
            init = proj
        elif init_strat == 'rand':
            init = np.random.rand(n, proj_dim)
        else:
            raise ValueError('Unknown initialisation strategy')
        unif_weights = np.ones_like(dists_f)
        if not os.path.exists(os.path.join(dir_res_raw, 'swiss_roll_full_proj_smacof_f_unif_' + param_str + '.npy')):
            proj_smacof_f_unif, _ = _mds_finsler.smacof(
                dists_f, randers_w_alpha=randers_w_alpha_embedding, metric=True,
                init=init, n_components=proj_dim, n_init=1, n_jobs=1,
                weight=unif_weights*1., max_iter=1000,
                pseudo_inv_solver="gmres", project_on_V=True,
            )
            np.save(os.path.join(dir_res_raw, 'swiss_roll_full_proj_smacof_f_unif_' + param_str + '.npy'), proj_smacof_f_unif)
        else:
            proj_smacof_f_unif = np.load(
                os.path.join(dir_res_raw, 'swiss_roll_full_proj_smacof_f_unif_' + param_str + '.npy'))

        fig_smacof_f_unif, ax_smacof_f_unif = utils.plot_proj_points(proj_smacof_f_unif, X=X, knn=knn, X_noiseless=X_noiseless, shape_type='swiss_roll', edge_alpha=1.)

        fig_smacof_f_unif.savefig(os.path.join(dir_res, 'swiss_roll_full_finsler_unif_' + param_str + '.pdf'))

        fig_smacof_f_unif_all, ax_smacof_f_unif_all = \
            utils.plot_proj_points(proj_smacof_f_unif, X=X, knn=knn, X_noiseless=X_noiseless, shape_type='swiss_roll', edge_alpha=1.,
                                   fig_ax=(fig_smacof_f_unif_all, ax_smacof_f_unif_all))

    fig_smacof_f_unif_all.savefig(os.path.join(dir_res, 'swiss_roll_full_finsler_unif_all.pdf'))

    plt.show()


if __name__ == '__main__':

    # import matplotlib
    # matplotlib.use('TkAgg')

    main_swiss_roll_full()

    plt.show()