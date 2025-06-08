import warnings

import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import KernelPCA
import scipy
import networkx as nx
import utils
import _mds_finsler
import matplotlib
import os
import umap


# Function to add edges with weights for a binary tree in NetworkX
def add_edges_binary_tree(G, node, depth, max_depth, weight_parent_to_child, weight_child_to_parent):
    if depth >= max_depth:
        return

    left_child = (node+1) * 2 - 1
    right_child = (node+1) * 2 + 1 - 1

    # Add node attribute: depth
    if depth == 0:
        G.add_node(node, depth=depth)
    # G.add_node(node, depth=depth)
    G.add_node(left_child, depth=depth + 1)
    G.add_node(right_child, depth=depth + 1)

    # Add edges with different weights for parent-child and child-parent
    if weight_parent_to_child is not None:
        G.add_edge(node, left_child, weight=weight_parent_to_child, edge_type='parent_to_child')
        G.add_edge(node, right_child, weight=weight_parent_to_child, edge_type='parent_to_child')
    if weight_child_to_parent is not None:
        G.add_edge(left_child, node, weight=weight_child_to_parent, edge_type='child_to_parent')
        G.add_edge(right_child, node, weight=weight_child_to_parent, edge_type='child_to_parent')

    add_edges_binary_tree(G, left_child, depth + 1, max_depth, weight_parent_to_child, weight_child_to_parent)
    add_edges_binary_tree(G, right_child, depth + 1, max_depth, weight_parent_to_child, weight_child_to_parent)


# Create a binary tree graph with edge weights
def create_weighted_binary_tree(depth, weight_parent_to_child=1., weight_child_to_parent=None):
    G = nx.DiGraph()  # Directed graph to show parent-child relationships
    add_edges_binary_tree(G, 0, 0, depth, weight_parent_to_child, weight_child_to_parent)
    return G

def add_depth_connections(G, w):
    for d in range(1, len(w)+1):
        for node_i in G.nodes():
            if G.nodes[node_i]['depth'] == d:
                for node_j in G.nodes():
                    if G.nodes[node_j]['depth'] == d:
                        if node_i != node_j:
                            G.add_edge(node_i, node_j, weight=w[d-1], edge_type='horizontal')
                            G.add_edge(node_j, node_i, weight=w[d-1], edge_type='horizontal')

# Function to plot the binary tree with arc-shaped edges and weights
def plot_graph_with_arcs_and_weights(G, pos=None):

    fig, ax = plt.subplots()

    if pos is None:
        pos = nx.spring_layout(G)  # Layout for a clean visualization

    # Drawing the edges in two separate sets to avoid overlap
    edges_forward = [(u, v) for u, v, d in G.edges(data=True) if d['edge_type'] == 'parent_to_child']
    edges_backward = [(u, v) for u, v, d in G.edges(data=True) if d['edge_type'] == 'child_to_parent']
    edges_horizontal = [(u, v) for u, v, d in G.edges(data=True) if d['edge_type'] == 'horizontal']

    # Plot only the nodes (vertices) without drawing edges
    # include tick marks
    nx.draw_networkx_nodes(G, pos, node_size=5, node_color='black', hide_ticks=False, ax=ax)
    # nx.draw_networkx_labels(G, pos, font_size=10, font_weight='bold', hide_ticks=False, ax=ax)

    # Draw forward (parent-to-child) and backward (child-to-parent) edges as arcs
    nx.draw_networkx_edges(G, pos, edgelist=edges_horizontal, connectionstyle="arc3,rad=0.", arrowstyle='-',
                           edge_color='green', label='Horizontal', hide_ticks=False, ax=ax, alpha=0.1)
    nx.draw_networkx_edges(G, pos, edgelist=edges_forward, connectionstyle="arc3,rad=0.2", arrowstyle='-|>',
                           edge_color='blue', label='Parent to Child', hide_ticks=False, ax=ax)
    nx.draw_networkx_edges(G, pos, edgelist=edges_backward, connectionstyle="arc3,rad=0.2", arrowstyle='-|>',
                           edge_color='red', label='Child to Parent', hide_ticks=False, ax=ax)

    # Add edge labels with weights and offset them
    # edge_labels_forward = {(u, v): d['weight'] for u, v, d in G.edges(data=True) if d['edge_type'] == 'parent_to_child'}
    # edge_labels_backward = {(u, v): d['weight'] for u, v, d in G.edges(data=True) if d['edge_type'] == 'child_to_parent'}
    # edge_labels_horizontal = {(u, v): d['weight'] for u, v, d in G.edges(data=True) if d['edge_type'] == 'horizontal'}
    #
    # Offset positions for labels to avoid overlap
    # pos_higher = {k: (v[0], v[1] + 0.05) for k, v in pos.items()}
    # pos_lower = {k: (v[0], v[1] - 0.05) for k, v in pos.items()}
    #
    # nx.draw_networkx_edge_labels(G, pos_higher, edge_labels=edge_labels_forward, hide_ticks=False, ax=ax)
    # nx.draw_networkx_edge_labels(G, pos_lower, edge_labels=edge_labels_backward, hide_ticks=False, ax=ax)
    # nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels_horizontal, hide_ticks=False, ax=ax)

    # plt.title("Binary Tree with Curved Edges and Weights")

    # set equal aspect ratio
    ax.set_aspect('equal')

    return fig, ax

#####################@


def main_binary_tree():

    # Hyperparameters
    tree_depth = 7                      # 3 to 7
    n = 2 ** (tree_depth + 1) - 1
    weight_parent_to_child = 0.5        # 0.5
    weight_child_to_parent = 1.5        # 1.5
    weight_horizontal = 0.1             # 0.1
    proj_dim = 2                        # 2
    randers_w_alpha_embedding = 0.5     # 0.5
    dir_res = 'res/'                    # 'res/'

    init_strat = 'rand'  # 'rand' | 'isomap', 'rand', 'isomap_with_rand'

    seed = 0  # 0
    np.random.seed(seed)

    if not os.path.exists(dir_res):
        os.makedirs(dir_res)

    param_str = (str(seed) + '_' + str(tree_depth) + '_' + str(proj_dim) + '_' + str() + '_' +
                 str(weight_parent_to_child) + '_' + str(weight_child_to_parent) + '_' +
                 str(weight_horizontal) + '_' +
                 str(randers_w_alpha_embedding) + '_' + init_strat)

    ############

    # Create base binary tree
    binary_tree = create_weighted_binary_tree(tree_depth,
                                              weight_parent_to_child=weight_parent_to_child,
                                              weight_child_to_parent=weight_child_to_parent)
    # fig_binary_tree, ax_binary_tree = plot_graph_with_arcs_and_weights(binary_tree)

    pure_binary_tree_top_down = create_weighted_binary_tree(tree_depth,
                                                            weight_parent_to_child=1., weight_child_to_parent=None)

    print('Computing spring layout without horizontal edges')

    # Method: Graph drawing by force‐directed placement, TMJ Fruchterman, EM Reingold
    p_spring_nohorizontal = nx.spring_layout(binary_tree)

    print('Adding horizontal edges between nodes at the same height')

    # Create symmetric edges between nodes at the same height
    w_horizontal = np.ones(tree_depth) * weight_horizontal
    add_depth_connections(binary_tree, w_horizontal)

    p_spring_horizontal = nx.spring_layout(binary_tree)

    print('Plotting binary tree with horizontal edges between nodes at the same height')

    fig_binary_tree_depth_connec_aswithout, ax_binary_tree_depth_connec_aswithout = \
        plot_graph_with_arcs_and_weights(binary_tree, pos=p_spring_nohorizontal)

    fig_binary_tree_depth_connec_with, ax_binary_tree_depth_connec_with = \
        plot_graph_with_arcs_and_weights(binary_tree, pos=p_spring_horizontal)

    ax_binary_tree_depth_connec_aswithout.axis('off')
    ax_binary_tree_depth_connec_with.axis('off')

    fig_binary_tree_depth_connec_aswithout.savefig(os.path.join(dir_res, 'binary_tree_depth_connec_aswithout_' + param_str + '.pdf'))
    fig_binary_tree_depth_connec_with.savefig(os.path.join(dir_res, 'binary_tree_depth_connec_with_' + param_str + '.pdf'))

    print('Compute distances')

    # # Compute distance matrix
    # Be careful with ordering as nx default takes the order of nodes in its nodelist computed from what appears in
    # edges, so not the correct ordering
    A = nx.adjacency_matrix(binary_tree, nodelist=np.sort(list(binary_tree.nodes))).toarray()
    A = scipy.sparse.csr_matrix(A)
    dists, _ = scipy.sparse.csgraph.shortest_path(A, method="auto", directed=True, return_predecessors=True)

    print('Isomap')
    # Code extracted from the Isomap method in scikit-learn/utils
    kernel_pca_ = KernelPCA(
        n_components=proj_dim,
        kernel="precomputed",
        eigen_solver="auto",
        tol=0,
        max_iter=None,
        n_jobs=None,
    ).set_output(transform="default")
    dists_sym = (dists + dists.T) / 2
    G = dists_sym ** 2
    G *= -0.5

    embedding_ = kernel_pca_.fit_transform(G)
    proj = embedding_
    # In Isomap's sklearn, they don't apply manually double centering, it is done inside kernel_pca_.fit_transform

    # Create position array for plotting
    proj_pos = {i: proj[i] for i in range(n)}

    fig_binary_tree_isomap, ax_binary_tree_isomap = plot_graph_with_arcs_and_weights(binary_tree, pos=proj_pos)
    ax_binary_tree_isomap.axis('off')

    # # Get x and y limits of plot and extend them a little bit
    # y_lim = ax_binary_tree_isomap.get_ylim()
    # ax_binary_tree_isomap.set_ylim(y_lim[0] - 0.05, y_lim[1] + 0.05)

    fig_binary_tree_isomap.savefig(os.path.join(dir_res, 'binary_tree_isomap_' + param_str + '.pdf'))

    # plt.show()

    print('Finsler SMACOF')
    if init_strat == 'rand':
        init = np.random.rand(n, proj_dim)
    elif init_strat == 'isomap':
        init = proj # @ np.array([[0, 1], [-1, 0]])  # Rotate x to y and y to -x
    elif init_strat == 'isomap_with_rand': # Noise added because isomap has collapsed the points
        init = (proj + np.random.randn(n, proj_dim) * 0.1) # @ np.array([[0, 1], [-1, 0]])  # Rotate x to y and y to -x
    unif_weights = np.ones_like(dists)
    # proj_smacof_f_unif, _ = _mds_finsler.smacof(
    proj_smacof_f_unif, _ = _mds_finsler_graph_debug.smacof(
        dists, randers_w_alpha=randers_w_alpha_embedding, metric=True,
        init=init, n_components=proj_dim, n_init=1, n_jobs=1,
        weight=unif_weights*1., max_iter=500,
        pseudo_inv_solver="gmres", project_on_V=True,
        check_monotony=False,
    )

    fig_binary_tree_smacof_f_unif, ax_binary_tree_smacof_f_unif = plot_graph_with_arcs_and_weights(binary_tree, pos=proj_smacof_f_unif)
    if tree_depth in [3, 5]:
        location = 'top_left'
    elif tree_depth in [2, 4, 6]:
        location = 'top_right'
    elif tree_depth in [7]:
        location = 'bottom_left'
    else:
        location = 'top_left'
    # matplotlib.rcParams['text.usetex'] = True
    # utils.plot_randers_w_arrow(proj_smacof_f_unif, ax_binary_tree_smacof_f_unif, shape_type='binary_tree', location=location)
    # # Get x and y limits of plot and extend them a little bit
    # x_lim = ax_binary_tree_smacof_f_unif.get_xlim()
    # ax_binary_tree_smacof_f_unif.set_xlim(x_lim[0] - 0.05, x_lim[1] + 0.05)

    ax_binary_tree_smacof_f_unif.axis('off')

    fig_binary_tree_smacof_f_unif.savefig(os.path.join(dir_res, 'binary_tree_smacof_f_unif_' + param_str + '.pdf'))


    print('Umap')
    reducer = umap.UMAP(n_components=proj_dim, metric='precomputed', random_state=seed)
    embedding_umap = reducer.fit_transform(dists_sym)

    fig_binary_tree_umap, ax_binary_tree_umap = plot_graph_with_arcs_and_weights(binary_tree, pos=embedding_umap)
    ax_binary_tree_umap.axis('off')

    fig_binary_tree_umap.savefig(os.path.join(dir_res, 'binary_tree_umap_' + param_str + '.pdf'))

    print('t-SNE')
    from sklearn.manifold import TSNE
    tsne = TSNE(n_components=proj_dim, metric='precomputed', init='random', random_state=seed)
    embedding_tsne = tsne.fit_transform(dists_sym)

    fig_binary_tree_tsne, ax_binary_tree_tsne = plot_graph_with_arcs_and_weights(binary_tree, pos=embedding_tsne)
    ax_binary_tree_tsne.axis('off')

    fig_binary_tree_tsne.savefig(os.path.join(dir_res, 'binary_tree_tsne_' + param_str + '.pdf'))

    plt.show(block=True)


if __name__ == '__main__':
    main_binary_tree()
    plt.show()