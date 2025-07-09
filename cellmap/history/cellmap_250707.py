import adjustText
import anndata
import IPython.display
import logging
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm
import matplotlib.colors
import matplotlib.animation as anm
import mpl_toolkits
import networkx as nx
import scanpy
import scipy
import scipy.sparse
import scipy.linalg
import scvelo as scv
import seaborn as sns
import sklearn.preprocessing
import sklearn.mixture
import sklearn.neighbors
import sklearn.linear_model
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
import plotly.offline
import umap


def create_graph(
    X,
    cutedge_vol=None,
    cutedge_length=None,
    cut_std=None,
    return_type="edges",
):
    tri_ = matplotlib.tri.Triangulation(X[:, 0], X[:, 1])
    X_src_, X_trg_ = X[tri_.edges[:, 0]], X[tri_.edges[:, 1]]
    length_edge_ = np.linalg.norm(X_src_ - X_trg_, axis=1)
    x1, y1 = X[tri_.triangles[:, 0], 0], X[tri_.triangles[:, 0], 1]
    x2, y2 = X[tri_.triangles[:, 1], 0], X[tri_.triangles[:, 1], 1]
    x3, y3 = X[tri_.triangles[:, 2], 0], X[tri_.triangles[:, 2], 1]
    vol_ = np.abs((x1 - x3) * (y2 - y3) - (x2 - x3) * (y1 - y3))
    length_ = np.max(
        [
            (x1 - x2) ** 2 + (y1 - y2) ** 2,
            (x2 - x3) ** 2 + (y2 - y3) ** 2,
            (x3 - x1) ** 2 + (y3 - y1) ** 2,
        ],
        axis=0,
    )
    if cut_std == None:
        std_delta_ = 0.1
        std_min_ = 1
        cut_std = std_min_
        while 1:
            if (
                len(
                    np.unique(
                        tri_.edges[length_edge_ < cut_std * np.std(length_edge_)]
                        .reshape(-1, 1)
                        .T[0]
                    )
                )
                == X.shape[0]
            ):
                break
            cut_std = cut_std + std_delta_
    if cutedge_vol == None:
        judge_vol_tri_ = vol_ < cut_std * np.std(vol_)
    else:
        judge_vol_tri_ = vol_ < np.percentile(vol_, 100 - cutedge_vol)
    if cutedge_length == None:
        judge_length_edge_ = length_edge_ < cut_std * np.std(length_edge_)
        judge_length_tri_ = length_ < cut_std * np.std(length_)
    else:
        judge_length_edge_ = length_edge_ < np.percentile(
            length_edge_, 100 - cutedge_length
        )
        judge_length_tri_ = length_ < np.percentile(length_edge_, 100 - cutedge_length)
    idx_mask_ = judge_vol_tri_ & judge_length_tri_
    tri_.set_mask(idx_mask_ == False)
    edge_tri_ = np.vstack(
        (
            np.vstack(
                (
                    tri_.triangles[idx_mask_][:, [0, 1]],
                    tri_.triangles[idx_mask_][:, [1, 2]],
                )
            ),
            tri_.triangles[idx_mask_][:, [2, 0]],
        )
    )
    edge_tri_sort_ = np.array([np.sort(e) for e in edge_tri_])
    # np.sort(edge_tri_sort_,axis=0),np.unique(edge_tri_sort_,axis=0).shape
    edges_, count_ = np.unique(edge_tri_sort_, axis=0, return_counts=True)
    idx_bd_ = np.unique(edges_[count_ == 1].reshape(1, -1)[0])
    if return_type == "edges":
        return edges_.T
    if return_type == "edges_bd":
        return edges_[:, 0], edges_[:, 1], idx_bd_
    if return_type == "triangles":
        return tri_, idx_mask_
    if return_type == "all":
        return tri_, idx_mask_, edges_[:, 0], edges_[:, 1], idx_bd_


def _check_arguments(adata, verbose=True, **kwargs):
    logger = logging.getLogger("argument checking")
    if verbose:
        logger.setLevel(logging.WARNING)
    else:
        logger.setLevel(logging.ERROR)

    if "exp_key" in kwargs.keys():
        if kwargs["exp_key"] != None:
            if (kwargs["exp_key"] not in adata.obsm.keys()) and (
                kwargs["exp_key"] not in adata.layers.keys()
            ):
                err_mssg = (
                    'The key "%s" was not found in adata.obsm.obsm. Please modify the argument "exp_key".'
                    % kwargs["exp_key"]
                )
                logger.exception(err_mssg)
                raise KeyError(err_mssg)

    if "exp_2d_key" in kwargs.keys():
        if (kwargs["exp_2d_key"] not in adata.obsm.keys()) and (
            kwargs["exp_2d_key"] not in adata.layers.keys()
        ):
            if "X_umap" in adata.obsm.keys():
                logger.warning(
                    'The key "%s" was not found in adata.obsm, but "X_umap" was found insted. "%s" was replaced with "X_umap".'
                    % (kwargs["exp_2d_key"], kwargs["exp_2d_key"])
                )
                kwargs["exp_2d_key"] = "X_umap"
            elif "X_tsne" in adata.obsm.keys():
                logger.warning(
                    'Warning: The key "%s" was not found in adata.obsm, but "X_tsne" was found insted. "%s" was replaced with "X_tsne".'
                    % (kwargs["exp_2d_key"], kwargs["exp_2d_key"])
                )
                kwargs["exp_2d_key"] = "X_tsne"
            elif "X_pca" in adata.obsm.keys():
                logger.warning(
                    'Warning: The key "%s" was not found in adata.obsm, but "X_pca" was found insted. "%s" was replaced with "X_tsne".'
                    % (kwargs["exp_2d_key"], kwargs["exp_2d_key"])
                )
                kwargs["exp_2d_key"] = "X_pca"
            else:
                raise KeyError(
                    'The key "%s" was not found in adata.obsm.obsm. Please modify the argument "exp_2d_key".'
                    % kwargs["exp_2d_key"]
                )

    if "vkey" in kwargs.keys():
        if (kwargs["vkey"] not in adata.obsm.keys()) and (
            kwargs["vkey"] not in adata.layers.keys()
        ):
            raise KeyError(
                'The key "%s" was not found in adata.obsm.obsm. Please modify the argument "vkey".'
                % kwargs["vkey"]
            )

    if "vel_2d_key" in kwargs.keys():
        if (kwargs["vel_2d_key"] not in adata.obsm.keys()) and (
            kwargs["vel_2d_key"] not in adata.layers.keys()
        ):
            if "velocity_umap" in adata.obsm.keys():
                logger.warning(
                    'The key "%s" was not found in adata.obsm, but "velocity_umap" was found insted. "%s" was replaced with "velocity_umap".'
                    % (kwargs["vel_2d_key"], kwargs["vel_2d_key"])
                )
                kwargs["vel_2d_key"] = "velocity_umap"
            elif "velocity_tsne" in adata.obsm.keys():
                logger.warning(
                    'Warning: The key "%s" was not found in adata.obsm, but "velocity_tsne" was found insted. "%s" was replaced with "velocity_tsne".'
                    % (kwargs["vel_2d_key"], kwargs["vel_2d_key"])
                )
                kwargs["vel_2d_key"] = "velocity_tsne"
            else:
                raise KeyError(
                    'The key "%s" was not found in adata.obsm.obsm. Please modify the argument "vel_2d_key".'
                    % kwargs["vel_2d_key"]
                )

    if "basis" in kwargs.keys():
        if ("X_%s" % kwargs["basis"] not in adata.obsm.keys()) and (
            "X_%s" % kwargs["basis"] not in adata.layers.keys()
        ):
            if "X_umap" in adata.obsm.keys():
                logger.warning(
                    'The key "%s" was not found in adata.obsm, but "X_umap" was found insted. "%s" was replaced with "X_umap".'
                    % (kwargs["basis"], kwargs["basis"])
                )
                kwargs["basis"] = "umap"
            elif "X_tsne" in adata.obsm.keys():
                logger.warning(
                    'Warning: The key "%s" was not found in adata.obsm, but "X_tsne" was found insted. "%s" was replaced with "X_tsne".'
                    % (kwargs["basis"], kwargs["basis"])
                )
                kwargs["basis"] = "tsne"
            else:
                raise KeyError(
                    'The key "%s" was not found in adata.obsm.obsm. Please modify the argument "exp_2d_key".'
                    % kwargs["basis"]
                )

    if "map_key" in kwargs.keys():
        if kwargs["map_key"] == None:
            kwargs["map_key"] = kwargs["exp_2d_key"]

    key_names = ["cluster_key", "potential_key"]
    for key in key_names:
        if key in kwargs.keys():
            if kwargs[key] not in adata.obs.keys():
                raise KeyError(
                    'The key "%s" was not found in adata.obs. Please modify the argument "%s".'
                    % (kwargs[key], key)
                )

    key_names = []
    for key in key_names:
        if key in kwargs.keys():
            if kwargs[key] not in adata.obsm.keys():
                raise KeyError(
                    'The key "%s" was not found in adata.obsm. Please modify the argument "%s".'
                    % (kwargs[key], key)
                )

    key_names = ["graph_key"]
    for key in key_names:
        if key in kwargs.keys():
            if kwargs[key] not in adata.uns.keys():
                raise KeyError(
                    'The key "%s" was not found in adata.uns. Please modify the argument "%s".'
                    % (kwargs[key], key)
                )

    key_names = ["expression_key"]
    for key in key_names:
        if key in kwargs.keys():
            if kwargs[key] != None:
                if (kwargs[key] not in adata.obsm.keys()) & (
                    kwargs[key] not in adata.layers.keys()
                ):
                    raise KeyError(
                        'The key "%s" was not found in adata.obsm or adata.layers. Please modify the argument "%s".'
                        % (kwargs[key], key)
                    )

    if "graph_method" in kwargs.keys():
        if kwargs["graph_method"] != None:
            if kwargs["graph_method"] not in ["Delauney", "knn"]:
                raise KeyError(
                    'The key "%s" was not found in adata.obsm or adata.layers. Please modify the argument "%s".'
                    % (kwargs[key], key)
                )

    key = "obs_key"
    if key in kwargs.keys():
        if type(kwargs[key]) == list:
            key_names = ["cluster_key", "potential_key"]
            for key_ in key_names:
                if key_ in kwargs.keys():
                    if kwargs[key_] in kwargs[key]:
                        # raise logger.warning('The key \"%s\" was multipled.' % (kwargs[key_]))
                        kwargs[key].remove(kwargs[key_])
            for arg in kwargs[key]:
                if arg not in adata.obs.keys():
                    logger.warning(
                        'The key "%s" was not found in adata.obs. The key "%s" is removed from "%s".'
                        % (arg, key, arg, key)
                    )
                    kwargs[key].remove(key)
            key_names = ["cluster_key", "potential_key"]
        elif kwargs[key] != None:
            raise TypeError("The argument %s should be a list or None")

    key = "genes"
    if key in kwargs.keys():
        if type(kwargs[key]) == list:
            for arg in kwargs[key]:
                if arg not in adata.var.index:
                    logger.warning(
                        'The gene "%s" was not found. The gene "%s" is removed from "%s".'
                        % (arg, arg, key)
                    )
                    kwargs[key].remove(arg)
        elif kwargs[key] != None:
            raise TypeError("The argument %s should be a list or None")

    return kwargs

def _set_expression_data(adata, exp_key):
    if exp_key == None:
        if scipy.sparse.issparse(adata.X):
            return adata.X.toarray()
        else:
            return adata.X
    elif exp_key in adata.layers.keys():
        return adata.layers[exp_key]
    elif exp_key in adata.obsm.keys():
        return adata.obsm[exp_key]
    else:
        err_mssg = (
            'The key "%s" was not found in adata.obsm.obsm. Please modify the argument "exp_key".'
            % exp_key
        )
        logger = logging.getLogger("argument checking")
        logger.exception(err_mssg)
        raise KeyError(err_mssg)


def cmap_earth(cv):
    c_min, c_max = 5, 95
    c_list = np.array(
        [
            "#0938BF",
            "#50D9FB",
            "#B7E5FA",
            "#98D685",
            "#fff5d1",
            "#997618",
            "#705B10",
            "#5C1D09",
        ]
    )
    c_level = np.array(
        [
            np.percentile(cv, (c_max - c_min) * (i) / len(c_list) + c_min)
            for i in range(len(c_list))
        ]
    )
    c_list = [
        "#0938BF",
        "#50D9FB",
        "#B7E5FA",
        "#98D685",
        "#F9EFCD",
        "#E0BB7D",
        "#D3A62D",
        "#997618",
        "#705B10",
        "#5F510D",
        "#A56453",
        "#5C1D09",
    ]
    c_level = np.percentile(cv,[0, 5, 20, 40, 60, 75, 80, 85, 90, 95, 99, 100])
    color = np.vstack((c_level, c_list)).T
    hight = 1000 * color[:, 0].astype(np.float32)
    hightnorm = sklearn.preprocessing.minmax_scale(hight)
    colornorm = []
    for no, norm in enumerate(hightnorm):
        colornorm.append([norm, color[no, 1]])
    colornorm[-1][0] = 1
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list(
        "earth", colornorm, N=hight.max() - hight.min() + 1
    )
    return cmap


def edge_velocity(
    X,
    vel,
    source,
    target,
    normalization=True,
):
    idx_vel = np.isnan(vel[0]) == False
    X1, X2 = X[:, idx_vel][source], X[:, idx_vel][target]
    V1, V2 = vel[:, idx_vel][source], vel[:, idx_vel][target]
    Dis = np.linalg.norm(X2 - X1, axis=1)
    Dis[Dis == 0] = 1
    # V1_p,V2_p = V1*(X2-X1),V2*(X2-X1)
    edge_vel = np.sum(0.5 * (V1 + V2) * (X2 - X1), axis=1) / Dis / np.sum(idx_vel)
    edge_vel[edge_vel<0]=0 
    if normalization:
        edge_vel_norm = np.linalg.norm(edge_vel)
        if edge_vel_norm > 0:
            edge_vel = edge_vel / edge_vel_norm
    return edge_vel


def Hodge_decomposition(
    adata,
    basis="umap",
    vkey="velocity",
    exp_key=None,
    potential_key="potential",
    rotation_key="rotation",
    vorticity_key="vorticity",
    streamfunc_key="streamfunc",
    graph_key="CellMap_graph",
    edge_vel_key="edge_velocity",
    graph_method="knn",  #'Delauney',
    HD_rate=0.0,
    n_neighbors=15,
    cutedge_vol=None,
    cutedge_length=None,
    cut_std=None,
    verbose=True,
    logscale_vel=True,
):
    """
    Hodge decomposition

    Parameters
    ----------
    adata: anndata (n_samples, n_features)

    exp_key: None or string
    """

    kwargs_arg = _check_arguments(
        adata,
        verbose=True,
        exp_key=exp_key,
        vkey=vkey,
        basis=basis,
        graph_method=graph_method,
    )
    exp_key, vkey, basis = (
        kwargs_arg["exp_key"],
        kwargs_arg["vkey"],
        kwargs_arg["basis"],
    )

    exp_2d_key_ = "X_%s" % basis
    vel_2d_key_ = "%s_%s" % (vkey, basis)
    pot_vkey_ = "%s_%s_%s" % (potential_key, vkey, basis)
    rot_vkey_ = "%s_%s_%s" % (rotation_key, vkey, basis)
    vor_key_ = "%s_%s" % (vorticity_key, basis)
    sl_key_ = "%s_%s" % (streamfunc_key, basis)
    pot_vor_key_ = "%s_%s_%s" % (potential_key, vorticity_key, basis)
    pot_sl_key_ = "%s_%s_%s" % (potential_key, streamfunc_key, basis)
    rot_vor_key_ = "%s_%s_%s" % (rotation_key, vorticity_key, basis)
    rot_sl_key_ = "%s_%s_%s" % (rotation_key, streamfunc_key, basis)

    exp_LD = (
        adata.obsm[exp_2d_key_][:, :2]
        if exp_2d_key_ in adata.obsm.keys()
        else adata.layers[exp_2d_key_][:, :2]
    )
    vel_LD = (
        adata.obsm[vel_2d_key_][:, :2]
        if vel_2d_key_ in adata.obsm.keys()
        else adata.layers[vel_2d_key_][:, :2]
    )

    exp_HD = _set_expression_data(adata, exp_key)

    ## Compute graph and edge velocities
    n_node_ = exp_HD.shape[0]
    idx_bd_ = []
    if graph_method == "Delauney":
        source, target, idx_bd_ = create_graph(
            exp_LD,
            cutedge_vol=cutedge_vol,
            cutedge_length=cutedge_length,
            cut_std=cut_std,
            return_type="edges_bd",
        )
    elif graph_method == "knn":
        knn = sklearn.neighbors.NearestNeighbors(
            n_neighbors=n_neighbors + 1, algorithm="kd_tree"
        )
        knn.fit(exp_LD)
        distances, indices = knn.kneighbors(exp_LD)
        distances, indices = distances[:, 1:], indices[:, 1:]
        source = np.ravel(
            np.repeat(np.arange(exp_LD.shape[0]).reshape((-1, 1)), n_neighbors, axis=1)
        )
        target = np.ravel(indices)

    if HD_rate > 0:
        vel_HD = adata.obsm[vkey] if vkey in adata.obsm.keys() else adata.layers[vkey]
        if logscale_vel:
            if "n_counts" in adata.obs.keys():
                vel_HD = (adata.obs["n_counts"].values * vel_HD.T).T / np.exp(exp_HD)
            else:
                vel_HD = (np.sum(exp_HD,axis=1) * vel_HD.T).T / np.exp(exp_HD)
        edge_vel_HD = edge_velocity(exp_HD, vel_HD, source, target)
    else:
        edge_vel_HD = 0

    if HD_rate < 1:
        edge_vel_LD = edge_velocity(exp_LD, vel_LD, source, target)
    else:
        edge_vel_LD = 0

    ## Solve potential
    n_edge_ = len(source)
    grad_mat = np.zeros([n_edge_, n_node_], dtype=float)
    grad_mat[tuple(np.vstack((np.arange(n_edge_), source)))] = -1
    grad_mat[tuple(np.vstack((np.arange(n_edge_), target)))] = 1
    div_mat = -grad_mat.T
    lap_ = (
        -(scipy.sparse.csr_matrix(div_mat))
        .dot(scipy.sparse.csr_matrix(grad_mat))
        .toarray()
    )
    edge_vel = (1 - HD_rate) * edge_vel_LD + HD_rate * edge_vel_HD
    source_term = div_mat @ edge_vel
    lap_inv_ = scipy.linalg.pinv(lap_)
    potential = lap_inv_ @ source_term
    pot_flow_ = -grad_mat @ potential
    rot_flow_ = edge_vel - pot_flow_
    adata.obs[potential_key] = potential - np.min(potential)

    # Compute potential & rotational flow
    vel_potential = np.zeros([adata.shape[0], 2], dtype=float)
    vel_rotation = np.zeros([adata.shape[0], 2], dtype=float)
    edge_vel_norm = np.linalg.norm(
        edge_velocity(exp_LD, vel_LD, source, target, normalization=False)
    )
    if graph_method == "Delauney":
        src_trg_ = np.hstack((source, target))
        trg_src_ = np.hstack((target, source))
        pot_flow_2_ = np.hstack((pot_flow_, -pot_flow_))
        rot_flow_2_ = np.hstack((rot_flow_, -rot_flow_))
        for i in range(adata.shape[0]):
            idx_ = src_trg_ == i
            dis_ = np.linalg.norm(
                exp_LD[src_trg_[idx_]] - exp_LD[trg_src_[idx_]], axis=1, ord=2
            )
            dis_[dis_ == 0] = 1
            ex_ = -(exp_LD[src_trg_[idx_]] - exp_LD[trg_src_[idx_]]).T / dis_
            vel_potential[i] = (
                4.0 * edge_vel_norm * np.mean(pot_flow_2_[idx_] * ex_, axis=1)
            )
            vel_rotation[i] = (
                4.0 * edge_vel_norm * np.mean(rot_flow_2_[idx_] * ex_, axis=1)
            )
        adata.obsm[pot_vkey_] = vel_potential
        adata.obsm[rot_vkey_] = vel_rotation
    elif graph_method == "knn":
        for i in range(adata.shape[0]):
            idx_ = source == i
            dis_ = np.linalg.norm(
                exp_LD[source[idx_]] - exp_LD[target[idx_]], axis=1, ord=2
            )
            dis_[dis_ == 0] = 1
            ex_ = -(exp_LD[source[idx_]] - exp_LD[target[idx_]]).T / dis_
            vel_potential[i] = (
                4.0 * edge_vel_norm * np.mean(pot_flow_[idx_] * ex_, axis=1)
            )
            vel_rotation[i] = (
                4.0 * edge_vel_norm * np.mean(rot_flow_[idx_] * ex_, axis=1)
            )
        adata.obsm[pot_vkey_] = vel_potential
        adata.obsm[rot_vkey_] = vel_rotation

    # vorticity_ = div_mat @ edge_velocity(exp_LD,np.vstack((vel_LD[:,1],-vel_LD[:,0])).T,source,target,normalization=False)
    div_ = np.linalg.norm(vel_LD, axis=1)
    div_[div_ == 0] = 1
    vorticity_ = div_mat @ edge_velocity(
        exp_LD,
        # np.vstack((vel_LD[:, 1] / div_, -vel_LD[:, 0] / div_)).T,
        np.vstack((vel_LD[:, 1] / div_, -vel_LD[:, 0] / div_)).T,
        source,
        target,
        normalization=False,
    )
    source_term_ = vorticity_
    streamfunc_ = -lap_inv_ @ source_term_
    adata.obs[vor_key_] = vorticity_
    adata.obs[sl_key_] = streamfunc_ - np.min(streamfunc_)

    # vorticity_ = div_mat @ edge_velocity(exp_LD,np.vstack((adata.obsm[pot_vkey_][:,1],-adata.obsm[pot_vkey_][:,0])).T,source,target,normalization=False)
    div_ = np.linalg.norm(adata.obsm[pot_vkey_], axis=1)
    div_[div_ == 0] = 1
    vorticity_ = div_mat @ edge_velocity(
        exp_LD,
        np.vstack(
            (adata.obsm[pot_vkey_][:, 1] / div_, -adata.obsm[pot_vkey_][:, 0] / div_)
        ).T,
        source,
        target,
        normalization=False,
    )
    source_term_ = vorticity_
    streamfunc_ = -lap_inv_ @ source_term_
    adata.obs[pot_vor_key_] = vorticity_
    adata.obs[pot_sl_key_] = streamfunc_ - np.min(streamfunc_)

    # vorticity_ = div_mat @ edge_velocity(exp_LD,np.vstack((adata.obsm[rot_vkey_][:,1],-adata.obsm[rot_vkey_][:,0])).T,source,target,normalization=False)
    div_ = np.linalg.norm(adata.obsm[rot_vkey_], axis=1)
    div_[div_ == 0] = 1
    vorticity_ = div_mat @ edge_velocity(
        exp_LD,
        np.vstack(
            (adata.obsm[rot_vkey_][:, 1] / div_, -adata.obsm[rot_vkey_][:, 0] / div_)
        ).T,
        source,
        target,
        normalization=False,
    )
    source_term_ = vorticity_
    streamfunc_ = -lap_inv_ @ source_term_
    adata.obs[rot_vor_key_] = vorticity_
    adata.obs[rot_sl_key_] = streamfunc_ - np.min(streamfunc_)

    adata.obs[rotation_key] = np.array(
        [
            np.mean(np.hstack((rot_flow_[source == i], -rot_flow_[target == i])))
            for i in range(adata.shape[0])
        ]
    )

    ## Contribution ratio
    log_ = {}
    log_["Contribution_ratio"] = {}
    norm_grad = np.linalg.norm(pot_flow_)
    norm_curl = np.linalg.norm(rot_flow_)
    log_["Contribution_ratio"]["Potential"] = "{:.2%}".format(
        norm_grad / (norm_grad + norm_curl)
    )
    log_["Contribution_ratio"]["Rotation"] = "{:.2%}".format(
        norm_curl / (norm_grad + norm_curl)
    )
    adata.uns["CellMap_log"] = log_
    if verbose:
        print(adata.uns["CellMap_log"])

    adata.uns[graph_key] = {"source": source, "target": target}
    adata.uns[edge_vel_key] = {
        edge_vel_key: edge_vel,
        edge_vel_key + "_pot": pot_flow_,
        edge_vel_key + "_rot": rot_flow_,
    }


def Hodge_decomposition_genes(
    adata,
    genes,
    basis="umap",
    vkey="velocity",
    exp_key=None,
    potential_key="potential",
    potential_vkey="potential_velocity",
    rotation_key="rotation",
    rotation_vkey="rotation_velocity",
    graph_method="knn",  #'Delauney',
    n_neighbors=10,
    cutedge_vol=None,
    cutedge_length=None,
    verbose=True,
    logscale_vel=True,
):
    """
    Hodge decomposition

    Parameters
    ----------
    adata: anndata (n_samples, n_features)

    exp_key: None or string
    """

    kwargs_arg = _check_arguments(
        adata,
        verbose=True,
        exp_key=exp_key,
        vkey=vkey,
        basis=basis,
        graph_method=graph_method,
    )
    exp_key, vkey, basis = (
        kwargs_arg["exp_key"],
        kwargs_arg["vkey"],
        kwargs_arg["basis"],
    )

    exp_2d_key_ = "X_%s" % basis
    vel_2d_key_ = "%s_%s" % (vkey, basis)
    pot_vkey_ = "%s_%s_%s" % (potential_key, vkey, basis)
    rot_vkey_ = "%s_%s_%s" % (rotation_key, vkey, basis)

    if exp_key == None:
        if scipy.sparse.issparse(adata.X):
            exp_HD = adata.X.toarray()
        else:
            exp_HD = adata.X
    elif exp_key in adata.obsm.keys():
        exp_HD = adata.obsm[exp_key]
    else:
        exp_HD = adata.layers[exp_key]

    vel_HD = adata.obsm[vkey] if vkey in adata.obsm.keys() else adata.layers[vkey]
    if logscale_vel:
        vel_HD = (1e4 * vel_HD.T / adata.obs["n_counts"].values).T / np.power(2, exp_HD)
    exp_LD = (
        adata.obsm[exp_2d_key_][:, :2]
        if exp_2d_key_ in adata.obsm.keys()
        else adata.layers[exp_2d_key_][:, :2]
    )

    n_node_ = exp_HD.shape[0]
    if graph_method == "Delauney":
        source, target = create_graph(
            exp_LD,
            cutedge_vol=cutedge_vol,
            cutedge_length=cutedge_length,
            return_type="edges",
        )
        # source, target = np.ravel(tri_.triangles[idx_tri][:,[0,1,2]]),np.ravel(tri_.triangles[idx_tri][:,[1,2,0]])
    elif graph_method == "knn":
        # pca = sklearn.decomposition.PCA()
        # exp_HD_pca = pca.fit_transform(exp_HD)
        # n_pca = np.min(np.arange(len(pca.explained_variance_ratio_))[np.cumsum(pca.explained_variance_ratio_)>contribution_rate])
        # knn = sklearn.neighbors.NearestNeighbors(n_neighbors=n_neighbors+1, algorithm='kd_tree')
        # knn.fit(exp_HD_pca[:,:n_pca])
        # distances, indices = knn.kneighbors(exp_HD_pca[:,:n_pca])
        knn = sklearn.neighbors.NearestNeighbors(
            n_neighbors=n_neighbors + 1, algorithm="kd_tree"
        )
        knn.fit(exp_LD)
        distances, indices = knn.kneighbors(exp_LD)
        distances, indices = distances[:, 1:], indices[:, 1:]
        source = np.ravel(
            np.repeat(np.arange(exp_LD.shape[0]).reshape((-1, 1)), n_neighbors, axis=1)
        )
        target = np.ravel(indices)

    n_edge_ = len(source)
    grad_mat = np.zeros([n_edge_, n_node_], dtype=float)
    grad_mat[tuple(np.vstack((np.arange(n_edge_), source)))] = -1
    grad_mat[tuple(np.vstack((np.arange(n_edge_), target)))] = 1
    div_mat = -grad_mat.T
    lap = -div_mat @ grad_mat
    lap_inv = np.linalg.pinv(lap)

    for gene in genes:
        X1, X2 = (
            exp_HD[:, adata.var.index == gene][source],
            exp_HD[:, adata.var.index == gene][target],
        )
        V1, V2 = (
            vel_HD[:, adata.var.index == gene][source],
            vel_HD[:, adata.var.index == gene][target],
        )
        Dis = np.linalg.norm(exp_HD[target] - exp_HD[source], axis=1)
        edge_vel = np.sum(0.5 * (V1 + V2) * (X2 - X1), axis=1) / Dis
        source_term = div_mat @ edge_vel
        potential = lap_inv @ source_term
        adata.obs[potential_key + "_Gene_%s" % gene] = potential - np.min(potential)


def view(
    adata,
    basis="umap",
    color_key="potential",
    cluster_key="clusters",
    show_graph=True,
    cutedge_vol=None,
    cutedge_length=None,
    title="",
    save=False,
    save_dir=None,
    save_filename="CellMap_view",
    figsize=None,
    fontsize_text=16,
    cbar=True,
    **kwargs,
):
    kwargs_arg = _check_arguments(adata, basis=basis)
    basis = kwargs_arg["basis"]
    basis_key = "X_%s" % basis

    if "cmap" not in kwargs:
        kwargs["cmap"] = cmap_earth(adata.obs[color_key])

    if cluster_key not in adata.obs.keys():
        cluster_key = None

    if figsize == None:
        figsize = (10, 6) if cbar else (8, 6)

    data_pos = adata.obsm[basis_key]
    fig, ax = plt.subplots(figsize=figsize)
    sc = ax.scatter(
        data_pos[:, 0], data_pos[:, 1], c=adata.obs[color_key], zorder=10, **kwargs
    )
    if show_graph:
        tri_ = create_graph(
            data_pos,
            cutedge_vol=cutedge_vol,
            cutedge_length=cutedge_length,
            return_type="triangles",
        )[0]
        ax.tripcolor(
            tri_, adata.obs[color_key], lw=0.5, zorder=0, alpha=0.3, cmap=kwargs["cmap"]
        )
    if cluster_key != None:
        if cluster_key in adata.obs.keys():
            cluster = adata.obs[cluster_key]
            for c in np.unique(cluster):
                txt = plt.text(
                    np.mean(data_pos[cluster == c], axis=0)[0],
                    np.mean(data_pos[cluster == c], axis=0)[1],
                    c,
                    fontsize=fontsize_text,
                    ha="center",
                    va="center",
                    fontweight="bold",
                    zorder=20,
                )
                txt.set_path_effects(
                    [matplotlib.patheffects.withStroke(linewidth=5, foreground="w")]
                )
        else:
            print('There is no cluster key "%s" in adata.obs' % cluster_key)
    ax.axis("off")
    ax.set_title(title, fontsize=18)
    if cbar:
        plt.colorbar(sc, aspect=20, pad=0.01, orientation="vertical").set_label(
            color_key, fontsize=20
        )
    if save:
        filename = (
            "%s" % (save_filename)
            if save_dir == None
            else "%s/%s" % (save_dir, save_filename)
        )
        fig.savefig(filename + ".png", bbox_inches="tight")


def view_contour(
    adata,
    basis="umap",
    color_key="potential",
    cluster_key="clusters",
    show_graph=True,
    cutedge_vol=None,
    cutedge_length=None,
    title="",
    save=False,
    save_dir=None,
    save_filename="CellMap_view",
    figsize=None,
    fontsize_text=16,
    cbar=True,
    **kwargs,
):
    kwargs_arg = _check_arguments(adata, basis=basis)
    basis = kwargs_arg["basis"]
    basis_key = "X_%s" % basis

    if "cmap" not in kwargs:
        kwargs["cmap"] = "jet"

    if cluster_key not in adata.obs.keys():
        cluster_key = None

    if figsize == None:
        figsize = (10, 6) if cbar else (8, 6)
    
    data_pos = adata.obsm[basis_key]
    tri_ = create_graph(
        data_pos,
        cutedge_vol=cutedge_vol,
        cutedge_length=cutedge_length,
        return_type="triangles",
    )[0]

    fig, ax = plt.subplots(1, 1, figsize=figsize, tight_layout=True)
    ax.axis("off")
    ax.set_title(title, fontsize=18)
    sc = ax.tripcolor(tri_, adata.obs[color_key], cmap=kwargs["cmap"])
    ax.tricontour(
        tri_,
        adata.obs[color_key],
        lw=0.2,
        alpha=0.2,
        levels=75,
        zorder=3,
        colors="k",
        cmap=None,
        ls="-",
    )
    ax.tricontour(
        tri_,
        adata.obs[color_key],
        lw=1,
        alpha=1,
        levels=15,
        zorder=3,
        colors="k",
        cmap=None,
        ls="-",
    )
    if cbar:
        plt.colorbar(sc, aspect=20, pad=0.01, orientation="vertical").set_label(
        color_key, fontsize=20
        )
    # ax.set_title(titles[i], fontsize=fontsize)
    if cluster_key != None:
        if cluster_key in adata.obs.keys():
            cluster = adata.obs[cluster_key]
            for c in np.unique(cluster):
                # plt.scatter(data_pos[cluster == c,0],data_pos[cluster == c,1],zorder=1,alpha=0.1,s=100)
                txt = ax.text(
                    np.mean(data_pos[cluster == c], axis=0)[0],
                    np.mean(data_pos[cluster == c], axis=0)[1],
                    c,
                    fontsize=20,
                    ha="center",
                    va="center",
                    fontweight="bold",
                    zorder=20,
                )
                txt.set_path_effects(
                    [matplotlib.patheffects.withStroke(linewidth=5, foreground="w")]
                )
    else:
        print('There is no cluster key "%s" in adata.obs' % cluster_key)
    if save:
        filename = (
            "%s" % (save_filename)
            if save_dir == None
            else "%s/%s" % (save_dir, save_filename)
        )
        fig.savefig(filename + ".png", bbox_inches="tight")

def view_cluster(
    adata,
    basis="umap",
    potential_key="potential",
    cluster_key="clusters",
    cutedge_vol=None,
    cutedge_length=None,
    n_points=1000,
    fontsize_text=16,
    seed=None,
    title="",
    **kwargs,
):
    kwargs_arg = _check_arguments(adata, basis=basis, potential_key=potential_key)
    basis = kwargs_arg["basis"]
    basis_key = "X_%s" % basis

    if "cmap" not in kwargs:
        kwargs["cmap"] = cmap_earth(adata.obs[potential_key])

    data_pos = adata.obsm[basis_key]
    fig, ax = plt.subplots(figsize=(8, 6))
    tri_ = create_graph(
        data_pos,
        cutedge_vol=cutedge_vol,
        cutedge_length=cutedge_length,
        return_type="triangles",
    )[0]
    sc = ax.tricontourf(
        tri_,
        adata.obs[potential_key],
        zorder=0,
        alpha=0.9,
        cmap=kwargs["cmap"],
        levels=100,
    )
    if cluster_key in adata.obs.keys():
        cluster = adata.obs[cluster_key]
        idx_random = np.zeros(cluster.shape, dtype=bool)
        np.random.seed(seed)
        idx_random[
            np.random.choice(
                len(idx_random), min(n_points, len(idx_random)), replace=False
            )
        ] = True
        cluster_set = np.unique(cluster)
        cmap_pt = (
            plt.get_cmap("tab10") if len(cluster_set) <= 10 else plt.get_cmap("tab20")
        )
        for i in range(len(cluster_set)):
            idx = (cluster == cluster_set[i]) & idx_random
            ax.scatter(
                data_pos[idx, 0],
                data_pos[idx, 1],
                zorder=10,
                alpha=0.8,
                edgecolor="w",
                color=cmap_pt(i),
                **kwargs,
            )
            txt = plt.text(
                np.mean(data_pos[cluster == cluster_set[i]], axis=0)[0],
                np.mean(data_pos[cluster == cluster_set[i]], axis=0)[1],
                cluster_set[i],
                color=cmap_pt(i),
                fontsize=fontsize_text,
                ha="center",
                va="center",
                fontweight="bold",
                zorder=20,
            )
            txt.set_path_effects([matplotlib.patheffects.withStroke(linewidth=5, foreground="w")])
    else:
        print('There is no cluster key "%s" in adata.obs' % cluster_key)
    ax.set_title(title, fontsize=18)
    plt.colorbar(sc, aspect=20, pad=0.01, orientation="vertical").set_label(
        potential_key, fontsize=20
    )
    ax.axis("off")


def view_surface(
    adata,
    basis="umap",
    color_key="potential",
    cluster_key=None,
    show_graph=False,
    cutedge_vol=None,
    cutedge_length=None,
    title="",
    **kwargs,
):
    kwargs_arg = _check_arguments(
        adata,
        basis=basis,
    )
    basis = kwargs_arg["basis"]
    basis_key = "X_%s" % basis

    if "cmap" not in kwargs:
        kwargs["cmap"] = cmap_earth(adata.obs[color_key])

    data_pos = adata.obsm[basis_key]
    tri_ = create_graph(
        data_pos,
        cutedge_vol=cutedge_vol,
        cutedge_length=cutedge_length,
        return_type="triangles",
    )[0]
    fig, ax = plt.subplots(figsize=(8, 6))
    cntr = ax.tricontourf(
        tri_, adata.obs[color_key], cmap=kwargs["cmap"], levels=100, zorder=2
    )
    fig.colorbar(cntr, shrink=0.75, orientation="vertical").set_label(
        color_key, fontsize=20
    )
    if show_graph:
        ax.triplot(tri_, color="w", lw=0.5, zorder=10, alpha=1)
    ax.set_xlim(
        np.min(data_pos[:, 0])
        - 0.02 * (np.max(data_pos[:, 0]) - np.min(data_pos[:, 0])),
        np.max(data_pos[:, 0])
        + 0.02 * (np.max(data_pos[:, 0]) - np.min(data_pos[:, 0])),
    )
    ax.set_ylim(
        np.min(data_pos[:, 1])
        - 0.02 * (np.max(data_pos[:, 1]) - np.min(data_pos[:, 1])),
        np.max(data_pos[:, 1])
        + 0.02 * (np.max(data_pos[:, 1]) - np.min(data_pos[:, 1])),
    )
    ax.tick_params(
        labelbottom=False,
        labelleft=False,
        labelright=False,
        labeltop=False,
        bottom=False,
        left=False,
        right=False,
        top=False,
    )
    ax.spines["right"].set_visible(False), ax.spines["top"].set_visible(
        False
    ), ax.spines["bottom"].set_visible(False), ax.spines["left"].set_visible(False)
    ax.set_title(title, fontsize=18)
    if cluster_key != None:
        texts = []
        if cluster_key in adata.obs.keys():
            cluster = adata.obs[cluster_key]
            for c in np.unique(cluster):
                txt = ax.text(
                    np.mean(data_pos[cluster == c], axis=0)[0],
                    np.mean(data_pos[cluster == c], axis=0)[1],
                    c,
                    fontsize=20,
                    ha="center",
                    va="center",
                    fontweight="bold",
                )
                txt.set_path_effects(
                    [matplotlib.patheffects.withStroke(linewidth=5, foreground="w")]
                )
                texts.append(txt)


def view_stream(
    adata,
    basis="umap",
    vkey="velocity",
    potential_key="potential",
    rotation_key="rotation",
    cluster_key="clusters",
    figsize=(24, 6),
    density=2,
    alpha=0.3,
    fontsize=18,
    legend_fontsize=18,
    **kwargs,
):
    kwargs_arg = _check_arguments(adata, basis=basis)
    basis = kwargs_arg["basis"]
    basis_key = "X_%s" % basis
    data_pos = adata.obsm[basis_key]
    pot_vkey_ = "%s_%s" % (potential_key, vkey)
    rot_vkey_ = "%s_%s" % (rotation_key, vkey)

    fig, ax = plt.subplots(1, 3, figsize=figsize, tight_layout=True,facecolor="w")
    scv.pl.velocity_embedding_stream(
        adata,
        basis=basis,
        vkey=vkey,
        title="RNA velocity",
        ax=ax[0],
        color=cluster_key,
        show=False,
        density=density,
        alpha=alpha,
        fontsize=fontsize,
        legend_fontsize=0,
        legend_loc=None,
        arrow_size=2,
        linewidth=2,
        **kwargs,
    )
    scv.pl.velocity_embedding_stream(
        adata,
        basis=basis,
        vkey=vkey,
        title="RNA velocity",
        ax=ax[0],
        color=cluster_key,
        show=False,
        density=density,
        alpha=alpha,
        fontsize=fontsize,
        legend_fontsize=0,
        legend_loc=None,
        arrow_size=2,
        linewidth=2,
        **kwargs,
    )
    scv.pl.velocity_embedding_stream(
        adata,
        basis=basis,
        vkey=pot_vkey_,
        title="Potential flow",
        ax=ax[1],
        color=cluster_key,
        show=False,
        density=density,
        alpha=alpha,
        fontsize=fontsize,
        legend_fontsize=0,
        legend_loc=None,
        arrow_size=2,
        linewidth=2,
        **kwargs,
    )
    scv.pl.velocity_embedding_stream(
        adata,
        basis=basis,
        vkey=rot_vkey_,
        title="Rotational flow",
        ax=ax[2],
        color=cluster_key,
        show=False,
        density=density,
        alpha=alpha,
        fontsize=fontsize,
        legend_fontsize=0,
        legend_loc=None,
        arrow_size=2,
        linewidth=2,
        **kwargs,
    )
    if cluster_key != None:
        cluster = adata.obs[cluster_key]
        for i in range(3):
            texts = []
            for c in np.unique(cluster):
                txt = ax[i].text(
                    np.mean(data_pos[cluster == c], axis=0)[0],
                    np.mean(data_pos[cluster == c], axis=0)[1],
                    c,
                    fontsize=legend_fontsize,
                    ha="center",
                    va="center",
                    fontweight="bold",
                    zorder=20,
                )
                txt.set_path_effects([matplotlib.patheffects.withStroke(linewidth=5, foreground="w")])
                texts.append(txt)


def view_stream_line(
    adata,
    basis="umap",
    contour_key="streamfunc",
    cluster_key="clusters",
    potential_key="potential",
    rotation_key="rotation",
    cutedge_vol=None,
    cutedge_length=None,
    title="",
    save=False,
    save_dir=None,
    save_filename="CellMap_stream_line",
    figsize=(24, 6),
    fontsize=18,
    cbar=False,
    **kwargs,
):
    kwargs_arg = _check_arguments(adata, basis=basis)
    basis = kwargs_arg["basis"]
    basis_key = "X_%s" % basis

    key_ = "%s_%s" % (contour_key, basis)
    pot_key_ = "%s_%s_%s" % (potential_key, contour_key, basis)
    rot_key_ = "%s_%s_%s" % (rotation_key, contour_key, basis)

    data_pos = adata.obsm[basis_key]
    tri_ = create_graph(
        data_pos,
        cutedge_vol=cutedge_vol,
        cutedge_length=cutedge_length,
        return_type="triangles",
    )[0]

    contour_keys = [key_, pot_key_, rot_key_]
    camps = [cmap_earth(adata.obs[key_]), "rainbow", "coolwarm"]
    titles = ["RNA velocity orbit", "Development orbit", "Periodic orbit"]

    fig, ax = plt.subplots(1, 3, figsize=figsize, tight_layout=True)
    for i in range(3):
        ax[i].axis("off")
        ax[i].set_title(title, fontsize=18)
        sc = ax[i].tripcolor(tri_, adata.obs[contour_keys[i]], cmap=camps[i])
        ax[i].tricontour(
            tri_,
            adata.obs[contour_keys[i]],
            lw=0.2,
            alpha=0.2,
            levels=75,
            zorder=3,
            colors="k",
            cmap=None,
            ls="-",
        )
        ax[i].tricontour(
            tri_,
            adata.obs[contour_keys[i]],
            lw=1,
            alpha=1,
            levels=15,
            zorder=3,
            colors="k",
            cmap=None,
            ls="-",
        )
        if cbar:
            plt.colorbar(sc, aspect=20, pad=0.01, orientation="vertical").set_label(
                contour_key, fontsize=20
            )
        ax[i].set_title(titles[i], fontsize=fontsize)
        if cluster_key != None:
            if cluster_key in adata.obs.keys():
                cluster = adata.obs[cluster_key]
                for c in np.unique(cluster):
                    # plt.scatter(data_pos[cluster == c,0],data_pos[cluster == c,1],zorder=1,alpha=0.1,s=100)
                    txt = ax[i].text(
                        np.mean(data_pos[cluster == c], axis=0)[0],
                        np.mean(data_pos[cluster == c], axis=0)[1],
                        c,
                        fontsize=20,
                        ha="center",
                        va="center",
                        fontweight="bold",
                        zorder=20,
                    )
                    txt.set_path_effects(
                        [matplotlib.patheffects.withStroke(linewidth=5, foreground="w")]
                    )
        else:
            print('There is no cluster key "%s" in adata.obs' % cluster_key)
    if save:
        filename = (
            "%s" % (save_filename)
            if save_dir == None
            else "%s/%s" % (save_dir, save_filename)
        )
        fig.savefig(filename + ".png", bbox_inches="tight")


def view_quiver(
    adata,
    basis="umap",
    vkey="velocity",
    potential_vkey="potential_velocity",
    rotation_vkey="rotation_velocity",
    cluster_key="clusters",
    alpha=0.3,
    fontsize=18,
    scale=1,
    quiver_rate=0.5,
    **kwargs,
):
    kwargs_arg = _check_arguments(adata, basis=basis)
    basis = kwargs_arg["basis"]
    basis_key = "X_%s" % basis
    vkey_ = "%s_%s" % (vkey, basis)
    pot_vkey_ = "%s_%s" % (potential_vkey, basis)
    rot_vkey_ = "%s_%s" % (rotation_vkey, basis)
    cluster_set = np.unique(adata.obs[cluster_key].values)
    cmap = plt.get_cmap("tab20")
    color = np.zeros(adata.shape[0], dtype=int)
    for j in range(len(cluster_set)):
        idx = adata.obs[cluster_key] == cluster_set[j]
        color[idx] = j
    fig, ax = plt.subplots(1, 3, figsize=(24, 6), tight_layout=True)
    for i in range(3):
        for j in range(len(cluster_set)):
            idx = adata.obs[cluster_key] == cluster_set[j]
            ax[i].scatter(
                adata.obsm[basis_key][idx, 0],
                adata.obsm[basis_key][idx, 1],
                s=200,
                alpha=alpha,
                label=cluster_set[j],
                color=cmap(j),
                zorder=0,
            )
            ax[i].text(
                np.mean(adata.obsm[basis_key][idx, 0]),
                np.mean(adata.obsm[basis_key][idx, 1]),
                cluster_set[j],
                fontsize=fontsize,
                ha="center",
                va="center",
                weight="bold",
            )
            txt = ax[i].text(
                np.mean(adata.obsm[basis_key][idx, 0]),
                np.mean(adata.obsm[basis_key][idx, 1]),
                cluster_set[j],
                fontsize=20,
                ha="center",
                va="center",
                fontweight="bold",
                zorder=20,
            )
            txt.set_path_effects([matplotlib.patheffects.withStroke(linewidth=5, foreground="w")])
    idx_qvr_ = np.random.choice(
        np.arange(adata.shape[0]), int(quiver_rate * adata.shape[0]), replace=False
    )
    ax[0].quiver(
        adata.obsm[basis_key][idx_qvr_, 0],
        adata.obsm[basis_key][idx_qvr_, 1],
        adata.obsm[vkey_][idx_qvr_, 0],
        adata.obsm[vkey_][idx_qvr_, 1],
        scale=scale,
        zorder=1,
        **kwargs,
    )
    ax[0].set_title("RNA velocity", fontsize=fontsize)
    ax[0].axis("off")
    ax[1].quiver(
        adata.obsm[basis_key][idx_qvr_, 0],
        adata.obsm[basis_key][idx_qvr_, 1],
        adata.obsm[pot_vkey_][idx_qvr_, 0],
        adata.obsm[pot_vkey_][idx_qvr_, 1],
        scale=scale,
        zorder=1,
        **kwargs,
    )
    ax[1].set_title("Potential flow", fontsize=fontsize)
    ax[1].axis("off")
    ax[2].quiver(
        adata.obsm[basis_key][idx_qvr_, 0],
        adata.obsm[basis_key][idx_qvr_, 1],
        adata.obsm[rot_vkey_][idx_qvr_, 0],
        adata.obsm[rot_vkey_][idx_qvr_, 1],
        scale=scale,
        zorder=1,
        **kwargs,
    )
    ax[2].set_title("Rotational flow", fontsize=fontsize)
    ax[2].axis("off")


def view_surface_genes(
    adata,
    genes,
    exp_key=None,
    basis="umap",
    vkey="velocity",
    potential_key="potential",
    graph_key="CellMap_graph",
    cluster_key=None,
    show_graph=False,
    cutedge_vol=None,
    cutedge_length=None,
    logscale_vel=True,
    **kwargs,
):
    kwargs_arg = _check_arguments(
        adata,
        basis=basis,
        potential_key=potential_key,
        graph_key=graph_key,
    )
    basis = kwargs_arg["basis"]
    basis_key = "X_%s" % basis

    if exp_key == None:
        if scipy.sparse.issparse(adata.X):
            exp_HD = adata.X.toarray()
        else:
            exp_HD = adata.X
    elif exp_key in adata.obsm.keys():
        exp_HD = adata.obsm[exp_key]
    else:
        exp_HD = adata.layers[exp_key]

    vel_HD = adata.obsm[vkey] if vkey in adata.obsm.keys() else adata.layers[vkey]
    if logscale_vel:
        vel_HD = (1e4 * vel_HD.T / adata.obs["n_counts"].values).T / np.power(2, exp_HD)
    if "cmap" not in kwargs:
        kwargs["cmap"] = cmap_earth(adata.obs[potential_key])

    data_pos = adata.obsm[basis_key]
    tri_ = create_graph(
        data_pos,
        cutedge_vol=cutedge_vol,
        cutedge_length=cutedge_length,
        return_type="triangles",
    )[0]

    for gene in genes:
        fig, ax = plt.subplots(1, 3, figsize=(45, 10))
        cntr = ax[0].tricontourf(
            tri_,
            np.squeeze(exp_HD[:, adata.var.index == gene]),
            cmap=kwargs["cmap"],
            levels=100,
            zorder=2,
        )
        fig.colorbar(cntr, shrink=0.75, orientation="vertical", ax=ax[0]).set_label(
            "gene expression", fontsize=20
        )
        ax[0].set_title("%s_expression" % gene, fontsize=18)
        cntr = ax[1].tricontourf(
            tri_,
            np.squeeze(vel_HD[:, adata.var.index == gene]),
            cmap=kwargs["cmap"],
            levels=100,
            zorder=2,
        )
        fig.colorbar(cntr, shrink=0.75, orientation="vertical", ax=ax[1]).set_label(
            "RNA velocity", fontsize=20
        )
        ax[1].set_title("%s_potential" % gene, fontsize=18)
        cntr = ax[2].tricontourf(
            tri_,
            adata.obs["%s_Gene_%s" % (potential_key, gene)],
            cmap=kwargs["cmap"],
            levels=100,
            zorder=2,
        )
        fig.colorbar(cntr, shrink=0.75, orientation="vertical", ax=ax[2]).set_label(
            potential_key, fontsize=20
        )
        ax[2].set_title("%s_potential" % gene, fontsize=18)
        for ax_i in range(3):
            if show_graph:
                ax[ax_i].triplot(tri_, color="w", lw=0.5, zorder=10, alpha=1)
            ax[ax_i].set_xlim(
                np.min(data_pos[:, 0])
                - 0.02 * (np.max(data_pos[:, 0]) - np.min(data_pos[:, 0])),
                np.max(data_pos[:, 0])
                + 0.02 * (np.max(data_pos[:, 0]) - np.min(data_pos[:, 0])),
            )
            ax[ax_i].set_ylim(
                np.min(data_pos[:, 1])
                - 0.02 * (np.max(data_pos[:, 1]) - np.min(data_pos[:, 1])),
                np.max(data_pos[:, 1])
                + 0.02 * (np.max(data_pos[:, 1]) - np.min(data_pos[:, 1])),
            )
            ax[ax_i].tick_params(
                labelbottom=False,
                labelleft=False,
                labelright=False,
                labeltop=False,
                bottom=False,
                left=False,
                right=False,
                top=False,
            )
            ax[ax_i].spines["right"].set_visible(False), ax[ax_i].spines[
                "top"
            ].set_visible(False), ax[ax_i].spines["bottom"].set_visible(False), ax[
                ax_i
            ].spines[
                "left"
            ].set_visible(
                False
            )
            if cluster_key != None:
                texts = []
                if cluster_key in adata.obs.keys():
                    cluster = adata.obs[cluster_key]
                    for c in np.unique(cluster):
                        txt = ax[ax_i].text(
                            np.mean(data_pos[cluster == c], axis=0)[0],
                            np.mean(data_pos[cluster == c], axis=0)[1],
                            c,
                            fontsize=20,
                            ha="center",
                            va="center",
                            fontweight="bold",
                        )
                        txt.set_path_effects(
                            [matplotlib.patheffects.withStroke(linewidth=5, foreground="w")]
                        )
                        texts.append(txt)


def view_3D(
    adata,
    basis="umap",
    potential_key="potential",
    cluster_key="clusters",
    cutedge_vol=None,
    cutedge_length=None,
    show_cells=False,
    show_shadow=True,
    shadow_alpha=0.2,
    title="Landscape",
    bgcolor="white",
    gridcolor="gray",
    seed=None,
    n_points=500,
    width=750,
    height=500,
    annote_fontsize=10,
    scene_aspectratio=dict(x=1.0, y=1.0, z=0.5),
    save=False,
    filename="CellMap_view_3D",
    camera=dict(eye=dict(x=1.2, y=-1.2, z=0.8)),
    **kwargs,
):
    kwargs_arg = _check_arguments(adata, basis=basis, potential_key=potential_key)
    basis = kwargs_arg["basis"]
    basis_key = "X_%s" % basis

    if "cmap" not in kwargs:
        kwargs["cmap"] = cmap_earth(adata.obs[potential_key])

    x, y, z = (
        adata.obsm[basis_key][:, 0],
        adata.obsm[basis_key][:, 1],
        adata.obs[potential_key],
    )

    c_list = [
        "#0938BF",
        "#50D9FB",
        "#B7E5FA",
        "#98D685",
        "#F9EFCD",
        "#E0BB7D",
        "#D3A62D",
        "#997618",
        "#705B10",
        "#5F510D",
        "#A56453",
        "#5C1D09",
    ]
    c_level = [0, 5, 20, 40, 60, 75, 80, 85, 90, 95, 99, 100]
    custom_cmap = [[0.01 * c_level[i], c_list[i]] for i in range(len(c_list))]

    tri_, idx_tri = create_graph(
        adata.obsm[basis_key],
        cutedge_vol=cutedge_vol,
        cutedge_length=cutedge_length,
        return_type="triangles",
    )
    triangles = tri_.triangles[idx_tri]

    idx = np.zeros(adata.shape[0], dtype=bool)
    np.random.seed(seed)
    idx[
        np.random.choice(adata.shape[0], min(n_points, adata.shape[0]), replace=False)
    ] = True
    shift = 0.01 * (max(z) - min(z))
    shadow = go.Mesh3d(
        x=x,
        y=y,
        z=np.zeros(adata.shape[0]) - np.min(z),
        i=triangles[:, 0],
        j=triangles[:, 1],
        k=triangles[:, 2],
        opacity=shadow_alpha,
        color="black",
    )

    if cluster_key in adata.obs.keys():
        clstr = adata.obs[cluster_key]
        clstr_set = np.unique(clstr)
        clstr_id = np.empty(adata.shape[0], dtype=int)
        text = np.array(
            [
                str(clstr[i]) + "<br>Potential: " + str(np.round(z[i], decimals=2))
                for i in range(adata.shape[0])
            ]
        )
        for i in range(len(clstr_set)):
            clstr_id[clstr == clstr_set[i]] = i
        cmap = plt.get_cmap("tab10")
        norm = plt.Normalize(vmin=0, vmax=10)
        color_mapped = cmap(norm(clstr_id[idx]))
        cells = go.Scatter3d(
            x=x[idx],
            y=y[idx],
            z=z[idx] + shift,
            mode="markers",
            marker=dict(size=2.5, color=color_mapped, opacity=1),
            text=text[idx],
            hovertemplate="X: %{x:.2f}<br>Y: %{y:.2f}<br>Z: %{z:.2f}<br>%{text}",
        )

        surf = go.Mesh3d(
            x=x,
            y=y,
            z=z,
            i=triangles[:, 0],
            j=triangles[:, 1],
            k=triangles[:, 2],
            intensity=z,
            colorscale=custom_cmap,
            text=text,
            opacity=1,
            hovertemplate="X: %{x:.2f}<br>Y: %{y:.2f}<br>Z: %{z:.2f}<br>%{text}",
        )

        annotations = [
            dict(
                showarrow=False,
                x=np.percentile(x[clstr == np.unique(clstr)[i]], 50),
                y=np.percentile(y[clstr == np.unique(clstr)[i]], 50),
                z=np.percentile(z[clstr == np.unique(clstr)[i]], 50),
                text="<b>%s<b>" % str(np.unique(clstr)[i]),
                font=dict(size=annote_fontsize, color="rgba(0,0,0,1)"),
                bgcolor="rgba(255,255,255,0.7)",
            )
            for i in range(len(np.unique(clstr)))
        ]
        layout = go.Layout(
            title=title,
            width=width,
            height=height,
            margin=dict(l=0, r=0, b=0, t=50),
            scene_camera=camera,
            scene=dict(
                annotations=annotations,
                xaxis_title=basis + "_1",
                yaxis_title=basis + "_2",
                zaxis_title=potential_key,
                xaxis=dict(backgroundcolor=bgcolor, gridcolor=gridcolor),
                yaxis=dict(backgroundcolor=bgcolor, gridcolor=gridcolor),
                zaxis=dict(backgroundcolor=bgcolor, gridcolor=gridcolor),
            ),
            meta=dict(),
            scene_aspectratio=scene_aspectratio,
        )
        data = [surf]
        if show_cells:
            data.append(cells)
        if show_shadow:
            data.append(shadow)
        fig = go.Figure(data=data, layout=layout)
        #                  )
    else:
        cells = go.Scatter3d(
            x=x[idx],
            y=y[idx],
            z=z[idx] + shift,
            mode="markers",
            marker=dict(
                size=2,
                color="gray",
            ),
        )
        surf = go.Mesh3d(
            x=x,
            y=y,
            z=z,
            i=triangles[:, 0],
            j=triangles[:, 1],
            k=triangles[:, 2],
            intensity=z,
            colorscale=custom_cmap,
            opacity=1,
        )

        layout = go.Layout(
            title=title,
            width=width,
            height=height,
            margin=dict(l=0, r=0, b=0, t=50),
            scene_camera=camera,
            scene=dict(
                xaxis_title=basis + "_1",
                yaxis_title=basis + "_2",
                zaxis_title=potential_key,
                xaxis=dict(backgroundcolor=bgcolor, gridcolor=gridcolor),
                yaxis=dict(backgroundcolor=bgcolor, gridcolor=gridcolor),
                zaxis=dict(backgroundcolor=bgcolor, gridcolor=gridcolor),
            ),
            meta=dict(),
            scene_aspectratio=dict(x=1.0, y=1.0, z=0.5),
        )
        data = [surf]
        if show_cells:
            data.append(cells)
        if show_shadow:
            data.append(shadow)
        fig = go.Figure(data=data, layout=layout)
    fig.show()

    if save:
        plotly.offline.plot(fig, filename=filename + ".html")


def view_surface_3D(
    adata,
    basis="umap",
    potential_key="potential",
    graph_key="CellMap_graph",
    cluster_key=None,
    cutedge_vol=1,
    cutedge_length=1,
    elev=30,
    azim=-60,
    plot_rate=0.3,
    title="",
    **kwargs,
):
    kwargs_arg = _check_arguments(
        adata,
        basis=basis,
        potential_key=potential_key,
        graph_key=graph_key,
    )
    basis = kwargs_arg["basis"]
    basis_key = "X_%s" % basis

    if "cmap" not in kwargs:
        kwargs["cmap"] = cmap_earth(adata.obs[potential_key])

    data_pos = adata.obsm[basis_key]
    tri_ = create_graph(
        data_pos,
        cutedge_vol=cutedge_vol,
        cutedge_length=cutedge_length,
        return_type="triangles",
    )[0]
    fig = plt.figure(figsize=(15, 15))
    ax = fig.add_subplot(111, projection="3d")
    cntr = ax.plot_trisurf(
        tri_, adata.obs[potential_key], cmap=kwargs["cmap"], zorder=2
    )
    ax.set_box_aspect(aspect=(1, 1, 0.8))
    fig.colorbar(cntr, shrink=0.5, orientation="vertical").set_label(
        potential_key, fontsize=20
    )
    ax.set_title(title, fontsize=18)
    if cluster_key != None:
        texts = []
        if cluster_key in adata.obs.keys():
            cluster = adata.obs[cluster_key]
            for c in np.unique(cluster):
                txt = ax.text(
                    np.mean(data_pos[cluster == c], axis=0)[0],
                    np.mean(data_pos[cluster == c], axis=0)[1],
                    np.mean(adata.obs[potential_key][cluster == c]),
                    c,
                    fontsize=15,
                    ha="center",
                    va="center",
                    fontweight="bold",
                    zorder=1000,
                )
                txt.set_path_effects(
                    [matplotlib.patheffects.withStroke(linewidth=5, foreground="w")]
                )
                texts.append(txt)
    ax.view_init(elev=elev, azim=azim)


def view_surface_3D_cluster(
    adata,
    basis="umap",
    potential_key="potential",
    graph_key="CellMap_graph",
    cluster_key="clusters",
    cutedge_vol=1,
    cutedge_length=1,
    elev=30,
    azim=-60,
    seed=None,
    n_points=500,
    title="",
    **kwargs,
):
    if cluster_key in adata.obs.keys():
        kwargs_arg = _check_arguments(
            adata,
            basis=basis,
            potential_key=potential_key,
            graph_key=graph_key,
        )
        basis = kwargs_arg["basis"]
        basis_key = "X_%s" % basis

        data_pos = adata.obsm[basis_key]
        tri_ = create_graph(
            data_pos,
            cutedge_vol=cutedge_vol,
            cutedge_length=cutedge_length,
            return_type="triangles",
        )[0]
        if "cmap" not in kwargs:
            kwargs["cmap"] = cmap_earth(adata.obs[potential_key])
        fig = plt.figure(figsize=(15, 15))
        ax = fig.add_subplot(111, projection="3d")
        cntr = ax.plot_trisurf(
            tri_, adata.obs[potential_key], cmap=kwargs["cmap"], zorder=2, alpha=0.9
        )  # ,cmap=cmap_CellMap,levels=100)
        ax.set_box_aspect(aspect=(1, 1, 0.8))
        ax.set_title(title, fontsize=18)
        fig.colorbar(cntr, shrink=0.5, orientation="vertical").set_label(
            potential_key, fontsize=20
        )
        cluster = adata.obs[cluster_key]
        idx = np.zeros(cluster.shape, dtype=bool)
        np.random.seed(seed)
        idx[np.random.choice(len(idx), min(n_points, len(idx)), replace=False)] = True
        cluster_set = np.unique(cluster)
        z_shift = 0.05 * np.abs(
            np.max(adata.obs[potential_key]) - np.min(adata.obs[potential_key])
        )
        if len(cluster_set) <= 10:
            cmap_pt = plt.get_cmap("tab10")
            vmin, vmax = 0, 10
        else:
            cmap_pt = plt.get_cmap("tab20")
            vmin, vmax = 0, 20
        id_color = np.empty(len(cluster), dtype=int)
        for i in range(len(cluster_set)):
            id_color[cluster == cluster_set[i]] = i
            txt = ax.text(
                np.mean(data_pos[cluster == cluster_set[i]], axis=0)[0],
                np.mean(data_pos[cluster == cluster_set[i]], axis=0)[1],
                np.max(adata.obs[potential_key][cluster == cluster_set[i]]),
                cluster_set[i],
                color=cmap_pt(i),
                fontsize=20,
                ha="center",
                va="center",
                fontweight="bold",
                zorder=1000,
            )
            txt.set_path_effects([matplotlib.patheffects.withStroke(linewidth=5, foreground="w")])
        kwargs["cmap"] = cmap_pt
        ax.scatter(
            data_pos[idx, 0],
            data_pos[idx, 1],
            adata.obs[potential_key][idx] + z_shift,
            c=id_color[idx],
            zorder=100,
            alpha=1,
            edgecolor="w",
            vmin=vmin,
            vmax=vmax,
            **kwargs,
        )
        ax.scatter(
            data_pos[idx, 0],
            data_pos[idx, 1],
            adata.obs[potential_key][idx] + z_shift * 0.5,
            color="k",
            zorder=10,
            alpha=0.1,
            vmin=vmin,
            vmax=vmax,
            **kwargs,
        )
        ax.view_init(elev=elev, azim=azim)
    else:
        print('There is no cluster key "%s" in adata.obs' % cluster_key)


def write(
    adata,
    filename="CellMap",
    basis="umap",
    vkey="velocity",
    exp_key=None,
    potential_key="potential",
    rotation_key="rotation",
    vorticity_key="vorticity",
    streamfunc_key="streamfunc",
    cluster_key="clusters",
    obs_key=None,
    genes=None,
    use_HVG=True,
    n_HVG=10,
):
    kwargs = _check_arguments(
        adata,
        basis=basis,
        potential_key=potential_key,
        cluster_key=cluster_key,
        obs_key=obs_key,
        genes=genes,
        expression_key=exp_key,
    )
    basis, obs_key, genes = kwargs["basis"], kwargs["obs_key"], kwargs["genes"]
    basis_key = "X_%s" % basis
    vkey_ = "%s_%s" % (vkey, basis)
    pot_key_ = "%s" % (potential_key)
    rot_key_ = "%s" % (rotation_key)
    pot_vkey_ = "%s_%s_%s" % (potential_key, vkey, basis)
    rot_vkey_ = "%s_%s_%s" % (rotation_key, vkey, basis)
    vol_key_ = "%s_%s_%s" % (potential_key, vorticity_key, basis)
    sl_key_ = "%s_%s_%s" % (potential_key, streamfunc_key, basis)
    pot_vol_key_ = "%s_%s_%s" % (potential_key, vorticity_key, basis)
    pot_sl_key_ = "%s_%s_%s" % (potential_key, streamfunc_key, basis)
    rot_vol_key_ = "%s_%s_%s" % (rotation_key, vorticity_key, basis)
    rot_sl_key_ = "%s_%s_%s" % (rotation_key, streamfunc_key, basis)

    data_exp = _set_expression_data(adata, exp_key)

    pd_out = pd.DataFrame(
        {
            "X": adata.obsm[basis_key][:, 0],
            "Y": adata.obsm[basis_key][:, 1],
            "Potential": adata.obs[pot_key_],
            "Annotation": adata.obs[cluster_key],
            "Rotation": adata.obs[rot_key_],
            "Streamline_Original": adata.obs[sl_key_],
            "Streamline_Potential": adata.obs[pot_sl_key_],
            "Streamline_Rotation": adata.obs[rot_sl_key_],
            "Vorticity_Original": adata.obs[vol_key_],
            "Vorticity_Potential": adata.obs[pot_vol_key_],
            "Vorticity_Rotation": adata.obs[rot_vol_key_],
            "Velocity_x": adata.obsm[vkey_][:, 0],
            "Velocity_y": adata.obsm[vkey_][:, 1],
            "Velocity_Potential_x": adata.obsm[pot_vkey_][:, 0],
            "Velocity_Potential_y": adata.obsm[pot_vkey_][:, 1],
            "Velocity_Rotation_x": adata.obsm[rot_vkey_][:, 0],
            "Velocity_Rotation_y": adata.obsm[rot_vkey_][:, 1],
        },
        index=adata.obs.index,
    )
    pd_out.index.name = "CellID"

    if obs_key != None:
        for arg in obs_key:
            pd_out.insert(len(pd_out.columns), arg, adata.obs[arg])

    if genes != None:
        for gene in genes:
            pd_out.insert(
                len(pd_out.columns), gene, data_exp[:, adata.var.index == gene]
            )

    if use_HVG:
        scanpy.pp.highly_variable_genes(adata)
        min_mean = np.percentile(
            np.mean(data_exp, axis=0)[np.mean(data_exp, axis=0) > 0],
            90,
        )
        idx_means = adata.var["means"] > min_mean
        for gene in adata.var.index[idx_means][
            np.argsort(adata.var["dispersions_norm"].values[idx_means])[::-1][:n_HVG]
        ]:
            pd_out.insert(
                len(pd_out.columns), "HVG_" + gene, data_exp[:, adata.var.index == gene]
            )

    print('succeeded in writing CellMapp data as "%s.csv"' % filename)
    print(
        "You can visualize the CDV file by CellMapp viewer https://yusuke-imoto-lab.github.io/CellMapViewer/CellMapViewer/viewer.html"
    )

    display(pd_out)

    pd_out.to_csv("%s.csv" % filename)


def view_pseudotime(
    adata,
    basis="umap",
    pseudotime_key = "pseudotime",
    figsize=(10, 6),
    cbar=True,
    cmap="jet",
    color_key = "Pseudo-time",
    save=False,
    save_dir=None,
    save_filename="CellMap_pseudotime",
):

    basis_key = "X_%s" % basis

    data_pos = adata.obsm[basis_key]

    fig, ax = plt.subplots(figsize=figsize)
    ax.scatter(data_pos[:,0],data_pos[:,1],zorder=0,color="gray",alpha=0.3,s=5)
    sc = ax.scatter(data_pos[:,0],data_pos[:,1],c=adata.uns[pseudotime_key],zorder=1,cmap=cmap,s=10)
    ax.axis("off")
    if cbar:
        plt.colorbar(sc, aspect=20, pad=0.01, orientation="vertical").set_label(
            color_key, fontsize=20
        )
    if save:
        filename = (
            "%s" % (save_filename)
            if save_dir == None
            else "%s/%s" % (save_dir, save_filename)
        )
        fig.savefig(filename + ".png", bbox_inches="tight")

def create_dgraph_potential(
    adata,
    basis="umap",
    map_key=None,
    potential_key="potential",
    cutedge_vol=None,
    cutedge_length=None,
):
    """
    Hodge decomposition

    Parameters
    ----------
    adata: anndata (n_samples, n_features)

    basis: ndarray or string
    """

    kwargs_arg = _check_arguments(adata, basis=basis, map_key=map_key)
    basis, map_key = kwargs_arg["basis"], kwargs_arg["map_key"]
    basis_key = "X_%s" % basis

    data_pos = adata.obsm[basis_key]
    tri_, idx_tri = create_graph(
        data_pos,
        cutedge_vol=cutedge_vol,
        cutedge_length=cutedge_length,
        return_mask=True,
    )
    triangles = tri_.triangles[idx_tri]
    n_node_ = data_pos.shape[0]
    graph_ = scipy.sparse.lil_matrix(np.zeros([n_node_, n_node_]))
    idx_set = [[0, 1], [1, 2], [2, 0]]
    # idx = np.isnan(data_vel[0])==False
    for id_x, id_y in idx_set:
        weight = (
            adata.obs[potential_key][triangles[:, id_y]].values
            - adata.obs[potential_key][triangles[:, id_x]].values
        )
        min_weight = np.percentile(np.abs(weight), 5)
        graph_[tuple(triangles[weight > min_weight][:, [id_x, id_y]].T[::-1])] = 1
        graph_[tuple(triangles[weight < -min_weight][:, [id_y, id_x]].T[::-1])] = 1
    return scipy.sparse.coo_matrix(graph_)


def create_dgraph(
    adata,
    basis="umap",
    vkey="velocity",
    cutedge_vol=None,
    cutedge_length=None,
):
    """
    Hodge decomposition

    Parameters
    ----------
    adata: anndata (n_samples, n_features)

    basis: ndarray or string
    """

    kwargs_arg = _check_arguments(adata, basis=basis, vkey=vkey, map_key=map_key)
    basis, vkey, map_key = (
        kwargs_arg["basis"],
        kwargs_arg["vkey"],
        kwargs_arg["map_key"],
    )
    basis_key = "X_%s" % basis

    data_pos = adata.obsm[basis_key]
    data_vel = adata.obsm["%s_%s" % (vkey, basis)]
    tri_, idx_tri = create_graph(
        data_pos,
        cutedge_vol=cutedge_vol,
        cutedge_length=cutedge_length,
        return_mask=True,
    )
    triangles = tri_.triangles[idx_tri]
    n_node_ = data_pos.shape[0]
    graph_ = scipy.sparse.lil_matrix(np.zeros([n_node_, n_node_]))
    idx_set = [[0, 1], [1, 2], [2, 0]]
    idx = np.isnan(data_vel[0]) == False
    for id_x, id_y in idx_set:
        X1 = data_pos[:, idx][triangles[:, id_x]]
        X2 = data_pos[:, idx][triangles[:, id_y]]
        V1 = data_vel[:, idx][triangles[:, id_x]]
        V2 = data_vel[:, idx][triangles[:, id_y]]
        weight = np.sum(0.5 * (V1 + V2) * (X2 - X1), axis=1)
        min_weight = np.percentile(np.abs(weight), 5)
        graph_[tuple(triangles[weight > min_weight][:, [id_x, id_y]].T[::-1])] = 1
        graph_[tuple(triangles[weight < -min_weight][:, [id_y, id_x]].T[::-1])] = 1
    return scipy.sparse.coo_matrix(graph_)


def calc_path(
    adata,
    source_cluster,
    target_clusters,
    n_cells=50,
    weight_rate=0.9,
    basis="umap",
    potential_key="potential",
    cluster_key="clusters",
    streamfunc_key="streamfunc",
    graph_method="Delauney",
    pseudotime_key = "pseudotime",
    path_key="path",
    n_neighbors=30,
    contribution_rate_pca=0.95,
    cutedge_vol=None,
    cutedge_length=None,
):
    if sum(adata.obs[cluster_key].values == source_cluster) == 0:
        raise KeyError("Cluster %s was not found" % source_cluster)
    for trg_ in target_clusters:
        if sum(adata.obs[cluster_key].values == source_cluster) == 0:
            raise KeyError("Cluster %s was not found" % trg_)

    basis_key = "X_%s" % basis
    pot_sl_key_ = "%s_%s_%s" % (potential_key, streamfunc_key, basis)

    data_pos = adata.obsm[basis_key]
    streamfunc_ = scipy.stats.zscore(adata.obs[pot_sl_key_])

    ## Compute graph and edge velocities
    if graph_method == "Delauney":
        source, target = create_graph(
            data_pos,
            cutedge_vol=cutedge_vol,
            cutedge_length=cutedge_length,
            return_type="edges",
        )
    elif graph_method == "knn":
        pca = sklearn.decomposition.PCA()
        exp_HD_pca = pca.fit_transform(data_pos)
        n_pca = np.min(
            np.arange(len(pca.explained_variance_ratio_))[
                np.cumsum(pca.explained_variance_ratio_) > contribution_rate_pca
            ]
        )
        knn = sklearn.neighbors.NearestNeighbors(
            n_neighbors=n_neighbors + 1, algorithm="kd_tree"
        )
        knn.fit(exp_HD_pca[:, :n_pca])
        distances, indices = knn.kneighbors(exp_HD_pca[:, :n_pca])
        distances, indices = distances[:, 1:], indices[:, 1:]
        source = np.ravel(
            np.repeat(
                np.arange(data_pos.shape[0]).reshape((-1, 1)), n_neighbors, axis=1
            )
        )
        target = np.ravel(indices)

    G = nx.DiGraph()
    G.add_weighted_edges_from(
        [(int(s), int(t), 1) for s, t in np.vstack((source, target)).T]
    )
    G.add_weighted_edges_from(
        [(int(t), int(s), -1) for s, t in np.vstack((source, target)).T]
    )
    edges_ = np.array(list(G.edges))
    weights_ = np.array([G[u][v]["weight"] for u, v in edges_])

    dis_mean = np.mean(np.linalg.norm(data_pos[source] - data_pos[target], axis=1))
    cmap_ = plt.get_cmap("tab10")
    # fig, ax = plt.subplots(figsize=figsize)
    # tri_ = create_graph(
    #     data_pos,
    #     cutedge_vol=cutedge_vol,
    #     cutedge_length=cutedge_length,
    #     return_type="triangles",
    # )[0]
    # ax.triplot(tri_, color="gray", zorder=0, alpha=0.2, lw=1)
    # clusters_ = adata.obs[cluster_key]
    # idx_ = clusters_ == source_cluster
    # ax.scatter(
    #     data_pos[idx_, 0],
    #     data_pos[idx_, 1],
    #     color="gray",
    #     zorder=10,
    #     marker="D",
    #     alpha=0.2,
    #     s=5,
    #     label=source_cluster + " (source)",
    # )
    # for i_trg_ in range(len(target_clusters)):
    #     idx_ = clusters_ == target_clusters[i_trg_]
    #     ax.scatter(
    #         data_pos[idx_, 0],
    #         data_pos[idx_, 1],
    #         color=cmap_(i_trg_),
    #         zorder=10,
    #         marker="o",
    #         alpha=0.2,
    #         s=5,
    #         label=target_clusters[i_trg_] + " (target)",
    #     )
    # leg = ax.legend(
    #     bbox_to_anchor=(1.05, 0.5),
    #     loc="center left",
    #     borderaxespad=0,
    #     fontsize=12,
    #     markerscale=3,
    # )
    # for lh in leg.legend_handles:
    #     lh.set_alpha(1)

    data_src_ = data_pos[adata.obs[cluster_key].values == source_cluster]
    center_src_ = np.mean(data_src_, axis=0)
    centrality_src_ = np.linalg.norm(data_src_ - center_src_, axis=1)
    src_set_all_ = np.arange(adata.shape[0])[
        adata.obs[cluster_key].values == source_cluster
    ][np.argsort(centrality_src_)]
    n_src_ = sum(adata.obs[cluster_key].values == source_cluster)
    path_all = {}
    for i_trg_ in range(len(target_clusters)):
        target_cluster = target_clusters[i_trg_]
        n_cells_ = np.min(
            [
                n_cells,
                sum(adata.obs[cluster_key].values == source_cluster),
                sum(adata.obs[cluster_key].values == target_cluster),
            ]
        )
        data_trg_ = data_pos[adata.obs[cluster_key].values == target_cluster]
        center_trg_ = np.mean(data_trg_, axis=0)
        centrality_trg_ = np.linalg.norm(data_trg_ - center_trg_, axis=1)
        n_trg_ = sum(adata.obs[cluster_key].values == target_cluster)
        idx_trg_ = np.arange(0, n_trg_, int(n_trg_ / n_cells_))[:n_cells_]
        trg_set_ = np.arange(adata.shape[0])[
            adata.obs[cluster_key].values == target_cluster
        ][np.argsort(centrality_trg_)][idx_trg_]
        # idx_src_= np.array([np.argmin(np.abs(streamfunc_[trg__] - streamfunc_[src_set_all_].values)) for trg__ in trg_set_])
        idx_src_ = np.arange(0, n_src_, int(n_src_ / n_cells_))[:n_cells_]
        src_set_ = src_set_all_[idx_src_]

        pathes, edges, weights, dists = [], [], [], []
        for src_, trg_ in np.vstack((src_set_, trg_set_)).T:
            weights_i_ = weight_rate * np.abs(
                # streamfunc_[trg_] - 0.5*(streamfunc_[edges_[:, 0]].values+streamfunc_[edges_[:, 1]].values)
                streamfunc_[trg_] - 0.5*streamfunc_[edges_[:, 0]] - 0.5*streamfunc_[edges_[:, 1]]
            ) + (1 - weight_rate) * np.exp(
                np.linalg.norm(data_pos[edges_[:, 0]] - data_pos[edges_[:, 1]], axis=1)
                / dis_mean
            )
            nx.set_edge_attributes(
                G, values=dict(zip(G.edges(), weights_i_)), name="weight"
            )
            path = nx.dijkstra_path(G, source=src_, target=trg_, weight="weight")
            pathes.append(path)
            edges.append(
                np.array([[path[i], path[i + 1]] for i in range(len(path) - 1)])
            )
            weights.append(
                (sum([G[path[i]][path[i + 1]]["weight"] for i in range(len(path) - 1)]))
                / sum(
                    [
                        np.linalg.norm(data_pos[path[i]] - data_pos[path[i + 1]])
                        for i in range(len(path) - 1)
                    ]
                )
            )
            dists.append(
                sum(
                    [
                        np.linalg.norm(data_pos[path[i]] - data_pos[path[i + 1]])
                        for i in range(len(path) - 1)
                    ]
                )
            )
        path_all[source_cluster + "_" + target_clusters[i_trg_]] = pathes

    adata.uns[path_key] = path_all

    counts = np.zeros(adata.shape[0])
    vals = np.zeros(adata.shape[0])
    pseudotime = np.repeat(None,adata.shape[0])
    for k_ in adata.uns[path_key].keys():
        for i_ in adata.uns[path_key][k_]:
            counts[i_] += 1
            vals[i_] += np.linspace(0,1,len(i_))
    pseudotime[counts>0] = vals[counts>0]/counts[counts>0]
    adata.uns[pseudotime_key] = pseudotime

def view_trajectory(
    adata,
    source_cluster,
    target_clusters,
    n_cells=50,
    weight_rate=0.9,
    basis="umap",
    potential_key="potential",
    cluster_key="clusters",
    streamfunc_key="streamfunc",
    graph_method="Delauney",
    path_key="path",
    n_neighbors=30,
    contribution_rate_pca=0.95,
    cutedge_vol=None,
    cutedge_length=None,
    figsize=(10, 8),
    save=False,
    save_dir=None,
    save_filename="Trajectory",
):
    kwargs_arg = _check_arguments(adata, verbose=True, basis=basis)
    basis = kwargs_arg["basis"]

    if path_key not in adata.uns.keys():
        calc_path(
            adata,
            source_cluster,
            target_clusters,
            n_cells=n_cells,
            weight_rate=weight_rate,
            basis=basis,
            potential_key="potential",
            cluster_key="clusters",
            streamfunc_key="streamfunc",
            graph_method="Delauney",
            path_key="path",
            n_neighbors=30,
            contribution_rate_pca=0.95,
            cutedge_vol=None,
            cutedge_length=None,
        )

    if sum(adata.obs[cluster_key].values == source_cluster) == 0:
        raise KeyError("Cluster %s was not found" % source_cluster)
    for trg_ in target_clusters:
        if sum(adata.obs[cluster_key].values == source_cluster) == 0:
            raise KeyError("Cluster %s was not found" % trg_)

    basis_key = "X_%s" % basis
    pot_sl_key_ = "%s_%s_%s" % (potential_key, streamfunc_key, basis)

    data_pos = adata.obsm[basis_key]
    # streamfunc_ = scipy.stats.zscore(adata.obs[pot_sl_key_])

    ## Compute graph and edge velocities
    if graph_method == "Delauney":
        source, target = create_graph(
            data_pos,
            cutedge_vol=cutedge_vol,
            cutedge_length=cutedge_length,
            return_type="edges",
        )
    elif graph_method == "knn":
        pca = sklearn.decomposition.PCA()
        exp_HD_pca = pca.fit_transform(data_pos)
        n_pca = np.min(
            np.arange(len(pca.explained_variance_ratio_))[
                np.cumsum(pca.explained_variance_ratio_) > contribution_rate_pca
            ]
        )
        knn = sklearn.neighbors.NearestNeighbors(
            n_neighbors=n_neighbors + 1, algorithm="kd_tree"
        )
        knn.fit(exp_HD_pca[:, :n_pca])
        distances, indices = knn.kneighbors(exp_HD_pca[:, :n_pca])
        distances, indices = distances[:, 1:], indices[:, 1:]
        source = np.ravel(
            np.repeat(
                np.arange(data_pos.shape[0]).reshape((-1, 1)), n_neighbors, axis=1
            )
        )
        target = np.ravel(indices)

    G = nx.DiGraph()
    G.add_weighted_edges_from(
        [(int(s), int(t), 1) for s, t in np.vstack((source, target)).T]
    )
    G.add_weighted_edges_from(
        [(int(t), int(s), -1) for s, t in np.vstack((source, target)).T]
    )
    edges_ = np.array(list(G.edges))
    weights_ = np.array([G[u][v]["weight"] for u, v in edges_])

    dis_mean = np.mean(np.linalg.norm(data_pos[source] - data_pos[target], axis=1))
    cmap_ = plt.get_cmap("tab10")
    fig, ax = plt.subplots(figsize=figsize)
    tri_ = create_graph(
        data_pos,
        cutedge_vol=cutedge_vol,
        cutedge_length=cutedge_length,
        return_type="triangles",
    )[0]
    ax.triplot(tri_, color="gray", zorder=0, alpha=0.2, lw=1)
    clusters_ = adata.obs[cluster_key]
    idx_ = clusters_ == source_cluster
    ax.scatter(
        data_pos[idx_, 0],
        data_pos[idx_, 1],
        color="gray",
        zorder=10,
        marker="D",
        alpha=0.2,
        s=5,
        label=source_cluster + " (source)",
    )
    for i_trg_ in range(len(target_clusters)):
        idx_ = clusters_ == target_clusters[i_trg_]
        ax.scatter(
            data_pos[idx_, 0],
            data_pos[idx_, 1],
            color=cmap_(i_trg_),
            zorder=10,
            marker="o",
            alpha=0.2,
            s=5,
            label=target_clusters[i_trg_] + " (target)",
        )
    leg = ax.legend(
        bbox_to_anchor=(1.05, 0.5),
        loc="center left",
        borderaxespad=0,
        fontsize=12,
        markerscale=3,
    )
    for lh in leg.legend_handles:
        lh.set_alpha(1)

    data_src_ = data_pos[adata.obs[cluster_key].values == source_cluster]
    center_src_ = np.mean(data_src_, axis=0)
    centrality_src_ = np.linalg.norm(data_src_ - center_src_, axis=1)
    src_set_all_ = np.arange(adata.shape[0])[
        adata.obs[cluster_key].values == source_cluster
    ][np.argsort(centrality_src_)]
    n_src_ = sum(adata.obs[cluster_key].values == source_cluster)
    # path_all = {}
    for i_trg_ in range(len(target_clusters)):
        target_cluster = target_clusters[i_trg_]
        n_cells_ = np.min(
            [
                n_cells,
                sum(adata.obs[cluster_key].values == source_cluster),
                sum(adata.obs[cluster_key].values == target_cluster),
            ]
        )
        data_trg_ = data_pos[adata.obs[cluster_key].values == target_cluster]
        center_trg_ = np.mean(data_trg_, axis=0)
        centrality_trg_ = np.linalg.norm(data_trg_ - center_trg_, axis=1)
        n_trg_ = sum(adata.obs[cluster_key].values == target_cluster)
        idx_trg_ = np.arange(0, n_trg_, int(n_trg_ / n_cells_))[:n_cells_]
        trg_set_ = np.arange(adata.shape[0])[
            adata.obs[cluster_key].values == target_cluster
        ][np.argsort(centrality_trg_)][idx_trg_]
        # idx_src_= np.array([np.argmin(np.abs(streamfunc_[trg__] - streamfunc_[src_set_all_].values)) for trg__ in trg_set_])
        idx_src_ = np.arange(0, n_src_, int(n_src_ / n_cells_))[:n_cells_]
        src_set_ = src_set_all_[idx_src_]

        # pathes, edges, weights, dists = [], [], [], []
        # for src_, trg_ in np.vstack((src_set_, trg_set_)).T:
        #     weights_i_ = weight_rate * np.abs(
        #         streamfunc_[trg_] - 0.5*(streamfunc_[edges_[:, 0]].values+streamfunc_[edges_[:, 1]].values)
        #         # 0.5*streamfunc_[trg_]+0.5*streamfunc_[src_] - streamfunc_[edges_[:, 1]]
        #     ) + (1 - weight_rate) * np.exp(
        #         np.linalg.norm(data_pos[edges_[:, 0]] - data_pos[edges_[:, 1]], axis=1)
        #         / dis_mean
        #     )
        #     nx.set_edge_attributes(
        #         G, values=dict(zip(G.edges(), weights_i_)), name="weight"
        #     )
        #     path = nx.dijkstra_path(G, source=src_, target=trg_, weight="weight")
        #     pathes.append(path)
        #     edges.append(
        #         np.array([[path[i], path[i + 1]] for i in range(len(path) - 1)])
        #     )
        #     weights.append(
        #         (sum([G[path[i]][path[i + 1]]["weight"] for i in range(len(path) - 1)]))
        #         / sum(
        #             [
        #                 np.linalg.norm(data_pos[path[i]] - data_pos[path[i + 1]])
        #                 for i in range(len(path) - 1)
        #             ]
        #         )
        #     )
        #     dists.append(
        #         sum(
        #             [
        #                 np.linalg.norm(data_pos[path[i]] - data_pos[path[i + 1]])
        #                 for i in range(len(path) - 1)
        #             ]
        #         )
        #     )
        pathes = adata.uns[path_key][source_cluster + "_" + target_clusters[i_trg_]]
        # path_all[source_cluster + "_" + target_clusters[i_trg_]] = pathes
        ax.scatter(
            data_pos[trg_set_, 0],
            data_pos[trg_set_, 1],
            color=cmap_(i_trg_),
            zorder=20,
            marker="o",
            s=30,
        )
        ax.scatter(
            data_pos[src_set_, 0],
            data_pos[src_set_, 1],
            color="gray",
            zorder=20,
            marker="D",
            s=30,
        )
        for i in range(n_cells_):
            ax.plot(
                data_pos[pathes[i], 0],
                data_pos[pathes[i], 1],
                color=cmap_(i_trg_),
                zorder=10,
                ls="-",
                lw=2,
                alpha=0.3,
            )
    ax.axis("off")

    # adata.uns[path_key] = path_all

    if save:
        filename = (
            "%s" % (save_filename)
            if save_dir == None
            else "%s/%s" % (save_dir, save_filename)
        )
        fig.savefig(filename + ".png", bbox_inches="tight")

        filename = (
            "%s_clusters" % (save_filename)
            if save_dir == None
            else "%s/%s" % (save_dir, save_filename)
        )
        fig.savefig(filename + ".png", bbox_inches="tight")

def dynamical_clustering(
    adata,
    target_clusters,
    TO_prob = 1e-10,
    basis="umap",
    cluster_key="clusters",
    streamfunc_key="streamfunc",
    figsize=(10, 8),
    fs_label = 16,
    fs_lagened = 16,
    save=False,
    save_dir=None,
    save_filename="Dynamical_clustering",
):
    
    basis_key = "X_%s" % basis
    sl_key_ = "%s_%s" % (streamfunc_key, basis)
    
    n_clusters_ = len(target_clusters)
    Gaussian_stats = {}
    Gaussian_pdf = np.empty([n_clusters_,adata.shape[0]],dtype=float)
    Gaussian_weight = np.empty(n_clusters_,dtype=float)
    
    for i_trg_ in range(n_clusters_):
        Gaussian_weight[i_trg_] = sum(adata.obs[cluster_key] == target_clusters[i_trg_])
    Gaussian_weight = Gaussian_weight/np.sum(Gaussian_weight)

    for i_trg_ in range(n_clusters_):
        cluster_ = target_clusters[i_trg_]
        idx_ = adata.obs[cluster_key] == cluster_
        Gaussian_stats[cluster_] = {}
        Gaussian_stats["mean"] = np.mean(adata.obs[sl_key_][idx_])
        Gaussian_stats["std"] = np.std(adata.obs[sl_key_][idx_])
        Gaussian_pdf[i_trg_] = Gaussian_weight[i_trg_]*scipy.stats.norm.pdf(adata.obs[sl_key_], Gaussian_stats["mean"], Gaussian_stats["std"])
    
    adata.uns["Dynamical_clustering"] = {"Gaussian_stats":Gaussian_stats,"Gaussian_pdf":Gaussian_pdf}
    
    idx_color_ = np.argmax(Gaussian_pdf,axis=0)
    cmap = plt.get_cmap("tab10")
    idx_color_[np.max(Gaussian_pdf,axis=0)<TO_prob] = -1
    
    pmin_,pmax_ = np.max(adata.obs[sl_key_]),np.min(adata.obs[sl_key_])
    for i_trg_ in range(n_clusters_):
        pmin_ = min(pmin_,np.min(adata.obs[sl_key_][adata.obs[cluster_key] == target_clusters[i_trg_]]))
        pmax_ = max(pmax_,np.max(adata.obs[sl_key_][adata.obs[cluster_key] == target_clusters[i_trg_]]))
    bins = np.linspace(pmin_,pmax_,50)
    x_range = np.linspace(pmin_-0.2*(pmax_-pmin_),pmax_+0.2*(pmax_-pmin_),100)
    fig = plt.figure(figsize=(8,4))
    ax1 = fig.add_subplot(111)
    ax2 = ax1.twinx()
    for i_trg_ in range(n_clusters_):
        cluster_ = target_clusters[i_trg_]
        idx_ = adata.obs[cluster_key] == cluster_
        ax1.hist(adata.obs[sl_key_][idx_],bins=bins,lw=1,edgecolor="k",label=target_clusters[i_trg_])
        pdf_values = Gaussian_weight[i_trg_]*scipy.stats.norm.pdf(x_range, np.mean(adata.obs[sl_key_][idx_]), np.std(adata.obs[sl_key_][idx_]))
        ax2.plot(x_range, pdf_values,lw=5,zorder=0,color="w")
        ax2.plot(x_range, pdf_values,lw=3,zorder=10)
    ax1.set_ylabel('Freaquency',fontsize=fs_label)
    ax1.set_xlabel('Orthogonal potential',fontsize=fs_label)
    ax2.set_ylabel('Probability',fontsize=fs_label)
    ax2.set_ylim([0,ax2.get_ylim()[1]])
    ax1.legend(bbox_to_anchor=(1.1, 1), loc='upper left',frameon=False,fontsize=fs_lagened)
    if save:
        filename = (
            "%s_Gaussian" % (save_filename)
            if save_dir == None
            else "%s/%s" % (save_dir, save_filename)
        )
        fig.savefig(filename + ".png", bbox_inches="tight")
    
    fig,ax = plt.subplots(figsize=figsize)
    ax.scatter(adata.obsm[basis_key][:,0],adata.obsm[basis_key][:,1],s=5,c="gray",alpha=0.1)
    for i in range(max(idx_color_)+1):
        idx_ = idx_color_ == i
        ax.scatter(adata.obsm[basis_key][idx_,0],adata.obsm[basis_key][idx_,1],s=10,c=cmap(i))
    for i_trg_ in range(n_clusters_):
        cluster_ = target_clusters[i_trg_]
        idx_ = adata.obs[cluster_key] == cluster_
        ax.scatter(adata.obsm[basis_key][idx_,0],adata.obsm[basis_key][idx_,1],s=50,c=cmap(i_trg_),edgecolors="k")
        txt = ax.text(np.mean(adata.obsm[basis_key][idx_,0]),np.mean(adata.obsm[basis_key][idx_,1]),target_clusters[i_trg_],fontsize=20,ha="center", va="center",fontweight="bold", zorder=20,)
        txt.set_path_effects([matplotlib.patheffects.withStroke(linewidth=5, foreground="w")])
    ax.axis("off")
    if save:
        filename = (
            "%s_clusters" % (save_filename)
            if save_dir == None
            else "%s/%s" % (save_dir, save_filename)
        )
        fig.savefig(filename + ".png", bbox_inches="tight")

def calc_dynamics(
    adata,
    source_cluster,
    target_clusters,
    path_key="path",
    path_scaled_key="path_scaled",
    exp_key=None,
    dynamics_key="dynamics",
    potential_dynamics_key = "potential_dynamics",
    culc_potential_dynamics_keys = ["potential"],
    n_div=100,
    degree=10,
):
    data_exp = _set_expression_data(adata, exp_key)
    
    path = adata.uns[path_key]
    path_scaled = {}
    for k_ in path.keys():
        path_scaled[k_] = np.array([np.array(p_)[np.linspace(0, len(p_)-1, n_div+1,dtype=int)] for p_ in adata.uns["path"][k_]])
    adata.uns[path_scaled_key] = path_scaled

    gene_dynamics = {"reg":{},"mean":{},"median":{},"std":{},"percentile_75":{},"percentile_25":{}}
    for i in range(len(path_scaled)):
        name_ = source_cluster + "_" + target_clusters[i]
        x_data = np.resize(np.linspace(0, 1, n_div+1),(n_div+1)*path_scaled[name_].shape[0])
        y_data = data_exp[path_scaled[name_]].reshape(path_scaled[name_].size,data_exp.shape[1])

        X = x_data[:, np.newaxis]
        plot_x = np.linspace(0, 1, n_div+1)
        poly = sklearn.preprocessing.PolynomialFeatures(degree=degree)
        model = sklearn.linear_model.LinearRegression()
        model.fit(poly.fit_transform(X), y_data)
        gd_i_ = model.predict(poly.fit_transform(plot_x[:, np.newaxis]))
        
        gd_i_[gd_i_ < 0] = 0
        gene_dynamics["reg"][name_] = gd_i_
        gene_dynamics["mean"][name_] = np.mean(data_exp[path_scaled[name_]],axis=0)
        gene_dynamics["median"][name_] = np.median(data_exp[path_scaled[name_]],axis=0)
        gene_dynamics["std"][name_] = np.std(data_exp[path_scaled[name_]],axis=0)
        gene_dynamics["percentile_75"][name_] = np.percentile(data_exp[path_scaled[name_]],75,axis=0)
        gene_dynamics["percentile_25"][name_] = np.percentile(data_exp[path_scaled[name_]],25,axis=0)
    adata.uns[dynamics_key] = gene_dynamics

    potential_dynamics = {}
    for k_ in culc_potential_dynamics_keys:
        potential_dynamics[k_] = {"reg":{},"mean":{},"median":{},"std":{},"percentile_75":{},"percentile_25":{}}
        for i in range(len(path_scaled)):
            name_ = source_cluster + "_" + target_clusters[i]
            x_data = np.resize(np.linspace(0, 1, n_div+1),(n_div+1)*path_scaled[name_].shape[0])
            y_data_all = adata.obs[[k_]].values
            y_data = y_data_all[path_scaled[name_]].reshape(path_scaled[name_].size,y_data_all.shape[1])

            X = x_data[:, np.newaxis]
            plot_x = np.linspace(0, 1, n_div+1)
            poly = sklearn.preprocessing.PolynomialFeatures(degree=degree)
            model = sklearn.linear_model.LinearRegression()
            model.fit(poly.fit_transform(X), y_data)
            gd_i_ = model.predict(poly.fit_transform(plot_x[:, np.newaxis]))
            
            gd_i_[gd_i_ < 0] = 0
            potential_dynamics[k_]["reg"][name_] = gd_i_[:,0]
            potential_dynamics[k_]["mean"][name_] = np.mean(y_data_all[path_scaled[name_]],axis=0)[:,0]
            potential_dynamics[k_]["median"][name_] = np.median(y_data_all[path_scaled[name_]],axis=0)[:,0]
            potential_dynamics[k_]["std"][name_] = np.std(y_data_all[path_scaled[name_]],axis=0)[:,0]
            potential_dynamics[k_]["percentile_75"][name_] = np.percentile(y_data_all[path_scaled[name_]],75,axis=0)[:,0]
            potential_dynamics[k_]["percentile_25"][name_] = np.percentile(y_data_all[path_scaled[name_]],25,axis=0)[:,0]
    adata.uns[potential_dynamics_key] = potential_dynamics

    print("Done the computation of dynamics")

# def calc_dynamics(
#     adata,
#     source_cluster,
#     target_clusters,
#     path_key="path",
#     path_scaled_key="path_scaled",
#     exp_key=None,
#     dynamics_key="dynamics",
#     n_div=100,
#     degree=10,
# ):
#     data_exp = _set_expression_data(adata, exp_key)
    
#     path = adata.uns[path_key]

#     path_scaled = {}
#     for k_ in path.keys():
#         path_scaled[k_] = np.array([np.array(p_)[np.linspace(0, len(p_)-1, n_div,dtype=int)] for p_ in adata.uns["path"][k_]])
#     adata.uns[path_scaled_key] = path_scaled

#     gene_dynamics_ = {}
#     for i in range(len(path_scaled)):
#         name_ = source_cluster + "_" + target_clusters[i]
#         x_data, y_data = np.empty(0, dtype=float), np.empty(
#             [0, adata.shape[1]], dtype=float
#         )
#         for pi in path_scaled[name_]:
#             x_data = np.append(x_data, np.linspace(0, 1, len(pi)))
#             y_data = np.vstack((y_data, data_exp[pi]))

#         X = x_data[:, np.newaxis]
#         poly = sklearn.preprocessing.PolynomialFeatures(degree=degree)
#         X_poly = poly.fit_transform(X)
#         model = sklearn.linear_model.LinearRegression()
#         model.fit(X_poly, y_data)
#         plot_x = np.linspace(0, 1, n_div + 1)
#         gd_i_ = model.predict(poly.fit_transform(plot_x[:, np.newaxis]))
#         gd_i_[gd_i_ < 0] = 0
#         gene_dynamics_[source_cluster + "_" + target_clusters[i]] = gd_i_
#     adata.uns[dynamics_key] = gene_dynamics_
#     print("Done the computation of gene dynamics")

def gene_dynamics_plot(
    adata,
    source_cluster,
    target_clusters,
    genes,
    path_key="path",
    path_scaled_key="path_scaled",
    exp_key=None,
    dynamics_key="dynamics",
    n_div=100,
    plot_center = "median",
    plot_errorbar = "percentile",
    smoothing = True,
    smoothing_sigma= 3,
    errbar_alpha = 0.15,
    figsize=(10, 4),
    fontsize_title=16,
    fontsize_label=14,
    fontsize_legend=12,
    legend = True,
    save=False,
    save_dir=None,
    save_filename="dynamics",
):

    if dynamics_key not in adata.uns.keys():
        calc_dynamics(
            adata,
            source_cluster,
            target_clusters,
            path_key=path_key,
            exp_key=exp_key,
            dynamics_key=dynamics_key,
            n_div=n_div,
        )

    cmap_ = plt.get_cmap("tab10")

    x_data = np.linspace(0, 1, n_div+1)
    y_data = adata.uns[dynamics_key][plot_center]
    if plot_errorbar == "percentile":
        err_top = adata.uns[dynamics_key]["percentile_75"]
        err_bottom = adata.uns[dynamics_key]["percentile_25"]
    if plot_errorbar == "std":
        err_top = adata.uns[dynamics_key]["mean"] + adata.uns[dynamics_key]["std"]
        err_bottom = adata.uns[dynamics_key]["mean"] - adata.uns[dynamics_key]["std"]

    for gene in genes:
        if gene in adata.var.index:
            fig,ax = plt.subplots(figsize=figsize)
            y_data_all = []
            for i in range(len(target_clusters)):
                name_ = source_cluster + "_" + target_clusters[i]
                y_,et_,eb_ = y_data[name_][:, adata.var.index == gene][:,0],err_top[name_][:, adata.var.index == gene][:,0],err_bottom[name_][:, adata.var.index == gene][:,0]
                if smoothing:
                    y_, et_, eb_ = [scipy.ndimage.gaussian_filter1d(arr, sigma=smoothing_sigma) for arr in (y_, et_, eb_)]
                ax.plot(x_data,y_, color=cmap_(i),label=target_clusters[i],lw=4,zorder=2)
                ax.plot(x_data,y_, color="w",lw=6,zorder=1)
                ax.fill_between(x_data,et_, eb_, color=cmap_(i), alpha=errbar_alpha,zorder=0)
                y_data_all = np.append(y_data_all, et_)
            ax.set_xlim(0,1)
            ax.set_ylim(0,np.max(y_data_all))
            if legend:
                ax.legend(
                    bbox_to_anchor=(1.05, 0.5),
                    loc="center left",
                    borderaxespad=0,
                    title="Target",
                    fontsize=fontsize_legend,
                    title_fontsize=fontsize_legend,
                )
            ax.set_xticks(
                [0, 0.25, 0.5, 0.75, 1],
                [
                    "Source (0)\n(%s)" % source_cluster,
                    "0.25",
                    "0.5",
                    "0.75",
                    "Target (1)",
                ],
                fontsize=fontsize_label,
            )
            ax.set_xticks(
                [0, 0.1, 0.2, 0.3, 0.4,0.5,0.6,0.7,0.8,0.9,1],
                [
                    "Source (0)\n(%s)" % source_cluster,
                    "0.1","0.2","0.3","0.4","0.5","0.6","0.7","0.8","0.9",
                    "Target (1)",
                ],
                fontsize=fontsize_label,
            )
            ax.set_yticks(ax.get_yticks(),ax.get_yticks(),fontsize=fontsize_label)
            ax.set_title(gene, fontsize=fontsize_title)
            ax.grid(ls="--",zorder=-1)
            plt.show()
            if save:
                filename = (
                    "%s_%s" % (save_filename, gene)
                    if save_dir == None
                    else "%s/%s_%s" % (save_dir, save_filename, gene)
                )
                fig.savefig(filename + ".png", bbox_inches="tight")
            plt.close()
        else:
            print('Gene "%s" was not found' % gene)



# def gene_dynamics_plot(
#     adata,
#     source_cluster,
#     target_clusters,
#     genes,
#     path_key="path",
#     exp_key=None,
#     dynamics_key="dynamics",
#     n_div=100,
#     figsize=(10, 4),
#     fontsize_title=16,
#     fontsize_label=14,
#     fontsize_legend=12,
#     legend = True,
#     save=False,
#     save_dir=None,
#     save_filename="dynamics",
# ):

#     if dynamics_key not in adata.uns.keys():
#         calc_dynamics(
#             adata,
#             source_cluster,
#             target_clusters,
#             path_key=path_key,
#             exp_key=exp_key,
#             dynamics_key=dynamics_key,
#             n_div=n_div,
#         )

#     data_exp = _set_expression_data(adata, exp_key)
#     path = adata.uns[path_key]
#     cmap_ = plt.get_cmap("tab10")

#     for gene in genes:
#         if gene in adata.var.index:
#             fig = plt.figure(figsize=figsize)
#             y_data_all = []
#             for i in range(len(target_clusters)):
#                 name_ = source_cluster + "_" + target_clusters[i]
#                 x_data, y_data = np.empty(0, dtype=float), np.empty(0, dtype=float)
#                 for pi in path[name_]:
#                     y_ = data_exp[:, adata.var.index == gene][pi].T[0]
#                     idx_ = y_ > 0
#                     x_data = np.append(x_data, np.linspace(0, 1, len(pi))[idx_])
#                     y_data = np.append(y_data, y_[idx_])
#                 if len(y_data):
#                     plt.scatter(x_data, y_data, color=cmap_(i), alpha=0.05, zorder=0)
#                 dynamics_ = adata.uns[dynamics_key][name_][
#                     :, adata.var.index == gene
#                 ]
#                 plot_x = np.linspace(0, 1, len(dynamics_))
#                 dynamics_[dynamics_ < 0] = 0
#                 plt.plot(plot_x, dynamics_, color="w", lw=8, zorder=1)
#                 plt.plot(
#                     plot_x,
#                     dynamics_,
#                     color=cmap_(i),
#                     lw=5,
#                     label=target_clusters[i],
#                     zorder=2,
#                 )
#                 y_data_all = np.append(y_data_all, y_data)
#             y_top_ = np.percentile(y_data_all, 99)
#             plt.ylim([-0.05 * y_top_, y_top_])
#             if legend:
#                 plt.legend(
#                     bbox_to_anchor=(1.05, 0.5),
#                     loc="center left",
#                     borderaxespad=0,
#                     title="Target",
#                     fontsize=fontsize_legend,
#                     title_fontsize=fontsize_legend,
#                 )
#             plt.xticks(
#                 [0, 0.25, 0.5, 0.75, 1],
#                 [
#                     "Source (0)\n(%s)" % source_cluster,
#                     "0.25",
#                     "0.5",
#                     "0.75",
#                     "Target (1)",
#                 ],
#                 fontsize=fontsize_label,
#             )
#             plt.title(gene, fontsize=fontsize_title)
#             plt.show()
#             if save:
#                 filename = (
#                     "%s_%s" % (save_filename, gene)
#                     if save_dir == None
#                     else "%s/%s_%s" % (save_dir, save_filename, gene)
#                 )
#                 fig.savefig(filename + ".png", bbox_inches="tight")
#             plt.close()
#         else:
#             print('Gene "%s" was not found' % gene)


# def DEG_dynamics(
#     adata,
#     source_cluster,
#     target_clusters,
#     path_key="path",
#     exp_key=None,
#     dynamics_key="dynamics",
#     gene_dynamics_stats_key="mean",
#     bifurcation_diagram_key="bifurcation_diagram",
#     target_genes=[],
#     n_div=100,
#     figsize=(14, 10),
#     fontsize_label=14,
#     fontsize_text=12,
#     fontsize_nDEG=18,
#     fontsize_legend=10,
#     DEG_min=1.0,
#     DEG_rate=0.3,
#     max_num_annotations=10,
#     max_num_legend=40,
#     interval=200,
#     show=True,
#     save=False,
#     save_dir=None,
#     save_filename="DEG_dynamics",
#     save_type="gif",
# ):
#     if dynamics_key not in adata.uns.keys():
#         calc_dynamics(
#             adata,
#             source_cluster,
#             target_clusters,
#             path_key=path_key,
#             exp_key=exp_key,
#             dynamics_key=dynamics_key,
#             n_div=n_div,
#         )

#     n_plot_ = int(len(target_clusters) * (len(target_clusters) - 1) / 2)
#     cmap_ = plt.get_cmap("tab10")
#     gene_dynamics_ = adata.uns[dynamics_key][gene_dynamics_stats_key]
#     matplotlib.rcParams["animation.embed_limit"] = 2**128
#     vlines = [0, 0.2, 0.4, 0.6, 0.8, 1]
#     vline_labels = np.append(
#         np.append("Source (0)", np.array(vlines)[1:-1]), "Target (1)"
#     )

#     def update(t, name_i_, name_j_, max_val_, lim, i, j, k):
#         print(
#             "\r...computing %s vs %s (%d/%d) %d/%d"
#             % (target_clusters[i], target_clusters[j], k, n_plot_, t + 1, n_div + 1),
#             end="",
#         )
#         idx_DEG_i_ = np.arange(adata.shape[1])[
#             (gene_dynamics_[name_j_][t] < gene_dynamics_[name_i_][t] - DEG_rate)
#             & (gene_dynamics_[name_i_][t] > DEG_min)
#         ]
#         idx_DEG_j_ = np.arange(adata.shape[1])[
#             (gene_dynamics_[name_i_][t] < gene_dynamics_[name_j_][t] - DEG_rate)
#             & (gene_dynamics_[name_j_][t] > DEG_min)
#         ]
#         idx_DEG_i_ = idx_DEG_i_[
#             np.argsort(
#                 gene_dynamics_[name_i_][t][idx_DEG_i_]
#                 - DEG_rate
#                 - gene_dynamics_[name_j_][t][idx_DEG_i_]
#             )[::-1]
#         ]
#         idx_DEG_j_ = idx_DEG_j_[
#             np.argsort(
#                 gene_dynamics_[name_j_][t][idx_DEG_j_]
#                 - DEG_rate
#                 - gene_dynamics_[name_i_][t][idx_DEG_j_]
#             )[::-1]
#         ]
#         if len(idx_DEG_i_) > max_num_annotations:
#             idx_DEG_ann_i_ = idx_DEG_i_[:max_num_annotations]
#         else:
#             idx_DEG_ann_i_ = idx_DEG_i_
#         if len(idx_DEG_j_) > max_num_annotations:
#             idx_DEG_ann_j_ = idx_DEG_j_[:max_num_annotations]
#         else:
#             idx_DEG_ann_j_ = idx_DEG_j_

#         if len(idx_DEG_i_) > max_num_legend:
#             idx_DEG_leg_i_ = idx_DEG_i_[:max_num_legend]
#         else:
#             idx_DEG_leg_i_ = idx_DEG_i_
#         if len(idx_DEG_j_) > max_num_legend:
#             idx_DEG_leg_j_ = idx_DEG_j_[:max_num_legend]
#         else:
#             idx_DEG_leg_j_ = idx_DEG_j_
#         ax1.cla()
#         ax2.cla()
#         ax3.cla()
#         name_i__ = source_cluster + "_" + target_clusters[0]
#         ax1.text(
#             0,
#             adata.uns[bifurcation_diagram_key][name_i_][0],
#             source_cluster + " ",
#             fontsize=fontsize_label,
#             va="center",
#             ha="right",
#         )
#         for i_ in range(len(target_clusters)):
#             name_i__ = source_cluster + "_" + target_clusters[i_]
#             if name_i__ not in [name_i_, name_j_]:
#                 y_ = adata.uns[bifurcation_diagram_key][name_i__]
#                 ax1.plot(
#                     np.linspace(0, 1, len(y_)),
#                     y_,
#                     lw=3,
#                     zorder=2,
#                     alpha=0.3,
#                     color=cmap_(i_),
#                 )
#                 ax1.text(
#                     1,
#                     y_[-1],
#                     " " + target_clusters[i_],
#                     fontsize=fontsize_label,
#                     va="center",
#                     ha="left",
#                     alpha=0.3,
#                 )
#         y_ = adata.uns[bifurcation_diagram_key][name_i_]
#         ax1.plot(np.linspace(0, 1, len(y_)), y_, lw=5, zorder=3, color=cmap_(i))
#         ax1.text(
#             1,
#             y_[-1],
#             " " + target_clusters[i],
#             fontsize=fontsize_label,
#             va="center",
#             ha="left",
#         )
#         y_ = adata.uns[bifurcation_diagram_key][name_j_]
#         ax1.plot(np.linspace(0, 1, len(y_)), y_, lw=5, zorder=3, color=cmap_(j))
#         ax1.text(
#             1,
#             y_[-1],
#             " " + target_clusters[j],
#             fontsize=fontsize_label,
#             va="center",
#             ha="left",
#         )
#         for vl in vlines:
#             ax1.axvline(vl, color="k", ls="--", lw=1, zorder=0)
#         ax1.axvline(t / n_div, color="r", ls="-", lw=2, zorder=3)
#         ax1.tick_params(axis="x", which="both", top=True)
#         ax1.spines["right"].set_visible(False)
#         ax1.spines["left"].set_visible(False)
#         ax1.spines["top"].set_visible(False)
#         ax1.spines["bottom"].set_visible(False)
#         ax1.xaxis.set_label_position("top")
#         ax1.xaxis.tick_top()
#         ax1.yaxis.set_visible(False)
#         ax1.set_xticks(vlines)
#         ax1.set_xticklabels(vline_labels, fontsize=fontsize_label)
#         ax2.set_title("Time = %.02f [s]" % (t / n_div))
#         ax2.scatter(
#             gene_dynamics_[name_i_][t],
#             gene_dynamics_[name_j_][t],
#             s=1,
#             color="gray",
#             zorder=1,
#         )
#         ax2.scatter(
#             gene_dynamics_[name_i_][t][idx_DEG_i_],
#             gene_dynamics_[name_j_][t][idx_DEG_i_],
#             color=cmap_(i),
#             zorder=2,
#             s=20,
#         )
#         ax2.scatter(
#             gene_dynamics_[name_i_][t][idx_DEG_j_],
#             gene_dynamics_[name_j_][t][idx_DEG_j_],
#             color=cmap_(j),
#             zorder=2,
#             s=20,
#         )
#         texts = []
#         for g in np.arange(adata.shape[1])[idx_DEG_ann_i_]:
#             tx_ = ax2.text(
#                 gene_dynamics_[name_i_][t][g],
#                 gene_dynamics_[name_j_][t][g],
#                 "_" + adata.var.index[g],
#                 color="k",
#                 zorder=2,
#                 fontsize=fontsize_text,
#             )
#             texts = np.append(texts, tx_)
#         for g in np.arange(adata.shape[1])[idx_DEG_ann_j_]:
#             tx_ = ax2.text(
#                 gene_dynamics_[name_i_][t][g],
#                 gene_dynamics_[name_j_][t][g],
#                 "_" + adata.var.index[g],
#                 color="k",
#                 zorder=2,
#                 fontsize=fontsize_text,
#             )
#             texts = np.append(texts, tx_)
#         if len(target_genes):
#             for gene_ in target_genes:
#                 idx_gene_ = adata.var.index == gene_
#                 ax2.scatter(
#                     gene_dynamics_[name_i_][t][idx_gene_],
#                     gene_dynamics_[name_j_][t][idx_gene_],
#                     s=20,
#                     color="red",
#                     zorder=2,
#                 )
#                 tx_ = ax2.text(
#                     gene_dynamics_[name_i_][t][idx_gene_],
#                     gene_dynamics_[name_j_][t][idx_gene_],
#                     "_" + gene_,
#                     color="r",
#                     zorder=2,
#                     fontsize=fontsize_text,
#                 )
#                 texts = np.append(texts, tx_)
#         legend_i_ = ""
#         for g in np.arange(adata.shape[1])[idx_DEG_leg_i_]:
#             legend_i_ += "(%.02f, %.02f)  %s\n" % (
#                 gene_dynamics_[name_i_][t][g],
#                 gene_dynamics_[name_j_][t][g],
#                 adata.var.index[g],
#             )
#         legend_j_ = ""
#         for g in np.arange(adata.shape[1])[idx_DEG_leg_j_]:
#             legend_j_ += "(%.02f, %.02f)  %s\n" % (
#                 gene_dynamics_[name_i_][t][g],
#                 gene_dynamics_[name_j_][t][g],
#                 adata.var.index[g],
#             )
#         ax2.text(
#             0.9 * (lim[1] - lim[0]) + lim[0],
#             0.1 * (lim[1] - lim[0]) + lim[0],
#             str(len(idx_DEG_i_)),
#             ha="center",
#             va="center",
#             fontsize=fontsize_nDEG,
#             color=cmap_(i),
#             fontweight="bold",
#             zorder=3,
#         )
#         ax2.text(
#             0.1 * (lim[1] - lim[0]) + lim[0],
#             0.9 * (lim[1] - lim[0]) + lim[0],
#             str(len(idx_DEG_j_)),
#             ha="center",
#             va="center",
#             fontsize=fontsize_nDEG,
#             color=cmap_(j),
#             fontweight="bold",
#             zorder=3,
#         )
#         ax2.fill_between(
#             lim,
#             lim - DEG_rate,
#             lim + DEG_rate,
#             facecolor="lightgray",
#             alpha=0.5,
#             zorder=0,
#         )
#         ax2.fill(
#             [-0.01 * max_val_, DEG_min, DEG_min, -0.01 * max_val_],
#             [-0.01 * max_val_, -0.01 * max_val_, DEG_min, DEG_min],
#             facecolor="lightgray",
#             alpha=0.5,
#             zorder=0,
#         )
#         ax2.set_xlabel(
#             target_clusters[i],
#             fontsize=fontsize_label,
#             color=cmap_(i),
#             fontweight="bold",
#         )
#         ax2.set_ylabel(
#             target_clusters[j],
#             fontsize=fontsize_label,
#             color=cmap_(j),
#             fontweight="bold",
#         )
#         ax2.set_xlim(lim)
#         ax2.set_ylim(lim)
#         ax2.grid(ls="--")
#         ax3.text(
#             0.0,
#             1,
#             target_clusters[i],
#             ha="left",
#             va="top",
#             fontsize=fontsize_legend,
#             color=cmap_(i),
#             zorder=3,
#             fontweight="bold",
#         )
#         ax3.text(
#             0.0,
#             0.97,
#             legend_i_,
#             ha="left",
#             va="top",
#             fontsize=fontsize_legend,
#             color=cmap_(i),
#             zorder=3,
#         )
#         ax3.text(
#             0.5,
#             1,
#             target_clusters[j],
#             ha="left",
#             va="top",
#             fontsize=fontsize_legend,
#             color=cmap_(j),
#             zorder=3,
#             fontweight="bold",
#         )
#         ax3.text(
#             0.5,
#             0.97,
#             legend_j_,
#             ha="left",
#             va="top",
#             fontsize=fontsize_legend,
#             color=cmap_(j),
#             zorder=3,
#         )
#         ax3.axis("off")

#     k = 0
#     for i in range(len(target_clusters)):
#         for j in range(i + 1, len(target_clusters)):
#             name_i_ = source_cluster + "_" + target_clusters[i]
#             name_j_ = source_cluster + "_" + target_clusters[j]
#             fig = plt.figure(figsize=figsize, tight_layout=True)
#             grid = plt.GridSpec(10, 14)
#             ax1 = fig.add_subplot(grid[0:2, 0:12])
#             ax2 = fig.add_subplot(grid[2:10, 0:8])
#             ax3 = fig.add_subplot(grid[2:10, 8:14])
#             max_val_ = max(
#                 np.max(gene_dynamics_[name_i_]), np.max(gene_dynamics_[name_j_])
#             )
#             lim = np.array([-0.01 * max_val_, 1.01 * max_val_])
#             k = k + 1
#             ani = anm.FuncAnimation(
#                 fig,
#                 update,
#                 interval=interval,
#                 fargs=(
#                     name_i_,
#                     name_j_,
#                     max_val_,
#                     lim,
#                     i,
#                     j,
#                     k,
#                 ),
#                 frames=n_div + 1,
#                 repeat=False,
#             )
#             if show == True:
#                 IPython.display.display(IPython.display.HTML(ani.to_jshtml()))
#             if save:
#                 filename = (
#                     "%s_%s_%s" % (save_filename, target_clusters[i], target_clusters[j])
#                     if save_dir == None
#                     else "%s/%s_%s_%s"
#                     % (save_dir, save_filename, target_clusters[i], target_clusters[j])
#                 )
#                 if len(target_genes):
#                     filename += "_TG" + str(len(target_genes))
#                 if save_type in ["gif", "video", "animetion"]:
#                     filename += ".gif"
#                     print("\nSaving gif animation as %s..." % filename)
#                     ani.save(filename)
#                 elif save_type in ["image", "png", "jpg", "jpeg"]:
#                     if save_type == "image":
#                         save_type = "png"
#                     print("\nSaving gif animation as %s" % filename)
#                     for t in range(n_div + 1):
#                         fig = plt.figure(figsize=figsize, tight_layout=True)
#                         grid = plt.GridSpec(10, 14)
#                         ax1 = fig.add_subplot(grid[0:2, 0:12])
#                         ax2 = fig.add_subplot(grid[2:10, 0:8])
#                         ax3 = fig.add_subplot(grid[2:10, 8:14])
#                         update(t, name_i_, name_j_, max_val_, lim, i, j, k)
#                         filename_ = "%s_%03d.%s" % (filename, t, save_type)
#                         fig.savefig(filename_, bbox_inches="tight")
#                         plt.close()
#             plt.close()

def DEG_dynamics(
    adata,
    source_cluster,
    target_clusters,
    path_key="path",
    exp_key=None,
    dynamics_key="dynamics",
    gene_dynamics_stats_key="mean",
    bifurcation_diagram_key="bifurcation_diagram",
    target_genes=[],
    n_div=100,
    figsize=(14, 10),
    fontsize_label=14,
    fontsize_text=12,
    fontsize_legend=10,
    PC = 1,
    Pval_thd = 2,
    FC_thd = 0.5,
    FC_cut = 1e-2,
    ps_active = 20,
    max_num_annotations=10,
    max_num_legend=40,
    interval=200,
    show=True,
    save=False,
    save_dir=None,
    save_filename="DEG_dynamics",
    save_type="gif",
):
    if dynamics_key not in adata.uns.keys():
        calc_dynamics(
            adata,
            source_cluster,
            target_clusters,
            path_key=path_key,
            exp_key=exp_key,
            dynamics_key=dynamics_key,
            n_div=n_div,
        )
    
    data_exp_ = _set_expression_data(adata, exp_key)
    n_plot_ = int(len(target_clusters) * (len(target_clusters) - 1) / 2)
    cmap_ = plt.get_cmap("tab10")
    gene_dynamics_ = adata.uns[dynamics_key][gene_dynamics_stats_key]
    matplotlib.rcParams["animation.embed_limit"] = 2**128
    vlines = [0, 0.2, 0.4, 0.6, 0.8, 1]
    vline_labels = np.append(
        np.append("Source (0)", np.array(vlines)[1:-1]), "Target (1)"
    )

    def update(t, name_i_, name_j_, max_val_, xlim_, ylim_, i, j, k):
        print(
            "\r...computing %s vs %s (%d/%d) %d/%d"
            % (target_clusters[i], target_clusters[j], k, n_plot_, t + 1, n_div+1),
            end="",
        )
        val_i_ = data_exp_[adata.uns['path_scaled'][name_i_][:,t]]
        val_j_ = data_exp_[adata.uns['path_scaled'][name_j_][:,t]]
        mean_i_ = np.mean(val_i_,axis=0)
        mean_j_ = np.mean(val_j_,axis=0)
        FC_ = np.nan_to_num(np.log2(mean_j_/mean_i_))
        FC_[(mean_i_<FC_cut) | (mean_j_<FC_cut)] = 0
        Pval_ = np.nan_to_num(-np.log10(scipy.stats.ttest_ind(val_i_, val_j_)[1]))
        
        idx_sig_i_ = np.arange(adata.shape[1])[(Pval_ > np.percentile(Pval_,100-Pval_thd)) & (FC_ < -FC_thd)]
        idx_sig_i_ = idx_sig_i_[np.argsort(FC_[idx_sig_i_])]
        idx_sig_j_ = np.arange(adata.shape[1])[(Pval_ > np.percentile(Pval_,100-Pval_thd)) & (FC_ > FC_thd)]
        idx_sig_j_ = idx_sig_j_[np.argsort(FC_[idx_sig_j_])[::-1]]

        ax1.cla()
        ax2.cla()
        ax3.cla()
        name_i__ = source_cluster + "_" + target_clusters[0]
        ax1.text(
            0,
            adata.uns[bifurcation_diagram_key][name_i_][0,PC-1],
            source_cluster + " ",
            fontsize=fontsize_label,
            va="center",
            ha="right",
        )
        for i_ in range(len(target_clusters)):
            name_i__ = source_cluster + "_" + target_clusters[i_]
            if name_i__ not in [name_i_, name_j_]:
                y_ = adata.uns[bifurcation_diagram_key][name_i__][:,PC-1]
                ax1.plot(
                    np.linspace(0, 1, len(y_)),
                    y_,
                    lw=3,
                    zorder=2,
                    alpha=0.3,
                    color=cmap_(i_),
                )
                ax1.text(
                    1,
                    y_[-1],
                    " " + target_clusters[i_],
                    fontsize=fontsize_label,
                    va="center",
                    ha="left",
                    alpha=0.3,
                )
        y_ = adata.uns[bifurcation_diagram_key][name_i_][:,PC-1]
        ax1.plot(np.linspace(0, 1, len(y_)), y_, lw=5, zorder=3, color=cmap_(i))
        ax1.text(
            1,
            y_[-1],
            " " + target_clusters[i],
            fontsize=fontsize_label,
            va="center",
            ha="left",
        )
        y_ = adata.uns[bifurcation_diagram_key][name_j_][:,PC-1]
        ax1.plot(np.linspace(0, 1, len(y_)), y_, lw=5, zorder=3, color=cmap_(j))
        ax1.text(
            1,
            y_[-1],
            " " + target_clusters[j],
            fontsize=fontsize_label,
            va="center",
            ha="left",
        )
        for vl in vlines:
            ax1.axvline(vl, color="k", ls="--", lw=1, zorder=0)
        ax1.axvline(t / n_div, color="r", ls="-", lw=2, zorder=3)
        ax1.tick_params(axis="x", which="both", top=True)
        ax1.spines["right"].set_visible(False)
        ax1.spines["left"].set_visible(False)
        ax1.spines["top"].set_visible(False)
        ax1.spines["bottom"].set_visible(False)
        ax1.xaxis.set_label_position("top")
        ax1.xaxis.tick_top()
        ax1.yaxis.set_visible(False)
        ax1.set_xticks(vlines)
        ax1.set_xticklabels(vline_labels, fontsize=fontsize_label)
        ax2.set_title("Time = %.02f [s]" % (t / n_div))
        ax2.scatter(FC_,Pval_,color="gray",s=1,zorder=1)
        ax2.scatter(FC_[idx_sig_i_],Pval_[idx_sig_i_],color=cmap_(i),s=ps_active,zorder=10)
        ax2.scatter(FC_[idx_sig_j_],Pval_[idx_sig_j_],color=cmap_(j),s=ps_active,zorder=10)
        for i_ in range(min(len(idx_sig_i_),max_num_annotations)):
            ax2.text(FC_[idx_sig_i_[i_]],Pval_[idx_sig_i_[i_]],adata.var.index[idx_sig_i_[i_]]+"_",zorder=20,fontsize=fontsize_text,ha="right")
        for i_ in range(min(len(idx_sig_j_),max_num_annotations)):
            ax2.text(FC_[idx_sig_j_[i_]],Pval_[idx_sig_j_[i_]],"_"+adata.var.index[idx_sig_j_[i_]],zorder=20,fontsize=fontsize_text)
        ax2.grid(ls="--",zorder=0)
        ax2.set_xlim(xlim_)
        ax2.set_ylim(ylim_)
        ax2.set_xlabel("Log2 Fold Change",fontsize=fontsize_label)
        ax2.set_ylabel("-Log10(p-value)",fontsize=fontsize_label)
        ax3.text(
            0.0,
            1,
            target_clusters[i],
            ha="left",
            va="top",
            fontsize=fontsize_legend,
            color=cmap_(i),
            zorder=3,
            fontweight="bold",
        )
        legend_i_ = ""
        for i_ in range(min(len(idx_sig_i_),max_num_legend)):
            g = idx_sig_i_[i_]
            legend_i_ += "(%.02f, %.02f)  %s\n" % (
                FC_[g],
                Pval_[g],
                adata.var.index[g],
            )
        ax3.text(
            0.0,
            0.97,
            legend_i_,
            ha="left",
            va="top",
            fontsize=fontsize_legend,
            color=cmap_(i),
            zorder=3,
        )
        ax3.text(
            0.5,
            1,
            target_clusters[j],
            ha="left",
            va="top",
            fontsize=fontsize_legend,
            color=cmap_(j),
            zorder=3,
            fontweight="bold",
        )
        legend_j_ = ""
        for i_ in range(min(len(idx_sig_j_),max_num_legend)):
            g = idx_sig_j_[i_]
            legend_j_ += "(%.02f, %.02f)  %s\n" % (
                FC_[g],
                Pval_[g],
                adata.var.index[g],
            )
        ax3.text(
            0.5,
            0.97,
            legend_j_,
            ha="left",
            va="top",
            fontsize=fontsize_legend,
            color=cmap_(j),
            zorder=3,
        )
        ax3.axis("off")

    k = 0
    for i in range(len(target_clusters)):
        for j in range(i + 1, len(target_clusters)):
            name_i_ = source_cluster + "_" + target_clusters[i]
            name_j_ = source_cluster + "_" + target_clusters[j]
            t = n_div
            val_i_ = data_exp_[adata.uns['path_scaled'][name_i_][:,t]]
            val_j_ = data_exp_[adata.uns['path_scaled'][name_j_][:,t]]
            FC_ = np.nan_to_num(np.log2(np.mean(val_i_,axis=0)/np.mean(val_j_,axis=0)))
            Pval_ = np.nan_to_num(-np.log10(scipy.stats.ttest_ind(val_i_, val_j_)[1]))
            xlim_ = [-np.max(np.abs(FC_))*1.3,np.max(np.abs(FC_))*1.3]
            ylim_ = [-0.5,np.max(Pval_)*1.01]
            fig = plt.figure(figsize=figsize, tight_layout=True)
            grid = plt.GridSpec(10, 14)
            ax1 = fig.add_subplot(grid[0:2, 0:12])
            ax2 = fig.add_subplot(grid[2:10, 0:8])
            ax3 = fig.add_subplot(grid[2:10, 8:14])
            max_val_ = max(
                np.max(gene_dynamics_[name_i_]), np.max(gene_dynamics_[name_j_])
            )
            k = k + 1
            ani = matplotlib.animation.FuncAnimation(
                fig,
                update,
                interval=interval,
                fargs=(
                    name_i_,
                    name_j_,
                    max_val_,
                    xlim_,
                    ylim_,
                    i,
                    j,
                    k,
                ),
                frames=n_div+1,
                repeat=False,
            )
            if show == True:
                IPython.display.display(IPython.display.HTML(ani.to_jshtml()))
            if save:
                filename = (
                    "%s_%s_%s" % (save_filename, target_clusters[i], target_clusters[j])
                    if save_dir == None
                    else "%s/%s_%s_%s"
                    % (save_dir, save_filename, target_clusters[i], target_clusters[j])
                )
                if len(target_genes):
                    filename += "_TG" + str(len(target_genes))
                if save_type in ["gif", "video", "animetion"]:
                    filename += ".gif"
                    print("\nSaving gif animation as %s..." % filename)
                    ani.save(filename)
                elif save_type in ["image", "png", "jpg", "jpeg"]:
                    if save_type == "image":
                        save_type = "png"
                    print("\nSaving gif animation as %s" % filename)
                    for t in range(n_div + 1):
                        fig = plt.figure(figsize=figsize, tight_layout=True)
                        grid = plt.GridSpec(10, 14)
                        ax1 = fig.add_subplot(grid[0:2, 0:12])
                        ax2 = fig.add_subplot(grid[2:10, 0:8])
                        ax3 = fig.add_subplot(grid[2:10, 8:14])
                        update(t, name_i_, name_j_, max_val_, xlim_, ylim_, i, j, k)
                        filename_ = "%s_%03d.%s" % (filename, t, save_type)
                        fig.savefig(filename_, bbox_inches="tight")
                        plt.close()
            plt.close()


def DEG_dynamics_clusters(
    adata,
    source_cluster,
    target_clusters,
    path_key="path",
    exp_key=None,
    dynamics_key="dynamics",
    bifurcation_diagram_key="bifurcation_diagram",
    target_genes=[],
    n_div=100,
    figsize=(14, 10),
    fontsize_label=14,
    fontsize_text=12,
    fontsize_nDEG=18,
    fontsize_legend=10,
    DEG_min=1.0,
    DEG_rate=0.3,
    max_num_annotations=10,
    max_num_legend=25,
    interval=200,
    save=False,
    save_dir=None,
    save_filename="DEG_dynamics",
    save_type="gif",
):
    if dynamics_key not in adata.uns.keys():
        calc_dynamics(
            adata,
            source_cluster,
            target_clusters,
            path_key=path_key,
            exp_key=exp_key,
            dynamics_key=dynamics_key,
            n_div=n_div,
        )

    n_plot_ = int(len(target_clusters) * (len(target_clusters) - 1) / 2)
    cmap_ = plt.get_cmap("tab10")
    cmap20_ = plt.get_cmap("tab20")
    gene_dynamics_ = adata.uns[dynamics_key]
    matplotlib.rcParams["animation.embed_limit"] = 2**128
    vlines = [0, 0.2, 0.4, 0.6, 0.8, 1]
    vline_labels = np.append(
        np.append("Source (0)", np.array(vlines)[1:-1]), "Target (1)"
    )

    def update(t, name_i_, name_j_, max_val_, lim, i, j, k):
        print(
            "\rcomputing %s vs %s (%d/%d) %d/%d"
            % (target_clusters[i], target_clusters[j], k, n_plot_, t + 1, n_div + 1),
            end="",
        )
        idx_DEG_i_ = np.arange(adata.shape[1])[
            (gene_dynamics_[name_j_][t] < gene_dynamics_[name_i_][t] - DEG_rate)
            & (gene_dynamics_[name_i_][t] > DEG_min)
        ]
        idx_DEG_j_ = np.arange(adata.shape[1])[
            (gene_dynamics_[name_i_][t] < gene_dynamics_[name_j_][t] - DEG_rate)
            & (gene_dynamics_[name_j_][t] > DEG_min)
        ]
        idx_DEG_i_ = idx_DEG_i_[
            np.argsort(
                gene_dynamics_[name_i_][t][idx_DEG_i_]
                - DEG_rate
                - gene_dynamics_[name_j_][t][idx_DEG_i_]
            )[::-1]
        ]
        idx_DEG_j_ = idx_DEG_j_[
            np.argsort(
                gene_dynamics_[name_j_][t][idx_DEG_j_]
                - DEG_rate
                - gene_dynamics_[name_i_][t][idx_DEG_j_]
            )[::-1]
        ]
        if len(idx_DEG_i_) > max_num_annotations:
            idx_DEG_ann_i_ = idx_DEG_i_[:max_num_annotations]
        else:
            idx_DEG_ann_i_ = idx_DEG_i_
        if len(idx_DEG_j_) > max_num_annotations:
            idx_DEG_ann_j_ = idx_DEG_j_[:max_num_annotations]
        else:
            idx_DEG_ann_j_ = idx_DEG_j_

        if len(idx_DEG_i_) > max_num_legend:
            idx_DEG_leg_i_ = idx_DEG_i_[:max_num_legend]
        else:
            idx_DEG_leg_i_ = idx_DEG_i_
        if len(idx_DEG_j_) > max_num_legend:
            idx_DEG_leg_j_ = idx_DEG_j_[:max_num_legend]
        else:
            idx_DEG_leg_j_ = idx_DEG_j_
        ax1.cla()
        ax2.cla()
        ax3.cla()
        name_i__ = source_cluster + "_" + target_clusters[0]
        ax1.text(
            0,
            adata.uns[bifurcation_diagram_key][name_i_][0],
            source_cluster + " ",
            fontsize=fontsize_label,
            va="center",
            ha="right",
        )
        for i_ in range(len(target_clusters)):
            name_i__ = source_cluster + "_" + target_clusters[i_]
            if name_i__ not in [name_i_, name_j_]:
                y_ = adata.uns[bifurcation_diagram_key][name_i__]
                ax1.plot(np.linspace(0, 1, len(y_)), y_, lw=3, zorder=2, alpha=0.3)
                ax1.text(
                    1,
                    y_[-1],
                    " " + target_clusters[i_],
                    fontsize=fontsize_label,
                    va="center",
                    ha="left",
                    alpha=0.3,
                )
        y_ = adata.uns[bifurcation_diagram_key][name_i_]
        ax1.plot(np.linspace(0, 1, len(y_)), y_, lw=5, zorder=3)
        ax1.text(
            1,
            y_[-1],
            " " + target_clusters[i],
            fontsize=fontsize_label,
            va="center",
            ha="left",
        )
        y_ = adata.uns[bifurcation_diagram_key][name_j_]
        ax1.plot(np.linspace(0, 1, len(y_)), y_, lw=5, zorder=3)
        ax1.text(
            1,
            y_[-1],
            " " + target_clusters[j],
            fontsize=fontsize_label,
            va="center",
            ha="left",
        )
        for vl in vlines:
            ax1.axvline(vl, color="k", ls="--", lw=1, zorder=0)
        ax1.axvline(t / n_div, color="r", ls="-", lw=2, zorder=3)
        ax1.tick_params(axis="x", which="both", top=True)
        ax1.spines["right"].set_visible(False)
        ax1.spines["left"].set_visible(False)
        ax1.spines["top"].set_visible(False)
        ax1.spines["bottom"].set_visible(False)
        ax1.xaxis.set_label_position("top")
        ax1.xaxis.tick_top()
        ax1.yaxis.set_visible(False)
        ax1.set_xticks(vlines)
        ax1.set_xticklabels(vline_labels, fontsize=fontsize_label)
        ax2.set_title("Time = %.02f [s]" % (t / n_div))
        ax2.scatter(
            gene_dynamics_[name_i_][t],
            gene_dynamics_[name_j_][t],
            s=1,
            color="gray",
            zorder=1,
        )
        for c_ in range(n_clusters):
            idx_ = (adata.var["clusters_" + name_i_] == c_) & (
                gene_dynamics_[name_i_][t] > gene_dynamics_[name_j_][t]
            )
            ax2.scatter(
                gene_dynamics_[name_i_][t][idx_],
                gene_dynamics_[name_j_][t][idx_],
                color=cmap20_(c_),
                zorder=2,
                s=20,
                label=str(c_ + 1),
                marker="o",
            )
            idx_ = (adata.var["clusters_" + name_j_] == c_) & (
                gene_dynamics_[name_i_][t] < gene_dynamics_[name_j_][t]
            )
            ax2.scatter(
                gene_dynamics_[name_i_][t][idx_],
                gene_dynamics_[name_j_][t][idx_],
                color=cmap20_(c_),
                zorder=2,
                s=20,
                marker="o",
            )
        ax2.legend(
            loc="lower left",
            bbox_to_anchor=(1.05, 0.0),
            ncol=5,
            title="Clusters",
            columnspacing=0.5,
        )
        texts = []
        for g in np.arange(adata.shape[1])[idx_DEG_ann_i_]:
            tx_ = ax2.text(
                gene_dynamics_[name_i_][t][g],
                gene_dynamics_[name_j_][t][g],
                "_" + adata.var.index[g],
                color="k",
                zorder=2,
                fontsize=fontsize_text,
            )
            texts = np.append(texts, tx_)
        for g in np.arange(adata.shape[1])[idx_DEG_ann_j_]:
            tx_ = ax2.text(
                gene_dynamics_[name_i_][t][g],
                gene_dynamics_[name_j_][t][g],
                "_" + adata.var.index[g],
                color="k",
                zorder=2,
                fontsize=fontsize_text,
            )
            texts = np.append(texts, tx_)
        if len(target_genes):
            for gene_ in target_genes:
                idx_gene_ = adata.var.index == gene_
                ax2.scatter(
                    gene_dynamics_[name_i_][t][idx_gene_],
                    gene_dynamics_[name_j_][t][idx_gene_],
                    s=20,
                    color="red",
                    zorder=2,
                )
                tx_ = ax2.text(
                    gene_dynamics_[name_i_][t][idx_gene_],
                    gene_dynamics_[name_j_][t][idx_gene_],
                    "_" + gene_,
                    color="r",
                    zorder=2,
                    fontsize=fontsize_text,
                )
                texts = np.append(texts, tx_)
        legend_i_ = ""
        for g in np.arange(adata.shape[1])[idx_DEG_leg_i_]:
            legend_i_ += "(%.02f, %.02f)  %s\n" % (
                gene_dynamics_[name_i_][t][g],
                gene_dynamics_[name_j_][t][g],
                adata.var.index[g],
            )
        legend_j_ = ""
        for g in np.arange(adata.shape[1])[idx_DEG_leg_j_]:
            legend_j_ += "(%.02f, %.02f)  %s\n" % (
                gene_dynamics_[name_i_][t][g],
                gene_dynamics_[name_j_][t][g],
                adata.var.index[g],
            )
        ax2.text(
            0.9 * (lim[1] - lim[0]) + lim[0],
            0.1 * (lim[1] - lim[0]) + lim[0],
            str(len(idx_DEG_i_)),
            ha="center",
            va="center",
            fontsize=fontsize_nDEG,
            color=cmap_(i),
            fontweight="bold",
            zorder=3,
        )
        ax2.text(
            0.1 * (lim[1] - lim[0]) + lim[0],
            0.9 * (lim[1] - lim[0]) + lim[0],
            str(len(idx_DEG_j_)),
            ha="center",
            va="center",
            fontsize=fontsize_nDEG,
            color=cmap_(j),
            fontweight="bold",
            zorder=3,
        )
        ax2.fill_between(
            lim,
            lim - DEG_rate,
            lim + DEG_rate,
            facecolor="lightgray",
            alpha=0.5,
            zorder=0,
        )
        ax2.fill(
            [-0.01 * max_val_, DEG_min, DEG_min, -0.01 * max_val_],
            [-0.01 * max_val_, -0.01 * max_val_, DEG_min, DEG_min],
            facecolor="lightgray",
            alpha=0.5,
            zorder=0,
        )
        ax2.set_xlabel(
            target_clusters[i],
            fontsize=fontsize_label,
            color=cmap_(i),
            fontweight="bold",
        )
        ax2.set_ylabel(
            target_clusters[j],
            fontsize=fontsize_label,
            color=cmap_(j),
            fontweight="bold",
        )
        ax2.set_xlim(lim)
        ax2.set_ylim(lim)
        ax2.grid(ls="--")
        ax3.text(
            0.0,
            1,
            target_clusters[i],
            ha="left",
            va="top",
            fontsize=fontsize_legend,
            color=cmap_(i),
            zorder=3,
            fontweight="bold",
        )
        ax3.text(
            0.0,
            0.97,
            legend_i_,
            ha="left",
            va="top",
            fontsize=fontsize_legend,
            color=cmap_(i),
            zorder=3,
        )
        ax3.text(
            0.5,
            1,
            target_clusters[j],
            ha="left",
            va="top",
            fontsize=fontsize_legend,
            color=cmap_(j),
            zorder=3,
            fontweight="bold",
        )
        ax3.text(
            0.5,
            0.97,
            legend_j_,
            ha="left",
            va="top",
            fontsize=fontsize_legend,
            color=cmap_(j),
            zorder=3,
        )
        ax3.axis("off")

    k = 0
    for i in range(len(target_clusters)):
        for j in range(i + 1, len(target_clusters)):
            name_i_ = source_cluster + "_" + target_clusters[i]
            name_j_ = source_cluster + "_" + target_clusters[j]
            fig = plt.figure(figsize=figsize, tight_layout=True)
            grid = plt.GridSpec(10, 14)
            ax1 = fig.add_subplot(grid[0:2, 0:12])
            ax2 = fig.add_subplot(grid[2:10, 0:8])
            ax3 = fig.add_subplot(grid[2:10, 8:14])
            max_val_ = max(
                np.max(gene_dynamics_[name_i_]), np.max(gene_dynamics_[name_j_])
            )
            lim = np.array([-0.01 * max_val_, 1.01 * max_val_])
            k = k + 1
            ani = anm.FuncAnimation(
                fig,
                update,
                interval=interval,
                fargs=(
                    name_i_,
                    name_j_,
                    max_val_,
                    lim,
                    i,
                    j,
                    k,
                ),
                frames=n_div + 1,
                repeat=False,
            )
            IPython.display.display(IPython.display.HTML(ani.to_jshtml()))
            plt.close()
            if save:
                filename = (
                    "%s_%s_%s" % (save_filename, target_clusters[i], target_clusters[j])
                    if save_dir == None
                    else "%s/%s_%s_%s"
                    % (save_dir, save_filename, target_clusters[i], target_clusters[j])
                )
                if len(target_genes):
                    filename += "_TG" + str(len(target_genes))
                if save_type in ["gif", "video", "animetion"]:
                    filename += ".gif"
                    print("\nSaving gif animation as %s" % filename)
                    ani.save(filename)
                elif save_type in ["image", "png", "jpg", "jpeg"]:
                    matplotlib.use("Agg")
                    if save_type == "image":
                        save_type = "png"
                    print("\nSaving gif animation as %s" % filename)
                    for t in range(n_div + 1):
                        fig = plt.figure(figsize=figsize, tight_layout=True)
                        grid = plt.GridSpec(10, 14)
                        ax1 = fig.add_subplot(grid[0:2, 0:12])
                        ax2 = fig.add_subplot(grid[2:10, 0:8])
                        ax3 = fig.add_subplot(grid[2:10, 8:14])
                        update(t, name_i_, name_j_, max_val_, lim, i, j, k)
                        filename_ = "%s_%03d.%s" % (filename, t, save_type)
                        fig.savefig(filename_, bbox_inches="tight")
                        plt.close()
                    matplotlib.use("module://matplotlib_inline.backend_inline")


# def calc_bifurcation_diagram(
#     adata,
#     source_cluster,
#     target_clusters,
#     path_key="path",
#     exp_key=None,
#     dynamics_key="dynamics",
#     bifurcation_diagram_key="bifurcation_diagram",
#     n_div=100,
#     PC=1,
# ):
#     if dynamics_key not in adata.uns.keys():
#         calc_dynamics(
#             adata,
#             source_cluster,
#             target_clusters,
#             path_key=path_key,
#             exp_key=exp_key,
#             dynamics_key=dynamics_key,
#             n_div=n_div,
#         )

#     name_i_ = source_cluster + "_" + target_clusters[0]
#     samples_ = np.empty(
#         [
#             len(target_clusters),
#             adata.uns[dynamics_key][name_i_].shape[0],
#             adata.uns[dynamics_key][name_i_].shape[1],
#         ],
#         dtype=float,
#     )
#     for i in range(len(target_clusters)):
#         name_i_ = source_cluster + "_" + target_clusters[i]
#         samples_[i] = adata.uns[dynamics_key][name_i_]
#     pca_ = sklearn.decomposition.PCA().fit(samples_[:, -1])
#     samples_pca_ = pca_.transform(np.concatenate(samples_))

#     bd_ = {}
#     for i in range(len(target_clusters)):
#         name_i_ = source_cluster + "_" + target_clusters[i]
#         bd_[name_i_] = samples_pca_[i * (n_div + 1) : (i + 1) * (n_div + 1), PC - 1]

#     adata.uns[bifurcation_diagram_key] = bd_


def calc_bifurcation_diagram(
    adata,
    source_cluster,
    target_clusters,
    cluster_key="clusters",
    path_key="path",
    path_scaled_key="path_scaled",
    exp_key=None,
    key_stats = "mean",
    dynamics_key="dynamics",
    bifurcation_diagram_key="bifurcation_diagram",
    n_div=100,
    n_components = 5,
):
    if dynamics_key not in adata.uns.keys():
        calc_dynamics(
            adata,
            source_cluster,
            target_clusters,
            path_key=path_key,
            exp_key=exp_key,
            dynamics_key=dynamics_key,
            n_div=n_div,
        )

    data_exp = _set_expression_data(adata, exp_key)
    path_scaled = adata.uns[path_scaled_key]
    name_i_ = source_cluster + "_" + target_clusters[0]
    samples_ = np.empty(
        [
            len(target_clusters),
            adata.uns[dynamics_key][key_stats][name_i_].shape[0],
            adata.uns[dynamics_key][key_stats][name_i_].shape[1],
        ],
        dtype=float,
    )
    for i in range(len(target_clusters)):
        name_i_ = source_cluster + "_" + target_clusters[i]
        samples_[i] = adata.uns[dynamics_key][key_stats][name_i_]
    # pca_ = sklearn.decomposition.PCA().fit(samples_[:, -1])
    # samples_pca_ = pca_.transform(np.concatenate(samples_))
    data_ = np.concatenate([data_exp[adata.obs[cluster_key] == k_] for k_ in target_clusters])
    n_components = min([n_components,data_.shape[1]])
    pca_ = sklearn.decomposition.PCA(n_components=n_components).fit(data_)
    samples_pca_ = pca_.transform(np.concatenate(samples_))
    data_all_ = np.array([data_exp[path_scaled[k_]] for k_ in path_scaled.keys()])
    data_all_pca_ = pca_.transform(data_all_.reshape(data_all_.shape[0]*data_all_.shape[1]*data_all_.shape[2],data_all_.shape[3]))
    data_all_pca_ = data_all_pca_.reshape(data_all_.shape[0],data_all_.shape[1],data_all_.shape[2],data_all_pca_.shape[1])

    bd_ = {}
    for i in range(len(target_clusters)):
        name_i_ = source_cluster + "_" + target_clusters[i]
        bd_[name_i_] = samples_pca_[i * (n_div+1) : (i + 1) * (n_div+1)]

    adata.uns[bifurcation_diagram_key] = bd_
    adata.uns[dynamics_key+"_PC"] = data_all_pca_

    # data_exp = _set_expression_data(adata, exp_key)
    # path_scaled = adata.uns[path_scaled_key]
    # data_all_ = np.array([data_exp[path_scaled[k_]] for k_ in path_scaled.keys()])
    # pca_ = sklearn.decomposition.PCA().fit(np.concatenate(data_all_)[:,-1])
    # data_all_pca_ = pca_.transform(data_all_.reshape(data_all_.shape[0]*data_all_.shape[1]*data_all_.shape[2],data_all_.shape[3]))
    # data_all_pca_ = data_all_pca_.reshape(data_all_.shape[0],data_all_.shape[1],data_all_.shape[2],data_all_pca_.shape[1])
    # data_all_pca_mean_ = np.mean(data_all_pca_,axis=1)


    # adata.uns[bifurcation_diagram_key] = {}
    # for i in range(len(target_clusters)):
    #     name_i_ = source_cluster + "_" + target_clusters[i]
    #     adata.uns[bifurcation_diagram_key][name_i_] = data_all_pca_mean_[i]

def bifurcation_diagram(
    adata,
    source_cluster,
    target_clusters,
    path_key="path",
    exp_key=None,
    dynamics_key="dynamics",
    bifurcation_diagram_key="bifurcation_diagram",
    n_div=100,
    figsize=(12, 3),
    errbar_key = "percentile",
    errbar_alpha = 0.15,
    fontsize_label=14,
    adjusttext=False,
    PC=1,
    save=False,
    save_dir=None,
    save_filename="Bifurcation_diagram",
):
    if dynamics_key not in adata.uns.keys():
        calc_dynamics(
            adata,
            source_cluster,
            target_clusters,
            path_key=path_key,
            exp_key=exp_key,
            dynamics_key=dynamics_key,
            n_div=n_div,
        )

    if bifurcation_diagram_key not in adata.uns.keys():
        calc_bifurcation_diagram(
            adata,
            source_cluster,
            target_clusters,
            path_key=path_key,
            exp_key=exp_key,
            dynamics_key=dynamics_key,
            bifurcation_diagram_key=bifurcation_diagram_key,
            n_div=n_div,
            PC=PC,
        )
    cmap_ = plt.get_cmap("tab10")
    vlines = [0, 0.1,0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
    vline_labels = np.append(
        np.append("Source (0)", np.array(vlines)[1:-1]), "Target (1)"
    )

    fig, ax = plt.subplots(figsize=figsize)
    name_i_ = source_cluster + "_" + target_clusters[0]
    ax.text(
        0,
        adata.uns[bifurcation_diagram_key][name_i_][:,PC-1][0],
        source_cluster + " ",
        fontsize=fontsize_label,
        va="center",
        ha="right",
    )
    et_,eb_ = np.mean(adata.uns[dynamics_key+"_PC"],axis=1)[:,:,PC-1],np.mean(adata.uns[dynamics_key+"_PC"],axis=1)[:,:,PC-1]
    if errbar_key == "percentile":
        et_,eb_ = np.percentile(adata.uns[dynamics_key+"_PC"],75,axis=1)[:,:,PC-1],np.percentile(adata.uns[dynamics_key+"_PC"],25,axis=1)[:,:,PC-1]
    elif errbar_key == "std":
        et_ = np.mean(adata.uns[dynamics_key+"_PC"],axis=1)[:,:,PC-1]+np.std(adata.uns[dynamics_key+"_PC"],axis=1)[:,:,PC-1]
        eb_ = np.mean(adata.uns[dynamics_key+"_PC"],axis=1)[:,:,PC-1]-np.std(adata.uns[dynamics_key+"_PC"],axis=1)[:,:,PC-1]
    texts = []
    for i in range(len(target_clusters)):
        name_i_ = source_cluster + "_" + target_clusters[i]
        y_ = adata.uns[bifurcation_diagram_key][name_i_][:,PC-1]
        x_ = np.linspace(0, 1, n_div+1)
        ax.plot(x_, y_, lw=5, zorder=2, color=cmap_(i))
        ax.fill_between(x_,et_[i], eb_[i], color=cmap_(i), alpha=errbar_alpha,zorder=0)
        texts = np.append(
            texts,
            ax.text(
                1,
                y_[-1],
                " " + target_clusters[i],
                fontsize=fontsize_label,
                va="center",
                ha="left",
            ),
        )
    # if adjusttext:
    #     adjustText.adjust_text(texts, arrowprops=dict(arrowstyle="-", color="k"))
    for vl in vlines:
        ax.axvline(vl, color="k", ls="--", lw=1, zorder=0,alpha=0.5)
    ax.tick_params(axis="x", which="both", top=True)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.xaxis.set_label_position("top")
    ax.xaxis.tick_top()
    ax.yaxis.set_visible(False)
    ax.set_xticks(vlines)
    ax.set_xticklabels(vline_labels, fontsize=fontsize_label)
    if save:
        filename = (
            "%s" % (save_filename)
            if save_dir == None
            else "%s/%s" % (save_dir, save_filename)
        )
        fig.savefig(filename + ".png", bbox_inches="tight")

def calc_landscape(
    adata,
    source_cluster,
    target_clusters,
    path_key = "path",
    path_scaled_key = "path_scaled",
    landscape_key = "landscape",
    exp_key=None,
    dynamics_key="dynamics",
    pot_key = "potential",
    n_div=100,
    n_grid = 100,
    n_grid_plot = 200,
    intp_method='linear',
    merge_step = 5,
    valley_range = 0.1,
    landscape_base = 0.05,
    PC=1,
):
    if dynamics_key not in adata.uns.keys():
        calc_dynamics(
            adata,
            source_cluster,
            target_clusters,
            path_key=path_key,
            exp_key=exp_key,
            dynamics_key=dynamics_key,
            n_div=n_div,
        )

    data_all_ = adata.uns[dynamics_key+"_PC"]
    x_lim_min,x_lim_max = np.min(data_all_[:,:,:,PC-1]), np.max(data_all_[:,:,:,PC-1])
    x_ = np.linspace(x_lim_min,x_lim_max, n_grid)
    z_plot = np.zeros([n_grid,n_div-merge_step])
    c_plot = np.zeros([n_grid,n_div-merge_step])
    pot_ = adata.obs[pot_key]

    for ti_ in range(n_div-merge_step):
        for i in range(len(data_all_)):
            name_i_ = source_cluster + "_" + target_clusters[i]
            data_ = data_all_[i,:,ti_:ti_+merge_step+1:,PC-1].flatten()
            pot_t_ = pot_[adata.uns[path_scaled_key][name_i_][:,ti_:ti_+merge_step+1].flatten()].mean()/pot_.max()
            kde_model = scipy.stats.gaussian_kde(data_)
            dens_ = kde_model(x_)
            z_ = pot_t_-valley_range*dens_/np.max(dens_)
            z_plot[:,ti_] += z_/len(data_all_)
            c_plot[:,ti_] += dens_/np.max(dens_)/len(data_all_)
        c_plot[:,ti_] = 1-c_plot[:,ti_]

    x1 = x_
    x2 = np.linspace(0,1,n_div-merge_step)
    X1, X2 = np.meshgrid(x1, x2)

    grid_x, grid_y = np.meshgrid(np.linspace(x_lim_min,x_lim_max,n_grid_plot+1), np.linspace(0,1,n_grid_plot+1))
    grid_z  = scipy.interpolate.griddata(np.array([X1.flatten(), X2.flatten()]).T, z_plot.T.flatten(), (grid_x, grid_y), method=intp_method)
    color_z = scipy.interpolate.griddata(np.array([X1.flatten(), X2.flatten()]).T, c_plot.T.flatten(), (grid_x, grid_y), method=intp_method)
    grid_z = (1-landscape_base)*(grid_z-grid_z.min())/(grid_z.max()-grid_z.min()) + landscape_base
    adata.uns[landscape_key] = [grid_x,grid_y,grid_z,color_z]


def view_landscape(
    adata,
    source_cluster,
    target_clusters,
    path_key = "path",
    path_scaled_key = "path_scaled",
    landscape_key = "landscape",
    exp_key=None,
    dynamics_key="dynamics",
    bifurcation_diagram_key="bifurcation_diagram",
    pot_key = "potential",
    n_div=100,
    n_grid = 100,
    n_grid_plot = 200,
    intp_method='linear',
    merge_step = 5,
    valley_range = 0.1,
    view_path = False,
    figsize=(10,10),
    fontsize_label = 12,
    fontize_text = 12,
    PC=1,
    view_init_elev=20,
    view_init_azim=100,
    view_timegrid = False,
    view_axis = True,
    view_clustername = True,
    timegrid_n = 10,
    timegrid_alpha = 0.5,
    timegrid_color = "k",
    timegrid_lw = 0.5,
    basecolor = matplotlib.cm.Oranges(0.99),
    basecolor_alpha = 0.9,
    save=False,
    save_csv = False,
    save_dir=None,
    save_filename="Landscape",
    save_dpi = 100
    
):
    if dynamics_key not in adata.uns.keys():
        calc_dynamics(
            adata,
            source_cluster,
            target_clusters,
            path_key=path_key,
            exp_key=exp_key,
            dynamics_key=dynamics_key,
            n_div=n_div,
        )

    if landscape_key not in adata.uns.keys():
        calc_landscape(
            adata,
            source_cluster,
            target_clusters,
            path_key = path_key,
            path_scaled_key = path_scaled_key,
            landscape_key = landscape_key,
            exp_key=path_key,
            dynamics_key=dynamics_key,
            pot_key = pot_key,
            n_div = n_div,
            n_grid = n_grid,
            n_grid_plot = n_grid_plot,
            intp_method = intp_method,
            merge_step = merge_step,
            valley_range = valley_range,
            PC = PC,
        )

    grid_x,grid_y,grid_z,color_z = adata.uns[landscape_key]
    x_lim_min,x_lim_max = grid_x.min(),grid_x.max()
    n_grid_plot = grid_x.shape[0]-1

    cmap = plt.get_cmap("tab10")
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='3d')

    ls = matplotlib.colors.LightSource(270, 45)
    rgb = ls.shade(color_z, cmap=matplotlib.cm.gist_earth, vert_exag=0.1, blend_mode='soft',vmin=-0.05, vmax=1.1)
    ax.plot_surface(grid_x, grid_y, grid_z, facecolors=rgb,  linewidth=0, zorder=0, alpha=1, edgecolor='none',rstride=1, cstride=1, antialiased=False, shade=False)

    px_,py_,pz_ = grid_x[:,-1],grid_y[:,-1],grid_z[:,-1]
    verts = [[px_[i], py_[i], pz_[i]] for i in range(len(px_))] + [[px_[i], py_[i], 0] for i in reversed(range(len(px_)))]
    poly = mpl_toolkits.mplot3d.art3d.Poly3DCollection([verts], alpha=basecolor_alpha, facecolors=basecolor, edgecolors='none',zorder=0)
    ax.add_collection3d(poly)

    px_,py_,pz_ = grid_x[:,0],grid_y[:,0],grid_z[:,0]
    verts = [[px_[i], py_[i], pz_[i]] for i in range(len(px_))] + [[px_[i], py_[i], 0] for i in reversed(range(len(px_)))]
    poly = mpl_toolkits.mplot3d.art3d.Poly3DCollection([verts], alpha=basecolor_alpha, facecolors=basecolor, edgecolors='none',zorder=0)
    ax.add_collection3d(poly)

    px_,py_,pz_ = grid_x[-1],grid_y[-1],grid_z[-1]
    verts = [[px_[i], py_[i], pz_[i]] for i in range(len(px_))] + [[px_[i], py_[i], 0] for i in reversed(range(len(px_)))]
    poly = mpl_toolkits.mplot3d.art3d.Poly3DCollection([verts], alpha=basecolor_alpha, facecolors=basecolor, edgecolors='none',zorder=0)
    ax.add_collection3d(poly)

    px_,py_,pz_ = grid_x[0],grid_y[0],grid_z[0]
    verts = [[px_[i], py_[i], pz_[i]] for i in range(len(px_))] + [[px_[i], py_[i], 0] for i in reversed(range(len(px_)))]
    poly = mpl_toolkits.mplot3d.art3d.Poly3DCollection([verts], alpha=basecolor_alpha, facecolors=basecolor, edgecolors='none',zorder=0)
    ax.add_collection3d(poly)

    if view_path:
        for i in range(len(target_clusters)):
            name_i_ = source_cluster + "_" + target_clusters[i]
            py_ = np.linspace(0,1,n_div+1)
            px_ = adata.uns[bifurcation_diagram_key][name_i_][:,PC-1]
            pz_ = np.array([grid_z.flatten()[np.argmin(np.linalg.norm(np.array([x__,y__])-np.array([grid_x.flatten(),grid_y.flatten()]).T,axis=1))] for x__,y__ in zip(px_,py_)])
            # print(px_)
            ax.plot(px_, py_, pz_, lw=4, color="w", zorder=10, alpha=0.8)
            ax.plot(px_, py_, pz_, lw=2, c=cmap(i), zorder=100)


    if view_clustername:
        text = ax.text(np.linspace(x_lim_min,x_lim_max, n_grid_plot)[np.argmin(grid_z[0])],0,np.max(grid_z[:,0]),source_cluster,fontsize=fontize_text,va="bottom",ha="center",zorder=100,fontweight="bold")
        text.set_path_effects([matplotlib.patheffects.withStroke(linewidth=3, foreground='white')])
        for i in range(len(target_clusters)):
            name_i_ = source_cluster + "_" + target_clusters[i]
            z_ = np.min(grid_z[-1])
            text = ax.text(adata.uns[bifurcation_diagram_key][name_i_][:,PC-1][-1],1.00,z_,
                    target_clusters[i],fontsize=fontize_text,va="top",ha="center",zorder=100,fontweight="bold",c=cmap(i))
            text.set_path_effects([matplotlib.patheffects.withStroke(linewidth=3, foreground='white')])

    if view_timegrid:
        for y__ in np.linspace(0,1,timegrid_n+1):
            idx__ = np.argmin(np.abs(grid_y[:,0]-y__))
            ax.plot(grid_x[0],y__,grid_z[idx__],zorder=100,c=timegrid_color,alpha=timegrid_alpha,lw=timegrid_lw,ls=(5,(10,5)))
            ax.text(grid_x[0,-1],y__,grid_z[idx__,-1],' {:.2g}'.format(y__),zorder=200)

    ax.set_xlim(x_lim_min,x_lim_max)
    ax.set_ylim(0,1.01)
    ax.set_zlim(0,1)
    ax.set_xlabel("State space (PC%s)" % PC,fontsize=fontsize_label)
    ax.set_ylabel("Time",fontsize=fontsize_label)
    ax.set_zlabel("Potential",fontsize=fontsize_label)
    ax.set_box_aspect((1, 1, 0.6))
    ax.view_init(elev=view_init_elev, azim=view_init_azim)
    ax.xaxis.pane.set_edgecolor('w')
    ax.yaxis.pane.set_edgecolor('w')
    ax.zaxis.pane.set_edgecolor('w')
    ax.xaxis.pane.set_facecolor("w")
    ax.yaxis.pane.set_facecolor("w")
    ax.zaxis.pane.set_facecolor("w")
    ax.grid(False)
    if view_axis == False:
        ax.axis("off")
    plt.show()

    if save:
        filename = (
            "%s" % (save_filename)
            if save_dir == None
            else "%s/%s" % (save_dir, save_filename)
        )
        fig.savefig(filename + ".png", bbox_inches="tight",dpi=save_dpi)
    if save_csv:
        filename = (
            "%s" % (save_filename)
            if save_dir == None
            else "%s/%s" % (save_dir, save_filename)
        )
        # pd.DataFrame({"X":grid_x.flatten(), "Y":grid_y.flatten(), "Potential":grid_z.flatten()})
        out_ = pd.DataFrame({"X":(grid_x.flatten()-x_lim_min)/(x_lim_max-x_lim_min), 
                             "Y":grid_y.flatten(), 
                             "Potential":grid_z.flatten(), 
                             "Potential_color":color_z.flatten(),
                             "Velocity_x":2*dZ_dx.flatten()/(x_lim_max-x_lim_min), 
                             "Velocity_y":dZ_dy.flatten()/(1-valley_range)})
        out_.to_csv(filename+".csv")

def view_landscape_3D(
    adata,
    source_cluster,
    target_clusters,
    path_key = "path",
    path_scaled_key = "path_scaled",
    landscape_key = "landscape",
    exp_key=None,
    dynamics_key="dynamics",
    bifurcation_diagram_key="bifurcation_diagram",
    pot_key = "potential",
    n_div=100,
    n_grid = 100,
    n_grid_plot = 200,
    intp_method='linear',
    merge_step = 5,
    valley_range = 0.1,
    fontsize_label = 18,
    fontize_text = 18,
    PC=1,
    view_timegrid = False,
    view_clustername = True,
    timegrid_n = 10,
    timegrid_alpha = 0.8,
    timegrid_color = "Black",
    timegrid_lw = 2,
    timegrid_fontsize = 14,
    save=False,
    save_dir=None,
    save_filename = "Landscape_3D",
    title="Landscape",
    bgcolor="white",
    gridcolor="gray",
    width=1200,
    height=750,
    scene_aspectratio=dict(x=1.0, y=1.0, z=0.5),
    filename="CellMap_view_3D",
    camera=dict(eye=dict(x=-0.6, y=1.25, z=0.5),up=dict(x=0, y=0, z=1),center=dict(x=0, y=0.2, z=-0.1)),
):
    if dynamics_key not in adata.uns.keys():
        calc_dynamics(
            adata,
            source_cluster,
            target_clusters,
            path_key=path_key,
            exp_key=exp_key,
            dynamics_key=dynamics_key,
            n_div=n_div,
        )

    if landscape_key not in adata.uns.keys():
        calc_landscape(
            adata,
            source_cluster,
            target_clusters,
            path_key = path_key,
            path_scaled_key = path_scaled_key,
            landscape_key = landscape_key,
            exp_key=path_key,
            dynamics_key=dynamics_key,
            pot_key = pot_key,
            n_div = n_div,
            n_grid = n_grid,
            n_grid_plot = n_grid_plot,
            intp_method = intp_method,
            merge_step = merge_step,
            valley_range = valley_range,
            PC = PC,
        )

    grid_x,grid_y,grid_z,color_z = adata.uns[landscape_key]
    x_lim_min,x_lim_max = grid_x.min(),grid_x.max()
    n_grid_plot = grid_x.shape[0]-1

    cmap_ = matplotlib.cm.get_cmap("gist_earth")
    n_cmap_ = 256
    indices = np.linspace(0, 1, n_cmap_)
    colorscale = []
    for i, val in enumerate(indices):
        rgba = cmap_(val)
        rgb_str = f'rgb({int(rgba[0]*255)}, {int(rgba[1]*255)}, {int(rgba[2]*255)})'
        colorscale.append([val, rgb_str])

    x_,y_,z_ = grid_x.flatten(), grid_y.flatten(), grid_z.flatten()
    triangles = matplotlib.tri.Triangulation(x_, y_).triangles
    data = []
    data.append(go.Mesh3d(
        x=x_,
        y=y_,
        z=z_,
        i=triangles[:, 0],
        j=triangles[:, 1],
        k=triangles[:, 2],
        intensity=color_z.flatten(),
        colorscale=colorscale,
        cmin=-0.05,
        cmax=1.1,
    ))

    n_ = len(grid_x[:,-1].flatten())
    x_= np.concatenate([grid_x[:,-1].flatten(),grid_x[:,-1].flatten()])
    y_ = np.concatenate([grid_y[:,-1].flatten(),grid_y[:,-1].flatten()])
    z_ = np.concatenate([np.repeat(0,grid_x.shape[1]),grid_z[:,-1].flatten()])
    triangles = np.concatenate([[[i,i+1,i+n_] for i in range(n_-1)],[[i+1,i+n_,i+n_+1] for i in range(n_-1)]])
    data.append(go.Mesh3d(
        x=x_,
        y=y_,
        z=z_,
        i=triangles[:, 0],
        j=triangles[:, 1],
        k=triangles[:, 2],
        color="#822804",
    ))

    n_ = len(grid_x[:,-1].flatten())
    x_= np.concatenate([grid_x[:,0].flatten(),grid_x[:,0].flatten()])
    y_ = np.concatenate([grid_y[:,-1].flatten(),grid_y[:,-1].flatten()])
    z_ = np.concatenate([np.repeat(0,grid_x.shape[1]),grid_z[:,0].flatten()])
    triangles = np.concatenate([[[i,i+1,i+n_] for i in range(n_-1)],[[i+1,i+n_,i+n_+1] for i in range(n_-1)]])
    data.append(go.Mesh3d(
        x=x_,
        y=y_,
        z=z_,
        i=triangles[:, 0],
        j=triangles[:, 1],
        k=triangles[:, 2],
        color="#822804",
    ))

    n_ = len(grid_y[0].flatten())
    x_= np.concatenate([grid_x[0].flatten(),grid_x[0].flatten()])
    y_ = np.concatenate([grid_y[0].flatten(),grid_y[0].flatten()])
    z_ = np.concatenate([np.repeat(0,grid_x.shape[1]),grid_z[0].flatten()])
    triangles = np.concatenate([[[i,i+1,i+n_] for i in range(n_-1)],[[i+1,i+n_,i+n_+1] for i in range(n_-1)]])
    data.append(go.Mesh3d(
        x=x_,
        y=y_,
        z=z_,
        i=triangles[:, 0],
        j=triangles[:, 1],
        k=triangles[:, 2],
        color="#822804",
    ))


    n_ = len(grid_y[0].flatten())
    x_= np.concatenate([grid_x[-1].flatten(),grid_x[-1].flatten()])
    y_ = np.concatenate([grid_y[-1].flatten(),grid_y[-1].flatten()])
    z_ = np.concatenate([np.repeat(0,grid_x.shape[1]),grid_z[-1].flatten()])
    triangles = np.concatenate([[[i,i+1,i+n_] for i in range(n_-1)],[[i+1,i+n_,i+n_+1] for i in range(n_-1)]])
    data.append(go.Mesh3d(
        x=x_,
        y=y_,
        z=z_,
        i=triangles[:, 0],
        j=triangles[:, 1],
        k=triangles[:, 2],
        color="#822804",
    ))
    annotations = []
    if view_clustername:
        annotations = [
            dict(
                showarrow=False,
                x=adata.uns[bifurcation_diagram_key][source_cluster + "_" + target_clusters[i]][:,PC-1][-1],
                y=1,
                z=np.min(grid_z[-1]),
                text="<b>%s<b>" % str(target_clusters[i]),
                font=dict(size=fontize_text, color="rgba(0,0,0,1)"),
                bgcolor="rgba(255,255,255,0.7)",
                xanchor='center',
                yanchor='top',
            )
            for i in range(len(target_clusters))
        ]
        annotations.append(
            dict(
                showarrow=False,
                x=np.linspace(x_lim_min,x_lim_max, n_grid_plot)[np.argmin(grid_z[0])],
                y=0,
                z=np.max(grid_z[:,0]),
                text="<b>%s<b>" % str(source_cluster),
                font=dict(size=fontize_text, color="rgba(0,0,0,1)"),
                bgcolor="rgba(255,255,255,0.7)",
                xanchor='center',
                yanchor='bottom',
            )
        )
    if view_timegrid:
        for y__ in np.linspace(0,1,timegrid_n+1):
            idx__ = np.argmin(np.abs(grid_y[:,0]-y__))
            data.append(
                go.Scatter3d(
                    x=grid_x[0],
                    y=[y__] * grid_x.shape[1],
                    z=grid_z[idx__]+5e-3,
                    mode="lines",
                    line=dict(
                        color=timegrid_color,
                        width=timegrid_lw,
                        dash="dash",
                    ),
                    opacity=timegrid_alpha,
                    showlegend=False,
                )
            )

            annotations.append(
                dict(
                    x=grid_x[0,-1],      
                    y=y__,
                    z=grid_z[idx__, -1], 
                    text=f"{y__:.2g}",  
                    showarrow=False, 
                    font=dict(color="black", size=timegrid_fontsize),
                    xanchor='left',
                    yanchor='bottom',
                )
            )


    layout = go.Layout(
        title=title,
        width=width,
        height=height,
        margin=dict(l=0, r=0, b=0, t=50),
        scene_camera=camera,
        scene=dict(
            annotations=annotations,
            xaxis=dict(
                title=dict(
                    text="State space (PC%s)" % PC,
                    font=dict(size=fontsize_label)
                ),
                backgroundcolor=bgcolor,
                gridcolor=gridcolor,
            ),
            yaxis=dict(
                title=dict(
                    text="Time",
                    font=dict(size=fontsize_label)
                ),
                backgroundcolor=bgcolor,
                gridcolor=gridcolor,
            ),
            zaxis=dict(
                title=dict(
                    text="Potential",
                    font=dict(size=fontsize_label)
                ),
                backgroundcolor=bgcolor,
                gridcolor=gridcolor,
            ),
        ),
        meta=dict(),
        scene_aspectratio=scene_aspectratio,
        coloraxis=dict(cmin=0, cmax=1),
    )

    fig = go.Figure(data=data, layout=layout)
    fig.show()

    if save:
        filename = (
            "%s" % (save_filename)
            if save_dir == None
            else "%s/%s" % (save_dir, save_filename)
        )
        plotly.offline.plot(fig, filename=filename + ".html")

# def view_landscape(
#     adata,
#     source_cluster,
#     target_clusters,
#     path_key = "path",
#     path_scaled_key = "path_scaled",
#     exp_key=None,
#     dynamics_key="dynamics",
#     bifurcation_diagram_key="bifurcation_diagram",
#     pot_key = "potential",
#     n_div=100,
#     n_grid = 100,
#     n_grid_plot = 200,
#     intp_method='linear',
#     merge_step = 5,
#     valley_range = 0.1,
#     view_path = False,
#     figsize=(10,10),
#     fontsize_label = 12,
#     fontize_text = 12,
#     PC=1,
#     view_init_elev=20,
#     view_init_azim=100,
#     view_timegrid = False,
#     view_axis = True,
#     view_clustername = True,
#     timegrid_n = 10,
#     timegrid_alpha = 0.5,
#     timegrid_color = "k",
#     timegrid_lw = 0.5,
#     save=False,
#     save_csv = False,
#     save_dir=None,
#     save_filename="Landscape",
#     save_dpi = 100
    
# ):
#     if dynamics_key not in adata.uns.keys():
#         calc_dynamics(
#             adata,
#             source_cluster,
#             target_clusters,
#             path_key=path_key,
#             exp_key=exp_key,
#             dynamics_key=dynamics_key,
#             n_div=n_div,
#         )

#     if bifurcation_diagram_key not in adata.uns.keys():
#         calc_bifurcation_diagram(
#             adata,
#             source_cluster,
#             target_clusters,
#             path_key=path_key,
#             exp_key=exp_key,
#             dynamics_key=dynamics_key,
#             bifurcation_diagram_key=bifurcation_diagram_key,
#             n_div=n_div,
#             PC=PC,
#         )

#     data_all_ = adata.uns[dynamics_key+"_PC"]
#     x_lim_min,x_lim_max = np.min(data_all_[:,:,:,PC-1]), np.max(data_all_[:,:,:,PC-1])
#     x_ = np.linspace(x_lim_min,x_lim_max, n_grid)
#     z_plot = np.zeros([n_grid,n_div-merge_step])
#     c_plot = np.zeros([n_grid,n_div-merge_step])
#     pot_ = adata.obs[pot_key]
#     cmap = plt.get_cmap("tab10")

#     for ti_ in range(n_div-merge_step):
#         for i in range(len(data_all_)):
#             name_i_ = source_cluster + "_" + target_clusters[i]
#             data_ = data_all_[i,:,ti_:ti_+merge_step+1:,PC-1].flatten()
#             pot_t_ = pot_[adata.uns[path_scaled_key][name_i_][:,ti_:ti_+merge_step+1].flatten()].mean()/pot_.max()
#             kde_model = scipy.stats.gaussian_kde(data_)
#             dens_ = kde_model(x_)
#             z_ = pot_t_-valley_range*dens_/np.max(dens_)
#             z_plot[:,ti_] += z_/len(data_all_)
#             c_plot[:,ti_] += dens_/np.max(dens_)/len(data_all_)
#         c_plot[:,ti_] = 1-c_plot[:,ti_]

#     x1 = x_
#     x2 = np.linspace(0,1,n_div-merge_step)
#     X1, X2 = np.meshgrid(x1, x2)

#     grid_x, grid_y = np.meshgrid(np.linspace(x_lim_min,x_lim_max,n_grid_plot+1), np.linspace(0,1,n_grid_plot+1))
#     grid_z  = scipy.interpolate.griddata(np.array([X1.flatten(), X2.flatten()]).T, z_plot.T.flatten(), (grid_x, grid_y), method=intp_method)
#     color_z = scipy.interpolate.griddata(np.array([X1.flatten(), X2.flatten()]).T, c_plot.T.flatten(), (grid_x, grid_y), method=intp_method)
#     dZ_dy, dZ_dx = np.gradient(grid_z, np.linspace(x_lim_min,x_lim_max,n_grid_plot+1), np.linspace(0,1,n_grid_plot+1))
#     vel_rate = 1
#     dZ_dy, dZ_dx = -vel_rate*(1-np.array(color_z,dtype=int))*dZ_dy, -vel_rate*(1-color_z)*dZ_dx
#     dZ_dx = scipy.interpolate.griddata(np.array([grid_x.flatten(), grid_y.flatten()]).T, dZ_dx.flatten(), (grid_x, grid_y), method="cubic")
#     dZ_dy = scipy.interpolate.griddata(np.array([grid_x.flatten(), grid_y.flatten()]).T, dZ_dy.flatten(), (grid_x, grid_y), method="cubic")

#     basecolor = matplotlib.cm.Oranges(0.99)
#     basecolor_alpha = 0.9

#     fig = plt.figure(figsize=figsize)
#     ax = fig.add_subplot(111, projection='3d')

#     ls = matplotlib.colors.LightSource(270, 45)
#     rgb = ls.shade(color_z, cmap=matplotlib.cm.gist_earth, vert_exag=0.1, blend_mode='soft',vmin=-0.05, vmax=1.1)
#     ax.plot_surface(grid_x, grid_y, grid_z, facecolors=rgb,  linewidth=0, zorder=0, alpha=1, edgecolor='none',rstride=1, cstride=1, antialiased=False, shade=False)

#     px_,py_,pz_ = grid_x[:,-1],grid_y[:,-1],grid_z[:,-1]
#     verts = [[px_[i], py_[i], pz_[i]] for i in range(len(px_))] + [[px_[i], py_[i], 0] for i in reversed(range(len(px_)))]
#     poly = mpl_toolkits.mplot3d.art3d.Poly3DCollection([verts], alpha=basecolor_alpha, facecolors=basecolor, edgecolors='none',zorder=0)
#     ax.add_collection3d(poly)

#     px_,py_,pz_ = grid_x[:,0],grid_y[:,0],grid_z[:,0]
#     verts = [[px_[i], py_[i], pz_[i]] for i in range(len(px_))] + [[px_[i], py_[i], 0] for i in reversed(range(len(px_)))]
#     poly = mpl_toolkits.mplot3d.art3d.Poly3DCollection([verts], alpha=basecolor_alpha, facecolors=basecolor, edgecolors='none',zorder=0)
#     ax.add_collection3d(poly)

#     px_,py_,pz_ = grid_x[-1],grid_y[-1],grid_z[-1]
#     verts = [[px_[i], py_[i], pz_[i]] for i in range(len(px_))] + [[px_[i], py_[i], 0] for i in reversed(range(len(px_)))]
#     poly = mpl_toolkits.mplot3d.art3d.Poly3DCollection([verts], alpha=basecolor_alpha, facecolors=basecolor, edgecolors='none',zorder=0)
#     ax.add_collection3d(poly)

#     px_,py_,pz_ = grid_x[0],grid_y[0],grid_z[0]
#     verts = [[px_[i], py_[i], pz_[i]] for i in range(len(px_))] + [[px_[i], py_[i], 0] for i in reversed(range(len(px_)))]
#     poly = mpl_toolkits.mplot3d.art3d.Poly3DCollection([verts], alpha=basecolor_alpha, facecolors=basecolor, edgecolors='none',zorder=0)
#     ax.add_collection3d(poly)

#     if view_path:
#         for i in range(len(target_clusters)):
#             name_i_ = source_cluster + "_" + target_clusters[i]
#             py_ = np.linspace(0,1,n_div+1)
#             px_ = adata.uns[bifurcation_diagram_key][name_i_][:,PC-1]
#             pz_ = np.array([grid_z.flatten()[np.argmin(np.linalg.norm(np.array([x__,y__])-np.array([grid_x.flatten(),grid_y.flatten()]).T,axis=1))] for x__,y__ in zip(px_,py_)])
#             # print(px_)
#             ax.plot(px_, py_, pz_, lw=4, color="w", zorder=10, alpha=0.8)
#             ax.plot(px_, py_, pz_, lw=2, c=cmap(i), zorder=100)


#     if view_clustername:
#         text = ax.text(np.linspace(x_lim_min,x_lim_max, n_grid_plot)[np.argmin(grid_z[0])],0,np.max(grid_z[:,0]),source_cluster,fontsize=fontize_text,va="bottom",ha="center",zorder=100,fontweight="bold")
#         text.set_path_effects([matplotlib.patheffects.withStroke(linewidth=3, foreground='white')])
#         for i in range(len(target_clusters)):
#             name_i_ = source_cluster + "_" + target_clusters[i]
#             z_ = np.min(grid_z[-1])
#             text = ax.text(adata.uns[bifurcation_diagram_key][name_i_][:,PC-1][-1],1.00,z_,
#                     target_clusters[i],fontsize=fontize_text,va="top",ha="center",zorder=100,fontweight="bold",c=cmap(i))
#             text.set_path_effects([matplotlib.patheffects.withStroke(linewidth=3, foreground='white')])

#     if view_timegrid:
#         for y__ in np.linspace(0,1,timegrid_n+1):
#             idx__ = np.argmin(np.abs(grid_y[:,0]-y__))
#             ax.plot(grid_x[0],y__,grid_z[idx__],zorder=100,c=timegrid_color,alpha=timegrid_alpha,lw=timegrid_lw,ls=(5,(10,5)))
#             ax.text(grid_x[0,-1],y__,grid_z[idx__,-1],' {:.2g}'.format(y__),zorder=200)

#     ax.set_xlim(x_lim_min,x_lim_max)
#     ax.set_ylim(0,1.01)
#     ax.set_zlim(0,1)
#     ax.set_xlabel("State space (PC%s)" % PC,fontsize=fontsize_label)
#     ax.set_ylabel("Time",fontsize=fontsize_label)
#     ax.set_zlabel("Potential",fontsize=fontsize_label)
#     ax.set_box_aspect((1, 1, 0.6))
#     ax.view_init(elev=view_init_elev, azim=view_init_azim)
#     ax.xaxis.pane.set_edgecolor('w')
#     ax.yaxis.pane.set_edgecolor('w')
#     ax.zaxis.pane.set_edgecolor('w')
#     ax.xaxis.pane.set_facecolor("w")
#     ax.yaxis.pane.set_facecolor("w")
#     ax.zaxis.pane.set_facecolor("w")
#     ax.grid(False)
#     if view_axis == False:
#         ax.axis("off")
#     plt.show()

#     if save:
#         filename = (
#             "%s" % (save_filename)
#             if save_dir == None
#             else "%s/%s" % (save_dir, save_filename)
#         )
#         fig.savefig(filename + ".png", bbox_inches="tight",dpi=save_dpi)
#     if save_csv:
#         filename = (
#             "%s" % (save_filename)
#             if save_dir == None
#             else "%s/%s" % (save_dir, save_filename)
#         )
#         # pd.DataFrame({"X":grid_x.flatten(), "Y":grid_y.flatten(), "Potential":grid_z.flatten()})
#         out_ = pd.DataFrame({"X":(grid_x.flatten()-x_lim_min)/(x_lim_max-x_lim_min), 
#                              "Y":grid_y.flatten(), 
#                              "Potential":grid_z.flatten(), 
#                              "Potential_color":color_z.flatten(),
#                              "Velocity_x":2*dZ_dx.flatten()/(x_lim_max-x_lim_min), 
#                              "Velocity_y":dZ_dy.flatten()/(1-valley_range)})
#         out_.to_csv(filename+".csv")


def calc_gene_atlas(
    adata,
    source_cluster,
    target_clusters,
    path_key="path",
    exp_key=None,
    dynamics_key="dynamics",
    gene_atlas_key="gene_atlas",
    n_div=100,
    n_neighbors=15,
    min_dist=0.3,
    seed=0,
    threshold_min=1,
    n_clusters=20,
    n_components=2,
):
    if dynamics_key not in adata.uns.keys():
        calc_dynamics(
            adata,
            source_cluster,
            target_clusters,
            path_key=path_key,
            exp_key=exp_key,
            dynamics_key=dynamics_key,
            n_div=n_div,
        )

    gene_dynamics_ = adata.uns[dynamics_key]
    gene_dynamics_all_ = np.empty([0, n_div + 1], dtype=float)
    gene_dynamics_all_norm_ = np.empty([0, n_div + 1], dtype=float)
    idx_gene_dynamics_ = [0]
    for i in range(len(target_clusters)):
        name_i_ = source_cluster + "_" + target_clusters[i]
        max_ = np.max(gene_dynamics_[name_i_], axis=0)
        idx_ = max_ > threshold_min
        adata.var["expressed_" + name_i_] = idx_
        idx_gene_dynamics_ = np.append(
            idx_gene_dynamics_, idx_gene_dynamics_[i] + sum(idx_)
        )
        gene_dynamics_all_ = np.vstack(
            (gene_dynamics_all_, gene_dynamics_[name_i_][:, idx_].T)
        )
        gene_dynamics_all_norm_ = np.vstack(
            (gene_dynamics_all_norm_, (gene_dynamics_[name_i_][:, idx_] / max_[idx_]).T)
        )

    umap_ = umap.UMAP(
        n_components=n_components,
        random_state=seed,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
    )
    gene_dynamics_all_umap_ = umap_.fit_transform(gene_dynamics_all_)

    data_ = gene_dynamics_all_umap_
    gm = sklearn.mixture.GaussianMixture(n_components=n_clusters, random_state=0).fit(
        data_
    )
    clusters_tmp_ = gm.predict(data_)
    pc1_ = sklearn.decomposition.PCA(n_components=1).fit_transform(data_)[:, 0]
    pc1_ = np.sign(pc1_ @ gene_dynamics_all_umap_[:, 0]) * pc1_
    pc1_order_ = np.argsort(
        [np.mean(pc1_[clusters_tmp_ == i]) for i in range(n_clusters)]
    )
    dict_sort_ = dict(zip(pc1_order_, np.unique(clusters_tmp_)))
    clusters_ = np.array([dict_sort_[c] for c in clusters_tmp_])

    texts_ = []
    index_ = []
    s_, e_ = 0, 0
    for i in range(len(target_clusters)):
        name_i_ = source_cluster + "_" + target_clusters[i]
        idx_ = adata.var["expressed_" + name_i_]
        gene_list_ = adata.var.index[idx_].values
        e_ += sum(idx_)
        txt_ = (
            gene_list_
            + "<br>"
            + target_clusters[i]
            + "<br>cluster "
            + np.array(clusters_[s_:e_] + 1, dtype=str)
        )  # + '<br><img src="'+image_+'" width="200">'
        texts_ = np.append(texts_, txt_)
        index_ = np.append(
            index_, source_cluster + "_" + target_clusters[i] + "_" + gene_list_
        )
        adata.var["clusters_" + name_i_] = -np.ones(adata.shape[1], dtype=int)
        adata.var["clusters_" + name_i_][idx_] = clusters_[s_:e_]
        s_ += sum(idx_)

    adata.uns[gene_atlas_key] = {
        "index": index_,
        "texts": texts_,
        "dynamics": gene_dynamics_all_,
        "dynamics_norm": gene_dynamics_all_norm_,
        "gene_atlas": gene_dynamics_all_umap_,
        "clusters": clusters_,
    }


def gene_atlas(
    adata,
    source_cluster,
    target_clusters,
    target_genes=[],
    dynamics_key="dynamics",
    gene_atlas_key="gene_atlas",
    normalization=False,
    n_div=100,
    n_neighbors=15,
    min_dist=0.3,
    seed=0,
    threshold_min=1,
    n_clusters=20,
    n_components=2,
    pt_size=5,
    save=False,
    save_dir=None,
    save_filename="Gene_atlas",
    save_type="html",
):
    if gene_atlas_key not in adata.uns.keys():
        calc_gene_atlas(
            adata,
            source_cluster,
            target_clusters,
            dynamics_key=dynamics_key,
            gene_atlas_key=gene_atlas_key,
            n_div=n_div,
            n_neighbors=n_neighbors,
            min_dist=min_dist,
            seed=seed,
            threshold_min=threshold_min,
            n_clusters=n_clusters,
            n_components=n_components,
        )
    elif n_clusters != len(np.unique(adata.uns[gene_atlas_key]["clusters"])):
        calc_gene_atlas(
            adata,
            source_cluster,
            target_clusters,
            dynamics_key=dynamics_key,
            gene_atlas_key=gene_atlas_key,
            n_div=n_div,
            n_neighbors=n_neighbors,
            min_dist=min_dist,
            seed=seed,
            threshold_min=threshold_min,
            n_clusters=n_clusters,
            n_components=n_components,
        )

    texts_ = adata.uns[gene_atlas_key]["texts"]
    if normalization:
        gene_dynamics_all_ = adata.uns[gene_atlas_key]["dynamics_norm"]
    else:
        gene_dynamics_all_ = adata.uns[gene_atlas_key]["dynamics"]
    gene_dynamics_all_umap_ = adata.uns[gene_atlas_key]["gene_atlas"]
    clusters_ = adata.uns[gene_atlas_key]["clusters"]
    cluster_set_ = np.unique(clusters_)

    x_data = gene_dynamics_all_umap_[:, 0]
    y_data = gene_dynamics_all_umap_[:, 1]

    color_clusters_ = np.array(
        [
            "rgb" + str(tuple(int(i * 255) for i in plt.get_cmap("tab20")(c % 20)))
            for c in clusters_
        ]
    )
    color_celltypes_ = np.empty(len(gene_dynamics_all_), dtype=object)

    annotations = [
        go.layout.Annotation(
            xref="paper",
            yref="paper",
            x=0.01,
            y=0.99,
            text="<b>Gene Atlas<b>",
            font=dict(size=18, color="white"),
            showarrow=False,
        )
    ]
    s_, e_ = 0, 0
    for i in range(len(target_clusters)):
        name_i_ = source_cluster + "_" + target_clusters[i]
        idx_ = adata.var["expressed_" + name_i_]
        e_ += sum(idx_)
        gene_list_ = adata.var.index[idx_].values
        color_celltypes_[s_:e_] = "rgba" + str(plt.get_cmap("tab10")(i % 10))
        for gene in target_genes:
            if gene in gene_list_:
                x_pos = x_data[np.arange(sum(idx_))[gene_list_ == gene][0] + s_]
                y_pos = y_data[np.arange(sum(idx_))[gene_list_ == gene][0] + s_]
                annotations.append(
                    go.layout.Annotation(
                        x=x_pos,
                        y=y_pos,
                        xref="x",
                        yref="y",
                        text="<b>%s_%s<b>" % (target_clusters[i], gene),
                        showarrow=True,
                        arrowhead=1,
                        arrowcolor="white",
                        font=dict(size=12, color="white"),
                    )
                )
        s_ += sum(idx_)

    data_clusters_ = [
        go.Scatter(
            x=x_data,
            y=y_data,
            mode="markers",
            marker=dict(
                color=color_clusters_,
                size=30,
                opacity=0.2,
            ),
            hoverinfo="skip",
            showlegend=False,
        )
    ]
    for c in cluster_set_:
        idx_ = clusters_ == c
        data_clusters_.append(
            go.Scatter(
                x=x_data[idx_],
                y=y_data[idx_],
                text=texts_[idx_],
                mode="markers",
                name="cluster " + str(c + 1),
                marker=dict(
                    color="rgb"
                    + str(tuple(int(i * 255) for i in plt.get_cmap("tab20")(c % 20))),
                    size=pt_size,
                    opacity=1,
                    line=dict(
                        color="white",
                        width=0.5,
                    ),
                ),
            )
        )

    s_, e_ = 0, 0
    data_celltypes_ = []
    for i in range(len(target_clusters)):
        name_i_ = source_cluster + "_" + target_clusters[i]
        idx_ = adata.var["expressed_" + name_i_]
        gene_list_ = adata.var.index[idx_].values
        e_ += sum(idx_)
        data_celltypes_.append(
            go.Scatter(
                x=x_data[s_:e_],
                y=y_data[s_:e_],
                text=texts_[s_:e_],
                mode="markers",
                name=target_clusters[i],
                marker=dict(
                    color=color_celltypes_[s_:e_],
                    size=pt_size,
                    opacity=1,
                    line=dict(
                        color="white",
                        width=0.5,
                    ),
                ),
            )
        )
        s_ += sum(idx_)

    layout = go.Layout(
        width=1200,
        height=800,
        plot_bgcolor="rgba(1,1,1,1)",
        paper_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(
            # showgrid=False,
            gridcolor="gray",
            gridwidth=1,
            griddash="dot",
            zeroline=False,
            showticklabels=False,
            layer="below traces",
        ),
        yaxis=dict(
            # showgrid=False,
            gridcolor="gray",
            gridwidth=1,
            griddash="dot",
            zeroline=False,
            showticklabels=False,
            layer="below traces",
        ),
        annotations=annotations,
    )

    fig1 = go.Figure(data=data_clusters_, layout=layout)
    pio.show(fig1)

    fig2 = go.Figure(data=data_celltypes_, layout=layout)
    pio.show(fig2)

    if save:
        filename = (
            "%s" % (save_filename)
            if save_dir == None
            else "%s/%s" % (save_dir, save_filename)
        )
        if save_type in ["png", "pdf", "svg", "eps"]:
            pio.write_image(fig1, filename + "_cluster." + save_type)
            pio.write_image(fig2, filename + "_celltype." + save_type)
        if save_type in ["html"]:
            plotly.offline.plot(fig1, filename=filename + "_cluster." + save_type)
            plotly.offline.plot(fig2, filename=filename + "_celltype." + save_type)


def gene_dynamics_clusters(
    adata,
    source_cluster,
    target_clusters,
    dynamics_key="dynamics",
    gene_atlas_key="gene_atlas",
    normalization=False,
    n_div=100,
    n_neighbors=15,
    min_dist=0.3,
    seed=0,
    threshold_min=1,
    n_clusters=20,
    n_components=2,
):
    if gene_atlas_key not in adata.uns.keys():
        calc_gene_atlas(
            adata,
            source_cluster,
            target_clusters,
            dynamics_key=dynamics_key,
            ene_atlas_key=gene_atlas_key,
            n_div=n_div,
            n_neighbors=n_neighbors,
            min_dist=min_dist,
            seed=seed,
            threshold_min=threshold_min,
            n_clusters=n_clusters,
            n_components=n_components,
        )

    if normalization:
        gene_dynamics_all_ = adata.uns[gene_atlas_key]["dynamics_norm"]
        ylabel_ = "normalized gene expression"
    else:
        gene_dynamics_all_ = adata.uns[gene_atlas_key]["dynamics"]
        ylabel_ = "gene expression"

    n_clusters = len(np.unique(adata.uns[gene_atlas_key]["clusters"]))
    for i in range(n_clusters):
        x_ = np.linspace(0, 1, n_div + 1)
        y_ = gene_dynamics_all_[adata.uns[gene_atlas_key]["clusters"] == i].T

        data_ = []
        for j in range(y_.shape[1]):
            data_.append(
                go.Scatter(
                    x=x_,
                    y=y_[:, j],
                    mode="lines",
                    text=adata.uns[gene_atlas_key]["texts"][
                        adata.uns[gene_atlas_key]["clusters"] == i
                    ][j],
                    name="",
                    # hoverinfo='skip',
                    showlegend=False,
                    opacity=0.8,
                    line=dict(
                        width=1,
                    ),
                )
            )
        layout = go.Layout(
            width=1200,
            height=400,
            plot_bgcolor="rgba(1,1,1,1)",
            paper_bgcolor="rgba(0,0,0,0)",
            xaxis=dict(
                range=[0, 1],
                # showgrid=False,
                title="Time",
                gridcolor="gray",
                gridwidth=1,
                griddash="dot",
                zeroline=False,
                # showticklabels=False,
                layer="below traces",
            ),
            yaxis=dict(
                # range=[0, 1],
                # showgrid=False,
                title="<b>Cluster %s</b><br>%s" % (str(i + 1), ylabel_),
                gridcolor="gray",
                gridwidth=1,
                griddash="dot",
                zeroline=False,
                # showticklabels=False,
                layer="below traces",
            ),
            # annotations=annotations
        )

        fig = go.Figure(data=data_, layout=layout)
        pio.show(fig)


def key_gene_dynamics(
    adata,
    source_cluster,
    target_clusters,
    time,
    n_genes=10,
    threshold_min=1,
    path_key="path",
    exp_key=None,
    dynamics_key="dynamics",
    gene_dynamics_stats_key="mean",
    n_div=100,
    fontsize_label=10,
    save=False,
    save_dir=None,
    save_filename="Key_gene_dynamics",
):
    if dynamics_key not in adata.uns.keys():
        calc_dynamics(
            adata,
            source_cluster,
            target_clusters,
            path_key=path_key,
            exp_key=exp_key,
            dynamics_key=dynamics_key,
            n_div=n_div,
        )

    idx_t_n_ = np.arange(n_div + 1)[(np.linspace(0, 1, n_div + 1) <= time)]
    idx_t_p_ = np.arange(n_div + 1)[(np.linspace(0, 1, n_div + 1) >= time)]

    columns_ = []
    for i in range(len(target_clusters)):
        for j in range(i + 1, len(target_clusters)):
            columns_ = np.append(
                columns_, target_clusters[i] + " vs " + target_clusters[j]
            )

    cmap_ = plt.get_cmap("tab10")
    vlines = [0, 0.2, 0.4, 0.6, 0.8, 1]
    sign_dict_ = {"1": "+", "-1": "-"}
    out_pd_ = pd.DataFrame(
        index=(np.arange(n_genes) + 1),
        columns=pd.MultiIndex.from_product([list(columns_), []]),
    )
    gene_dynamics_ = adata.uns[dynamics_key][gene_dynamics_stats_key]
    for i in range(len(target_clusters)):
        for j in range(i + 1, len(target_clusters)):
            name_i_ = source_cluster + "_" + target_clusters[i]
            name_j_ = source_cluster + "_" + target_clusters[j]

            max_i_, max_j_ = np.max(gene_dynamics_[name_i_], axis=0), np.max(
                gene_dynamics_[name_j_], axis=0
            )
            idx_max_ = (max_i_ > threshold_min) & (max_j_ > threshold_min)

            vol_p_ = np.sum(
                np.abs(
                    gene_dynamics_[name_i_][idx_t_p_]
                    - gene_dynamics_[name_j_][idx_t_p_]
                ),
                axis=0,
            )
            vol_n_ = np.sum(
                np.abs(
                    gene_dynamics_[name_i_][idx_t_n_]
                    - gene_dynamics_[name_j_][idx_t_n_]
                ),
                axis=0,
            )
            diff_ = vol_p_ / (1e-5 + vol_n_)
            diff_order_ = np.argsort(diff_[idx_max_])[::-1]

            out_pd_[
                (target_clusters[i] + " vs " + target_clusters[j], "gene")
            ] = adata.var.index[idx_max_][diff_order_[:n_genes]]
            sign_ = [
                int(
                    np.sign(
                        np.sum(
                            gene_dynamics_[name_i_][:, idx_max_][:, diff_order_[i_]][
                                idx_t_p_
                            ]
                            - gene_dynamics_[name_j_][:, idx_max_][:, diff_order_[i_]][
                                idx_t_p_
                            ]
                        )
                    )
                )
                for i_ in range(n_genes)
            ]
            out_pd_[
                (target_clusters[i] + " vs " + target_clusters[j], target_clusters[i])
            ] = [sign_dict_[str(s_)] for s_ in sign_]
            out_pd_[
                (target_clusters[i] + " vs " + target_clusters[j], target_clusters[j])
            ] = [sign_dict_[str(-s_)] for s_ in sign_]

            vline_labels = np.append(
                np.append("Source (0)\n%s" % source_cluster, np.array(vlines)[1:-1]),
                "Target (1)",
            )
            for i_ in range(n_genes):
                fig, ax = plt.subplots(1, 1, figsize=(8, 2), tight_layout=True)
                ax.plot(
                    np.linspace(0, 1, n_div + 1),
                    gene_dynamics_[name_i_][:, idx_max_][:, diff_order_[i_]],
                    color=cmap_(i),
                    zorder=2,
                )
                ax.text(
                    1,
                    gene_dynamics_[name_i_][:, idx_max_][:, diff_order_[i_]][-1],
                    " " + target_clusters[i],
                    fontsize=fontsize_label,
                    va="center",
                    ha="left",
                )
                ax.plot(
                    np.linspace(0, 1, n_div + 1),
                    gene_dynamics_[name_j_][:, idx_max_][:, diff_order_[i_]],
                    color=cmap_(j),
                    zorder=2,
                )
                ax.text(
                    1,
                    gene_dynamics_[name_j_][:, idx_max_][:, diff_order_[i_]][-1],
                    " " + target_clusters[j],
                    fontsize=fontsize_label,
                    va="center",
                    ha="left",
                )
                ax.set_title(adata.var.index[idx_max_][diff_order_[i_]])
                ax.axvline(time, color="r", zorder=1)
                ax.text(
                    time,
                    0.95,
                    str(time) + " ",
                    color="r",
                    zorder=1,
                    va="top",
                    ha="right",
                    transform=ax.transAxes,
                )
                ax.set_xlim([0, 1])
                ax.set_xticks(vlines)
                ax.set_xticklabels(vline_labels, fontsize=fontsize_label)
                ax.spines["top"].set_visible(False)
                ax.spines["right"].set_visible(False)
                if save:
                    filename = (
                        "%s" % (save_filename)
                        if save_dir == None
                        else "%s/%s" % (save_dir, save_filename)
                    )
                    filename += target_clusters[i] + "_" + target_clusters[j]
                    filename += (
                        "_{}".format(round(time, len(str(n_div))))
                        + "_"
                        + "%02d_" % (i_ + 1)
                        + adata.var.index[idx_max_][diff_order_[i_]]
                    )
                    fig.savefig(filename + ".png", bbox_inches="tight")
    display(out_pd_)
    
def calc_GRN(
    adata,
    source_cluster,
    target_clusters,
    path_key="path",
    exp_key=None,
    grn_key="GRN",
    n_components=20,
):
    
    data_exp = _set_expression_data(adata, exp_key)

    adata.uns[grn_key] = {}
    for i in range(len(target_clusters)):
        name_i_ = source_cluster + "_" + target_clusters[i]
        adata.uns[grn_key][name_i_] = np.empty([adata.shape[1], adata.shape[1]])
        X_ = np.empty([adata.shape[1], 0], dtype=float)
        Y_ = np.empty([adata.shape[1], 0], dtype=float)
        ALL_ = np.empty([adata.shape[1], 0], dtype=float)
        for k in range(len(adata.uns[path_key][name_i_])):
            idx_cells_ = np.array(adata.uns[path_key][name_i_][k])
            n_nodes_ = len(data_exp[idx_cells_])
            X_ = np.hstack((X_, data_exp[idx_cells_][:-1].T))
            Y_ = np.hstack((Y_, data_exp[idx_cells_][1:].T))
            ALL_ = np.hstack((ALL_, data_exp[idx_cells_].T))
        pca_ = sklearn.decomposition.PCA(n_components=n_components)
        pca_.fit(ALL_.T)

        X_Lasso = pca_.transform(X_.T)
        Y_Lasso = pca_.transform(Y_.T)

        alphas_cv = np.logspace(-2, 4, num=20)
        clf_cv = sklearn.linear_model.MultiTaskLassoCV(
            alphas=alphas_cv, cv=3, fit_intercept=False
        )
        clf_cv.fit(X_Lasso, Y_Lasso)
        clf = sklearn.linear_model.Lasso(alpha=clf_cv.alpha_)
        clf.fit(X_Lasso, Y_Lasso)
        A_pca_ = pca_.components_.T @ clf.coef_ @ pca_.components_
        adata.uns[grn_key][name_i_] = {"GRN_matrix": A_pca_, "reg_param": clf_cv.alpha_}


def view_GRN(
    adata,
    source_cluster,
    target_clusters,
    exp_key=None,
    grn_key="GRN",
    genes=None,
    n_genes=20,
    save=False,
    save_dir=None,
    save_filename="GRN",
):
    if genes == None:
        data_exp = _set_expression_data(adata, exp_key)
        data_exp_var_div_mean = np.nan_to_num(
            np.var(data_exp, axis=0) / np.mean(data_exp, axis=0)
        )
        idx_rank_ = np.argsort(data_exp_var_div_mean)[::-1][:n_genes]
        genes = adata.var.index[idx_rank_]

    for i in range(len(target_clusters)):
        name_i_ = source_cluster + "_" + target_clusters[i]
        df_A_ = pd.DataFrame(
            adata.uns[grn_key][name_i_]["GRN_matrix"],
            index=adata.var.index,
            columns=adata.var.index,
        )
        vmin = min(-0.01, np.percentile(adata.uns[grn_key][name_i_]["GRN_matrix"], 5))
        vmax = max(0.01, np.percentile(adata.uns[grn_key][name_i_]["GRN_matrix"], 95))
        fig, ax = plt.subplots(1, 1, figsize=(10, 8), tight_layout=True)
        sns.heatmap(
            df_A_[genes].loc[genes].T,
            cmap="bwr",
            ax=ax,
            robust=True,
            vmin=vmin,
            vmax=vmax,
            center=0,
        )
        ax.set_title(name_i_)
        ax.set_xlabel("Target")
        ax.set_ylabel("Source")
        if save:
            filename = (
                "%s_%s" % (save_filename,name_i_)
                if save_dir == None
                else "%s/%s" % (save_dir, save_filename)
            )
            fig.savefig(filename + ".png", bbox_inches="tight")