# from adjustText import adjust_text
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
from matplotlib import patheffects as PathEffects
import networkx as nx
import scanpy
import scipy
import scvelo as scv
import sklearn.preprocessing
import sklearn.mixture
import sklearn.neighbors
import sklearn.linear_model
# from sklearn.linear_model import LinearRegression
# from sklearn.preprocessing import PolynomialFeatures
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
import plotly.offline
import umap




def create_graph(
    X,
    cutedge_vol = None,
    cutedge_length = None,
    cut_std = None,
    return_type = 'edges',
):
    tri_ = matplotlib.tri.Triangulation(X[:,0],X[:,1])
    X_src_,X_trg_ = X[tri_.edges[:,0]],X[tri_.edges[:,1]]
    length_edge_ = np.linalg.norm(X_src_-X_trg_,axis=1)
    x1,y1 = X[tri_.triangles[:,0],0],X[tri_.triangles[:,0],1]
    x2,y2 = X[tri_.triangles[:,1],0],X[tri_.triangles[:,1],1]
    x3,y3 = X[tri_.triangles[:,2],0],X[tri_.triangles[:,2],1]
    vol_ = np.abs((x1-x3)*(y2-y3)-(x2-x3)*(y1-y3))
    length_ = np.max([(x1-x2)**2+(y1-y2)**2,(x2-x3)**2+(y2-y3)**2,(x3-x1)**2+(y3-y1)**2],axis=0)
    if cut_std == None:
        std_delta_ = 0.1
        std_min_ = 1
        cut_std = std_min_
        while 1:
            if len(np.unique(tri_.edges[length_edge_ < cut_std*np.std(length_edge_)].reshape(-1,1).T[0])) == X.shape[0]:
                break
            cut_std = cut_std + std_delta_
    if cutedge_vol == None:
        judge_vol_tri_ = vol_ < cut_std*np.std(vol_)
    else:
        judge_vol_tri_ = vol_ < np.percentile(vol_,100-cutedge_vol)
    if cutedge_length == None:
        judge_length_edge_ = length_edge_ < cut_std*np.std(length_edge_)
        judge_length_tri_= length_ < cut_std*np.std(length_)
    else:
        judge_length_edge_ = length_edge_ < np.percentile(length_edge_,100-cutedge_length)
        judge_length_tri_ = length_ < np.percentile(length_edge_,100-cutedge_length)
    idx_mask_ = judge_vol_tri_ & judge_length_tri_
    tri_.set_mask(idx_mask_==False)
    edge_tri_ = np.vstack((np.vstack((tri_.triangles[idx_mask_][:,[0,1]],tri_.triangles[idx_mask_][:,[1,2]])),tri_.triangles[idx_mask_][:,[2,0]]))
    edge_tri_sort_ = np.array([np.sort(e) for e in edge_tri_])
    np.sort(edge_tri_sort_,axis=0),np.unique(edge_tri_sort_,axis=0).shape
    edges_,count_ = np.unique(edge_tri_sort_,axis=0,return_counts=True)
    idx_bd_ = np.unique(edges_[count_==1].reshape(1,-1)[0])
    if return_type == 'edges': return edges_.T
    if return_type == 'edges_bd': return edges_[:,0],edges_[:,1],idx_bd_
    if return_type == 'triangles': return tri_,idx_mask_



def check_arguments(
    adata,
    verbose = True,
    **kwargs
):
    logger = logging.getLogger("argument checking")
    if verbose:
        logger.setLevel(logging.WARNING)
    else:
        logger.setLevel(logging.ERROR)
    
    if 'exp_key' in kwargs.keys():
        if kwargs['exp_key'] != None:
            if (kwargs['exp_key'] not in adata.obsm.keys()) and (kwargs['exp_key'] not in adata.layers.keys()):
                err_mssg = 'The key \"%s\" was not found in adata.obsm.obsm. Please modify the argument \"exp_key\".' % kwargs['exp_key']
                logger.exception(err_mssg)
                raise KeyError(err_mssg)
    
    if 'exp_2d_key' in kwargs.keys():
        if (kwargs['exp_2d_key'] not in adata.obsm.keys()) and (kwargs['exp_2d_key'] not in adata.layers.keys()):
            if 'X_umap' in adata.obsm.keys():
                logger.warning('The key \"%s\" was not found in adata.obsm, but \"X_umap\" was found insted. \"%s\" was replaced with \"X_umap\".' % (kwargs['exp_2d_key'],kwargs['exp_2d_key']))
                kwargs['exp_2d_key'] = 'X_umap'
            elif 'X_tsne' in adata.obsm.keys():
                logger.warning('Warning: The key \"%s\" was not found in adata.obsm, but \"X_tsne\" was found insted. \"%s\" was replaced with \"X_tsne\".' % (kwargs['exp_2d_key'],kwargs['exp_2d_key']))
                kwargs['exp_2d_key'] = 'X_tsne'
            elif 'X_pca' in adata.obsm.keys():
                logger.warning('Warning: The key \"%s\" was not found in adata.obsm, but \"X_pca\" was found insted. \"%s\" was replaced with \"X_tsne\".' % (kwargs['exp_2d_key'],kwargs['exp_2d_key']))
                kwargs['exp_2d_key'] = 'X_pca'
            else:
                raise KeyError('The key \"%s\" was not found in adata.obsm.obsm. Please modify the argument \"exp_2d_key\".' % kwargs['exp_2d_key'])
    
    if 'vkey' in kwargs.keys():
        if (kwargs['vkey'] not in adata.obsm.keys()) and (kwargs['vkey'] not in adata.layers.keys()):
            raise KeyError('The key \"%s\" was not found in adata.obsm.obsm. Please modify the argument \"vkey\".' % kwargs['vkey'])
    
    if 'vel_2d_key' in kwargs.keys():
        if (kwargs['vel_2d_key'] not in adata.obsm.keys()) and (kwargs['vel_2d_key'] not in adata.layers.keys()):
            if 'velocity_umap' in adata.obsm.keys():
                logger.warning('The key \"%s\" was not found in adata.obsm, but \"velocity_umap\" was found insted. \"%s\" was replaced with \"velocity_umap\".' % (kwargs['vel_2d_key'],kwargs['vel_2d_key']))
                kwargs['vel_2d_key'] = 'velocity_umap'
            elif 'velocity_tsne' in adata.obsm.keys():
                logger.warning('Warning: The key \"%s\" was not found in adata.obsm, but \"velocity_tsne\" was found insted. \"%s\" was replaced with \"velocity_tsne\".' % (kwargs['vel_2d_key'],kwargs['vel_2d_key']))
                kwargs['vel_2d_key'] = 'velocity_tsne'
            else:
                raise KeyError('The key \"%s\" was not found in adata.obsm.obsm. Please modify the argument \"vel_2d_key\".' % kwargs['vel_2d_key'])
    
    if 'basis' in kwargs.keys():
        if ('X_%s' % kwargs['basis'] not in adata.obsm.keys()) and ('X_%s' % kwargs['basis'] not in adata.layers.keys()):
            if 'X_umap' in adata.obsm.keys():
                logger.warning('The key \"%s\" was not found in adata.obsm, but \"X_umap\" was found insted. \"%s\" was replaced with \"X_umap\".' % (kwargs['basis'],kwargs['basis']))
                kwargs['basis'] = 'umap'
            elif 'X_tsne' in adata.obsm.keys():
                logger.warning('Warning: The key \"%s\" was not found in adata.obsm, but \"X_tsne\" was found insted. \"%s\" was replaced with \"X_tsne\".' % (kwargs['basis'],kwargs['basis']))
                kwargs['basis'] = 'tsne'
            else:
                raise KeyError('The key \"%s\" was not found in adata.obsm.obsm. Please modify the argument \"exp_2d_key\".' % kwargs['basis'])

    if 'map_key' in kwargs.keys():
        if kwargs['map_key'] == None:
            kwargs['map_key'] = kwargs['exp_2d_key']
    
    key_names = ['cluster_key','potential_key']
    for key in key_names:
        if key in kwargs.keys():
            if kwargs[key] not in adata.obs.keys():
                raise KeyError('The key \"%s\" was not found in adata.obs. Please modify the argument \"%s\".' % (kwargs[key],key))
    
    key_names = []
    for key in key_names:
        if key in kwargs.keys():
            if kwargs[key] not in adata.obsm.keys():
                raise KeyError('The key \"%s\" was not found in adata.obsm. Please modify the argument \"%s\".' % (kwargs[key],key))
    
    key_names = ['graph_key']
    for key in key_names:
        if key in kwargs.keys():
            if kwargs[key] not in adata.uns.keys():
                raise KeyError('The key \"%s\" was not found in adata.uns. Please modify the argument \"%s\".' % (kwargs[key],key))
    
    key_names = ['expression_key']
    for key in key_names:
        if key in kwargs.keys():
            if kwargs[key] != None:
                if (kwargs[key] not in adata.obsm.keys()) & (kwargs[key] not in adata.layers.keys()):
                    raise KeyError('The key \"%s\" was not found in adata.obsm or adata.layers. Please modify the argument \"%s\".' % (kwargs[key],key))
    
    if 'graph_method' in kwargs.keys():
        if kwargs['graph_method'] != None:
            if kwargs['graph_method'] not in ['Delauney','knn']:
                raise KeyError('The key \"%s\" was not found in adata.obsm or adata.layers. Please modify the argument \"%s\".' % (kwargs[key],key))

    key = 'obs_key'
    if key in kwargs.keys():
        if type(kwargs[key]) == list:
            key_names = ['cluster_key','potential_key']
            for key_ in key_names:
                if key_ in kwargs.keys():
                    if kwargs[key_] in kwargs[key]:
                        # raise logger.warning('The key \"%s\" was multipled.' % (kwargs[key_]))
                        kwargs[key].remove(kwargs[key_])
            for arg in kwargs[key]:
                if arg not in adata.obs.keys():
                    logger.warning('The key \"%s\" was not found in adata.obs. The key \"%s\" is removed from \"%s\".' % (arg,key,arg,key))
                    kwargs[key].remove(key)
            key_names = ['cluster_key','potential_key']
        elif kwargs[key] != None:
            raise TypeError('The argument %s should be a list or None')
    
    key = 'genes'
    if key in kwargs.keys():
        if type(kwargs[key]) == list:
            for arg in kwargs[key]:
                if arg not in adata.var.index:
                    logger.warning('The gene \"%s\" was not found. The gene \"%s\" is removed from \"%s\".' % (arg,arg,key))
                    kwargs[key].remove(arg)
        elif kwargs[key] != None:
            raise TypeError('The argument %s should be a list or None')

    return kwargs

def pt(cv,k):
    return np.percentile(cv,k)

def cmap_earth(cv):
    #c_list  = np.array(['#0938BF','#50D9FB','#B7E5FA','#98D685','#36915c','#F9EFCD','#E0BB7D','#D3A62D','#997618','#705B10','#5F510D','#A56453','#5C1D09'])
    c_min,c_max = 5,95
    c_list  = np.array(['#0938BF','#50D9FB','#B7E5FA','#98D685','#fff5d1','#997618','#705B10','#5C1D09'])
    # c_list  = np.array(['#0938BF','#0938BF','#50D9FB','#B7E5FA','#98D685','#F9EFCD','#E0BB7D','#D3A62D','#997618','#705B10','#5F510D','#A56453','#5C1D09','#5C1D09'])
    c_level = np.array([np.percentile(cv,(c_max-c_min)*(i)/len(c_list)+c_min) for i in range(len(c_list))])
    c_list  = ['#0938BF','#50D9FB','#B7E5FA','#98D685','#F9EFCD','#E0BB7D','#D3A62D','#997618','#705B10','#5F510D','#A56453','#5C1D09']
    c_level = [pt(cv,0),pt(cv,5),pt(cv,20),pt(cv,40),pt(cv,60),pt(cv,75),pt(cv,80),pt(cv,85),pt(cv,90),pt(cv,95),pt(cv,99),pt(cv,100)]
    # c_level = np.array([np.percentile(cv,100*(i)/len(c_list)) for i in range(len(c_list))])
    # c_level = np.array([i*(np.max(cv)-np.min(cv))/len(c_list) + np.min(cv) for i in range(len(c_list))])
    color = np.vstack((c_level,c_list)).T
    hight = 1000*color[:,0].astype(np.float32)
    hightnorm = sklearn.preprocessing.minmax_scale(hight)
    colornorm = []
    for no, norm in enumerate(hightnorm):
        colornorm.append([norm, color[no, 1]])
    colornorm[-1][0] = 1
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list('earth', colornorm, N=hight.max()-hight.min()+1)
    return cmap

def edge_velocity(
        X,
        vel,
        source,
        target,
        normalization = True,
):
    idx_vel = np.isnan(vel[0])==False
    X1,X2 = X[:,idx_vel][source],X[:,idx_vel][target]
    V1,V2 = vel[:,idx_vel][source],vel[:,idx_vel][target]
    Dis = np.linalg.norm(X2-X1,axis=1)
    Dis[Dis==0] = 1
    # V1_p,V2_p = V1*(X2-X1),V2*(X2-X1)
    # V1_p[V1_p<0] = 0
    # V2_p[V2_p<0] = 0
    # edge_vel = np.sum(0.5*(V1_p+V2_p),axis=1)/Dis/np.sum(idx_vel)
    edge_vel = np.sum(0.5*(V1+V2)*(X2-X1),axis=1)/Dis/np.sum(idx_vel)
    if normalization:
        edge_vel_norm = np.linalg.norm(edge_vel)
        if edge_vel_norm > 0: edge_vel= edge_vel/edge_vel_norm
    return edge_vel

def Hodge_decomposition(
    adata,
    basis = 'umap',
    vkey  = 'velocity',
    exp_key = None,
    potential_key = 'potential',
    rotation_key = 'rotation',
    vorticity_key = 'vorticity',
    streamfunc_key = 'streamfunc',
    graph_key = 'CellMap_graph',
    edge_vel_key = 'edge_velocity',
    graph_method = 'knn',#'Delauney',
    HD_rate = 0.0,
    n_neighbors = 10,
    contribution_rate_pca = 0.95,
    cutedge_vol  = None,
    cutedge_length = None,
    cut_std = None,
    verbose = True,
    logscale_vel = True,
    ):
    """
    Hodge decomposition

    Parameters
    ----------
    adata: anndata (n_samples, n_features)
    
    exp_key: None or string
    """
    
    kwargs_arg = check_arguments(adata, verbose = True, exp_key=exp_key, vkey = vkey, basis=basis, graph_method=graph_method)
    exp_key,vkey,basis = kwargs_arg['exp_key'],kwargs_arg['vkey'],kwargs_arg['basis']
    
    exp_2d_key_ = 'X_%s' % basis
    vel_2d_key_ = '%s_%s' % (vkey,basis)
    pot_vkey_ = '%s_%s_%s' % (potential_key,vkey,basis)
    rot_vkey_ = '%s_%s_%s' % (rotation_key,vkey,basis)
    vor_key_ = '%s_%s' % (vorticity_key,basis)
    sl_key_ = '%s_%s' % (streamfunc_key,basis)
    pot_vor_key_ = '%s_%s_%s' % (potential_key,vorticity_key,basis)
    pot_sl_key_ = '%s_%s_%s' % (potential_key,streamfunc_key,basis)
    rot_vor_key_ = '%s_%s_%s' % (rotation_key,vorticity_key,basis)
    rot_sl_key_ = '%s_%s_%s' % (rotation_key,streamfunc_key,basis)
    
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
        vel_HD = (adata.obs['n_counts'].values*vel_HD.T).T/np.exp(exp_HD)
    exp_LD = adata.obsm[exp_2d_key_][:,:2] if exp_2d_key_ in adata.obsm.keys() else adata.layers[exp_2d_key_][:,:2]
    vel_LD = adata.obsm[vel_2d_key_][:,:2] if vel_2d_key_ in adata.obsm.keys() else adata.layers[vel_2d_key_][:,:2]
    
    ## Compute graph and edge velocities
    n_node_ = exp_HD.shape[0]
    idx_bd_  = []
    if graph_method == 'Delauney':
        source, target, idx_bd_ = create_graph(exp_LD,cutedge_vol=cutedge_vol,cutedge_length=cutedge_length,cut_std=cut_std,return_type='edges_bd')
    elif graph_method == 'knn':
        knn = sklearn.neighbors.NearestNeighbors(n_neighbors=n_neighbors+1, algorithm='kd_tree')
        knn.fit(exp_LD)
        distances, indices = knn.kneighbors(exp_LD)
        distances, indices = distances[:,1:], indices[:,1:]
        source = np.ravel(np.repeat(np.arange(exp_HD.shape[0]).reshape((-1, 1)),n_neighbors,axis=1))
        target = np.ravel(indices)
    
    if HD_rate > 0:
        edge_vel_HD = edge_velocity(exp_HD,vel_HD,source,target)
    else:
        edge_vel_HD = 0
    
    if HD_rate < 1:
        edge_vel_LD = edge_velocity(exp_LD,vel_LD,source,target)
    else:
        edge_vel_LD = 0
    
    ## Solve potential
    n_edge_ = len(source)
    grad_mat = np.zeros([n_edge_,n_node_],dtype=float)
    grad_mat[tuple(np.vstack((np.arange(n_edge_),source)))] = -1
    grad_mat[tuple(np.vstack((np.arange(n_edge_),target)))] = 1
    div_mat = -grad_mat.T
    lap = -np.dot(div_mat,grad_mat)
    edge_vel = (1-HD_rate)*edge_vel_LD+HD_rate*edge_vel_HD
    source_term = np.dot(div_mat,edge_vel)
    lap_inv_ = np.linalg.pinv(lap)
    potential = np.dot(lap_inv_,source_term)
    pot_flow_ = -np.dot(grad_mat,potential)
    rot_flow_ = edge_vel - pot_flow_
    adata.obs[potential_key] = potential - np.min(potential)


    # Compute potential & rotational flow
    vel_potential = np.zeros([adata.shape[0],2],dtype=float)
    vel_rotation = np.zeros([adata.shape[0],2],dtype=float)
    edge_vel_norm = np.linalg.norm(edge_velocity(exp_LD,vel_LD,source,target,normalization=False))
    if graph_method == 'Delauney':
        src_trg_ = np.hstack((source,target))
        trg_src_ = np.hstack((target,source))
        pot_flow_2_ = np.hstack((pot_flow_,-pot_flow_))
        rot_flow_2_ = np.hstack((rot_flow_,-rot_flow_))
        for i in range(adata.shape[0]):
            idx_ = src_trg_ == i
            dis_ = np.linalg.norm(exp_LD[src_trg_[idx_]]-exp_LD[trg_src_[idx_]],axis=1,ord=2)
            dis_[dis_==0] = 1
            ex_ = -(exp_LD[src_trg_[idx_]]-exp_LD[trg_src_[idx_]]).T/dis_
            vel_potential[i] = 4.*edge_vel_norm*np.mean(pot_flow_2_[idx_]*ex_,axis=1)
            vel_rotation[i]  = 4.*edge_vel_norm*np.mean(rot_flow_2_[idx_]*ex_,axis=1)
        adata.obsm[pot_vkey_] = vel_potential
        adata.obsm[rot_vkey_] = vel_rotation
    elif graph_method == 'knn':
        for i in range(adata.shape[0]):
            idx_ = source == i
            dis_ = np.linalg.norm(exp_LD[source[idx_]]-exp_LD[target[idx_]],axis=1,ord=2)
            dis_[dis_==0] = 1
            ex_ = -(exp_LD[source[idx_]]-exp_LD[target[idx_]]).T/dis_
            vel_potential[i] = 4.*edge_vel_norm*np.mean(pot_flow_[idx_]*ex_,axis=1)
            vel_rotation[i]  = 4.*edge_vel_norm*np.mean(rot_flow_[idx_]*ex_,axis=1)
        adata.obsm[pot_vkey_] = vel_potential
        adata.obsm[rot_vkey_] = vel_rotation
    
    
    # vorticity_ = np.dot(div_mat,edge_velocity(exp_LD,np.vstack((vel_LD[:,1],-vel_LD[:,0])).T,source,target,normalization=False))
    div_ = np.linalg.norm(vel_LD,axis=1)
    div_[div_==0] = 1
    vorticity_ = np.dot(div_mat,edge_velocity(exp_LD,np.vstack((vel_LD[:,1]/div_,-vel_LD[:,0]/div_)).T,source,target,normalization=False))
    source_term_ = vorticity_
    streamfunc_ = -np.dot(lap_inv_,source_term_)
    adata.obs[vor_key_] = vorticity_
    adata.obs[sl_key_]  = streamfunc_-np.min(streamfunc_)

    # vorticity_ = np.dot(div_mat,edge_velocity(exp_LD,np.vstack((adata.obsm[pot_vkey_][:,1],-adata.obsm[pot_vkey_][:,0])).T,source,target,normalization=False))
    div_ = np.linalg.norm(adata.obsm[pot_vkey_],axis=1)
    div_[div_==0] = 1
    vorticity_ = np.dot(div_mat,edge_velocity(exp_LD,np.vstack((adata.obsm[pot_vkey_][:,1]/div_,-adata.obsm[pot_vkey_][:,0]/div_)).T,source,target,normalization=False))
    source_term_ = vorticity_
    streamfunc_ = -np.dot(lap_inv_,source_term_)
    adata.obs[pot_vor_key_] = vorticity_
    adata.obs[pot_sl_key_] = streamfunc_-np.min(streamfunc_)

    # vorticity_ = np.dot(div_mat,edge_velocity(exp_LD,np.vstack((adata.obsm[rot_vkey_][:,1],-adata.obsm[rot_vkey_][:,0])).T,source,target,normalization=False))
    div_ = np.linalg.norm(adata.obsm[rot_vkey_],axis=1)
    div_[div_==0] = 1
    vorticity_ = np.dot(div_mat,edge_velocity(exp_LD,np.vstack((adata.obsm[rot_vkey_][:,1]/div_,-adata.obsm[rot_vkey_][:,0]/div_)).T,source,target,normalization=False))
    source_term_ = vorticity_
    streamfunc_ = -np.dot(lap_inv_,source_term_)
    adata.obs[rot_vor_key_] = vorticity_
    adata.obs[rot_sl_key_] = streamfunc_-np.min(streamfunc_)

    adata.obs[rotation_key] = np.array([np.mean(np.hstack((rot_flow_[source==i],-rot_flow_[target==i]))) for i in range(adata.shape[0])])

    ## Contribution ratio
    log_ = {}
    log_["Contribution_ratio"] = {}
    norm_grad = np.linalg.norm(pot_flow_)
    norm_curl = np.linalg.norm(rot_flow_)
    log_["Contribution_ratio"]['Potential'] = '{:.2%}'.format(norm_grad/(norm_grad+norm_curl))
    log_["Contribution_ratio"]['Rotation']  = '{:.2%}'.format(norm_curl/(norm_grad+norm_curl))
    adata.uns['CellMap_log'] = log_
    if verbose: print(adata.uns['CellMap_log'])

    adata.uns[graph_key] = {'source':source,'target':target}
    adata.uns[edge_vel_key] = {edge_vel_key:edge_vel,edge_vel_key+'_pot':pot_flow_,edge_vel_key+'_rot':rot_flow_}

def Hodge_decomposition_genes(
    adata,
    genes,
    basis = 'umap',
    vkey  = 'velocity',
    exp_key = None,
    potential_key = 'potential',
    potential_vkey = 'potential_velocity',
    rotation_key = 'rotation',
    rotation_vkey = 'rotation_velocity',
    graph_method = 'knn',#'Delauney',
    n_neighbors = 10,
    cutedge_vol  = None,
    cutedge_length = None,
    verbose = True,
    logscale_vel = True,
    ):
    """
    Hodge decomposition

    Parameters
    ----------
    adata: anndata (n_samples, n_features)
    
    exp_key: None or string
    """
    
    kwargs_arg = check_arguments(adata, verbose = True, exp_key=exp_key, vkey = vkey, basis=basis, graph_method=graph_method)
    exp_key,vkey,basis = kwargs_arg['exp_key'],kwargs_arg['vkey'],kwargs_arg['basis']
    
    exp_2d_key_ = 'X_%s' % basis
    vel_2d_key_ = '%s_%s' % (vkey,basis)
    pot_vkey_ = '%s_%s' % (potential_vkey,basis)
    rot_vkey_ = '%s_%s' % (rotation_vkey,basis)
    
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
        vel_HD = (1e+4*vel_HD.T/adata.obs['n_counts'].values).T/np.power(2,exp_HD)
    exp_LD = adata.obsm[exp_2d_key_][:,:2] if exp_2d_key_ in adata.obsm.keys() else adata.layers[exp_2d_key_][:,:2]
    
    n_node_ = exp_HD.shape[0]
    if graph_method == 'Delauney':
        source, target = create_graph(exp_LD,cutedge_vol=cutedge_vol,cutedge_length=cutedge_length,return_type='edges')
        # source, target = np.ravel(tri_.triangles[idx_tri][:,[0,1,2]]),np.ravel(tri_.triangles[idx_tri][:,[1,2,0]])
    elif graph_method == 'knn':
        # pca = sklearn.decomposition.PCA()
        # exp_HD_pca = pca.fit_transform(exp_HD)
        # n_pca = np.min(np.arange(len(pca.explained_variance_ratio_))[np.cumsum(pca.explained_variance_ratio_)>contribution_rate])
        # knn = sklearn.neighbors.NearestNeighbors(n_neighbors=n_neighbors+1, algorithm='kd_tree')
        # knn.fit(exp_HD_pca[:,:n_pca])
        # distances, indices = knn.kneighbors(exp_HD_pca[:,:n_pca])
        knn = sklearn.neighbors.NearestNeighbors(n_neighbors=n_neighbors+1, algorithm='kd_tree')
        knn.fit(exp_LD)
        distances, indices = knn.kneighbors(exp_LD)
        distances, indices = distances[:,1:], indices[:,1:]
        source = np.ravel(np.repeat(np.arange(exp_HD.shape[0]).reshape((-1, 1)),n_neighbors,axis=1))
        target = np.ravel(indices)
    
    n_edge_ = len(source)
    grad_mat = np.zeros([n_edge_,n_node_],dtype=float)
    grad_mat[tuple(np.vstack((np.arange(n_edge_),source)))] = -1
    grad_mat[tuple(np.vstack((np.arange(n_edge_),target)))] = 1
    div_mat = -grad_mat.T
    lap = -np.dot(div_mat,grad_mat)
    lap_inv = np.linalg.pinv(lap)
    
    for gene in genes:
        X1,X2 = exp_HD[:,adata.var.index == gene][source],exp_HD[:,adata.var.index == gene][target]
        V1,V2 = vel_HD[:,adata.var.index == gene][source],vel_HD[:,adata.var.index == gene][target]
        Dis = np.linalg.norm(exp_HD[target]-exp_HD[source],axis=1)
        edge_vel = np.sum(0.5*(V1+V2)*(X2-X1),axis=1)/Dis
        source_term = np.dot(div_mat,edge_vel)
        potential = np.dot(lap_inv,source_term)
        adata.obs[potential_key+'_Gene_%s' % gene] = potential - np.min(potential)

def view(
    adata,
    basis = 'umap',
    color_key = 'potential',
    cluster_key = 'clusters',
    show_graph = True,
    cutedge_vol  = None,
    cutedge_length = None,
    title = '',
    save = False,
    save_dir = None,
    save_filename = 'CellMap_view',
    figsize = None,
    fontsize_text = 16,
    cbar = True,
    **kwargs
    ):
    
    kwargs_arg = check_arguments(adata, basis=basis)
    basis = kwargs_arg['basis']
    basis_key = 'X_%s' % basis
    
    if 'cmap' not in kwargs:
        kwargs['cmap'] = cmap_earth(adata.obs[color_key])

    if cluster_key not in adata.obs.keys():
        cluster_key = None
    
    if figsize == None:
        figsize = (10,6) if cbar else (8,6)
        

    data_pos = adata.obsm[basis_key]
    fig,ax = plt.subplots(figsize=figsize)
    sc = ax.scatter(data_pos[:,0],data_pos[:,1],c=adata.obs[color_key],zorder=10,**kwargs)
    if show_graph:
        tri_ = create_graph(data_pos,cutedge_vol=cutedge_vol,cutedge_length=cutedge_length,return_type='triangles')[0]
        ax.tripcolor(tri_,adata.obs[color_key],lw=0.5,zorder=0,alpha=0.3,cmap=kwargs['cmap'])
    if cluster_key != None:
        if cluster_key in adata.obs.keys():
            cluster = adata.obs[cluster_key]
            for c in np.unique(cluster):
                txt = plt.text(np.mean(data_pos[cluster == c],axis=0)[0],np.mean(data_pos[cluster == c],axis=0)[1],c,fontsize=fontsize_text,ha='center', va='center',fontweight='bold',zorder=20)
                txt.set_path_effects([PathEffects.withStroke(linewidth=5, foreground='w')])
        else:
            print('There is no cluster key \"%s\" in adata.obs' % cluster_key)
    ax.axis('off')
    ax.set_title(title,fontsize=18)
    if cbar: plt.colorbar(sc,aspect=20, pad=0.01, orientation='vertical').set_label(color_key,fontsize=20)
    if save:
        filename = '%s' % (save_filename) if save_dir == None else '%s/%s' % (save_dir,save_filename)
        fig.savefig(filename+'.png', bbox_inches='tight')


def view_cluster(
    adata,
    basis = 'umap',
    potential_key = 'potential',
    cluster_key = 'clusters',
    cutedge_vol  = None,
    cutedge_length = None,
    n_points = 1000,
    fontsize_text = 16,
    seed = None,
    title = '',
    **kwargs
    ):
    
    kwargs_arg = check_arguments(adata,basis = basis, potential_key=potential_key)
    basis = kwargs_arg['basis']
    basis_key = 'X_%s' % basis
    
    if 'cmap' not in kwargs:
        kwargs['cmap'] = cmap_earth(adata.obs[potential_key])

    data_pos = adata.obsm[basis_key]
    fig,ax = plt.subplots(figsize=(8,6))
    tri_ = create_graph(data_pos,cutedge_vol=cutedge_vol,cutedge_length=cutedge_length,return_type='triangles')[0]
    sc = ax.tricontourf(tri_,adata.obs[potential_key],zorder=0,alpha=0.9,cmap=kwargs['cmap'],levels=100)
    if cluster_key in adata.obs.keys():
        cluster = adata.obs[cluster_key]
        idx_random = np.zeros(cluster.shape,dtype=bool)
        np.random.seed(seed)
        idx_random[np.random.choice(len(idx_random),min(n_points,len(idx_random)),replace=False)] = True
        cluster_set = np.unique(cluster)
        cmap_pt = plt.get_cmap("tab10") if len(cluster_set) <= 10 else plt.get_cmap("tab20")
        for i in range(len(cluster_set)):
            idx = (cluster == cluster_set[i]) & idx_random
            ax.scatter(data_pos[idx,0],data_pos[idx,1],zorder=10,alpha=0.8,edgecolor='w',color=cmap_pt(i),**kwargs)
            txt = plt.text(np.mean(data_pos[cluster == cluster_set[i]],axis=0)[0],np.mean(data_pos[cluster == cluster_set[i]],axis=0)[1],cluster_set[i]
                           ,color=cmap_pt(i),fontsize=fontsize_text,ha='center', va='center',fontweight='bold',zorder=20)
            txt.set_path_effects([PathEffects.withStroke(linewidth=5, foreground='w')])
    else:
        print('There is no cluster key \"%s\" in adata.obs' % cluster_key)
    ax.set_title(title,fontsize=18)
    plt.colorbar(sc,aspect=20, pad=0.01, orientation='vertical').set_label(potential_key,fontsize=20)
    ax.axis('off')


def view_surface(
    adata,
    basis = 'umap',
    color_key = 'potential',
    cluster_key = None,
    show_graph = False,
    cutedge_vol  = None,
    cutedge_length = None,
    title = '',
    **kwargs
    ):
    
    kwargs_arg = check_arguments(adata,
                             basis = basis,
                            )
    basis = kwargs_arg['basis']
    basis_key = 'X_%s' % basis

    if 'cmap' not in kwargs:
        kwargs['cmap'] = cmap_earth(adata.obs[color_key])
    
    data_pos = adata.obsm[basis_key]
    tri_ = create_graph(data_pos,cutedge_vol=cutedge_vol,cutedge_length=cutedge_length,return_type='triangles')[0]
    fig,ax = plt.subplots(figsize=(8,6))
    cntr = ax.tricontourf(tri_,adata.obs[color_key],cmap=kwargs['cmap'],levels=100,zorder=2)
    fig.colorbar(cntr, shrink=0.75, orientation='vertical').set_label(color_key,fontsize=20)
    if show_graph: ax.triplot(tri_,color='w',lw=0.5,zorder=10,alpha=1)
    ax.set_xlim(np.min(data_pos[:,0])-0.02*(np.max(data_pos[:,0])-np.min(data_pos[:,0])),np.max(data_pos[:,0])+0.02*(np.max(data_pos[:,0])-np.min(data_pos[:,0])))
    ax.set_ylim(np.min(data_pos[:,1])-0.02*(np.max(data_pos[:,1])-np.min(data_pos[:,1])),np.max(data_pos[:,1])+0.02*(np.max(data_pos[:,1])-np.min(data_pos[:,1])))
    ax.tick_params(labelbottom=False,labelleft=False,labelright=False,labeltop=False,bottom=False,left=False,right=False,top=False)
    ax.spines['right'].set_visible(False),ax.spines['top'].set_visible(False),ax.spines['bottom'].set_visible(False),ax.spines['left'].set_visible(False)
    ax.set_title(title,fontsize=18)
    if cluster_key != None:
        texts = []
        if cluster_key in adata.obs.keys():
            cluster = adata.obs[cluster_key]
            for c in np.unique(cluster):
                txt = ax.text(np.mean(data_pos[cluster == c],axis=0)[0],np.mean(data_pos[cluster == c],axis=0)[1],c,fontsize=20,ha='center', va='center',fontweight='bold')
                txt.set_path_effects([PathEffects.withStroke(linewidth=5, foreground='w')])
                texts.append(txt)

def view_stream(
    adata,
    basis = 'umap',
    vkey = 'velocity',
    potential_vkey = 'potential_velocity',
    rotation_vkey = 'rotation_velocity',
    cluster_key = 'clusters',
    figsize=(24,6),
    density = 2,
    alpha = 0.3,
    fontsize = 18,
    legend_fontsize = 18,
    **kwargs
    ):
    
    kwargs_arg = check_arguments(adata, basis = basis)
    basis = kwargs_arg['basis']
    basis_key = 'X_%s' % basis
    data_pos = adata.obsm[basis_key]
    cluster = adata.obs[cluster_key]
    
    fig,ax = plt.subplots(1,3,figsize=figsize,tight_layout=True)
    scv.pl.velocity_embedding_stream(adata,basis=basis,vkey=vkey, title='RNA velocity',ax=ax[0],color=cluster_key,
                                     show=False,density=density,alpha=alpha,fontsize=fontsize,legend_fontsize=0, legend_loc=None,arrow_size=2,linewidth=2,**kwargs)
    scv.pl.velocity_embedding_stream(adata,basis=basis,vkey=potential_vkey, title='Potential flow',ax=ax[1],color=cluster_key,
                                     show=False,density=density,alpha=alpha,fontsize=fontsize,legend_fontsize=0, legend_loc=None,arrow_size=2,linewidth=2,**kwargs)
    scv.pl.velocity_embedding_stream(adata,basis=basis,vkey=rotation_vkey, title='Rotational flow',ax=ax[2],color=cluster_key,
                                     show=False,density=density,alpha=alpha,fontsize=fontsize,legend_fontsize=0, legend_loc=None,arrow_size=2,linewidth=2,**kwargs)
    for i in range(3):
        texts = []
        for c in np.unique(cluster):
            txt = ax[i].text(np.mean(data_pos[cluster == c],axis=0)[0],np.mean(data_pos[cluster == c],axis=0)[1],c,fontsize=legend_fontsize,ha='center', va='center',fontweight='bold',zorder=20)
            txt.set_path_effects([PathEffects.withStroke(linewidth=5, foreground='w')])
            texts.append(txt)


def view_stream_line(
    adata,
    basis = 'umap',
    contour_key = 'streamfunc',
    cluster_key = 'clusters',
    potential_key = 'potential',
    rotation_key = 'rotation',
    cutedge_vol  = None,
    cutedge_length = None,
    title = '',
    save = False,
    save_dir = None,
    save_filename = 'CellMap_stream_line',
    figsize = (24,6),
    fontsize = 18,
    cbar = False,
    **kwargs
    ):
    
    kwargs_arg = check_arguments(adata,basis = basis)
    basis = kwargs_arg['basis']
    basis_key = 'X_%s' % basis
    
    key_ = '%s_%s' % (contour_key,basis)
    pot_key_ = '%s_%s_%s' % (potential_key,contour_key,basis)
    rot_key_ = '%s_%s_%s' % (rotation_key,contour_key,basis)
    
    
    data_pos = adata.obsm[basis_key]
    tri_ = create_graph(data_pos,cutedge_vol=cutedge_vol,cutedge_length=cutedge_length,return_type='triangles')[0]
    
    contour_keys = [key_, pot_key_, rot_key_]
    camps = [cmap_earth(adata.obs[key_]),'rainbow','coolwarm']
    titles = ['RNA velocity orbit','Development orbit','Periodic orbit']
    
    fig,ax = plt.subplots(1,3,figsize=figsize,tight_layout=True)
    for i in range(3):
        ax[i].axis('off')
        ax[i].set_title(title,fontsize=18)
        sc = ax[i].tripcolor(tri_,adata.obs[contour_keys[i]],cmap=camps[i])
        ax[i].tricontour(tri_,adata.obs[contour_keys[i]],lw=0.2,alpha=0.2,levels=75,zorder=3,colors='k',cmap=None,ls='-')
        ax[i].tricontour(tri_,adata.obs[contour_keys[i]],lw=1,alpha=1,levels=15,zorder=3,colors='k',cmap=None,ls='-')
        if cbar: plt.colorbar(sc,aspect=20, pad=0.01, orientation='vertical').set_label(contour_key,fontsize=20)
        ax[i].set_title(titles[i],fontsize=fontsize)
        if cluster_key != None:
            if cluster_key in adata.obs.keys():
                cluster = adata.obs[cluster_key]
                for c in np.unique(cluster):
                    # plt.scatter(data_pos[cluster == c,0],data_pos[cluster == c,1],zorder=1,alpha=0.1,s=100)
                    txt = ax[i].text(np.mean(data_pos[cluster == c],axis=0)[0],np.mean(data_pos[cluster == c],axis=0)[1],c,fontsize=20,ha='center', va='center',fontweight='bold',zorder=20)
                    txt.set_path_effects([PathEffects.withStroke(linewidth=5, foreground='w')])
        else:
            print('There is no cluster key \"%s\" in adata.obs' % cluster_key)
    if save:
        filename = '%s' % (save_filename) if save_dir == None else '%s/%s' % (save_dir,save_filename)
        fig.savefig(filename+'.png', bbox_inches='tight')


def view_quiver(
    adata,
    basis = 'umap',
    vkey = 'velocity',
    potential_vkey = 'potential_velocity',
    rotation_vkey = 'rotation_velocity',
    cluster_key='clusters',
    alpha=0.3,
    fontsize = 18,
    scale=1,
    quiver_rate = 0.5,
    **kwargs
    ):
    
    kwargs_arg = check_arguments(adata,basis = basis)
    basis = kwargs_arg['basis']
    basis_key = 'X_%s' % basis
    vkey_ = '%s_%s' % (vkey,basis)
    pot_vkey_ = '%s_%s' % (potential_vkey,basis)
    rot_vkey_ = '%s_%s' % (rotation_vkey,basis)
    cluster_set = np.unique(adata.obs[cluster_key].values)
    cmap = plt.get_cmap("tab20")
    color = np.zeros(adata.shape[0],dtype=int)
    for j in range(len(cluster_set)):
        idx = adata.obs[cluster_key] == cluster_set[j]
        color[idx] = j
    fig,ax = plt.subplots(1,3,figsize=(24,6),tight_layout=True)
    for i in range(3):
        for j in range(len(cluster_set)):
            idx = adata.obs[cluster_key] == cluster_set[j]
            ax[i].scatter(adata.obsm[basis_key][idx,0],adata.obsm[basis_key][idx,1],s=200,alpha=alpha,label=cluster_set[j],color=cmap(j),zorder=0)
            ax[i].text(np.mean(adata.obsm[basis_key][idx,0]),np.mean(adata.obsm[basis_key][idx,1]),cluster_set[j],fontsize=fontsize,
                       ha='center',va='center',weight='bold')
            txt = ax[i].text(np.mean(adata.obsm[basis_key][idx,0]),np.mean(adata.obsm[basis_key][idx,1]),cluster_set[j],fontsize=20,ha='center', va='center',fontweight='bold',zorder=20)
            txt.set_path_effects([PathEffects.withStroke(linewidth=5, foreground='w')])
    idx_qvr_ = np.random.choice(np.arange(adata.shape[0]),int(quiver_rate*adata.shape[0]),replace=False)
    ax[0].quiver(adata.obsm[basis_key][idx_qvr_,0],adata.obsm[basis_key][idx_qvr_,1],adata.obsm[vkey_][idx_qvr_,0],adata.obsm[vkey_][idx_qvr_,1],scale=scale,zorder=1,**kwargs)
    ax[0].set_title('RNA velocity',fontsize=fontsize)
    ax[0].axis('off')
    ax[1].quiver(adata.obsm[basis_key][idx_qvr_,0],adata.obsm[basis_key][idx_qvr_,1],adata.obsm[pot_vkey_][idx_qvr_,0],adata.obsm[pot_vkey_][idx_qvr_,1],scale=scale,zorder=1,**kwargs)
    ax[1].set_title('Potential flow',fontsize=fontsize)
    ax[1].axis('off')
    ax[2].quiver(adata.obsm[basis_key][idx_qvr_,0],adata.obsm[basis_key][idx_qvr_,1],adata.obsm[rot_vkey_][idx_qvr_,0],adata.obsm[rot_vkey_][idx_qvr_,1],scale=scale,zorder=1,**kwargs)
    ax[2].set_title('Rotational flow',fontsize=fontsize)
    ax[2].axis('off')

def view_surface_genes(
    adata,
    genes,
    exp_key = None,
    basis = 'umap',
    vkey  = 'velocity',
    potential_key = 'potential',
    graph_key = 'CellMap_graph',
    cluster_key = None,
    show_graph = False,
    cutedge_vol  = None,
    cutedge_length = None,
    logscale_vel = True,
    **kwargs,
    ):
    
    kwargs_arg = check_arguments(adata,
                             basis = basis,
                             potential_key = potential_key,
                             graph_key = graph_key,
                            )
    basis = kwargs_arg['basis']
    basis_key = 'X_%s' % basis
    
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
        vel_HD = (1e+4*vel_HD.T/adata.obs['n_counts'].values).T/np.power(2,exp_HD)
    if 'cmap' not in kwargs:
        kwargs['cmap'] = cmap_earth(adata.obs[potential_key])
    
    data_pos = adata.obsm[basis_key]
    tri_ = create_graph(data_pos,cutedge_vol=cutedge_vol,cutedge_length=cutedge_length,return_type='triangles')[0]
    
    for gene in genes:
        fig,ax = plt.subplots(1,3,figsize=(45,10))
        cntr = ax[0].tricontourf(tri_,np.squeeze(exp_HD[:,adata.var.index==gene]),cmap=kwargs['cmap'],levels=100,zorder=2)
        fig.colorbar(cntr, shrink=0.75, orientation='vertical',ax=ax[0]).set_label('gene expression',fontsize=20)
        ax[0].set_title('%s_expression' % gene,fontsize=18)
        cntr = ax[1].tricontourf(tri_,np.squeeze(vel_HD[:,adata.var.index==gene]),cmap=kwargs['cmap'],levels=100,zorder=2)
        fig.colorbar(cntr, shrink=0.75, orientation='vertical',ax=ax[1]).set_label('RNA velocity',fontsize=20)
        ax[1].set_title('%s_potential' % gene,fontsize=18)
        cntr = ax[2].tricontourf(tri_,adata.obs['%s_Gene_%s' % (potential_key,gene)],cmap=kwargs['cmap'],levels=100,zorder=2)
        fig.colorbar(cntr, shrink=0.75, orientation='vertical',ax=ax[2]).set_label(potential_key,fontsize=20)
        ax[2].set_title('%s_potential' % gene,fontsize=18)
        for ax_i in range(3):
            if show_graph: ax[ax_i].triplot(tri_,color='w',lw=0.5,zorder=10,alpha=1)
            ax[ax_i].set_xlim(np.min(data_pos[:,0])-0.02*(np.max(data_pos[:,0])-np.min(data_pos[:,0])),np.max(data_pos[:,0])+0.02*(np.max(data_pos[:,0])-np.min(data_pos[:,0])))
            ax[ax_i].set_ylim(np.min(data_pos[:,1])-0.02*(np.max(data_pos[:,1])-np.min(data_pos[:,1])),np.max(data_pos[:,1])+0.02*(np.max(data_pos[:,1])-np.min(data_pos[:,1])))
            ax[ax_i].tick_params(labelbottom=False,labelleft=False,labelright=False,labeltop=False,bottom=False,left=False,right=False,top=False)
            ax[ax_i].spines['right'].set_visible(False),ax[ax_i].spines['top'].set_visible(False),ax[ax_i].spines['bottom'].set_visible(False),ax[ax_i].spines['left'].set_visible(False)
            if cluster_key != None:
                texts = []
                if cluster_key in adata.obs.keys():
                    cluster = adata.obs[cluster_key]
                    for c in np.unique(cluster):
                        txt = ax[ax_i].text(np.mean(data_pos[cluster == c],axis=0)[0],np.mean(data_pos[cluster == c],axis=0)[1],c,fontsize=20,ha='center', va='center',fontweight='bold')
                        txt.set_path_effects([PathEffects.withStroke(linewidth=5, foreground='w')])
                        texts.append(txt)

def view_3D(
    adata,
    basis = 'umap',
    potential_key = 'potential',
    cluster_key ='clusters',
    cutedge_vol  = None,
    cutedge_length = None,
    show_cells = False,
    show_shadow = True,
    shadow_alpha = 0.2,
    title = 'Landscape',
    bgcolor = "white",
    gridcolor = "gray",
    seed = None,
    n_points = 500,
    save = False,
    filename = 'CellMap_view_3D',
    **kwargs
    ):
    
    kwargs_arg = check_arguments(adata, basis=basis, potential_key=potential_key)
    basis = kwargs_arg['basis']
    basis_key = 'X_%s' % basis
    
    if 'cmap' not in kwargs:
        kwargs['cmap'] = cmap_earth(adata.obs[potential_key])

    x,y,z = adata.obsm[basis_key][:,0], adata.obsm[basis_key][:,1],adata.obs[potential_key]
    
    c_list  = ['#0938BF','#50D9FB','#B7E5FA','#98D685','#F9EFCD','#E0BB7D','#D3A62D','#997618','#705B10','#5F510D','#A56453','#5C1D09']
    c_level = [0,5,20,40,60,75,80,85,90,95,99,100]
    custom_cmap = [[0.01*c_level[i],c_list[i]] for i in range(len(c_list))]

    tri_,idx_tri = create_graph(adata.obsm[basis_key],cutedge_vol=cutedge_vol,cutedge_length=cutedge_length,return_type = 'triangles')
    triangles = tri_.triangles[idx_tri]

    camera = dict(eye=dict(x=1.2, y=-1.2, z=1.0))
    idx = np.zeros(adata.shape[0],dtype=bool)
    np.random.seed(seed)
    idx[np.random.choice(adata.shape[0],min(n_points,adata.shape[0]),replace=False)] = True
    shift = 0.01*(max(z)-min(z))
    shadow = go.Mesh3d(
        x=x,
        y=y,
        z=np.zeros(adata.shape[0]),
        i=triangles[:, 0],
        j=triangles[:, 1],
        k=triangles[:, 2],
        opacity=shadow_alpha,
        color='black',
    )

    if cluster_key in adata.obs.keys():
        clstr = adata.obs[cluster_key]
        clstr_set = np.unique(clstr)
        clstr_id = np.empty(adata.shape[0],dtype=int)
        text = np.array([clstr[i]+'<br>Potential: '+str(np.round(z[i],decimals=2)) for i in range(adata.shape[0])])
        for i in range(len(clstr_set)):
            clstr_id[clstr == clstr_set[i]] = i
        cmap = plt.get_cmap('tab10')
        norm = plt.Normalize(vmin=0,vmax=10)
        color_mapped = cmap(norm(clstr_id[idx]))
        cells = go.Scatter3d(
            x=x[idx],
            y=y[idx],
            z=z[idx]+shift,
            mode='markers',
            marker=dict(
                size=2.5,
                color=color_mapped,
                opacity=1
            ),
            text=text[idx],
            hovertemplate='X: %{x:.2f}<br>Y: %{y:.2f}<br>Z: %{z:.2f}<br>%{text}'
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
            hovertemplate='X: %{x:.2f}<br>Y: %{y:.2f}<br>Z: %{z:.2f}<br>%{text}'
        )
        
        annotations = [dict(
            showarrow=False,
            x=np.percentile(x[clstr == np.unique(clstr)[i]],50),
            y=np.percentile(y[clstr == np.unique(clstr)[i]],50),
            z=np.percentile(z[clstr == np.unique(clstr)[i]],50),
            text="<b>%s<b>" % str(np.unique(clstr)[i]),
            font=dict(size=14,color='rgba(0,0,0,1)'),bgcolor="rgba(255,255,255,0.7)") for i in range(len(np.unique(clstr)))]
        layout = go.Layout(
            title = title,
            width=1500,
            height=1000,
            margin=dict(l=0,r=0, b=0,t=50),
            scene_camera=camera,
            scene=dict(annotations=annotations,xaxis_title=basis+"_1",yaxis_title=basis+"_2",zaxis_title=potential_key,
                     xaxis = dict(backgroundcolor=bgcolor,gridcolor=gridcolor),
                     yaxis = dict(backgroundcolor=bgcolor,gridcolor=gridcolor),
                     zaxis = dict(backgroundcolor=bgcolor,gridcolor=gridcolor),
            ),
            meta=dict(),
            scene_aspectratio=dict(x=1.0, y=1.0, z=0.5),
        )
        data = [surf]
        if show_cells: data.append(cells)
        if show_shadow: data.append(shadow)
        fig = go.Figure(data=data, layout=layout)
        #                  )
    else:
        cells = go.Scatter3d(
            x=x[idx],
            y=y[idx],
            z=z[idx]+shift,
            mode='markers',
            marker=dict(
                size=2,
                color='gray',
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
            opacity=1
        )

        layout = go.Layout(
            title = title,
            width=1500,
            height=1000,
            margin=dict(l=0,r=0, b=0,t=50),
            scene_camera=camera,
            scene=dict(xaxis_title=basis+"_1",yaxis_title=basis+"_2",zaxis_title=potential_key,
                     xaxis = dict(backgroundcolor=bgcolor,gridcolor=gridcolor),
                     yaxis = dict(backgroundcolor=bgcolor,gridcolor=gridcolor),
                     zaxis = dict(backgroundcolor=bgcolor,gridcolor=gridcolor),
            ),
            meta=dict(),
            scene_aspectratio=dict(x=1.0, y=1.0, z=0.5),
        )
        data = [surf]
        if show_cells: data.append(cells)
        if show_shadow: data.append(shadow)
        fig = go.Figure(data=data, layout=layout)
    fig.show()
    
    if save: plotly.offline.plot(fig, filename=filename+'.html')

def view_surface_3D(
    adata,
    basis = 'umap',
    potential_key = 'potential',
    graph_key = 'CellMap_graph',
    cluster_key = None,
    cutedge_vol  = 1,
    cutedge_length = 1,
    elev = 30,
    azim = -60,
    plot_rate = 0.3,
    title = '',
    **kwargs
    ):
    
    kwargs_arg = check_arguments(adata,
                             basis = basis,
                             potential_key = potential_key,
                             graph_key = graph_key,
                            )
    basis = kwargs_arg['basis']
    basis_key = 'X_%s' % basis
    
    if 'cmap' not in kwargs:
        kwargs['cmap'] = cmap_earth(adata.obs[potential_key])

    data_pos = adata.obsm[basis_key]
    tri_ = create_graph(data_pos,cutedge_vol=cutedge_vol,cutedge_length=cutedge_length,return_type='triangles')[0]
    fig = plt.figure(figsize=(15,15))
    ax = fig.add_subplot(111, projection='3d')
    cntr = ax.plot_trisurf(tri_,adata.obs[potential_key],cmap=kwargs['cmap'],zorder=2)
    ax.set_box_aspect(aspect = (1,1,0.8))
    fig.colorbar(cntr, shrink=0.5, orientation='vertical').set_label(potential_key,fontsize=20)
    ax.set_title(title,fontsize=18)
    if cluster_key != None:
        texts = []
        if cluster_key in adata.obs.keys():
            cluster = adata.obs[cluster_key]
            for c in np.unique(cluster):
                txt = ax.text(np.mean(data_pos[cluster == c],axis=0)[0],np.mean(data_pos[cluster == c],axis=0)[1],np.mean(adata.obs[potential_key][cluster == c]),c,fontsize=15,ha='center', va='center',fontweight='bold',zorder=1000)
                txt.set_path_effects([PathEffects.withStroke(linewidth=5, foreground='w')])
                texts.append(txt)
    ax.view_init(elev=elev, azim=azim)


def view_surface_3D_cluster(
    adata,
    basis = 'umap',
    potential_key = 'potential',
    graph_key = 'CellMap_graph',
    cluster_key = 'clusters',
    cutedge_vol  = 1,
    cutedge_length = 1,
    elev = 30,
    azim = -60,
    seed = None,
    n_points = 500,
    title = '',
    **kwargs
    ):
    
    if cluster_key in adata.obs.keys():
        kwargs_arg = check_arguments(adata,
                                 basis = basis,
                                 potential_key = potential_key,
                                 graph_key = graph_key,
                                )
        basis = kwargs_arg['basis']
        basis_key = 'X_%s' % basis

        data_pos = adata.obsm[basis_key]
        tri_ = create_graph(data_pos,cutedge_vol=cutedge_vol,cutedge_length=cutedge_length,return_type='triangles')[0]
        if 'cmap' not in kwargs:
            kwargs['cmap'] = cmap_earth(adata.obs[potential_key])
        fig = plt.figure(figsize=(15,15))
        ax = fig.add_subplot(111, projection='3d')
        cntr = ax.plot_trisurf(tri_,adata.obs[potential_key],cmap=kwargs['cmap'],zorder=2,alpha=0.9)#,cmap=cmap_CellMap,levels=100)
        ax.set_box_aspect(aspect = (1,1,0.8))
        ax.set_title(title,fontsize=18)
        fig.colorbar(cntr, shrink=0.5, orientation='vertical').set_label(potential_key,fontsize=20)
        cluster = adata.obs[cluster_key]
        idx = np.zeros(cluster.shape,dtype=bool)
        np.random.seed(seed)
        idx[np.random.choice(len(idx),min(n_points,len(idx)),replace=False)] = True
        cluster_set = np.unique(cluster)
        z_shift = 0.05*np.abs( np.max(adata.obs[potential_key]) - np.min(adata.obs[potential_key]))
        if len(cluster_set) <= 10:
            cmap_pt = plt.get_cmap("tab10")
            vmin,vmax = 0,10
        else:
            cmap_pt = plt.get_cmap("tab20")
            vmin,vmax = 0,20
        id_color = np.empty(len(cluster),dtype=int)
        for i in range(len(cluster_set)):
            id_color[cluster == cluster_set[i]] = i
            txt = ax.text(np.mean(data_pos[cluster == cluster_set[i]],axis=0)[0],
                           np.mean(data_pos[cluster == cluster_set[i]],axis=0)[1],
                           np.max(adata.obs[potential_key][cluster == cluster_set[i]]),cluster_set[i]
                           ,color=cmap_pt(i),fontsize=20,ha='center', va='center',fontweight='bold',zorder=1000)
            txt.set_path_effects([PathEffects.withStroke(linewidth=5, foreground='w')])
        kwargs['cmap'] = cmap_pt
        ax.scatter(data_pos[idx,0],data_pos[idx,1],adata.obs[potential_key][idx]+z_shift,c=id_color[idx],zorder=100,alpha=1,edgecolor='w',vmin=vmin,vmax=vmax,**kwargs)
        ax.scatter(data_pos[idx,0],data_pos[idx,1],adata.obs[potential_key][idx]+z_shift*0.5,color='k',zorder=10,alpha=0.1,vmin=vmin,vmax=vmax,**kwargs)
        ax.view_init(elev=elev, azim=azim);
    else:
        print('There is no cluster key \"%s\" in adata.obs' % cluster_key)


def write(
    adata,
    filename = 'CellMap',
    basis = 'umap',
    vkey  = 'velocity',
    exp_key = None,
    potential_key = 'potential',
    rotation_key = 'rotation',
    vorticity_key = 'vorticity',
    streamfunc_key = 'streamfunc',
    cluster_key = 'clusters',
    obs_key = None,
    genes = None,
    use_HVG = True,
    n_HVG = 10,
):
    kwargs = check_arguments(adata,basis=basis,potential_key=potential_key,cluster_key=cluster_key,obs_key=obs_key,genes=genes,expression_key=exp_key)
    basis,obs_key,genes = kwargs['basis'],kwargs['obs_key'],kwargs['genes']
    basis_key = 'X_%s' % basis
    vkey_ = '%s_%s' % (vkey,basis)
    pot_key_ = '%s' % (potential_key)
    rot_key_ = '%s' % (rotation_key)
    pot_vkey_ = '%s_%s_%s' % (potential_key,vkey,basis)
    rot_vkey_ = '%s_%s_%s' % (rotation_key,vkey,basis)
    vol_key_ = '%s_%s_%s' % (potential_key,vorticity_key,basis)
    sl_key_ = '%s_%s_%s' % (potential_key,streamfunc_key,basis)
    pot_vol_key_ = '%s_%s_%s' % (potential_key,vorticity_key,basis)
    pot_sl_key_ = '%s_%s_%s' % (potential_key,streamfunc_key,basis)
    rot_vol_key_ = '%s_%s_%s' % (rotation_key,vorticity_key,basis)
    rot_sl_key_ = '%s_%s_%s' % (rotation_key,streamfunc_key,basis)
    
    
    if exp_key == None:
        if scipy.sparse.issparse(adata.X): data_exp = adata.X.toarray()
        else: data_exp = adata.X
    else:
        data_exp = adata.layers[exp_key]
    
    pd_out = pd.DataFrame({
        'X':adata.obsm[basis_key][:,0],'Y':adata.obsm[basis_key][:,1],
        'Potential':adata.obs[pot_key_],
        'Annotation':adata.obs[cluster_key],
        'Rotation':adata.obs[rot_key_],
        'Streamline_Original':adata.obs[sl_key_],
        'Streamline_Potential':adata.obs[pot_sl_key_],
        'Streamline_Rotation':adata.obs[rot_sl_key_],
        'Vorticity_Original':adata.obs[vol_key_],
        'Vorticity_Potential':adata.obs[pot_vol_key_],
        'Vorticity_Rotation':adata.obs[rot_vol_key_],
        'Velocity_x':adata.obsm[vkey_][:,0],
        'Velocity_y':adata.obsm[vkey_][:,1],
        'Velocity_Potential_x':adata.obsm[pot_vkey_][:,0],
        'Velocity_Potential_y':adata.obsm[pot_vkey_][:,1],
        'Velocity_Rotation_x':adata.obsm[rot_vkey_][:,0],
        'Velocity_Rotation_y':adata.obsm[rot_vkey_][:,1],
    },index=adata.obs.index)
    pd_out.index.name='CellID'
    
    if obs_key != None:
        for arg in obs_key:
            pd_out.insert(len(pd_out.columns), arg, adata.obs[arg])
    
    if genes != None:
        for gene in genes:
            pd_out.insert(len(pd_out.columns), gene, data_exp[:,adata.var.index == gene])
    
    if use_HVG:
        scanpy.pp.highly_variable_genes(adata)
        min_mean = np.percentile(np.mean(adata.X.toarray(),axis=0)[np.mean(adata.X.toarray(),axis=0)>0],90)
        idx_means = adata.var['means'] > min_mean
        for gene in adata.var.index[idx_means][np.argsort(adata.var['dispersions_norm'].values[idx_means])[::-1][:n_HVG]]:
            pd_out.insert(len(pd_out.columns), 'HVG_'+gene, data_exp[:,adata.var.index == gene])
    
    print('succeeded in writing CellMapp data as \"%s.csv\"' % filename)
    print('You can visualize the CDV file by CellMapp viewer https://yusuke-imoto-lab.github.io/CellMapViewer/CellMapViewer/viewer.html')

    display(pd_out)
    
    pd_out.to_csv('%s.csv' % filename)


def create_dgraph_potential(
    adata,
    basis = 'umap',
    map_key = None,
    potential_key = 'potential',
    cutedge_vol  = None,
    cutedge_length = None,
    ):
    """
    Hodge decomposition

    Parameters
    ----------
    adata: anndata (n_samples, n_features)
    
    basis: ndarray or string
    """
    
    kwargs_arg = check_arguments(adata,
                             basis = basis,
                             map_key = map_key
                            )
    basis,map_key = kwargs_arg['basis'],kwargs_arg['map_key']
    basis_key = 'X_%s' % basis
    
    data_pos = adata.obsm[basis_key]
    tri_,idx_tri = create_graph(data_pos,cutedge_vol=cutedge_vol,cutedge_length=cutedge_length,return_mask = True)
    triangles = tri_.triangles[idx_tri]
    n_node_ = data_pos.shape[0]
    graph_ = scipy.sparse.lil_matrix(np.zeros([n_node_,n_node_]))
    idx_set = [[0,1],[1,2],[2,0]]
    # idx = np.isnan(data_vel[0])==False
    for id_x,id_y in idx_set:
        weight = adata.obs[potential_key][triangles[:,id_y]].values - adata.obs[potential_key][triangles[:,id_x]].values
        min_weight = np.percentile(np.abs(weight),5)
        graph_[tuple(triangles[weight>min_weight][:,[id_x,id_y]].T[::-1])] = 1
        graph_[tuple(triangles[weight<-min_weight][:,[id_y,id_x]].T[::-1])] = 1
    return scipy.sparse.coo_matrix(graph_)


def create_dgraph(
    adata,
    basis = 'umap',
    vkey  = 'velocity',
    cutedge_vol  = None,
    cutedge_length = None,
    ):
    """
    Hodge decomposition

    Parameters
    ----------
    adata: anndata (n_samples, n_features)
    
    basis: ndarray or string
    """
    
    kwargs_arg = check_arguments(adata,
                             basis = basis,
                             vkey = vkey,
                             map_key = map_key
                            )
    basis,vkey,map_key = kwargs_arg['basis'],kwargs_arg['vkey'],kwargs_arg['map_key']
    basis_key = 'X_%s' % basis
    
    data_pos = adata.obsm[basis_key]
    data_vel = adata.obsm['%s_%s' % (vkey,basis)]
    tri_,idx_tri = create_graph(data_pos,cutedge_vol=cutedge_vol,cutedge_length=cutedge_length,return_mask = True)
    triangles = tri_.triangles[idx_tri]
    n_node_ = data_pos.shape[0]
    graph_ = scipy.sparse.lil_matrix(np.zeros([n_node_,n_node_]))
    idx_set = [[0,1],[1,2],[2,0]]
    idx = np.isnan(data_vel[0])==False
    for id_x,id_y in idx_set:
        X1 = data_pos[:,idx][triangles[:,id_x]]
        X2 = data_pos[:,idx][triangles[:,id_y]]
        V1 = data_vel[:,idx][triangles[:,id_x]]
        V2 = data_vel[:,idx][triangles[:,id_y]]
        weight = np.sum(0.5*(V1+V2)*(X2-X1),axis=1)
        min_weight = np.percentile(np.abs(weight),5)
        graph_[tuple(triangles[weight>min_weight][:,[id_x,id_y]].T[::-1])] = 1
        graph_[tuple(triangles[weight<-min_weight][:,[id_y,id_x]].T[::-1])] = 1
    return scipy.sparse.coo_matrix(graph_)



def view_trajectory(
    adata,
    source_cluster,
    target_clusters,
    n_cells = 50,
    register = 1,
    basis = 'umap',
    potential_key = 'potential',
    cluster_key = 'clusters',
    streamfunc_key = 'streamfunc',
    graph_method = 'Delauney',
    path_key = 'path',
    n_neighbors = 10,
    contribution_rate_pca = 0.95,
    cutedge_vol  = None,
    cutedge_length = None,
    figsize = (10,8),
    save = False,
    save_dir = None,
    save_filename = 'trajectory',
):

    kwargs_arg = check_arguments(adata, verbose = True, basis=basis)
    basis = kwargs_arg['basis']

    if sum(adata.obs[cluster_key].values == source_cluster) == 0:
        raise KeyError('Cluster %s was not found' % source_cluster)
    for trg_ in target_clusters:
        if sum(adata.obs[cluster_key].values == source_cluster) == 0:
            raise KeyError('Cluster %s was not found' % trg_)

    basis_key = 'X_%s' % basis
    pot_sl_key_ = '%s_%s_%s' % (potential_key,streamfunc_key,basis)

    data_pos = adata.obsm[basis_key]
    streamfunc_ = adata.obs[pot_sl_key_]

    ## Compute graph and edge velocities
    if graph_method == 'Delauney':
        tri_ = create_graph(data_pos,cutedge_vol=cutedge_vol,cutedge_length=cutedge_length,return_type='triangles')[0]
        source, target = create_graph(data_pos,cutedge_vol=cutedge_vol,cutedge_length=cutedge_length,return_type='edges')
    elif graph_method == 'knn':
        pca = sklearn.decomposition.PCA()
        exp_HD_pca = pca.fit_transform(data_pos)
        n_pca = np.min(np.arange(len(pca.explained_variance_ratio_))[np.cumsum(pca.explained_variance_ratio_)>contribution_rate_pca])
        knn = sklearn.neighbors.NearestNeighbors(n_neighbors=n_neighbors+1, algorithm='kd_tree')
        knn.fit(exp_HD_pca[:,:n_pca])
        distances, indices = knn.kneighbors(exp_HD_pca[:,:n_pca])
        distances, indices = distances[:,1:], indices[:,1:]
        source = np.ravel(np.repeat(np.arange(data_pos.shape[0]).reshape((-1, 1)),n_neighbors,axis=1))
        target = np.ravel(indices)
    
    # def cost(data_pos,s,t,g,trg_,reg):
    #     return np.exp(-g*reg)*np.linalg.norm(data_pos[s]-data_pos[t])*np.exp(np.linalg.norm(data_pos[s]-data_pos[t])/dis_mean)

    G = nx.DiGraph()
    G.add_weighted_edges_from([(int(s),int(t),1) for s,t in np.vstack((source,target)).T])
    G.add_weighted_edges_from([(int(t),int(s),-1) for s,t in np.vstack((source,target)).T])
    edges_ = np.array(list(G.edges))
    weights_ = np.array([G[u][v]['weight'] for u, v in edges_])

    dis_mean = np.mean(np.linalg.norm(data_pos[source]-data_pos[target],axis=1))
    cmap_ = plt.get_cmap("tab10")
    fig,ax = plt.subplots(figsize=figsize)
    ax.triplot(tri_,color='gray',zorder=0,alpha=0.2,lw=1)
    clusters_ = adata.obs[cluster_key]
    idx_ = clusters_ == source_cluster
    ax.scatter(data_pos[idx_,0],data_pos[idx_,1],color='gray',zorder=10,marker='D',alpha=0.2,s=5,label=source_cluster+' (source)')
    for i_trg_ in range(len(target_clusters)):
        idx_ = clusters_ == target_clusters[i_trg_]
        ax.scatter(data_pos[idx_,0],data_pos[idx_,1],color=cmap_(i_trg_),zorder=10,marker='o',alpha=0.2,s=5,label=target_clusters[i_trg_]+' (target)')
    leg = ax.legend(bbox_to_anchor=(1.05, 0.5), loc='center left', borderaxespad=0, fontsize=12,markerscale=3)
    for lh in leg.legend_handles: lh.set_alpha(1)

    data_src_ = data_pos[adata.obs[cluster_key].values == source_cluster]
    center_src_ = np.mean(data_src_,axis=0)
    centrality_src_ = np.linalg.norm(data_src_-center_src_,axis=1)
    src_set_all_ = np.arange(adata.shape[0])[adata.obs[cluster_key].values == source_cluster][np.argsort(centrality_src_)]
    n_src_ = sum(adata.obs[cluster_key].values == source_cluster)
    path_all = {}
    for i_trg_ in range(len(target_clusters)):
        target_cluster = target_clusters[i_trg_]
        n_cells_ = np.min([n_cells,sum(adata.obs[cluster_key].values == source_cluster),sum(adata.obs[cluster_key].values == target_cluster)])
        data_trg_ = data_pos[adata.obs[cluster_key].values == target_cluster]
        center_trg_ = np.mean(data_trg_,axis=0)
        centrality_trg_ = np.linalg.norm(data_trg_-center_trg_,axis=1)
        n_trg_ = sum(adata.obs[cluster_key].values == target_cluster)
        idx_trg_ = np.arange(0,n_trg_,int(n_trg_/n_cells_))[:n_cells_]
        trg_set_ = np.arange(adata.shape[0])[adata.obs[cluster_key].values == target_cluster][np.argsort(centrality_trg_)][idx_trg_]
        # idx_src_= np.array([np.argmin(np.abs(streamfunc_[trg__] - streamfunc_[src_set_all_].values)) for trg__ in trg_set_])
        idx_src_ = np.arange(0,n_src_,int(n_src_/n_cells_))[:n_cells_]
        src_set_ = src_set_all_[idx_src_]

        pathes,edges,weights,dists  = [],[],[],[]
        for src_,trg_ in np.vstack((src_set_,trg_set_)).T:
            # G.add_weighted_edges_from([(int(s),int(t),np.exp(-g*register)*np.linalg.norm(streamfunc_[trg_]-streamfunc_[int(t)])*np.exp(np.linalg.norm(data_pos[int(s)]-data_pos[int(t)])/dis_mean)) for s,t,g in np.vstack((source,target,grad_)).T])
            # G.add_weighted_edges_from([(int(t),int(s), np.exp(g*register)*np.linalg.norm(streamfunc_[trg_]-streamfunc_[int(t)])*np.exp(np.linalg.norm(data_pos[int(s)]-data_pos[int(t)])/dis_mean)) for s,t,g in np.vstack((source,target,grad_)).T])
            # weights_ = np.hstack((np.exp(-grad_*register)*np.abs(streamfunc_[trg_]-streamfunc_[target])*np.exp(np.linalg.norm(data_pos[source]-data_pos[target],axis=1)/dis_mean),np.exp(grad_*register)*np.abs(streamfunc_[trg_]-streamfunc_[target])*np.exp(np.linalg.norm(data_pos[source]-data_pos[target],axis=1)/dis_mean)))
            weights_i_ = np.exp(-weights_*(adata.obs[potential_key][edges_[:,0]].values - adata.obs[potential_key][edges_[:,1]].values)*register)*np.abs(streamfunc_[trg_]-streamfunc_[edges_[:,1]])*np.exp(np.linalg.norm(data_pos[edges_[:,0]]-data_pos[edges_[:,1]],axis=1)/dis_mean)
            nx.set_edge_attributes(G, values=dict(zip(G.edges(), weights_i_)), name='weight')
            path = nx.dijkstra_path(G, source=src_, target=trg_, weight='weight')
            pathes.append(path)
            edges.append(np.array([[path[i], path[i+1]] for i in range(len(path)-1)]))
            weights.append((sum([G[path[i]][path[i+1]]['weight'] for i in range(len(path)-1)]))/sum([np.linalg.norm(data_pos[path[i]]-data_pos[path[i+1]]) for i in range(len(path)-1)]))
            dists.append(sum([np.linalg.norm(data_pos[path[i]]-data_pos[path[i+1]]) for i in range(len(path)-1)]))
        path_all[source_cluster+'_'+target_clusters[i_trg_]] = pathes
        ax.scatter(data_pos[trg_set_,0],data_pos[trg_set_,1],color=cmap_(i_trg_),zorder=20,marker='o',s=30)
        ax.scatter(data_pos[src_set_,0],data_pos[src_set_,1],color='gray',zorder=20,marker='D',s=30)
        for i in range(n_cells_):
            ax.plot(data_pos[pathes[i],0],data_pos[pathes[i],1],color=cmap_(i_trg_),zorder=10,ls='-',lw=2,alpha=0.3)
    ax.axis('off')
    adata.uns[path_key] = path_all
    if save:
        filename = '%s' % (save_filename) if save_dir == None else '%s/%s' % (save_dir,save_filename)
        fig.savefig(filename+'.png', bbox_inches='tight')
    # G = nx.Graph()
    # for i in range(len(source)):
    #     G.add_edge(source[i],target[i],w=np.linalg.norm(data_pos[source[i]]-data_pos[target[i]]))
    # degree_centrality = nx.degree_centrality(G)

    # G = nx.DiGraph()
    # dis_mean = np.mean(np.linalg.norm(data_pos[source]-data_pos[target],axis=1))

    # def cost(data_pos,s,t,g,reg):
    #     # return np.exp(-g*reg*(degree_centrality[s]+degree_centrality[t]))*np.linalg.norm(data_pos[s]-data_pos[t])
    #     return np.exp(-g*reg)*np.linalg.norm(data_pos[s]-data_pos[t])*np.exp(np.linalg.norm(data_pos[s]-data_pos[t])/dis_mean)
    
    # grad_ = adata.obs[potential_key][source].values - adata.obs[potential_key][target].values
    # G.add_weighted_edges_from([(int(s),int(t),cost(data_pos,int(s),int(t),g,register)) for s,t,g in np.vstack((source[grad_>0],target[grad_>0],grad_[grad_>0])).T])
    # G.add_weighted_edges_from([(int(t),int(s),cost(data_pos,int(s),int(t),-g,register)) for s,t,g in np.vstack((source[grad_>0],target[grad_>0],grad_[grad_>0])).T])
    # G.add_weighted_edges_from([(int(t),int(s),cost(data_pos,int(s),int(t),-g,register)) for s,t,g in np.vstack((source[grad_<0],target[grad_<0],grad_[grad_<0])).T])
    # G.add_weighted_edges_from([(int(s),int(t),cost(data_pos,int(s),int(t),g,register)) for s,t,g in np.vstack((source[grad_<0],target[grad_<0],grad_[grad_<0])).T])
    
    # cmap_ = plt.get_cmap("tab10")
    # figsize = (10,8)
    # fig,ax = plt.subplots(figsize=figsize)
    # ax.triplot(tri_,color='gray',zorder=0,alpha=0.2,lw=1)
    # clusters_ = adata.obs[cluster_key]
    # idx_ = clusters_ == source_cluster
    # ax.scatter(data_pos[idx_,0],data_pos[idx_,1],color='gray',zorder=10,marker='D',alpha=0.2,s=5,label=source_cluster+' (source)')
    # for i_trg_ in range(len(target_clusters)):
    #     idx_ = clusters_ == target_clusters[i_trg_]
    #     ax.scatter(data_pos[idx_,0],data_pos[idx_,1],color=cmap_(i_trg_),zorder=10,marker='o',alpha=0.2,s=5,label=target_clusters[i_trg_]+' (target)')
    # leg = ax.legend(bbox_to_anchor=(1.05, 0.5), loc='center left', borderaxespad=0, fontsize=12,markerscale=3)
    # for lh in leg.legend_handles: lh.set_alpha(1)

    # data_src_ = data_pos[adata.obs[cluster_key].values == source_cluster]
    # center_src_ = np.mean(data_src_,axis=0)
    # centrality_src_ = np.linalg.norm(data_src_-center_src_,axis=1)
    # src_set_all_ = np.arange(adata.shape[0])[adata.obs[cluster_key].values == source_cluster][np.argsort(centrality_src_)]
    # n_src_ = sum(adata.obs[cluster_key].values == source_cluster)
    # path_all = {}
    # for i_trg_ in range(len(target_clusters)):
    #     target_cluster = target_clusters[i_trg_]
    #     n_cells_ = np.min([n_cells,sum(adata.obs[cluster_key].values == source_cluster),sum(adata.obs[cluster_key].values == target_cluster)])
    #     data_trg_ = data_pos[adata.obs[cluster_key].values == target_cluster]
    #     center_trg_ = np.mean(data_trg_,axis=0)
    #     centrality_trg_ = np.linalg.norm(data_trg_-center_trg_,axis=1)
    #     n_trg_ = sum(adata.obs[cluster_key].values == target_cluster)
    #     idx_trg_ = np.arange(0,n_trg_,int(n_trg_/n_cells_))[:n_cells_]
    #     trg_set_ = np.arange(adata.shape[0])[adata.obs[cluster_key].values == target_cluster][np.argsort(centrality_trg_)][idx_trg_]
    #     idx_src_ = np.arange(0,n_src_,int(n_src_/n_cells_))[:n_cells_]
    #     src_set_ = src_set_all_[idx_src_]

    #     pathes,edges,weights,dists  = [],[],[],[]
    #     for src_,trg_ in np.vstack((src_set_,trg_set_)).T:
    #         path = nx.dijkstra_path(G, source=src_, target=trg_, weight='weight')
    #         pathes.append(path)
    #         edges.append(np.array([[path[i], path[i+1]] for i in range(len(path)-1)]))
    #         weights.append((sum([G[path[i]][path[i+1]]['weight'] for i in range(len(path)-1)]))/sum([np.linalg.norm(data_pos[path[i]]-data_pos[path[i+1]]) for i in range(len(path)-1)]))
    #         dists.append(sum([np.linalg.norm(data_pos[path[i]]-data_pos[path[i+1]]) for i in range(len(path)-1)]))
    #     path_all[source_cluster+'_'+target_clusters[i_trg_]] = pathes
    #     ax.scatter(data_pos[trg_set_,0],data_pos[trg_set_,1],color=cmap_(i_trg_),zorder=20,marker='o',s=20)
    #     for i in range(n_cells_):
    #         ax.plot(data_pos[pathes[i],0],data_pos[pathes[i],1],color=cmap_(i_trg_),zorder=10,ls=':')
    # ax.scatter(data_pos[src_set_,0],data_pos[src_set_,1],color='gray',zorder=20,marker='D',s=30)
    # ax.axis('off')
    # adata.uns[path_key] = path_all


def calc_gene_dynamics(
        adata,
        source_cluster,
        target_clusters,
        path_key = 'path',
        exp_key = None,
        gene_dynamics_key = 'gene_dynamics',
        n_div = 100,
        degree = 10,
    ):

    if exp_key == None:
        if scipy.sparse.issparse(adata.X): data_exp = adata.X.toarray()
        else: data_exp = adata.X
    else:
        data_exp = adata.layers[exp_key]
    path = adata.uns[path_key]

    gene_dynamics_ = {}
    for i in range(len(path)):
        name_ = source_cluster+'_'+target_clusters[i]
        x_data,y_data = np.empty(0,dtype=float),np.empty([0,adata.shape[1]],dtype=float)
        for pi in path[name_]:
            x_data = np.append(x_data,np.linspace(0,1,len(pi)))
            y_data = np.vstack((y_data,data_exp[pi]))

        X = x_data[:, np.newaxis]
        poly = sklearn.preprocessing.PolynomialFeatures(degree=degree)
        X_poly = poly.fit_transform(X)
        model = sklearn.linear_model.LinearRegression()
        model.fit(X_poly, y_data)
        plot_x = np.linspace(0,1,n_div+1)
        gd_i_ = model.predict(poly.fit_transform(plot_x[:, np.newaxis]))
        gd_i_[gd_i_ <0] = 0
        gene_dynamics_[source_cluster+'_'+target_clusters[i]] = gd_i_
    adata.uns[gene_dynamics_key] = gene_dynamics_
    print('Done the computation of gene dynamics')
    

def gene_dynamics_plot(
    adata,
    source_cluster,
    target_clusters,
    genes,
    path_key = 'path',
    exp_key = None,
    gene_dynamics_key = 'gene_dynamics',
    n_div = 100,
    fontsize_title = 16,
    fontsize_label = 14,
    fontsize_legend = 12,
    save = False,
    save_dir = None,
    save_filename = 'gene_dynamics_plot',
):
    
    # kwargs_arg = check_arguments(adata, verbose=True)

    if gene_dynamics_key not in adata.uns.keys():
        calc_gene_dynamics(adata, source_cluster, target_clusters, path_key=path_key, exp_key=exp_key, gene_dynamics_key=gene_dynamics_key, n_div=n_div)

    if exp_key == None:
        if scipy.sparse.issparse(adata.X): data_exp = adata.X.toarray()
        else: data_exp = adata.X
    else:
        data_exp = adata.layers[exp_key]
    path = adata.uns[path_key]
    cmap_ = plt.get_cmap("tab10")

    for gene in genes:
        if gene in adata.var.index:
            fig = plt.figure(figsize=(10,6))
            y_data_all = []
            for i in range(len(target_clusters)):
                name_ = source_cluster+'_'+target_clusters[i]
                x_data,y_data = np.empty(0,dtype=float),np.empty(0,dtype=float)
                for pi in path[name_]:
                    y_ = data_exp[:,adata.var.index==gene][pi].T[0]
                    idx_ = y_>0
                    x_data = np.append(x_data,np.linspace(0,1,len(pi))[idx_])
                    y_data = np.append(y_data,y_[idx_])
                if len(y_data):
                    plt.scatter(x_data, y_data,color=cmap_(i),alpha=0.05,zorder=0)
                dynamics_ = adata.uns[gene_dynamics_key][name_][:,adata.var.index==gene]
                plot_x = np.linspace(0,1,len(dynamics_))
                dynamics_[dynamics_ < 0] = 0
                plt.plot(plot_x, dynamics_,color='w',lw=8,zorder=1)
                plt.plot(plot_x, dynamics_,color=cmap_(i),lw=5,label=target_clusters[i],zorder=2)
                y_data_all = np.append(y_data_all,y_data)
            y_top_ = np.percentile(y_data_all,99)
            plt.ylim([-0.05*y_top_,y_top_])
            plt.legend(bbox_to_anchor=(1.05, 0.5), loc='center left', borderaxespad=0,title='Target', fontsize=fontsize_legend, title_fontsize=fontsize_legend)
            plt.xticks([0,0.25,0.5,0.75,1],['Source (0)\n(%s)' % source_cluster,'0.25','0.5','0.75','Target (1)'],fontsize=fontsize_label)
            plt.title(gene,fontsize=fontsize_title)
            plt.show()
            if save:
                filename = '%s_%s' % (save_filename,gene) if save_dir == None else '%s/%s_%s' % (save_dir,save_filename,gene)
                fig.savefig(filename+'.png', bbox_inches='tight')
            plt.close()
        else:
            print('Gene \"%s\" was not found' % gene)



def DEG_dynamics(
        adata,
        source_cluster,
        target_clusters,
        path_key = 'path',
        exp_key = None,
        gene_dynamics_key = 'gene_dynamics',
        bifurcation_diagram_key = 'bifurcation_diagram',
        target_genes = [],
        n_div = 100,
        fontsize_label = 14,
        fontsize_text = 12,
        fontsize_nDEG = 18,
        fontsize_legend = 10,
        DEG_min = 1.0,
        DEG_rate = 0.3,
        max_num_annotations = 10,
        max_num_legend = 40,
        save = False,
        save_dir = None,
        save_filename = 'DEG_dynamics',
        save_type = 'gif',
    ):

    if gene_dynamics_key not in adata.uns.keys():
        calc_gene_dynamics(adata, source_cluster, target_clusters, path_key=path_key, exp_key=exp_key, gene_dynamics_key=gene_dynamics_key, n_div=n_div)

    n_plot_ = int(len(target_clusters)*(len(target_clusters)-1)/2)
    cmap_ = plt.get_cmap("tab10")
    gene_dynamics_ = adata.uns[gene_dynamics_key]
    matplotlib.rcParams['animation.embed_limit'] = 2**128
    vlines = [0,0.2,0.4,0.6,0.8,1]
    vline_labels = np.append(np.append('Source (0)',np.array(vlines)[1:-1]),'Target (1)')
    def update(t,name_i_,name_j_,max_val_,lim,i,j,k):
        print('\rcomputing %s vs %s (%d/%d) %d/%d' % (target_clusters[i],target_clusters[j],k,n_plot_,t+1,n_div+1),end='')
        idx_DEG_i_ = np.arange(adata.shape[1])[(gene_dynamics_[name_j_][t] < gene_dynamics_[name_i_][t] - DEG_rate) & (gene_dynamics_[name_i_][t] > DEG_min)]
        idx_DEG_j_ = np.arange(adata.shape[1])[(gene_dynamics_[name_i_][t] < gene_dynamics_[name_j_][t] - DEG_rate) & (gene_dynamics_[name_j_][t] > DEG_min)]
        idx_DEG_i_ = idx_DEG_i_[np.argsort(gene_dynamics_[name_i_][t][idx_DEG_i_]- DEG_rate - gene_dynamics_[name_j_][t][idx_DEG_i_])[::-1]]
        idx_DEG_j_ = idx_DEG_j_[np.argsort(gene_dynamics_[name_j_][t][idx_DEG_j_]- DEG_rate - gene_dynamics_[name_i_][t][idx_DEG_j_])[::-1]]
        if len(idx_DEG_i_) > max_num_annotations:
            idx_DEG_ann_i_ = idx_DEG_i_[:max_num_annotations]
        else:
            idx_DEG_ann_i_  = idx_DEG_i_
        if len(idx_DEG_j_) > max_num_annotations:
            idx_DEG_ann_j_ = idx_DEG_j_[:max_num_annotations]
        else:
            idx_DEG_ann_j_  = idx_DEG_j_

        if len(idx_DEG_i_) > max_num_legend:
            idx_DEG_leg_i_ = idx_DEG_i_[:max_num_legend]
        else:
            idx_DEG_leg_i_  = idx_DEG_i_
        if len(idx_DEG_j_) > max_num_legend:
            idx_DEG_leg_j_ = idx_DEG_j_[:max_num_legend]
        else:
            idx_DEG_leg_j_  = idx_DEG_j_
        ax1.cla()
        ax2.cla()
        ax3.cla()
        name_i__ = source_cluster + '_' + target_clusters[0]
        ax1.text(0,adata.uns[bifurcation_diagram_key][name_i_][0],source_cluster+' ',fontsize=fontsize_label,va='center',ha='right')
        for i_ in range(len(target_clusters)):
            name_i__ = source_cluster + '_' + target_clusters[i_]
            if name_i__ not in [name_i_,name_j_]:
                y_ = adata.uns[bifurcation_diagram_key][name_i__]
                ax1.plot(np.linspace(0,1,len(y_)),y_,lw=3,zorder=2,alpha=0.3,color=cmap_(i_))
                ax1.text(1,y_[-1],' '+target_clusters[i_],fontsize=fontsize_label,va='center',ha='left',alpha=0.3)
        y_ = adata.uns[bifurcation_diagram_key][name_i_]
        ax1.plot(np.linspace(0,1,len(y_)),y_,lw=5,zorder=3,color=cmap_(i))
        ax1.text(1,y_[-1],' '+target_clusters[i],fontsize=fontsize_label,va='center',ha='left')
        y_ = adata.uns[bifurcation_diagram_key][name_j_]
        ax1.plot(np.linspace(0,1,len(y_)),y_,lw=5,zorder=3,color=cmap_(j))
        ax1.text(1,y_[-1],' '+target_clusters[j],fontsize=fontsize_label,va='center',ha='left')
        for vl in vlines:
            ax1.axvline(vl,color='k',ls='--',lw=1,zorder=0)
        ax1.axvline(t/n_div,color='r',ls='-',lw=2,zorder=3)
        ax1.tick_params(axis='x', which='both', top=True)
        ax1.spines['right'].set_visible(False)
        ax1.spines['left'].set_visible(False)
        ax1.spines['top'].set_visible(False)
        ax1.spines['bottom'].set_visible(False)
        ax1.xaxis.set_label_position('top')
        ax1.xaxis.tick_top()
        ax1.yaxis.set_visible(False)
        ax1.set_xticks(vlines)
        ax1.set_xticklabels(vline_labels,fontsize=fontsize_label)
        ax2.set_title('Time = %.02f [s]' % (t/n_div))
        ax2.scatter(gene_dynamics_[name_i_][t],gene_dynamics_[name_j_][t],s=1,color="gray",zorder=1)
        ax2.scatter(gene_dynamics_[name_i_][t][idx_DEG_i_],gene_dynamics_[name_j_][t][idx_DEG_i_],color=cmap_(i),zorder=2,s=20)
        ax2.scatter(gene_dynamics_[name_i_][t][idx_DEG_j_],gene_dynamics_[name_j_][t][idx_DEG_j_],color=cmap_(j),zorder=2,s=20)
        texts = []
        for g in np.arange(adata.shape[1])[idx_DEG_ann_i_]:
            tx_ = ax2.text(gene_dynamics_[name_i_][t][g],gene_dynamics_[name_j_][t][g],'_'+adata.var.index[g],color="k",zorder=2,fontsize=fontsize_text)
            texts = np.append(texts,tx_)
        for g in np.arange(adata.shape[1])[idx_DEG_ann_j_]:
            tx_ = ax2.text(gene_dynamics_[name_i_][t][g],gene_dynamics_[name_j_][t][g],'_'+adata.var.index[g],color="k",zorder=2,fontsize=fontsize_text)
            texts = np.append(texts,tx_)
        if len(target_genes):
            for gene_ in target_genes:
                idx_gene_ = adata.var.index == gene_
                ax2.scatter(gene_dynamics_[name_i_][t][idx_gene_],gene_dynamics_[name_j_][t][idx_gene_],s=20,color="red",zorder=2)
                tx_ = ax2.text(gene_dynamics_[name_i_][t][idx_gene_],gene_dynamics_[name_j_][t][idx_gene_],'_'+gene_,color="r",zorder=2,fontsize=fontsize_text)
                texts = np.append(texts,tx_)
        legend_i_ = ''
        for g in np.arange(adata.shape[1])[idx_DEG_leg_i_]: legend_i_ += '(%.02f, %.02f)  %s\n' % (gene_dynamics_[name_i_][t][g],gene_dynamics_[name_j_][t][g],adata.var.index[g])
        legend_j_ = ''
        for g in np.arange(adata.shape[1])[idx_DEG_leg_j_]: legend_j_ += '(%.02f, %.02f)  %s\n' % (gene_dynamics_[name_i_][t][g],gene_dynamics_[name_j_][t][g],adata.var.index[g])
        ax2.text(0.9*(lim[1]-lim[0])+lim[0], 0.1*(lim[1]-lim[0])+lim[0], str(len(idx_DEG_i_)) , ha='center', va='center', fontsize=fontsize_nDEG,color=cmap_(i), fontweight="bold",zorder=3)
        ax2.text(0.1*(lim[1]-lim[0])+lim[0], 0.9*(lim[1]-lim[0])+lim[0], str(len(idx_DEG_j_)) , ha='center', va='center', fontsize=fontsize_nDEG,color=cmap_(j), fontweight="bold",zorder=3)
        ax2.fill_between(lim, lim-DEG_rate, lim+DEG_rate, facecolor='lightgray',  alpha=0.5,zorder=0)
        ax2.fill([-0.01*max_val_, DEG_min, DEG_min, -0.01*max_val_], [-0.01*max_val_, -0.01*max_val_, DEG_min, DEG_min], facecolor='lightgray', alpha=0.5,zorder=0)
        ax2.set_xlabel(target_clusters[i],fontsize=fontsize_label,color=cmap_(i), fontweight="bold")
        ax2.set_ylabel(target_clusters[j],fontsize=fontsize_label,color=cmap_(j), fontweight="bold")
        ax2.set_xlim(lim)
        ax2.set_ylim(lim)
        ax2.grid(ls='--')
        ax3.text(0.0,1,target_clusters[i], ha='left', va='top', fontsize=fontsize_legend,color=cmap_(i),zorder=3, fontweight="bold")
        ax3.text(0.0,0.97,legend_i_, ha='left', va='top', fontsize=fontsize_legend,color=cmap_(i),zorder=3)
        ax3.text(0.5,1,target_clusters[j], ha='left', va='top', fontsize=fontsize_legend,color=cmap_(j),zorder=3, fontweight="bold")
        ax3.text(0.5,0.97,legend_j_, ha='left', va='top', fontsize=fontsize_legend,color=cmap_(j),zorder=3)
        ax3.axis('off')
        
    k = 0
    for i in range(len(target_clusters)):
        for j in range(i+1,len(target_clusters)):
            name_i_ = source_cluster+'_'+target_clusters[i]
            name_j_ = source_cluster+'_'+target_clusters[j]
            # fig,ax = plt.subplots(1,2,figsize=(14,8),gridspec_kw={'width_ratios': [4,3]},tight_layout=True)
            fig = plt.figure(figsize=(14,10),tight_layout=True)
            grid = plt.GridSpec(10,14)
            ax1 = fig.add_subplot(grid[0:2,0:12])
            ax2 = fig.add_subplot(grid[2:10,0:8])
            ax3 = fig.add_subplot(grid[2:10,8:14])
            max_val_ = max(np.max(gene_dynamics_[name_i_]),np.max(gene_dynamics_[name_j_]))
            lim = np.array([-0.01*max_val_,1.01*max_val_])
            k = k+1
            ani = anm.FuncAnimation(fig,update,interval=200,fargs=(name_i_,name_j_,max_val_,lim,i,j,k,),frames=n_div+1)
            IPython.display.display(IPython.display.HTML(ani.to_jshtml()))
            if save:
                filename = '%s_%s_%s' % (save_filename,target_clusters[i],target_clusters[j]) if save_dir == None else '%s/%s_%s_%s' % (save_dir,save_filename,target_clusters[i],target_clusters[j])
                if len(target_genes): filename += '_TG' + str(len(target_genes))
                if save_type in ['gif','video','animetion']:
                    filename += '.gif'
                    print('\nSaving gif animation as %s...' % filename)
                    ani.save(filename)
                elif save_type in ['image','png','jpg','jpeg']:
                    if save_type == 'image': save_type = 'png'
                    print('\nSaving gif animation as %s' % filename)
                    for t in range(n_div+1):
                        fig = plt.figure(figsize=(14,10),tight_layout=True)
                        grid = plt.GridSpec(10,14)
                        ax1 = fig.add_subplot(grid[0:2,0:12])
                        ax2 = fig.add_subplot(grid[2:10,0:8])
                        ax3 = fig.add_subplot(grid[2:10,8:14])
                        update(t,name_i_,name_j_,max_val_,lim,i,j,k)
                        filename_ = '%s_%03d.%s' % (filename,t,save_type)
                        fig.savefig(filename_, bbox_inches='tight')
                        plt.close()
            plt.close()

def DEG_dynamics_clusters(
        adata,
        source_cluster,
        target_clusters,
        path_key = 'path',
        exp_key = None,
        gene_dynamics_key = 'gene_dynamics',
        bifurcation_diagram_key = 'bifurcation_diagram',
        target_genes = [],
        n_div = 100,
        fontsize_label = 14,
        fontsize_text = 12,
        fontsize_nDEG = 18,
        fontsize_legend = 10,
        DEG_min = 1.0,
        DEG_rate = 0.3,
        max_num_annotations = 10,
        max_num_legend = 25,
        save = False,
        save_dir = None,
        save_filename = 'DEG_dynamics',
        save_type = 'gif',
    ):

    if gene_dynamics_key not in adata.uns.keys():
        calc_gene_dynamics(adata, source_cluster, target_clusters, path_key=path_key, exp_key=exp_key, gene_dynamics_key=gene_dynamics_key, n_div=n_div)

    n_plot_ = int(len(target_clusters)*(len(target_clusters)-1)/2)
    cmap_ = plt.get_cmap("tab10")
    cmap20_ = plt.get_cmap("tab20")
    gene_dynamics_ = adata.uns[gene_dynamics_key]
    matplotlib.rcParams['animation.embed_limit'] = 2**128
    vlines = [0,0.2,0.4,0.6,0.8,1]
    vline_labels = np.append(np.append('Source (0)',np.array(vlines)[1:-1]),'Target (1)')
    def update(t,name_i_,name_j_,max_val_,lim,i,j,k):
        print('\rcomputing %s vs %s (%d/%d) %d/%d' % (target_clusters[i],target_clusters[j],k,n_plot_,t+1,n_div+1),end='')
        idx_DEG_i_ = np.arange(adata.shape[1])[(gene_dynamics_[name_j_][t] < gene_dynamics_[name_i_][t] - DEG_rate) & (gene_dynamics_[name_i_][t] > DEG_min)]
        idx_DEG_j_ = np.arange(adata.shape[1])[(gene_dynamics_[name_i_][t] < gene_dynamics_[name_j_][t] - DEG_rate) & (gene_dynamics_[name_j_][t] > DEG_min)]
        idx_DEG_i_ = idx_DEG_i_[np.argsort(gene_dynamics_[name_i_][t][idx_DEG_i_]- DEG_rate - gene_dynamics_[name_j_][t][idx_DEG_i_])[::-1]]
        idx_DEG_j_ = idx_DEG_j_[np.argsort(gene_dynamics_[name_j_][t][idx_DEG_j_]- DEG_rate - gene_dynamics_[name_i_][t][idx_DEG_j_])[::-1]]
        if len(idx_DEG_i_) > max_num_annotations:
            idx_DEG_ann_i_ = idx_DEG_i_[:max_num_annotations]
        else:
            idx_DEG_ann_i_  = idx_DEG_i_
        if len(idx_DEG_j_) > max_num_annotations:
            idx_DEG_ann_j_ = idx_DEG_j_[:max_num_annotations]
        else:
            idx_DEG_ann_j_  = idx_DEG_j_

        if len(idx_DEG_i_) > max_num_legend:
            idx_DEG_leg_i_ = idx_DEG_i_[:max_num_legend]
        else:
            idx_DEG_leg_i_  = idx_DEG_i_
        if len(idx_DEG_j_) > max_num_legend:
            idx_DEG_leg_j_ = idx_DEG_j_[:max_num_legend]
        else:
            idx_DEG_leg_j_  = idx_DEG_j_
        ax1.cla()
        ax2.cla()
        ax3.cla()
        name_i__ = source_cluster + '_' + target_clusters[0]
        ax1.text(0,adata.uns[bifurcation_diagram_key][name_i_][0],source_cluster+' ',fontsize=fontsize_label,va='center',ha='right')
        for i_ in range(len(target_clusters)):
            name_i__ = source_cluster + '_' + target_clusters[i_]
            if name_i__ not in [name_i_,name_j_]:
                y_ = adata.uns[bifurcation_diagram_key][name_i__]
                ax1.plot(np.linspace(0,1,len(y_)),y_,lw=3,zorder=2,alpha=0.3)
                ax1.text(1,y_[-1],' '+target_clusters[i_],fontsize=fontsize_label,va='center',ha='left',alpha=0.3)
        y_ = adata.uns[bifurcation_diagram_key][name_i_]
        ax1.plot(np.linspace(0,1,len(y_)),y_,lw=5,zorder=3)
        ax1.text(1,y_[-1],' '+target_clusters[i],fontsize=fontsize_label,va='center',ha='left')
        y_ = adata.uns[bifurcation_diagram_key][name_j_]
        ax1.plot(np.linspace(0,1,len(y_)),y_,lw=5,zorder=3)
        ax1.text(1,y_[-1],' '+target_clusters[j],fontsize=fontsize_label,va='center',ha='left')
        for vl in vlines:
            ax1.axvline(vl,color='k',ls='--',lw=1,zorder=0)
        ax1.axvline(t/n_div,color='r',ls='-',lw=2,zorder=3)
        ax1.tick_params(axis='x', which='both', top=True)
        ax1.spines['right'].set_visible(False)
        ax1.spines['left'].set_visible(False)
        ax1.spines['top'].set_visible(False)
        ax1.spines['bottom'].set_visible(False)
        ax1.xaxis.set_label_position('top')
        ax1.xaxis.tick_top()
        ax1.yaxis.set_visible(False)
        ax1.set_xticks(vlines)
        ax1.set_xticklabels(vline_labels,fontsize=fontsize_label)
        ax2.set_title('Time = %.02f [s]' % (t/n_div))
        ax2.scatter(gene_dynamics_[name_i_][t],gene_dynamics_[name_j_][t],s=1,color="gray",zorder=1)
        for c_ in range(n_clusters):
            idx_ = (adata.var['clusters_'+name_i_] == c_) & (gene_dynamics_[name_i_][t] > gene_dynamics_[name_j_][t])
            ax2.scatter(gene_dynamics_[name_i_][t][idx_],gene_dynamics_[name_j_][t][idx_],color=cmap20_(c_),zorder=2,s=20,label=str(c_+1),marker='o')
            idx_ = (adata.var['clusters_'+name_j_] == c_) & (gene_dynamics_[name_i_][t] < gene_dynamics_[name_j_][t])
            ax2.scatter(gene_dynamics_[name_i_][t][idx_],gene_dynamics_[name_j_][t][idx_],color=cmap20_(c_),zorder=2,s=20,marker='o')
        ax2.legend(loc='lower left', bbox_to_anchor=(1.05, 0.0), ncol=5,  title = "Clusters", columnspacing=0.5)
        texts = []
        for g in np.arange(adata.shape[1])[idx_DEG_ann_i_]:
            tx_ = ax2.text(gene_dynamics_[name_i_][t][g],gene_dynamics_[name_j_][t][g],'_'+adata.var.index[g],color="k",zorder=2,fontsize=fontsize_text)
            texts = np.append(texts,tx_)
        for g in np.arange(adata.shape[1])[idx_DEG_ann_j_]:
            tx_ = ax2.text(gene_dynamics_[name_i_][t][g],gene_dynamics_[name_j_][t][g],'_'+adata.var.index[g],color="k",zorder=2,fontsize=fontsize_text)
            texts = np.append(texts,tx_)
        if len(target_genes):
            for gene_ in target_genes:
                idx_gene_ = adata.var.index == gene_
                ax2.scatter(gene_dynamics_[name_i_][t][idx_gene_],gene_dynamics_[name_j_][t][idx_gene_],s=20,color="red",zorder=2)
                tx_ = ax2.text(gene_dynamics_[name_i_][t][idx_gene_],gene_dynamics_[name_j_][t][idx_gene_],'_'+gene_,color="r",zorder=2,fontsize=fontsize_text)
                texts = np.append(texts,tx_)
        legend_i_ = ''
        for g in np.arange(adata.shape[1])[idx_DEG_leg_i_]: legend_i_ += '(%.02f, %.02f)  %s\n' % (gene_dynamics_[name_i_][t][g],gene_dynamics_[name_j_][t][g],adata.var.index[g])
        legend_j_ = ''
        for g in np.arange(adata.shape[1])[idx_DEG_leg_j_]: legend_j_ += '(%.02f, %.02f)  %s\n' % (gene_dynamics_[name_i_][t][g],gene_dynamics_[name_j_][t][g],adata.var.index[g])
        ax2.text(0.9*(lim[1]-lim[0])+lim[0], 0.1*(lim[1]-lim[0])+lim[0], str(len(idx_DEG_i_)) , ha='center', va='center', fontsize=fontsize_nDEG,color=cmap_(i), fontweight="bold",zorder=3)
        ax2.text(0.1*(lim[1]-lim[0])+lim[0], 0.9*(lim[1]-lim[0])+lim[0], str(len(idx_DEG_j_)) , ha='center', va='center', fontsize=fontsize_nDEG,color=cmap_(j), fontweight="bold",zorder=3)
        ax2.fill_between(lim, lim-DEG_rate, lim+DEG_rate, facecolor='lightgray',  alpha=0.5,zorder=0)
        ax2.fill([-0.01*max_val_, DEG_min, DEG_min, -0.01*max_val_], [-0.01*max_val_, -0.01*max_val_, DEG_min, DEG_min], facecolor='lightgray', alpha=0.5,zorder=0)
        ax2.set_xlabel(target_clusters[i],fontsize=fontsize_label,color=cmap_(i), fontweight="bold")
        ax2.set_ylabel(target_clusters[j],fontsize=fontsize_label,color=cmap_(j), fontweight="bold")
        ax2.set_xlim(lim)
        ax2.set_ylim(lim)
        ax2.grid(ls='--')
        ax3.text(0.0,1,target_clusters[i], ha='left', va='top', fontsize=fontsize_legend,color=cmap_(i),zorder=3, fontweight="bold")
        ax3.text(0.0,0.97,legend_i_, ha='left', va='top', fontsize=fontsize_legend,color=cmap_(i),zorder=3)
        ax3.text(0.5,1,target_clusters[j], ha='left', va='top', fontsize=fontsize_legend,color=cmap_(j),zorder=3, fontweight="bold")
        ax3.text(0.5,0.97,legend_j_, ha='left', va='top', fontsize=fontsize_legend,color=cmap_(j),zorder=3)
        ax3.axis('off')
        
    k = 0
    for i in range(len(target_clusters)):
        for j in range(i+1,len(target_clusters)):
            name_i_ = source_cluster+'_'+target_clusters[i]
            name_j_ = source_cluster+'_'+target_clusters[j]
            # fig,ax = plt.subplots(1,2,figsize=(14,8),gridspec_kw={'width_ratios': [4,3]},tight_layout=True)
            fig = plt.figure(figsize=(14,10),tight_layout=True)
            grid = plt.GridSpec(10,14)
            ax1 = fig.add_subplot(grid[0:2,0:12])
            ax2 = fig.add_subplot(grid[2:10,0:8])
            ax3 = fig.add_subplot(grid[2:10,8:14])
            max_val_ = max(np.max(gene_dynamics_[name_i_]),np.max(gene_dynamics_[name_j_]))
            lim = np.array([-0.01*max_val_,1.01*max_val_])
            k = k+1
            ani = anm.FuncAnimation(fig,update,interval=200,fargs=(name_i_,name_j_,max_val_,lim,i,j,k,),frames=n_div+1)
            IPython.display.display(IPython.display.HTML(ani.to_jshtml()))
            plt.close()
            if save:
                filename = '%s_%s_%s' % (save_filename,target_clusters[i],target_clusters[j]) if save_dir == None else '%s/%s_%s_%s' % (save_dir,save_filename,target_clusters[i],target_clusters[j])
                if len(target_genes): filename += '_TG' + str(len(target_genes))
                if save_type in ['gif','video','animetion']:
                    filename += '.gif'
                    print('\nSaving gif animation as %s' % filename)
                    ani.save(filename)
                elif save_type in ['image','png','jpg','jpeg']:
                    matplotlib.use('Agg')
                    if save_type == 'image': save_type = 'png'
                    print('\nSaving gif animation as %s' % filename)
                    for t in range(n_div+1):
                        fig = plt.figure(figsize=(14,10),tight_layout=True)
                        grid = plt.GridSpec(10,14)
                        ax1 = fig.add_subplot(grid[0:2,0:12])
                        ax2 = fig.add_subplot(grid[2:10,0:8])
                        ax3 = fig.add_subplot(grid[2:10,8:14])
                        update(t,name_i_,name_j_,max_val_,lim,i,j,k)
                        filename_ = '%s_%03d.%s' % (filename,t,save_type)
                        fig.savefig(filename_, bbox_inches='tight')
                        plt.close()
                    matplotlib.use('module://matplotlib_inline.backend_inline')
                    

def calc_bifurcation_diagram(
        adata,
        source_cluster,
        target_clusters,
        path_key = 'path',
        exp_key = None,
        gene_dynamics_key = 'gene_dynamics',
        bifurcation_diagram_key = 'bifurcation_diagram',
        n_div = 100,
        PC = 1,
    ):

    if gene_dynamics_key not in adata.uns.keys():
        calc_gene_dynamics(adata, source_cluster, target_clusters, path_key=path_key, exp_key=exp_key, gene_dynamics_key=gene_dynamics_key, n_div=n_div)

    name_i_ = source_cluster + '_' + target_clusters[0]
    samples_ = np.empty([len(target_clusters),adata.uns['gene_dynamics'][name_i_].shape[0],adata.uns['gene_dynamics'][name_i_].shape[1]],dtype=float)
    for i in range(len(target_clusters)):
        name_i_ = source_cluster + '_' + target_clusters[i]
        samples_[i] = adata.uns['gene_dynamics'][name_i_]
    pca_ = sklearn.decomposition.PCA().fit(samples_[:,-1])
    samples_pca_ = pca_.transform(np.concatenate(samples_))

    bd_ = {}
    for i in range(len(target_clusters)):
        name_i_ = source_cluster + '_' + target_clusters[i]
        bd_[name_i_] = samples_pca_[i*(n_div+1):(i+1)*(n_div+1),PC-1]
    
    adata.uns[bifurcation_diagram_key] = bd_

def bifurcation_diagram(
        adata,
        source_cluster,
        target_clusters,
        path_key = 'path',
        exp_key = None,
        gene_dynamics_key = 'gene_dynamics',
        bifurcation_diagram_key = 'bifurcation_diagram',
        n_div = 100,
        fontsize_label = 14,
        adjusttext =False,
        PC = 1,
        save = False,
        save_dir = None,
        save_filename = 'bifurcation_diagram',
    ):

    if gene_dynamics_key not in adata.uns.keys():
        calc_gene_dynamics(adata, source_cluster, target_clusters, path_key=path_key, exp_key=exp_key, gene_dynamics_key=gene_dynamics_key, n_div=n_div)
    
    if bifurcation_diagram_key not in adata.uns.keys():
        calc_bifurcation_diagram(
        adata, source_cluster, target_clusters, path_key=path_key, exp_key=exp_key, gene_dynamics_key=gene_dynamics_key, 
            bifurcation_diagram_key=bifurcation_diagram_key, n_div=n_div, PC=PC)

    vlines = [0,0.2,0.4,0.6,0.8,1]
    vline_labels = np.append(np.append('Source (0)',np.array(vlines)[1:-1]),'Target (1)')

    fig,ax = plt.subplots(figsize=(12,3))
    name_i_ = source_cluster + '_' + target_clusters[0]
    ax.text(0,adata.uns[bifurcation_diagram_key][name_i_][0],source_cluster+' ',fontsize=fontsize_label,va='center',ha='right')
    texts = []
    for i in range(len(target_clusters)):
        name_i_ = source_cluster + '_' + target_clusters[i]
        y_ = adata.uns[bifurcation_diagram_key][name_i_]
        ax.plot(np.linspace(0,1,n_div+1),y_,lw=5,zorder=2)
        texts = np.append(texts,ax.text(1,y_[-1],' '+target_clusters[i],fontsize=fontsize_label,va='center',ha='left'))
    if adjusttext: adjustText.adjust_text(texts, arrowprops=dict(arrowstyle='-', color='k'))
    for vl in vlines:
        ax.axvline(vl,color='k',ls='--',lw=1,zorder=0)
    ax.tick_params(axis='x', which='both', top=True)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.xaxis.set_label_position('top')
    ax.xaxis.tick_top()
    ax.yaxis.set_visible(False)
    ax.set_xticks(vlines)
    ax.set_xticklabels(vline_labels,fontsize=fontsize_label)
    if save:
        filename = '%s' % (save_filename) if save_dir == None else '%s/%s' % (save_dir,save_filename)
        fig.savefig(filename+'.png', bbox_inches='tight')

def calc_gene_atlas(
        adata,
        source_cluster,
        target_clusters,
        path_key = 'path',
        exp_key = None,
        gene_dynamics_key = 'gene_dynamics',
        gene_atlas_key = 'gene_atlas',
        n_div = 100,
        n_neighbors = 15,
        min_dist = 0.3,
        seed = 0,
        threshold_min = 1,
        n_clusters = 20,
        n_components = 2,
    ):

    if gene_dynamics_key not in adata.uns.keys():
        calc_gene_dynamics(adata, source_cluster, target_clusters, path_key=path_key, exp_key=exp_key, gene_dynamics_key=gene_dynamics_key, n_div=n_div)

    gene_dynamics_ = adata.uns[gene_dynamics_key]
    gene_dynamics_all_ = np.empty([0,n_div+1],dtype=float)
    gene_dynamics_all_norm_ = np.empty([0,n_div+1],dtype=float)
    idx_gene_dynamics_ = [0]
    for i in range(len(target_clusters)):
        name_i_ = source_cluster + '_' + target_clusters[i]
        max_ = np.max(gene_dynamics_[name_i_],axis=0)
        idx_ = max_ > threshold_min
        adata.var['expressed_'+name_i_] = idx_
        idx_gene_dynamics_ = np.append(idx_gene_dynamics_,idx_gene_dynamics_[i]+sum(idx_))
        gene_dynamics_all_ = np.vstack((gene_dynamics_all_,gene_dynamics_[name_i_][:,idx_].T))
        gene_dynamics_all_norm_ = np.vstack((gene_dynamics_all_norm_,(gene_dynamics_[name_i_][:,idx_]/max_[idx_]).T))


    umap_ = umap.UMAP(n_components=n_components,random_state=seed,n_neighbors=n_neighbors,min_dist=min_dist)
    gene_dynamics_all_umap_ = umap_.fit_transform(gene_dynamics_all_)

    data_ = gene_dynamics_all_umap_
    gm = sklearn.mixture.GaussianMixture(n_components=n_clusters,random_state=0).fit(data_)
    clusters_tmp_ = gm.predict(data_)
    pc1_ = sklearn.decomposition.PCA(n_components=1).fit_transform(data_)[:,0]
    pc1_ = np.sign(np.dot(pc1_,gene_dynamics_all_umap_[:,0]))*pc1_
    pc1_order_ = np.argsort([np.mean(pc1_[clusters_tmp_==i]) for i in range(n_clusters)])
    dict_sort_ = dict(zip(pc1_order_,np.unique(clusters_tmp_)))
    clusters_ = np.array([dict_sort_[c] for c in clusters_tmp_])

    texts_ = []
    index_ = []
    s_,e_ = 0,0
    for i in range(len(target_clusters)):
        name_i_ = source_cluster + '_' + target_clusters[i]
        idx_ = adata.var['expressed_'+name_i_]
        gene_list_ = adata.var.index[idx_].values
        e_ += sum(idx_)
        txt_ = gene_list_ + '<br>' + target_clusters[i]+'<br>cluster '+ np.array(clusters_[s_:e_]+1,dtype=str)#+ '<br><img src="'+image_+'" width="200">'
        texts_= np.append(texts_,txt_)
        index_ = np.append(index_,source_cluster + '_' + target_clusters[i] + '_' + gene_list_)
        adata.var['clusters_'+name_i_] = -np.ones(adata.shape[1],dtype=int)
        adata.var['clusters_'+name_i_][idx_] = clusters_[s_:e_]
        s_ += sum(idx_)

    adata.uns[gene_atlas_key] = {
        'index':index_,
        'texts':texts_,
        'dynamics':gene_dynamics_all_,
        'dynamics_norm':gene_dynamics_all_norm_,
        'gene_atlas':gene_dynamics_all_umap_,
        'clusters': clusters_,
    }

def gene_atlas(
        adata,
        source_cluster,
        target_clusters,
        target_genes = [],
        gene_dynamics_key = 'gene_dynamics',
        gene_atlas_key = 'gene_atlas',
        normalization = False,
        n_div = 100,
        n_neighbors = 15,
        min_dist = 0.3,
        seed = 0,
        threshold_min = 1,
        n_clusters = 20,
        n_components = 2,
        pt_size = 5,
        save = False,
        save_dir = None,
        save_filename = 'gene_atlas',
        save_type = 'html',
    ):
    if gene_atlas_key not in adata.uns.keys():
        calc_gene_atlas(adata,source_cluster,target_clusters,gene_dynamics_key = gene_dynamics_key, gene_atlas_key=gene_atlas_key,
            n_div = n_div,n_neighbors = n_neighbors,min_dist = min_dist,seed = seed,threshold_min=threshold_min, 
            n_clusters = n_clusters,n_components = n_components,
        )
    elif n_clusters != len(np.unique(adata.uns[gene_atlas_key]['clusters'])):
        calc_gene_atlas(adata,source_cluster,target_clusters,gene_dynamics_key = gene_dynamics_key, gene_atlas_key = gene_atlas_key,
            n_div=n_div,n_neighbors = n_neighbors,min_dist = min_dist,seed = seed,threshold_min=threshold_min, 
            n_clusters = n_clusters,n_components = n_components,
        )
    
    texts_ = adata.uns[gene_atlas_key]['texts']
    if normalization:
        gene_dynamics_all_ = adata.uns[gene_atlas_key]['dynamics_norm']
    else:
        gene_dynamics_all_ = adata.uns[gene_atlas_key]['dynamics']
    gene_dynamics_all_umap_ = adata.uns[gene_atlas_key]['gene_atlas']
    clusters_ = adata.uns[gene_atlas_key]['clusters']
    cluster_set_ = np.unique(clusters_)

    x_data = gene_dynamics_all_umap_[:,0]
    y_data = gene_dynamics_all_umap_[:,1]

    color_clusters_ = np.array(['rgb'+str(tuple(int(i*255) for i in plt.get_cmap("tab20")(c%20))) for c in clusters_])
    color_celltypes_ = np.empty(len(gene_dynamics_all_),dtype=object)

    
    annotations = [
        go.layout.Annotation(
            xref='paper',
            yref='paper',
            x=0.01,
            y=0.99,
            text='<b>Gene Atlas<b>',
            font=dict(size=18,color='white'),
            showarrow=False,
        )
    ]
    s_,e_ = 0,0
    for i in range(len(target_clusters)):
        name_i_ = source_cluster + '_' + target_clusters[i]
        idx_ = adata.var['expressed_'+name_i_]
        e_ += sum(idx_)
        gene_list_ = adata.var.index[idx_].values
        color_celltypes_[s_:e_] = 'rgba'+str(plt.get_cmap("tab10")(i%10))
        for gene in target_genes:
            if gene in gene_list_:
                x_pos = x_data[np.arange(sum(idx_))[gene_list_==gene][0]+s_]
                y_pos = y_data[np.arange(sum(idx_))[gene_list_==gene][0]+s_]
                annotations.append(
                    go.layout.Annotation(
                        x=x_pos,
                        y=y_pos,
                        xref='x',
                        yref='y',
                        text='<b>%s_%s<b>' % (target_clusters[i],gene),
                        showarrow=True,
                        arrowhead=1,
                        arrowcolor='white',
                        font=dict(size=12,color='white'),
                    )
                )
        s_ += sum(idx_)

    data_clusters_ = [
        go.Scatter(
            x=x_data,
            y=y_data,
            mode='markers',
            marker=dict(
                color=color_clusters_,
                size=30,
                opacity=0.2,
            ),
            hoverinfo='skip',
            showlegend=False,
        )
    ]
    for c in cluster_set_:
        idx_ = clusters_ ==c
        data_clusters_.append(
            go.Scatter(
                x=x_data[idx_],
                y=y_data[idx_],
                text = texts_[idx_],
                mode='markers',
                name='cluster '+str(c+1),
                marker=dict(
                    color='rgb'+str(tuple(int(i*255) for i in plt.get_cmap("tab20")(c%20))),
                    size=pt_size,
                    opacity=1,
                    line=dict(
                        color='white',
                        width=0.5,
                    ),
                ),
            )
        )

    s_,e_ = 0,0
    data_celltypes_ = []
    for i in range(len(target_clusters)):
        name_i_ = source_cluster + '_' + target_clusters[i]
        idx_ = adata.var['expressed_'+name_i_]
        gene_list_ = adata.var.index[idx_].values
        e_ += sum(idx_)
        data_celltypes_.append(
            go.Scatter(
                x=x_data[s_:e_],
                y=y_data[s_:e_],
                text = texts_[s_:e_],
                mode='markers',
                name=target_clusters[i],
                marker=dict(
                    color=color_celltypes_[s_:e_],
                    size=pt_size,
                    opacity=1,
                    line=dict(
                        color='white',
                        width=0.5,
                    ),
                )
            )
        )
        s_ += sum(idx_)


    layout = go.Layout(
        width=1200,
        height=800,
        plot_bgcolor='rgba(1,1,1,1)',
        paper_bgcolor='rgba(0,0,0,0)',
        xaxis=dict(
            # showgrid=False,
            gridcolor='gray',
            gridwidth=1,
            griddash='dot',
            zeroline=False,
            showticklabels=False,
            layer='below traces',
        ),
        yaxis=dict(
            # showgrid=False,
            gridcolor='gray',
            gridwidth=1,
            griddash='dot',
            zeroline=False,
            showticklabels=False,
            layer='below traces',
        ),
        annotations=annotations
    )

    fig1 = go.Figure(data=data_clusters_, layout=layout)
    pio.show(fig1)

    fig2 = go.Figure(data=data_celltypes_, layout=layout)
    pio.show(fig2)

    if save:
        filename = '%s' % (save_filename) if save_dir == None else '%s/%s' % (save_dir,save_filename)
        if save_type in ['png','pdf','svg','eps']:
            pio.write_image(fig1, filename + '_cluster.' + save_type)
            pio.write_image(fig2, filename + '_celltype.' + save_type)
        if save_type in ['html']:
            plotly.offline.plot(fig1, filename=filename + '_cluster.' + save_type)
            plotly.offline.plot(fig2, filename=filename + '_celltype.' + save_type)



def gene_dynamics_clusters(
        adata,
        source_cluster,
        target_clusters,
        gene_dynamics_key = 'gene_dynamics',
        gene_atlas_key = 'gene_atlas',
        normalization = False,
        n_div = 100,
        n_neighbors = 15,
        min_dist = 0.3,
        seed = 0,
        threshold_min = 1,
        n_clusters = 20,
        n_components = 2,
    ):
    
    if gene_atlas_key not in adata.uns.keys():
        calc_gene_atlas(adata,source_cluster,target_clusters,gene_dynamics_key = gene_dynamics_key,ene_atlas_key = gene_atlas_key,
            n_div = n_div,n_neighbors = n_neighbors,min_dist = min_dist,seed = seed,threshold_min=threshold_min, 
            n_clusters = n_clusters,n_components = n_components,
        )
    
    if normalization:
        gene_dynamics_all_ = adata.uns[gene_atlas_key]['dynamics_norm']
        ylabel_ = 'normalized gene expression'
    else:
        gene_dynamics_all_ = adata.uns[gene_atlas_key]['dynamics']
        ylabel_ = 'gene expression'
    
    n_clusters = len(np.unique(adata.uns[gene_atlas_key]['clusters']))
    for i in range(n_clusters):
        x_ = np.linspace(0,1,n_div+1)
        y_ = gene_dynamics_all_[adata.uns[gene_atlas_key]['clusters']==i].T

        data_ = []
        for j in range(y_.shape[1]):
            data_.append(
                go.Scatter(
                    x=x_,
                    y=y_[:,j],
                    mode = "lines",
                    text = adata.uns[gene_atlas_key]['texts'][adata.uns[gene_atlas_key]['clusters']==i][j],
                    name = '',
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
            plot_bgcolor='rgba(1,1,1,1)',
            paper_bgcolor='rgba(0,0,0,0)',
            xaxis=dict(
                range=[0, 1],
                # showgrid=False,
                title = 'Time',
                gridcolor='gray',
                gridwidth=1,
                griddash='dot',
                zeroline=False,
                # showticklabels=False,
                layer='below traces',
            ),
            yaxis=dict(
                # range=[0, 1],
                # showgrid=False,
                title = '<b>Cluster %s</b><br>%s' % (str(i+1),ylabel_),
                gridcolor='gray',
                gridwidth=1,
                griddash='dot',
                zeroline=False,
                # showticklabels=False,
                layer='below traces',
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
        n_genes = 10,
        threshold_min = 1,
        path_key = 'path',
        exp_key = None,
        gene_dynamics_key = 'gene_dynamics',
        n_div = 100,
        fontsize_label = 10,
        save = False,
        save_dir = None,
        save_filename = 'key_gene_dynamics',
    ):

    if gene_dynamics_key not in adata.uns.keys():
        calc_gene_dynamics(adata, source_cluster, target_clusters, path_key=path_key, exp_key=exp_key, gene_dynamics_key=gene_dynamics_key, n_div=n_div)

    idx_t_n_ = np.arange(n_div+1)[(np.linspace(0,1,n_div+1) <= time)]
    idx_t_p_ = np.arange(n_div+1)[(np.linspace(0,1,n_div+1) >= time)]

    columns_ = []
    for i in range(len(target_clusters)):
        for j in range(i+1,len(target_clusters)):
            columns_ = np.append(columns_,target_clusters[i]+'_vs_'+target_clusters[j])

    cmap_ = plt.get_cmap("tab10")
    vlines = [0,0.2,0.4,0.6,0.8,1]
    sign_dict_ = {'1':'+','-1':'-'}
    out_pd_ = pd.DataFrame(index=(np.arange(n_genes)+1),columns=pd.MultiIndex.from_product([list(columns_),[]]))
    for i in range(len(target_clusters)):
        for j in range(i+1,len(target_clusters)):
            
            name_i_ = source_cluster + '_' + target_clusters[i]
            name_j_ = source_cluster + '_' + target_clusters[j]
            gene_dynamics_ = adata.uns[gene_dynamics_key]

            max_i_,max_j_ = np.max(gene_dynamics_[name_i_],axis=0),np.max(gene_dynamics_[name_j_],axis=0)
            idx_max_ = (max_i_ > threshold_min) & (max_j_ > threshold_min)

            vol_p_ = np.sum(np.abs(gene_dynamics_[name_i_][idx_t_p_]-gene_dynamics_[name_j_][idx_t_p_]),axis=0)
            vol_n_ = np.sum(np.abs(gene_dynamics_[name_i_][idx_t_n_]-gene_dynamics_[name_j_][idx_t_n_]),axis=0)
            diff_ = vol_p_/(1e-5+vol_n_)
            diff_order_ = np.argsort(diff_[idx_max_])[::-1]

            out_pd_[(target_clusters[i]+'_vs_'+target_clusters[j],'gene')] = adata.var.index[idx_max_][diff_order_[:n_genes]]
            sign_ = [int(np.sign(np.sum(gene_dynamics_[name_i_][:,idx_max_][:,diff_order_[i_]][idx_t_p_]-gene_dynamics_[name_j_][:,idx_max_][:,diff_order_[i_]][idx_t_p_]))) for i_ in range(n_genes)]
            out_pd_[(target_clusters[i]+'_vs_'+target_clusters[j],target_clusters[i])] = [sign_dict_[str(s_)] for s_ in sign_]
            out_pd_[(target_clusters[i]+'_vs_'+target_clusters[j],target_clusters[j])] = [sign_dict_[str(-s_)] for s_ in sign_]

            vline_labels = np.append(np.append('Source (0)\n%s' % source_cluster,np.array(vlines)[1:-1]),'Target (1)')
            for i_ in range(n_genes):
                fig,ax = plt.subplots(1,1,figsize=(8,2),tight_layout=True)
                ax.plot(np.linspace(0,1,n_div+1),gene_dynamics_[name_i_][:,idx_max_][:,diff_order_[i_]],color=cmap_(i),zorder=2)
                ax.text(1,gene_dynamics_[name_i_][:,idx_max_][:,diff_order_[i_]][-1],' '+target_clusters[i],fontsize=fontsize_label,va='center',ha='left')
                ax.plot(np.linspace(0,1,n_div+1),gene_dynamics_[name_j_][:,idx_max_][:,diff_order_[i_]],color=cmap_(j),zorder=2)
                ax.text(1,gene_dynamics_[name_j_][:,idx_max_][:,diff_order_[i_]][-1],' '+target_clusters[j],fontsize=fontsize_label,va='center',ha='left')
                ax.set_title(adata.var.index[idx_max_][diff_order_[i_]])
                ax.axvline(time,color='r',zorder=1)
                ax.text(time,0.95,str(time)+' ',color='r',zorder=1,va='top',ha='right',transform=ax.transAxes)
                ax.set_xlim([0,1])
                ax.set_xticks(vlines)
                ax.set_xticklabels(vline_labels,fontsize=fontsize_label)
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                if save:
                    filename = '%s' % (save_filename) if save_dir == None else '%s/%s' % (save_dir,save_filename)
                    filename += target_clusters[i] + '_' + target_clusters[j]
                    filename += '_{}'.format(round(time, len(str(n_div)))) + '_' + '%02d_' % (i_+1) + adata.var.index[idx_max_][diff_order_[i_]]
                    fig.savefig(filename+'.png', bbox_inches='tight')
    display(out_pd_)