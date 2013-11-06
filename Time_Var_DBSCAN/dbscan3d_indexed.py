# -*- coding: utf-8 -*-
"""
DBSCAN: Density-Based Spatial Clustering of Applications with Noise

Modified from Sklearn libraries.
Modified Author: Eric Carlson 
Author email:    erccarls@ucsc.edu
"""
import numpy as np
from sklearn.base import BaseEstimator, ClusterMixin
from sklearn.metrics import pairwise_distances

def dbscan3(X, eps, min_samples, timeScale=1, metric='euclidean',indexing=False):
    """Perform DBSCAN clustering from vector array or distance matrix.

    Parameters
    ----------
    X: array [X, Y, T] where X,Y,T are a single coordinate vector.
    eps: float
        The maximum distance between two samples for them to be considered
        as in the same neighborhood.
    min_samples: int
        The number of samples in a neighborhood for a point to be considered
        as a core point.
    metric: string
        Compute distances in 'euclidean', or 'spherical' coordinate space

    Returns
    -------
    core_samples: array [n_core_samples]
        Indices of core samples.

    labels : array [n_samples]
        Cluster labels for each point.  Noisy samples are given the label -1.

    References
    ----------
    Ester, M., H. P. Kriegel, J. Sander, and X. Xu, “A Density-Based
    Algorithm for Discovering Clusters in Large Spatial Databases with Noise”.
    In: Proceedings of the 2nd International Conference on Knowledge Discovery
    and Data Mining, Portland, OR, AAAI Press, pp. 226–231. 1996
    """
    
    X = np.asarray(X,dtype = np.float32)    # convert to numpy array
    XX,XY,XT = X[:,0],X[:,1],X[:,2] # Separate spatial component
    n = np.shape(X)[0]   # Number of points
    where = np.where     # Defined for quicker calls
    square = np.square   # Defined for quicker calls
    sin = np.sin
    cos = np.cos
    arctan = np.arctan
    sqrt = np.sqrt
    
    # Initially, all samples are noise.
    labels = -np.ones(n)
    neighborhoods = [] 

    #===========================================================================
    # Refine the epislon neighborhoods (compute the euclidean distances)
    #===========================================================================
    if (metric == 'euclidean'):
    #===========================================================================
    # Select points within bounding box of width 2*epsilon centered on point
    # of interest.
    #===========================================================================    
        def epsQuery(i):
            xcut = np.logical_and(XX <= XX[i]+eps,XX >= XX[i]-eps)
            ycut = np.logical_and(XY <= XY[i]+eps,XY >= XY[i]-eps)
            tcut = np.logical_and(XT <= (XT[i]+eps*float(timeScale)),XT >= (XT[i]-eps*float(timeScale)))
            cut = np.logical_and(xcut,ycut)
            cut = np.logical_and(cut,tcut)
            return where(cut==True)[0]
        # Run a crude epsQuery for all points
        neighborhoods = [epsQuery(i) for i in range(0,n)]
        # Compute distances using numpy vector methods          
        neighborhoods = [(neighborhoods[i][where( square( XX[neighborhoods[i]] - XX[i]) + square(XY[neighborhoods[i]] - XY[i]) <= eps*eps)[0]]) if len(neighborhoods[i])>=min_samples else neighborhoods[i] for i in range(0,n)]
    #TODO: implement spherical refinement for real data
    elif (metric=='spherical'):
        XX = np.deg2rad(XX)
        XY = np.deg2rad(XY)
        tcut = np.logical_and(XT <= XT[i]+eps*float(timeScale),XT >= XT[i]-eps*float(timeScale))
        def Vincenty_Form(XX,XY,i):
            #XX are lats XY is longitudes, i is the point of interest
            dPhi = XX-XX[i]
            dLam = XY-XY[i]
            return arctan(np.divide(sqrt( square( np.multiply(cos(XX)),sin(dPhi)) + square(cos(XX[i])*sin(XX)-np.multiply(sin(XX[i])*cos(XX),cos(dLam))) ) , (sin(XX[i])*sin(XX)+cos(XX[i])*np.multiply(cos(XX), cos(dPhi)))))
        # Compute arc length, and also cut on time
        neighborhoods = [np.logical_and(where(Vincenty_Form(XX,XY,i) <= np.deg2rad(eps))[0], tcut==True) for i in range(0,n)]
        

    #======================================================
    # From here the algorithm is essentially the same as sklearn
    #======================================================
    core_samples = [] # A list of all core samples found.
    label_num = 0 # label_num is the label given to the new cluster

    # Look at all samples and determine if they are core.
    # If they are then build a new cluster from them.
    for index in range(0,n):
        if labels[index] != -1 or len(neighborhoods[index]) < min_samples:
            # This point is already classified, or not enough for a core point.
            continue
        core_samples.append(index)

        labels[index] = label_num
        # candidates for new core samples in the cluster.
        candidates = [index]
        while len(candidates) > 0:
            new_candidates = []
            # A candidate is a core point in the current cluster that has
            # not yet been used to expand the current cluster.
            for c in candidates:
                noise = np.where(labels[neighborhoods[c]] == -1)[0]
                noise = neighborhoods[c][noise]
                labels[noise] = label_num
                for neighbor in noise:
                    # check if its a core point as well
                    if len(neighborhoods[neighbor]) >= min_samples:
                        # is new core point
                        new_candidates.append(neighbor)
                        core_samples.append(neighbor)
            # Update candidates for next round of cluster expansion.
            candidates = new_candidates
        # Current cluster finished.
        # Next core point found will start a new cluster.
        label_num += 1
    #print "Core Samples", len(core_samples), " Distance Comps: ", ndist
    return core_samples, labels



def dbscan3_indexed(X, eps, min_samples, timeScale=1, metric='euclidean',indexing=True):
    """Perform DBSCAN clustering from vector array or distance matrix.

    Parameters
    ----------
    X: array [X, Y, T] where X,Y,T are a single coordinate vector.
    eps: float
        The maximum distance between two samples for them to be considered
        as in the same neighborhood.
    min_samples: int
        The number of samples in a neighborhood for a point to be considered
        as a core point.
    metric: string
        Compute distances in 'euclidean', or 'spherical' coordinate space

    Returns
    -------
    core_samples: array [n_core_samples]
        Indices of core samples.

    labels : array [n_samples]
        Cluster labels for each point.  Noisy samples are given the label -1.

    References
    ----------
    Ester, M., H. P. Kriegel, J. Sander, and X. Xu, “A Density-Based
    Algorithm for Discovering Clusters in Large Spatial Databases with Noise”.
    In: Proceedings of the 2nd International Conference on Knowledge Discovery
    and Data Mining, Portland, OR, AAAI Press, pp. 226–231. 1996
    """
    
    X = np.asarray(X,dtype = np.float32)    # convert to numpy array
    XX,XY,XT = X[:,0],X[:,1],X[:,2] # Separate spatial component
    n = np.shape(X)[0]   # Number of points
    where = np.where     # Defined for quicker calls
    square = np.square   # Defined for quicker calls
    sin = np.sin
    cos = np.cos
    arctan = np.arctan
    sqrt = np.sqrt

    #===========================================================================
    # Refine the epislon neighborhoods (compute the euclidean distances)
    #===========================================================================
    if (metric == 'euclidean'):
        ############################################################################
        # In this section we create assign each point in the input array an index i,j
        # which locates the points on a large scale grid.  The epsilon query will then
        # only compute distances to points in the neighboring grid points.
        # For a Euclidean metric, we want a grid with elements of width ~epsilon.
        # This will have to be refined for spherical metrics since the poles produce
        # anomalous behavior.
        startX, stopX = np.min(XX), np.max(XX) # widths
        startY, stopY = np.min(XY), np.max(XY)
        GridSizeX, GridSizeY = int(np.ceil((stopX-startX)/eps)), int((np.ceil((stopY-startY)/eps)))
        
        Xidx, Yidx = np.floor(np.divide((XX-startX),eps)).astype(int), np.floor(np.divide((XY-startY),eps)).astype(int) # Grid indices for all points.
        # Iterate through each grid element and add indices 
        Grid = np.empty(shape=(GridSizeX,GridSizeY),dtype=object)
        for i in range(GridSizeX):
            Xtrue = np.where((Xidx==i))[0] # indices where x condition is met
            for j in range(GridSizeY):
                #find indicies 
                Ytrue = np.where((Yidx[Xtrue]==j))[0]
                Grid[i][j] = Xtrue[Ytrue]            
    
        def epsQuery(k):
            """ Returns the indicies of all points within a crude epsilon """  
            i,j = Xidx[k],Yidx[k]
            il,ih = i-1, i+2
            jl,jh = j-1, j+2
            if jl<0  : jl=0
            if il<0  : il=0
            if ih>=GridSizeX: ih=-1
            if jh>=GridSizeY: jh=-1
            idx = np.array([item for sublist in [item for sublist in Grid[il:ih,jl:jh] for item in sublist] for item in sublist])
            tcut = np.logical_and(XT[idx] <= (XT[k]+eps*float(timeScale)),XT[idx] >= (XT[k]-eps*float(timeScale)))
            idx = idx[np.where(tcut)[0]]
            return idx
        
        neighborhoods = [epsQuery(i) for i in range(0,n)]
        # Compute distances using numpy vector methods          
        neighborhoods = [(neighborhoods[i][where( square( XX[neighborhoods[i]] - XX[i]) + square(XY[neighborhoods[i]] - XY[i]) <= eps*eps)[0]]) if len(neighborhoods[i])>=min_samples else neighborhoods[i] for i in range(0,n)]
    #TODO: implement spherical refinement for real data
    elif (metric=='spherical'):
        XX = np.deg2rad(XX)
        XY = np.deg2rad(XY)
        tcut = np.logical_and(XT <= XT[i]+eps*float(timeScale),XT >= XT[i]-eps*float(timeScale))
        def Vincenty_Form(XX,XY,i):
            #XX are lats XY is longitudes, i is the point of interest
            dPhi = XX-XX[i]
            dLam = XY-XY[i]
            return arctan(np.divide(sqrt( square( np.multiply(cos(XX)),sin(dPhi)) + square(cos(XX[i])*sin(XX)-np.multiply(sin(XX[i])*cos(XX),cos(dLam))) ) , (sin(XX[i])*sin(XX)+cos(XX[i])*np.multiply(cos(XX), cos(dPhi)))))
        # Compute arc length, and also cut on time
        neighborhoods = [np.logical_and(where(Vincenty_Form(XX,XY,i) <= np.deg2rad(eps))[0], tcut==True) for i in range(0,n)]
        
    # Initially, all samples are noise.
    labels = -np.ones(n)
    #======================================================
    # From here the algorithm is essentially the same as sklearn
    #======================================================
    core_samples = [] # A list of all core samples found.
    label_num = 0 # label_num is the label given to the new cluster

    # Look at all samples and determine if they are core.
    # If they are then build a new cluster from them.
    for index in range(0,n):
        if labels[index] != -1 or len(neighborhoods[index]) < min_samples:
            # This point is already classified, or not enough for a core point.
            continue
        core_samples.append(index)

        labels[index] = label_num
        # candidates for new core samples in the cluster.
        candidates = [index]
        while len(candidates) > 0:
            new_candidates = []
            # A candidate is a core point in the current cluster that has
            # not yet been used to expand the current cluster.
            for c in candidates:
                noise = np.where(labels[neighborhoods[c]] == -1)[0]
                noise = neighborhoods[c][noise]
                labels[noise] = label_num
                for neighbor in noise:
                    # check if its a core point as well
                    if len(neighborhoods[neighbor]) >= min_samples:
                        # is new core point
                        new_candidates.append(neighbor)
                        core_samples.append(neighbor)
            # Update candidates for next round of cluster expansion.
            candidates = new_candidates
        # Current cluster finished.
        # Next core point found will start a new cluster.
        label_num += 1
    #print "Core Samples", len(core_samples), " Distance Comps: ", ndist
    return core_samples, labels

class DBSCAN(BaseEstimator, ClusterMixin):
#class DBSCAN(BaseEstimator):
    """Perform DBSCAN clustering from vector array or distance matrix.

    DBSCAN - Density-Based Spatial Clustering of Applications with Noise.
    Finds core samples of high density and expands clusters from them.
    Good for data which contains clusters of similar density.

    Parameters
    ----------
    eps : float, optional
        The maximum distance between two samples for them to be considered
        as in the same neighborhood.
    min_samples : int, optional
        The number of samples in a neighborhood for a point to be considered
        as a core point.
    metric : string, or callable
        'euclidean' or 'spherical', where 'spherical is in degrees.

    Attributes
    ----------
    `core_sample_indices_` : array, shape = [n_core_samples]
        Indices of core samples.

    `components_` : array, shape = [n_core_samples, n_features]
        Copy of each core sample found by training.

    `labels_` : array, shape = [n_samples]
        Cluster labels for each point in the dataset given to fit().
        Noisy samples are given the label -1.

    Notes
    -----
    See examples/plot_dbscan.py for an example.

    References
    ----------
    Ester, M., H. P. Kriegel, J. Sander, and X. Xu, “A Density-Based
    Algorithm for Discovering Clusters in Large Spatial Databases with Noise”.
    In: Proceedings of the 2nd International Conference on Knowledge Discovery
    and Data Mining, Portland, OR, AAAI Press, pp. 226–231. 1996
    """

    def __init__(self, eps=0.5, min_samples=5, timeScale=1, metric='euclidean',indexing = True):
        self.eps = eps
        self.min_samples = min_samples
        self.metric = metric
        self.timeScale = timeScale
        self.indexing = indexing
        
    def fit(self, X, **params):
        """Perform DBSCAN clustering from vector array or distance matrix.

        Parameters
        ----------
        X: array [n_samples X (or lat), n_samples Y (or long),n_samples T]
        """
        if self.indexing== True:
            self.core_sample_indices_, self.labels_ = dbscan3_indexed(X,**self.get_params())
        else:
            self.core_sample_indices_, self.labels_ = dbscan3(X,**self.get_params())
        return self

