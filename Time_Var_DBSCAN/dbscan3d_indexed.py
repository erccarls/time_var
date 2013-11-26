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
import multiprocessing as mp
from multiprocessing import pool
from functools import partial
import itertools
#from numba import autojit


def dbscan3(X, eps, min_samples, timeScale=1, metric='euclidean',indexing=False):
    """Perform DBSCAN clustering from vector array or distance matrix.

    Parameters
    ----------
    X: array [X, Y, T] where X,Y,T are a single coordinate vector. (lat, long, time for spherical)
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



def dbscan3_indexed(X, eps, min_samples, timeScale, metric,indexing):
#def dbscan3_indexed(X, eps, min_samples, timeScale=1, metric='euclidean',indexing=True):
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
    
    ############################################################################
    # In this section we create assign each point in the input array an index i,j
    # which locates the points on a large scale grid.  The epsilon query will then
    # only compute distances to points in the neighboring grid points.
    # For a Euclidean metric, we want a grid with elements of width ~epsilon.
    # This will have to be refined for spherical metrics since the poles produce
    # anomalous behavior.
    startX, stopX = np.min(XX), np.max(XX) # widths
    startY, stopY = np.min(XY), np.max(XY)
    eps2=eps*2. # Larger grids is a speed advantage in most cases (i.e. takes longer to index with finer grid)
    # We choose rectangles of 6x6 epsilons and return all grid elements when query point is within 10 degrees of a pole.
    GridSizeX, GridSizeY = int(np.ceil((stopX-startX)/eps2)), int((np.ceil((stopY-startY)/eps2)))
    Xidx, Yidx = np.floor(np.divide((XX-startX),eps2)).astype(int), np.floor(np.divide((XY-startY),eps2)).astype(int) # Grid indices for all points.
    # Iterate through each grid element and add indices
    Grid = np.empty(shape=(GridSizeX,GridSizeY),dtype=object)
    for i in range(GridSizeX):
        Xtrue = np.where((Xidx==i))[0] # indices where x condition is met
        for j in range(GridSizeY):
            #find indicies 
            Ytrue = np.where((Yidx[Xtrue]==j))[0]
            Grid[i][j] = Xtrue[Ytrue]        
    #==============================================================================

    
    #===========================================================================
    # Refine the epislon neighborhoods (compute the euclidean distances)
    #===========================================================================
    if (metric == 'euclidean'):
        EPS_PARTIAL = partial(__epsQueryThread,  Xidx=Xidx,Yidx=Yidx,GridSizeX=GridSizeX,GridSizeY=GridSizeY,Grid=Grid,XT=XT,XX=XX,XY=XY,timeScale=timeScale,eps=eps,min_samples=min_samples)    
        neighborhoods = map(EPS_PARTIAL,range(0,n)) # Call mutithreaded map.
        #p = pool.Pool(mp.cpu_count()) # Allocate thread pool
        #neighborhoods = p.map(EPS_PARTIAL,range(0,n)) # Call mutithreaded map.
        #p.close()  # Kill pool after jobs complete.  required to free memory.
        #p.join()   # wait for jobs to finish.

    elif (metric=='spherical'):
        #=========================================================================================
        # First query the grid
        # find the grid point boundaries within 12 deg of pole, return all longitudes and all latitudes above/below
        high_grid_lat = np.floor((84.-startX)/eps2)  # Grid queries for elements above this latitude should return all longitudes
        low_grid_lat  = np.ceil((-84.-startX)/eps2) # Grid queries for elements below this latitude should return all longitudes
        # ensure that these are within the grid bounds.
        if low_grid_lat<0: low_grid_lat=0
        if high_grid_lat >= GridSizeX: high_grid_lat=GridSizeX-1
        #def get_grid_points(k):  
        def __epsilonQuerySpherical(k):  
            if indexing == True:
                i,j = Xidx[k],Yidx[k]
                il,ih = i-1, i+2 # select neighboring grid indices.
                if (XX[k]<85 and XX[k]>-85):
                    jl,jh = int(j-1./np.sin(np.abs(np.deg2rad(90-XX[k])))), int(j+2./np.abs(np.sin(np.deg2rad(90-XX[k])))) # np.sin(np.deg2rad(90-84))
                # if within 10 degrees of either pole, return all points above or below
                if il<=low_grid_lat:
                    jl,jh  = 0,GridSizeY # select all longitudes
                    il,ih  = 0,low_grid_lat+1
                if ih>=high_grid_lat: 
                    jl,jh  = 0,GridSizeY # select all longitudes
                    il,ih  = high_grid_lat-1, GridSizeX
                idx = []
                # if we span the line of 0 longitude, we need to break into 2 chunks.
                if (jl<0 or jh > GridSizeY):
                    
                    idx = idx + [item for sublist in [item for sublist2 in Grid[il:ih,int(-2-2):] for item in sublist2] for item in sublist]
                    idx = np.array(idx + [item for sublist in [item for sublist2 in Grid[il:ih,0:2+int(2.)] for item in sublist2] for item in sublist])
                else:
                    idx = np.array([item for sublist in [item for sublist2 in Grid[il:ih,jl:jh] for item in sublist2] for item in sublist])
            
            if indexing==False: idx = np.array(range(0,n)).astype(int) # select all points
            
            #Compute real arc lengths for these points.
            idx = np.append(idx,k).astype(int)
            j = -1 # index of original point in reduced list
            x = np.deg2rad(XX[idx])
            y = np.deg2rad(XY[idx])
            dPhi = x-x[j] # lat 
            dLam = y-y[j] # lon
            # Distances using Vincenty's formula for arc length on a great circle.
            d = np.arctan2(sqrt( square(cos(x)*sin(dLam) ) + square(cos(x[j])*sin(x)-sin(x[j])*cos(x)*cos(dLam)) ) , sin(x[j])*sin(x)+cos(x[j])*cos(x)*cos(dLam) )
            # Find where within time constraints
            tcut = np.logical_and(XT[idx] <= XT[k]+eps*float(timeScale),XT[idx] >= XT[k]-eps*float(timeScale))
            rcut = d<np.deg2rad(eps)
            return idx[np.where(np.logical_and(rcut, tcut)==True)[0]] # This now contains indices of points in the eps neighborhood 
        #=========================================================================================
        neighborhoods = [ __epsilonQuerySpherical(k) for k in range(0,n)]
        
    # Initially, all samples are noise.
    labels = -np.ones(n)
    #======================================================
    # From here the algorithm is essentially the same as sklearn
    #======================================================
    core_samples = [] # A list of all core samples found.
    label_num = 0 # label_num is the label given to the new clust

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
    return core_samples, labels



def __epsQueryThread(k,Xidx,Yidx,GridSizeX,GridSizeY,Grid,XX,XY,XT,timeScale,eps,min_samples):
    """ Returns the epsilon neighborhood of a point for euclidean metric"""  
    i,j = Xidx[k],Yidx[k]
    il,ih = i-1, i+2
    jl,jh = j-1, j+2
    if jl<0  : jl=0
    if il<0  : il=0
    if ih>=GridSizeX: ih=-1
    if jh>=GridSizeY: jh=-1
    idx = np.array([item for sublist in [item for sublist2 in Grid[il:ih,jl:jh] for item in sublist2] for item in sublist])
    if len(idx) !=0:
        tcut = np.logical_and(XT[idx] <= (XT[k]+eps*timeScale),XT[idx] >= (XT[k]-eps*float(timeScale)))
        tcut = np.where(tcut==True)[0]
        if len(tcut)!=0:
            try:
                idx = idx[tcut] #original indices meeting tcut  This is the rough eps neighborhood
            except:
                print 'Error with idx', idx                    
            # Compute actual distances using numpy vector methods                
            return idx[np.where( np.square( XX[idx] - XX[k]) + np.square(XY[idx] - XY[k]) <= eps*eps)[0]]
        else: return np.array([])
    else: return np.array([])
    
    

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
        self.core_sample_indices_, self.labels_ = dbscan3_indexed(X,**self.get_params())
        return self

