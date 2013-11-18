# -*- coding: utf-8 -*-
"""
DBSCAN: Density-Based Spatial Clustering of Applications with Noise

Modified from Sklearn libraries.  
"""
import warnings
import numpy as np
from sklearn.base import BaseEstimator, ClusterMixin
from sklearn.metrics import pairwise_distances
from sklearn.utils import check_random_state
from functools import partial
from scipy import weave

def dbscan3(X, eps=0.5, min_samples=5, timeScale=1, metric='euclidean', indexing =True):
    """Perform DBSCAN clustering from vector array or distance matrix.

    Parameters
    ----------
    X: array [n_samples, n_samples]
    eps: float, optional
        The maximum distance between two samples for them to be considered
        as in the same neighborhood.
    min_samples: int, optional
        The number of samples in a neighborhood for a point to be considered
        as a core point.
    metric: string
        Compute distances in euclidean, or spherical coordinate space

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
    XX = X[:,0] # Separate spatial component
    XY = X[:,1] # Separate spatial component
    XT = X[:,2] # Separate time Component and rescale the times.
    n = np.shape(X)[0]   # Number of points
    deg2rad = np.pi/180. # Conversion
    where = np.where
    sum=np.sum
    square = np.square
    
    
    ############################################################################
    # This section loops through the rtree neighborhood and checks if points are
    # really in neighborhood.  Note R-tree returns a rectangular window so we will
    # always need to do this without modifying the rtree package.  It also allows
    # for precise computation of the central angle in spherical coordinates.  
    def refine_nhood(nhood,index,eps):
        if (metric == 'spherical'):
            # Use Haverside Formula http://en.wikipedia.org/wiki/Great-circle_distance
            x1, y1 = float(X[index][0]), float(X[index][1])
            x2 = np.asarray([X[i][0] for i in nhood])
            y2 = np.asarray([X[i][1] for i in nhood])
            n2 = len(x2)
            support = "#include <math.h>"
            epsRad = eps*deg2rad
            code = """
            py::list ret;
            for (int i=0;i<n2;i++){
                double d = 2.*asin(sqrt(pow(sin( deg2rad*.5*(y1-y2[i])),2.) + cos(deg2rad*y1)*cos(deg2rad*y2[i])*pow(sin(deg2rad*.5*(x1-x2[i])),2.)));
                if (d <= epsRad) ret.append(nhood[i]);
            }
            return_val = ret;
            """
            return np.array(weave.inline(code,['n2','x2','y2','nhood','x1','y1','epsRad','deg2rad'],support_code = support, libraries = ['m']))    

    # Initially, all samples are noise.
    labels = -np.ones(n)

    #===========================================================================
    # Select points within bounding box of width 2*epsilon centered on point
    # of interest.  This gives a quick way to only compute distances for nearest
    # neighbors.
    #===========================================================================    
    
    def epsQuery(i):
        xcut = np.logical_and(XX <= XX[i]+eps,XX >= XX[i]-eps)
        ycut = np.logical_and(XY <= XY[i]+eps,XY >= XY[i]-eps)
        tcut = np.logical_and(XT <= XT[i]+eps*float(timeScale),XT >= XT[i]-eps*float(timeScale))
        cut = np.logical_and(xcut,ycut)
        cut = np.logical_and(cut,tcut)
        return np.where(cut==True)[0]
    
    neighborhoods = [epsQuery(i) for i in range(0,n)]
        
    #===========================================================================
    # Refine the epislon neighborhoods (compute the euclidean distances)
    #===========================================================================
    if (metric == 'euclidean'):
        # Compute distances using numpy vector methods ONLY if neighborhood size 
        # is already greater than min_samples, otherwise don't bother.          
        neighborhoods = [(neighborhoods[i][where( square( XX[neighborhoods[i]] - XX[i]) + square(XY[neighborhoods[i]] - XY[i]) <= eps*eps)[0]]) if len(neighborhoods[i])>=min_samples else neighborhoods[i] for i in range(0,n)]
    #TODO: implement spherical refinement for real data
    
    
    
    
    #======================================================
    # From here the algorithm is essentially the same
    #======================================================
    # A list of all core samples found.
    core_samples = []
    # label_num is the label given to the new cluster
    label_num = 0

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





def dbscan(X, eps=0.5, min_samples=5, metric='euclidean', indexing = False):
    """Perform DBSCAN clustering from vector array or distance matrix.

    Parameters
    ----------
    X: array [n_samples, n_samples] or [n_samples, n_features]
        Array of distances between samples, or a feature array.
        The array is treated as a feature array unless the metric is given as
        'precomputed'.
    eps: float, optional
        The maximum distance between two samples for them to be considered
        as in the same neighborhood.
    min_samples: int, optional
        The number of samples in a neighborhood for a point to be considered
        as a core point.
    metric: string, or callable
        The metric to use when calculating distance between instances in a
        feature array. If metric is a string or callable, it must be one of
        the options allowed by metrics.pairwise.calculate_distance for its
        metric parameter.
        If metric is "precomputed", X is assumed to be a distance matrix and
        must be square.
    numCPU: number of cpus.  If < 0. numSystemCPU+1+numCPU are used   
    Returns
    -------
    core_samples: array [n_core_samples]
        Indices of core samples.

    labels : array [n_samples]
        Cluster labels for each point.  Noisy samples are given the label -1.

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
    import time
    start = time.time()
    X = np.asarray(X)
    n = X.shape[0]
    # If index order not given, create random order.
    D = pairwise_distances(X, metric=metric)
    
    #from scipy.spatial import distance
    #deg2rad = np.pi/180.
    #def haverside(u,v):
    #    return 2.*np.arccos(np.sqrt(  np.square(np.sin(deg2rad*.5*(u[1]-v[1])) ) + np.cos(deg2rad*u[1])*np.cos(deg2rad*v[1]) * np.square(np.sin(deg2rad*.5 *(u[0]-v[1]))) ))    
    #D = distance.squareform(distance.pdist(X, haverside))
     
    # Calculate neighborhood for all samples. This leaves the original point
    # in, which needs to be considered later (i.e. point i is the
    # neighborhood of point i. While True, its useless information)
    neighborhoods = [np.where(x <= eps)[0] for x in D]
    
    # Initially, all samples are noise.
    labels = -np.ones(n)
    # A list of all core samples found.
    core_samples = []
    # label_num is the label given to the new cluster
    label_num = 0
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
                #print noise, neighborhoods[c][noise]
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
    #print "Core Samples", len(core_samples)," Distance Comps: ", n**2
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
        The metric to use when calculating distance between instances in a
        feature array. If metric is a string or callable, it must be one of
        the options allowed by metrics.pairwise.calculate_distance for its
        metric parameter.
        If metric is "precomputed", X is assumed to be a distance matrix and
        must be square.
    random_state : numpy.RandomState, optional
        The generator used to initialize the centers. Defaults to numpy.random.

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

    def __init__(self, eps=0.5, min_samples=5, timeScale=1, metric='euclidean', indexing=None):
        self.eps = eps
        self.min_samples = min_samples
        self.metric = metric
        self.indexing = indexing
        self.timeScale = timeScale
        
    def fit(self, X, **params):
        """Perform DBSCAN clustering from vector array or distance matrix.

        Parameters
        ----------
        X: array [n_samples, n_samples] or [n_samples, n_features]
            Array of distances between samples, or a feature array.
            The array is treated as a feature array unless the metric is
            given as 'precomputed'.
        """
#        if (self.indexing == None):
#            # automatically choose based on photon count.  Use indexing automatically for high photons counts
#            if len(X)>2000:
#                self.core_sample_indices_, self.labels_ = dbscan3(X,
#                                                         **self.get_params())
#            else:
#                self.core_sample_indices_, self.labels_ = dbscan(X,
#                                                         **self.get_params())
#        if (self.indexing ==True):
#            self.core_sample_indices_, self.labels_ = dbscan3(X,
#                                                         **self.get_params())
#        elif (self.indexing == False):
#            self.core_sample_indices_, self.labels_ = dbscan(X,
#                                                         **self.get_params())

        self.core_sample_indices_, self.labels_ = dbscan3(X,**self.get_params())
        return self

    
