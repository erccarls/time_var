'''
Statistics and plotting tools for Fermi MC tools

Created on Jul 24, 2012
@author: Eric Carlson
'''

import matplotlib.pyplot as plt  #@UnresolvedImport
import cPickle as pickle
import numpy as np
import scipy.cluster as cluster
from matplotlib.backends.backend_pdf import PdfPages
import DBSCAN
import multiprocessing as mp
from multiprocessing import pool
from functools import partial
import ClusterResult


def DBSCAN_Compute_Clusters(mcSims, eps, timeScale, min_samples ,nCorePoints = 3, numAnalyze=0, sigMethod ='isotropic', BGDensity=0, TotalTime=48,inner=1.25,outer=2.0, fileout = '',numProcs = 1, plot=False):
    '''
    Main DBSCAN cluster method.  Input a list of simulation outputs and output a list of clustering properties for each simulation.
    Inputs:
        mcSims: this is the output from the runMC methods (i.e. the list of simulation outputs) (python object not filename)
        eps: DBSCAN epsilon parameter
        min_samples: min num points in epislon neighborhood for the DBSCAN algorithm.
    Optional Inputs:
        nCorePoints=3: After DBSCAN is run, there must be at least this many points for a cluster to not be thrown out.
        numAnalyze=0 : number of simulations to analyze out of list.  default is 0 which analyzes all simulations passed
        fileout=''   : if not empty string, store all the clustering info in a pickle file.
        indexing     : if None, automatically choose fastest method. True, always uses grid index, if False always computes full distance matrix and requires much more memory.
    Returns:
        dbScanResults: a tuple (labels) for each simulation
            clusterReturn: For each cluster, a list of points in that cluster
            labels: A list of points with cluster identification number.  -1 corresponds to noise.
                    NOTE: ALL BORDER POINTS ARE CONSIDERED NOISE.  See DBSCAN.py for info if you need
                    Border points.
    '''
    
    ###############################################################
    # Compute Clusters Using DBSCAN     
    ###############################################################
    # Check number to analyze
    if ((numAnalyze == 0) or (numAnalyze > len(mcSims))):
        numAnalyze =len(mcSims)
    # Define methods for mapping
    
    DBSCAN_PARTIAL = partial(__DBSCAN_THREAD,  eps=eps, min_samples=min_samples,timeScale=timeScale,nCorePoints = nCorePoints,plot=plot)
    if numProcs>1:
        p = pool.Pool(numProcs) # Allocate thread pool
        dbscanResults = p.map(DBSCAN_PARTIAL, mcSims[:numAnalyze]) # Call mutithreaded map.
        p.close()  # Kill pool after jobs complete.  required to free memory.
        p.join()   # wait for jobs to finish.
    else:
        # Serial Version.  Only use for debugging
        dbscanResults = map(DBSCAN_PARTIAL, mcSims[:numAnalyze])
    
    ################################################################
    # Compute cluster properties for each cluster in each simulation
    ################################################################
    PROPS_PARTIAL = partial( __Cluster_Properties_Thread,BGDensity=BGDensity,TotalTime=TotalTime, inner=inner,outer=outer,sigMethod=sigMethod)
    if numProcs>1:
        p = pool.Pool(numProcs) # Allocate thread pool 
        ClusterResults = p.map(PROPS_PARTIAL,zip(dbscanResults,mcSims))
        #ClusterResults = parmap(PROPS_PARTIAL,zip(dbscanResults,mcSims))
        p.close()  # Kill pool after jobs complete.  required to free memory.
        p.join()   # wait for jobs to finish.
    else:
    
        ClusterResults = map(PROPS_PARTIAL,zip(dbscanResults,mcSims))
    
    if (fileout != ''): pickle.dump(ClusterResults, open(fileout,'wb')) # Write to file if requested
    return ClusterResults
        


def Mean_Significance(ClusterResults):
    """Given a set of ClusterResults, calculates the mean of the "mean significance weighted by number of cluster members" """
    return np.mean([np.ma.average(CR.Sigs, weights=CR.Members) for CR in ClusterResults])

def Mean_Radius(ClusterResults):
    """Given a set of ClusterResults, calculates the mean of the "mean Radius weighted by significance" """
    return np.mean([np.ma.average(CR.SizeR, weights=CR.Sigs) for CR in ClusterResults])    

def Mean_SizeT(ClusterResults):
    """Given a set of ClusterResults, calculates the mean of the "mean temporal size weighted by significance" """
    return np.mean([np.ma.average(CR.SizeT, weights=CR.Sigs) for CR in ClusterResults])        
        
def Mean_Members(ClusterResults):
    """Given a set of ClusterResults, calculates the mean of the "mean number of cluster members weighted by significance" """
    return np.mean([np.ma.average(CR.Members, weights=CR.Sigs) for CR in ClusterResults])   
        
def Mean_Clusters(ClusterResults,sig_cut=0.):
    """Given a set of ClusterResults, calculates the mean number of detected clusters with significance greater than sig_cut"""
    return np.mean([np.shape(np.where(CR.Sigs>=sig_cut)[0])[0]  for CR in ClusterResults])
    


####################################################################################################
#
# Internal Methods 
#
####################################################################################################

def __DBSCAN_THREAD(sim, eps, min_samples,timeScale,nCorePoints,indexing= None,plot=False):    
        X = np.transpose([sim[0],sim[1],sim[2]])
        return DBSCAN.RunDBScan3D(X, eps, N_min=min_samples, TimeScale = timeScale, N_CorePoints=nCorePoints, plot=plot)      

def __Cluster_Properties_Thread(input,BGDensity,TotalTime, inner,outer,sigMethod):
    labels,sim = input
    
    
    clusters   = np.array(np.int_(np.unique(labels)[1:])) # want to ignore the -1 for noise so ignore first element
    CRLabels = clusters   
    # Some beurocracy because of way numpy array typecast handles things
    arrlen = np.shape(np.unique(labels))[0]
    if arrlen == 1: 
        CRNumClusters = 0 # Number of clusters found in the simulation
        return ClusterResult.ClusterResult()
    elif arrlen != 2: CRNumClusters = np.array(np.shape(clusters))[0] # Number of clusters found in the simulation
    elif arrlen == 2: 
        CRNumClusters = 1 # Number of clusters found in the simulation
        CRLabels, clusters = [clusters,], [clusters,]
    
    CRCoords = [__Get_Cluster_Coords(sim, labels, cluster) for cluster in clusters] # contains coordinate triplets for each cluster in clusters.
    # Compute significances
    if sigMethod == 'isotropic':
        CRSigsMethod = 'isotropic'
        CRSigs  = np.array([DBSCAN.Compute_Cluster_Significance_3d_Isotropic(CRCoords[cluster], BGDensity = BGDensity, TotalTime=TotalTime) for cluster in range(len(clusters))])
    elif sigMethod =='annulus':
        CRSigsMethod = 'annulus'
        CRSigs   = np.array([DBSCAN.Compute_Cluster_Significance_3d_Annulus(CRCoords[cluster], np.transpose(sim), inner=inner, outer=outer) for cluster in range(len(clusters))])
    # Compute sizes and centroids
    CRSizeR,CRSizeT,CRCentX,CRCentY,CRCentT = np.transpose([__Cluster_Size(CRCoords[cluster]) for cluster in range(len(clusters))])
    CRMembers = np.array([np.count_nonzero(labels==cluster) for cluster in clusters]) # count the number of points in each cluster.
    
    CR = ClusterResult.ClusterResult(Labels=CRLabels, Coords=CRCoords, CentX=CRCentX, CentY=CRCentY, CentT=CRCentT, SizeR=CRSizeR, SizeT=CRSizeT, Members=CRMembers, Sigs=CRSigs, SigsMethod=CRSigsMethod, NumClusters=CRNumClusters)  # initialize new cluster results object

    return CR


def __Get_Cluster_Coords(sim,labels, cluster_index):
    """Returns a set of coordinate triplets for cluster 'cluster_index' given input vectors [X (1xn),Y(1xn),T(1xn)] in sim, and a set of corresponding labels"""
    idx = np.where(labels==cluster_index)[0] # find indices of points which are in the given cluster
    return np.transpose(np.array(sim))[:][idx] # select out those points and return the transpose (which provides (x,y,t) triplets for each point 
        
def __Cluster_Size(cluster_coords):
    """Returns sizeR (95% containment), sizeT (full containment), given a set of cluster coordinate triplets""" 
    x,y,t = np.transpose(cluster_coords) # Reformat input
    centX,centY,centT = np.mean(x), np.mean(y),np.mean(t) # Compute Centroid
    r = np.sqrt(np.square(x-centX)+np.square(y-centY))  # Build list of radii from cluster centroid
    # Sort the list and choose the radius where the cumulative count is >95%
    
    countIndex = int(np.ceil(0.95*np.shape(r)[0]-1)) 
    clusterRadius = np.sort(r)[countIndex]   # choose the radius at this index
    sizeT = np.max(t)-np.min(t) 
    return (clusterRadius, sizeT, centX, centY, centT)
