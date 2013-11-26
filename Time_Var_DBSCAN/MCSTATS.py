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
import scipy.linalg as la


def DBSCAN_Compute_Clusters(mcSims, eps, timeScale, min_samples ,nCorePoints = 3, numAnalyze=0, sigMethod ='isotropic', BGDensity=0, TotalTime=48,inner=1.25,outer=2.0, fileout = '',numProcs = 1, plot=False,indexing=True,metric='euclidean'):
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
        metric       : 'euclidean' or 'spherical' default is euclidean
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
    
    DBSCAN_PARTIAL = partial(__DBSCAN_THREAD,  eps=eps, min_samples=min_samples,timeScale=timeScale,nCorePoints = nCorePoints,plot=plot,indexing=indexing,metric=metric)
    if numProcs>1:
        p = pool.Pool(numProcs) # Allocate thread pool
        dbscanResults = p.map(DBSCAN_PARTIAL, mcSims[:numAnalyze]) # Call mutithreaded map.
        p.close()  # Kill pool after jobs complete.  required to free memory.
        p.join()   # wait for jobs to finish.
    else:
    #    # Serial Version.  Only use for debugging
        dbscanResults = map(DBSCAN_PARTIAL, mcSims[:numAnalyze])
    #dbscanResults = map(DBSCAN_PARTIAL, mcSims[:numAnalyze])
    
    ################################################################
    # Compute cluster properties for each cluster in each simulation
    ################################################################
    PROPS_PARTIAL = partial( __Cluster_Properties_Thread,BGDensity=BGDensity,TotalTime=TotalTime, inner=inner,outer=outer,sigMethod=sigMethod,metric=metric)
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
    """Given a set of ClusterResults, calculates the mean of the "mean Radius weighted by Members" """
    return np.mean([np.ma.average(CR.Size95X, weights=CR.Members) for CR in ClusterResults])    

def Mean_SizeT(ClusterResults):
    """Given a set of ClusterResults, calculates the mean of the "mean temporal size weighted by Members" """
    return np.mean([np.ma.average(CR.Size95T, weights=CR.Members) for CR in ClusterResults])        

def Mean_SizeSigR(ClusterResults):
    """Given a set of ClusterResults, calculates the mean of the "stdev of radius from centroid weighted by Members" """
    return np.mean([np.ma.average(CR.MedR, weights=CR.Members) for CR in ClusterResults]) 

def Mean_SizeSigT(ClusterResults):
    """Given a set of ClusterResults, calculates the mean of the "stdev temporal dist from centroid weighted by Members" """
    return np.mean([np.ma.average(CR.MedT, weights=CR.Members) for CR in ClusterResults])
        
def Mean_Members(ClusterResults):
    """Given a set of ClusterResults, calculates the mean of the "mean number of cluster members weighted by Members" """
    return np.mean([np.ma.average(CR.Members, weights=CR.Sigs) for CR in ClusterResults])   
        
def Mean_Clusters(ClusterResults,sig_cut=0.):
    """Given a set of ClusterResults, calculates the mean number of detected clusters with significance greater than sig_cut"""
    return np.mean([np.count_nonzero(CR.Sigs >= sig_cut)  for CR in ClusterResults])
    


####################################################################################################
#
# Internal Methods 
#
####################################################################################################
def __DBSCAN_THREAD(sim, eps, min_samples,timeScale,nCorePoints,plot=False,indexing=True,metric='euclidean'):    
        X = np.transpose([sim[0],sim[1],sim[2]])
        return DBSCAN.RunDBScan3D(X, eps, N_min=min_samples, TimeScale = timeScale, N_CorePoints=nCorePoints, plot=plot,indexing=indexing,metric=metric)      

def __Cluster_Properties_Thread(input,BGDensity,TotalTime, inner,outer,sigMethod,metric):
    labels,sim = input
    idx=np.where(labels!=-1)[0] # ignore the noise points
    clusters = np.unique(labels[idx]).astype(int)
#    clusters   = np.array(np.int_(np.unique(labels)[1:])) # want to ignore the -1 for noise so ignore first element
    CRLabels = np.array(labels)

    # Some beurocracy because of way numpy array typecast handles things
    arrlen = len(np.unique(labels))
    if 0 not in labels: # no clusters 
        CR = ClusterResult.ClusterResult(Labels=[], Coords=[], 
                                     CentX=[]    , CentY=[]    , CentT=[], 
                                     Sig95X=[]  , Sig95Y=[]  , Sig95T=[], 
                                     Size95X=[], Size95Y=[], Size95T=[], 
                                     MedR=[]      , MedT=[],
                                     Members=[], Sigs=[], 
                                     SigsMethod=[], NumClusters=[],PA=[])  # initialize new cluster results object
        CRNumClusters = 0 # Number of clusters found in the simulation
        return CR
    elif arrlen != 2: CRNumClusters = np.array(np.shape(clusters))[0] # Number of clusters found in the simulation
    elif arrlen == 2: 
        CRNumClusters = 1 # Number of clusters found in the simulation
        CRLabels, clusters = [np.array(labels),], [clusters,]
    
    CRCoords = [__Get_Cluster_Coords(sim, labels, cluster) for cluster in clusters] # contains coordinate triplets for each cluster in clusters.

   # Compute sizes and centroids
    if metric=='euclidean':
        CRSize95X, CRSize95Y, CRSize95T, CRPA, CRMedR, CRMedT, CRCentX,CRCentY,CRCentT,CRSig95X,CRSig95Y,CRSig95T = np.transpose([__Cluster_Size(CRCoords[cluster]) for cluster in range(len(clusters))])
    elif metric=='spherical':
        CRSize95X, CRSize95Y, CRSize95T, CRPA, CRMedR, CRMedT, CRCentX,CRCentY,CRCentT,CRSig95X,CRSig95Y,CRSig95T = np.transpose([__Cluster_Size_Spherical(CRCoords[cluster]) for cluster in range(len(clusters))])
    else: print 'Invalid metric: ' , str(metric)
    # Compute significances
    if sigMethod == 'isotropic':
        CRSigsMethod = 'isotropic'
        CRSigs  = np.array([DBSCAN.Compute_Cluster_Significance_3d_Isotropic(CRCoords[cluster], BGDensity = BGDensity, TotalTime=TotalTime) for cluster in range(len(clusters))])
    elif sigMethod =='annulus':
        CRSigsMethod = 'annulus'
        CRSigs   = np.array([DBSCAN.Compute_Cluster_Significance_3d_Annulus(CRCoords[cluster], np.transpose(sim), inner=inner, outer=outer) for cluster in range(len(clusters))])
    else: print 'Invalid significance evaluation method: ' , str(sigMethod)
    
    CRMembers = np.array([len(CRCoords[cluster]) for cluster in range(len(clusters))]) # count the number of points in each cluster.
    # Input into cluster results instance
    CR = ClusterResult.ClusterResult(Labels=np.array(CRLabels), Coords=CRCoords, 
                                     CentX=CRCentX    , CentY=CRCentY    , CentT=CRCentT, 
                                     Sig95X=CRSig95X  , Sig95Y=CRSig95Y  , Sig95T=CRSig95T, 
                                     Size95X=CRSize95X, Size95Y=CRSize95Y, Size95T=CRSize95T, 
                                     MedR=CRMedR      , MedT=CRMedT,
                                     Members=CRMembers, Sigs=CRSigs, 
                                     SigsMethod=CRSigsMethod, NumClusters=CRNumClusters,PA=CRPA)  # initialize new ClusterResults instance
    return CR
    


def __Get_Cluster_Coords(sim,labels, cluster_index):
    """Returns a set of coordinate triplets for cluster 'cluster_index' given input vectors [X (1xn),Y(1xn),T(1xn)] in sim, and a set of corresponding labels"""
    idx = np.where(labels==cluster_index)[0] # find indices of points which are in the given cluster
    return np.transpose(np.array(sim))[:][idx] # select out those points and return the transpose (which provides (x,y,t) triplets for each point 




def __Cluster_Size(cluster_coords):
    """Returns basic cluster properties, given a set of cluster coordinate triplets""" 
    X,Y,T = np.transpose(cluster_coords)
    CentX0,CentY0,CentT0 = np.mean(X),np.mean(Y), np.mean(T)
    X, Y = X-CentX0, Y-CentY0 
    
    # Singular Value Decomposition
    U,S,V = la.svd((X,Y))
    # Rotate to align x-coord to principle component
    x = U[0][0]*X + U[0][1]*Y
    y = U[1][0]*X + U[1][1]*Y
    
    # Compute weighted average and stdev in rotated frame
    weights = np.divide(1,np.sqrt(np.square(x)+np.square(y))) # weight by 1/r 
    CentX, CentY = np.average(x,weights=weights), np.average(y,weights=weights)
    SigX,SigY = np.sqrt(np.average(np.square(x-CentX), weights=weights)), np.sqrt(np.average(np.square(y-CentY), weights=weights))
    # Find Position Angle
    xref,yref = np.dot(U,[0,1])
    theta = -np.rad2deg(np.arctan2(xref,yref))
    POSANG = theta+90.
    
    CentX ,CentY = np.dot(la.inv(U),(CentX,CentY)) #Translate the updated centroid into the original frame
    CENTX,CENTY =  CentX+CentX0,CentY+CentY0          # Add this update to the old centroids
    SIG95X,SIG95Y = 2*SigX/np.sqrt(np.shape(x)[0]),2*SigY/np.sqrt(np.shape(x)[0]) 
    SIZE95X, SIZE95Y = 2*S/np.sqrt(len(X)) 
    
    r = np.sqrt(np.square(X-CentX)+np.square(Y-CentY))  # Build list of radii from cluster centroid
    SIG95T = np.std(T)/np.sqrt(np.shape(r)[0])
    dt = np.abs(T-CentT0)
    countIndexT = int(np.ceil(0.95*np.shape(dt)[0]-1))
    SIZE95T = np.sort(dt)[countIndexT]   # choose the radius at this index

    return SIZE95X, SIZE95Y,SIZE95T, POSANG, np.median(r), np.median(dt), CENTX,CENTY,CentT0,SIG95X,SIG95Y,SIG95T


def __Cluster_Size_Spherical(cluster_coords):
    """Returns basic cluster properties, given a set of cluster coordinate triplets""" 
    X,Y,T = np.transpose(cluster_coords)
    # Map to cartesian
    X, Y = np.deg2rad(X), np.deg2rad(Y)
    x = np.cos(X) * np.cos(Y)
    y = np.cos(X) * np.sin(Y)
    z = np.sin(X)

    # Compute Cartesian Centroids
    CentX0,CentY0,CentZ0,CentT0 = np.mean(x),np.mean(y), np.mean(z), np.mean(T)
    r = np.sqrt(CentX0**2 + CentY0**2 + CentZ0**2)
    
    # Rotate away Z components so we are in the x,z plane and properly oriented with galactic coordinates
    def rotation_matrix(axis,theta):
        axis = axis/np.sqrt(np.dot(axis,axis))
        a = np.cos(theta/2.)
        b,c,d = -axis*np.sin(theta/2.)
        return np.array([[a*a+b*b-c*c-d*d, 2*(b*c-a*d), 2*(b*d+a*c)],
                         [2*(b*c+a*d), a*a+c*c-b*b-d*d, 2*(c*d-a*b)],
                         [2*(b*d-a*c), 2*(c*d+a*b), a*a+d*d-b*b-c*c]])
    # Pick axes
    n = np.array([CentX0, CentY0,CentZ0])/r

    n=np.array(n)/np.sqrt(np.dot(n,n))
    nz = np.array([0.,0.,1.])
    if (n!=nz).all():axis = np.cross(nz,n) 
    else: axis = np.array([0,1,0])

    theta = np.pi/2.-np.arccos(np.dot(nz,n))

    R1 = rotation_matrix(axis,-theta)
    print np.dot(R1,n)
    theta2 = np.pi/2-np.arccos(np.dot(R1,n)[0])
    R2 = rotation_matrix(nz,theta2)
    R  = np.dot(R2,R1)
    def rotate(n):
        n = n/np.sqrt(np.dot(n,n))
        return np.dot(R,n)

    # rotate all the vectors (Y component should be zero for all)
    X,Z,Y = np.rad2deg(np.transpose([rotate(np.transpose((x,y,z))[i]) for i in range(len(x))]))
    
    #from matplotlib import pyplot as plt
    #plt.figure(0)
    #plt.scatter(X,Y)
    #plt.axis('equal')
    #plt.show()  
    
    # Convert Centroids back to lat/long in radians
    CentY0 = np.rad2deg(np.arctan2(CentY0, CentX0))
    CentX0 = np.rad2deg(np.arcsin(CentZ0/r))
     
    # Singular Value Decomposition
    U,S,V = la.svd((X,Y))
    
    # Rotate to align x-coord to principle component
    x = U[0][0]*X + U[0][1]*Y
    y = U[1][0]*X + U[1][1]*Y
    
    # Compute weighted average and stdev in rotated frame
    weights = np.divide(1,np.sqrt(np.square(x)+np.square(y))) # weight by 1/r 
    CentX, CentY = np.average(x,weights=weights), np.average(y,weights=weights)
    CentX,CentY =     CentX0, CentY0
    SigX,SigY = np.sqrt(np.average(np.square(x-CentX), weights=weights)), np.sqrt(np.average(np.square(y-CentY), weights=weights))
    
    # Find Position Angle
    xref,yref = np.dot(U,[0,1])
    theta = -np.rad2deg(np.arctan2(xref,yref))
    POSANG = theta+90.
    
   
    #CentX ,CentY = np.dot(la.inv(U),(CentX,CentY)) #Translate the updated centroid into the original frame
    #CENTX,CENTY =  CentX+CentX0,CentY+CentY0          # Add this update to the old centroids
    SIG95X,SIG95Y = 2*SigX/np.sqrt(np.shape(x)[0]),2*SigY/np.sqrt(np.shape(x)[0]) 
    SIZE95X, SIZE95Y = 2*S/np.sqrt(len(X)) 
    #r = np.sqrt(np.square(X-CENTX)+np.square(y-CENTY))  # Build list of radii from cluster centroid
    r = np.sqrt(np.square(X-CentX0)+np.square(y-CentY0))  # Build list of radii from cluster centroid
    
    SIG95T = np.std(T)/np.sqrt(np.shape(r)[0])
    dt = np.abs(T-CentT0)
    countIndexT = int(np.ceil(0.95*np.shape(dt)[0]-1))
    SIZE95T = np.sort(dt)[countIndexT]   # choose the radius at this index

    return SIZE95X, SIZE95Y,SIZE95T, POSANG, np.median(r), np.median(dt), CentX0,CentY0,CentT0,SIG95X,SIG95Y,SIG95T



                   
