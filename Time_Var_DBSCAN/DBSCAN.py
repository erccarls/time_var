#===============================================================================
# DBSCAN.py: Methods for running clustering algorithm and computing some cluster
#  statistics such as significance, background counts.
# Author: Eric Carlson
#===============================================================================
import numpy as np
#import dbscan3d_indexed_cython as dbscan3d_indexed
import dbscan3d_indexed
import math 

def RunDBScan3D(X,eps,N_min,TimeScale,N_CorePoints =3 , plot=False,indexing=True, metric='euclidean'):
    """
    Runs DBSCAN3D on photon list, X, of coordinate triplets using parameters eps and n_samples defining the search radius and the minimum number of events.
    
    Inputs:
        X:         a list of coordinate pairs (n x 3) for each photon
        eps:       DBSCAN epsilon parameter
        N_min:     DBSCAN core point threshold.  Must have AT LEAST n_samples points per eps-neighborhood to be core point
        TimeScale: This number multiplies the epsilon along the time dimension.
    Optional inputs 
        N_CorePoints: Number of core points required to form a cluster.  This is different than the N_min parameter.
        plot:         For debugging, this allows a a 3-d visualization of the clusters
        
    Returns: 
        Labels: A list of cluster labels corresponding to X.  -1 Implies noise.
    """
    #===========================================================================
    # Compute DBSCAN
    #===========================================================================
    db = dbscan3d_indexed.DBSCAN(eps, min_samples=N_min, timeScale=TimeScale,indexing=indexing,metric=metric).fit(X)
    core_samples = db.core_sample_indices_ # Select only core points.
    labels = db.labels_                    # Assign Cluster Labels
    # Get the cluster labels for each core point
    coreLabels = [labels[i] for i in core_samples]
    # Count clusters with > nCore, core points
    validClusters = [i if coreLabels.count(i) >= N_CorePoints else None for i in set(coreLabels)]
    # relabel points that are not in valid clusters as noise.  If you want border points, comment out this line
    labels = np.array([label if label in validClusters else -1 for label in labels])
    
    #===========================================================================
    # # Plot result
    #===========================================================================
    if (plot == True):   
        import pylab as pl
        from itertools import cycle
        from mpl_toolkits.mplot3d import Axes3D
        pl.close('all')
        fig = pl.figure(1)
        pl.clf()
        fig = pl.gcf()
        ax = fig.gca(projection='3d')
        # Black removed and is used for noise instead.
        colors = cycle('bgrcmybgrcmybgrcmybgrcmy')
        
        for k, col in zip(set(labels), colors):
            if k == -1:
                # Black used for noise.
                col = 'k'
                markersize = 6
            class_members = [index[0] for index in np.argwhere(labels == k)]

            #cluster_core_samples = [index for index in core_samples
            #                        if labels[index] == k]
            for index in class_members:
                x = X[index]
                if index in core_samples and k != -1:
                    markersize = 6
                    ax.scatter(x[0],x[1],x[2], color=col,s=markersize)
                else:
                    markersize = 2
                    ax.scatter(x[0], x[1],x[2], c=col,s=2)
        pl.axis('equal')
        pl.xlabel(r'$l$ [$^\circ$]')
        pl.ylabel(r'$b$ [$^\circ$]')
        pl.show()

    return labels
    


def Compute_Cluster_Significance_3d_Isotropic(X, BGDensity, TotalTime):
    """
    Takes input list of coordinate triplets (in angular scale) and computes the cluster significance based on a background model.
    
    Inputs:
        -X is a tuple containing a coordinate pair for each point in a cluster.  
        -BGDensity is the 2-d number of events per square degree integrated over the background.  Time is handled seperately
        -TotalTime is the full simulation time in units of the time dimension.  Usually months
        
    returns significance
    """
    # Default to zero significance
    if (len(X)==1):return 0
    # Otherwise.......
    x,y,t = np.transpose(X) # Reformat input
    centX,centY,centT = np.mean(x), np.mean(y),np.mean(t) # Compute Centroid
    r = np.sqrt(  np.square(x-centX) + np.square(y-centY) ) # Build list of radii from cluster centroid
    countIndex = int(math.ceil(0.95*np.shape(r)[0]-1)) # Sort the list and choose the radius where the cumulative count is >95% 
    clusterRadius = np.sort(r)[countIndex]   # choose the radius at this index 
    N_bg = np.pi * clusterRadius**2. * BGDensity # Use isotropic density to compute the background expectation
    dT = (np.max(t)-np.min(t))/float(TotalTime) # Rescale according to the total time.
    ######################################################
    # Evaluate significance as defined by Li & Ma (1983).  N_cl corresponds to N_on, N_bg corresponds to N_off
    if dT > .01: # This would be a 15 day period     
        N_bg,N_cl = N_bg*dT, countIndex
        if N_cl/(N_cl+N_bg)<1e-20 or N_bg/(N_cl+N_bg)<1e-20:
            return 0
        S2 = 2.0*(N_cl*math.log(2.0*N_cl/(N_cl+N_bg)) + N_bg*math.log(2.0*N_bg/(N_cl+N_bg)))
        if S2>0.0:
            return math.sqrt(S2)   
        else:
            return 0
    else: return 0
    
    
    
def Compute_Cluster_Significance_3d_Annulus(X_cluster,X_all,inner=1.25, outer=2.0):
    """
    Takes input list of coordinate triplets for the cluster and for the entire simulation and computes the cluster size.
    Next, the background level is computed by drawing an annulus centered on the the cluster with inner and outer radii 
    specified as a fraction of the initial radius.  Then the significance is calculated.  The cluster is cylindrical
    with the axis aligned temproally.  Similarly, the background annulus is taken over a cylindrical shell and is 
    computed over the range of times in X_all (thus if the background is time varying, this will average that).  
    
    Inputs:
        -X_cluster: A tuple containing a coordinate triplet (x,y,z) for each point in a cluster.  
        -X_all:     A tuple containing coordinate triplets for background events, typically just all events.
        -inner:    The inner radius of the background annulus in fraction of cluster radius.
        -outer:    The outer radius of the background annulus in fraction of cluster radius.
        
    return:
        Cluster significance from Li & Ma (1985)
    """
    # Default to zero significance
    if (len(X_cluster)==1):return 0
    # Otherwise.......
    x,y,t = np.transpose(X_cluster) # Reformat input
    x_all,y_all,t_all = X_all # Reformat input
    centX,centY,centT = np.mean(x), np.mean(y),np.mean(t) # Compute Centroid
    minT,maxT = np.min(t), np.max(t) # find the window of times for the cluster
    r = np.sqrt(np.square(x-centX)+np.square(y-centY)) # Build list of radii from cluster centroid
    countIndex = int(math.ceil(0.95*np.shape(r)[0]-1)) # Sort the list and choose the radius where the cumulative count is >95% 
    clusterRadius = np.sort(r)[countIndex]             # choose the radius at this index 
    min_T_all, max_T_all = np.min(t_all),np.max(t_all)
    ################################################################
    # Estimate the background count
    AnnulusVolume = np.pi* ((outer*clusterRadius)**2 -(inner*clusterRadius)**2)*(max_T_all-min_T_all)
    r_all = np.sqrt(np.square(x_all-centX)+np.square(y_all-centY)) # compute all points radius from the centroid. 
    r_cut = np.logical_and(r_all>clusterRadius*inner,r_all<clusterRadius*outer)# Count the number of points within the annulus and find BG density
    idx =  np.where(r_cut==True)[0] # pick points in the annulus
    BGDensity = np.shape(idx)[0]/AnnulusVolume # Density = counts / annulus volume 
    ######################################################
    # Evaluate significance as defined by Li & Ma (1983).  N_cl corresponds to N_on, N_bg corresponds to N_off
    N_bg = np.pi * clusterRadius**2. * (maxT-minT)* BGDensity # BG count = cluster volume*bgdensity
    N_cl = countIndex # Number of counts in cluster.
    # Ensure log args are greater than 0.
    if N_cl/(N_cl+N_bg) <= 0 or N_bg/(N_cl+N_bg) <= 0: return 0 
    S2 = 2.0*(N_cl*math.log(2.0*N_cl/(N_cl+N_bg)) + N_bg*math.log(2.0*N_bg/(N_cl+N_bg)))
    if S2>0.:
        return math.sqrt(S2)   
    else:
        return 0.
 
                    
    
    

