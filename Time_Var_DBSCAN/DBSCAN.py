#===============================================================================
# DBSCAN.py: Methods for running clustering algorithm and computing some cluster
#  statistics such as significance, background counts, and clustering scale.
# Author: Eric Carlson
# Updated: 03-01-2013
#===============================================================================
import numpy as np
from scipy.spatial import distance
from sklearn.cluster import DBSCAN
import dbscan_indexed
from sklearn import metrics
import pickle, math
from scipy import weave
from scipy.weave import converters
import gc 
from mpl_toolkits.mplot3d import Axes3D

def RunDBScan(X,eps,n_samples,timeScale,nCorePoints =3 ,indexing = None, plot=False):
    """
    Runs DBScan on Number-observations-length vector X of coordinate pairs using parameters eps and n_samples defining the search radius and the minimum number of events.
    
    Inputs:
        X:         a list of coordinate pairs (n x 2) for each photon
        eps:       DBSCAN epsilon parameter
        n_samples: DBSCAN core point threshold.  Must have AT LEAST n_samples points per eps-neighborhood to be core point
    Optional inputs 
        nCorePoints: Number of core points required to form a cluster
        indexing:    None: Automatically choose based on number of events
                     True: Grid based indexing (Generally faster and *MUCH* more memory efficient) 
                     False: No indexing.  Computes entire distance matrix.  Faster for < 1000 photons
    Returns: 
    (clusterReturn,labels): a tuple of lists of coordinate pairs for points in each cluster found (core points only, not border points, and labels for each point)
    """
    #===========================================================================
    # Compute DBSCAN
    #===========================================================================
    db = dbscan_indexed.DBSCAN(eps, min_samples=n_samples, timeScale=timeScale, indexing = indexing).fit(X)
    
    core_samples = db.core_sample_indices_ # Select only core points.
    labels = db.labels_                    # Assign Cluster Labels
    
    # Get the cluster labels for each core point
    coreLabels = [labels[i] for i in core_samples]
        
    # Count clusters with > nCore, core points
    validClusters = [] 
    [validClusters.append(i) if coreLabels.count(i) >= nCorePoints else None for i in set(coreLabels)]
    
    # relabel points that are not in valid clusters as noise.  If you want border points, comment out this line
    labels = np.array([label if label in validClusters else -1 for label in labels])
    # For each cluster build a list of the core sample coordinates
    X = np.asanyarray(X)
    clusterReturn = [X[np.where((labels == cluster))[0]] for cluster in validClusters]
    
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

            cluster_core_samples = [index for index in core_samples
                                    if labels[index] == k]
            
            
            for index in class_members:
                x = X[index]
                if index in core_samples and k != -1:
                    markersize = 6
                    
                    ax.scatter(x[0],x[1],x[2], color=col,s=markersize)
                    
                    #fig.gca().add_artist(pl.Circle((x[0],x[1]),eps,fc = 'none',ec = col))
                    # Plot sphere
#                    
#                    u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
#                    x1=np.cos(u)*np.sin(v)+x
#                    y1=np.sin(u)*np.sin(v)+y
#                    z1=np.cos(v)+z
#                    ax.plot_wireframe(x1, y1, z1, color="r")
                else:
                    markersize = 2
                    ax.scatter(x[0], x[1],x[2], c=col,s=2)
        pl.axis('equal')
        pl.xlabel(r'$l$ [$^\circ$]')
        pl.ylabel(r'$b$ [$^\circ$]')
        pl.show()

    return (clusterReturn, labels)
    


#===============================================================================
# Internal: Integrates the background template
#===============================================================================
def Evaluate_BG_Contribution(x,y,radius, BGTemplate, numBGEvents, flatLevel = 0): 
    """
    # There is an unresolved bug with this code.  DO NOT USE IN CURRENT FORM
    Integrate the background template and return the expected event count.
    
    Inputs:
     -x,y are the centroid of the cluster 
     -radius is the radius of integration in pixels
     -BGTemplate is the pickled background template used.
     -numBGEvents is the total number of expected background events for the entire angular region being considered. (.75 times)
    
    Returns:
    -count: expected number of background events in region.
    """
    #===========================================================================
    # There is an unresolved bug with this code.  DO NOT USE IN CURRENT FORM 
    #===========================================================================
    # Rescale the BG template so that the integral directly gives the event count.
    BGTemplate = np.array(BGTemplate)/float(np.sum(BGTemplate))*(1.0-flatLevel)
    BGTemplate += flatLevel/np.shape(BGTemplate)[0]**2.0  # Add flat Backgronud
    BGTemplate = float(numBGEvents)*BGTemplate
    
    # Specify data types for weave
    size = len(BGTemplate[0])
    radius = int(round(radius))
    x,y = float(x),float(y)
    start = int(-radius-1)
    

    # Integrate annulus
    code = """
        double ret = 0.;
        for (int i= start; i<-start ; i++){
            for (int j= start; j<-start ; j++){
                if ((i*i+j*j <= radius*radius) && ((0<=(i+x)<size) && (0<=(j+y)<size))){
                    ret += BGTemplate((int)(j+y), (int) (i+x));
                }
            }
        }
        return_val = ret;
    """
    return float(weave.inline(code,['radius','BGTemplate','size','x','y','start'], compiler='gcc', type_converters = converters.blitz)) 
            
    
def Compute_Cluster_Significance_Isotropic(X, BGTemplate,BGDensity, totalPhotons,outputSize=300, angularSize = 10.0,flatLevel = 0):
    """
    Takes input list of coordinate pairs (in angular scale) and computes the cluster significance based on a background model.
    
    Inputs:
        -X is a tuple containing a coordinate pair for each point in a cluster.  
        -BGTemplate is the pickled background array used.
        -totalPhotons is the total number of all photons.  This is used to estimate the background count.
        
        -flatLevel: What fraction of background is isotropic? 1 for completely isotropic 
        -BG: Average fraction of total photons that are background.
    returns significance
    """
    # Default to zero significance
    if (len(X)==1):return 0
    
    # Otherwise.......
    x,y = np.transpose(X) # Reformat input
    numBGEvents = BGDensity*angularSize**2 # Number of expected background events.  Based on power law extrapolation from 10-300 GeV excluding 120-140 GeV
    ppa = float(outputSize)/float(angularSize) # pixels per degree
    centX,centY = np.mean(x), np.mean(y) # Compute Centroid
    
    # Build list of radii from cluster centroid
    r = [math.sqrt((x[i]-centX)**2 + (y[i]-centY)**2) for i in range(len(x))]
    
    # Sort the list and choose the radius where the cumulative count is >95%
    countIndex = int(math.ceil(0.95*len(r)-1)) 
    clusterRadius = np.sort(r)[countIndex]   # choose the radius at this index 

    # Estimate the background count
    #N_bg = Evaluate_BG_Contribution(centX*ppa+outputSize/2.0,centY*ppa+outputSize/2.0,clusterRadius*ppa,BGTemplate,numBGEvents, flatLevel = flatLevel)
    
    
    # For now just use isotropic density
    BGDensity = numBGEvents / angularSize**2
    N_bg = np.pi * clusterRadius**2. * BGDensity                 
    #N_cl = countIndex - N_bg
    
    # FIXED
    N_cl = countIndex
    #print 'N_on, N_off' ,N_cl, N_bg
    
    ######################################################
    # Evaluate significance as defined by Li & Ma (1983). 
    # N_cl corresponds to N_on, N_bg corresponds to N_off
    if N_cl/(N_cl+N_bg)<1e-20 or N_bg/(N_cl+N_bg)<1e-20:
        return 0
    S2 = 2.0*(N_cl*math.log(2.0*N_cl/(N_cl+N_bg))     +      N_bg*math.log(2.0*N_bg/(N_cl+N_bg)))
    if S2>0.0:
        return math.sqrt(S2)   
    else:
        return 0
    
    
    
def Compute_Cluster_Significance_3d_Isotropic(X, BGDensity, totalTime):
    """
    Takes input list of coordinate pairs (in angular scale) and computes the cluster significance based on a background model.
    
    Inputs:
        -X is a tuple containing a coordinate pair for each point in a cluster.  
        -BGDensity is the 2-d number of events per square degree integrated over the background.
        -totalTime is the full simulation time in units of the scale factor
        
    returns significance
    """
    # Default to zero significance
    if (len(X)==1):return 0
    
    # Otherwise.......
    x,y,t = np.transpose(X) # Reformat input
    
    centX,centY,centT = np.mean(x), np.mean(y),np.mean(t) # Compute Centroid
    
    # Build list of radii from cluster centroid
    r = [math.sqrt((x[i]-centX)**2 + (y[i]-centY)**2) for i in range(len(x))]
    
    # Sort the list and choose the radius where the cumulative count is >95%
    countIndex = int(math.ceil(0.95*len(r)-1)) 
    clusterRadius = np.sort(r)[countIndex]   # choose the radius at this index 

    # Estimate the background count
    #N_bg = Evaluate_BG_Contribution(centX*ppa+outputSize/2.0,centY*ppa+outputSize/2.0,clusterRadius*ppa,BGTemplate,numBGEvents, flatLevel = flatLevel)
    
    
    # For now just use isotropic density
    N_bg = np.pi * clusterRadius**2. * BGDensity
    
    # Rescale according to the total time.
     
    dT = (np.max(t)-np.min(t))/float(totalTime)
    if dT > .01: # This would be a 15 day period
        #print np.min(t)-np.max(t), totalTime, dT
        N_bg = N_bg*dT
                    
        #N_cl = countIndex - N_bg
        
        # FIXED
        N_cl = countIndex
        #print 'N_on, N_off' ,N_cl, N_bg
        
        ######################################################
        # Evaluate significance as defined by Li & Ma (1983). 
        # N_cl corresponds to N_on, N_bg corresponds to N_off
        if N_cl/(N_cl+N_bg)<1e-20 or N_bg/(N_cl+N_bg)<1e-20:
            return 0
        S2 = 2.0*(N_cl*math.log(2.0*N_cl/(N_cl+N_bg))     +      N_bg*math.log(2.0*N_bg/(N_cl+N_bg)))
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
    
    
    # Build list of radii from cluster centroid
    r = np.sqrt(np.square(x-centX)+np.square(y-centY))
    
    # Sort the list and choose the radius where the cumulative count is >95%
    countIndex = int(math.ceil(0.95*np.shape(r)[0]-1)) 
    clusterRadius = np.sort(r)[countIndex]   # choose the radius at this index 
    min_T_all, max_T_all = np.min(t_all),np.max(t_all)
    # Estimate the background count
    #N_bg = Evaluate_BG_Contribution(centX*ppa+outputSize/2.0,centY*ppa+outputSize/2.0,clusterRadius*ppa,BGTemplate,numBGEvents, flatLevel = flatLevel)

    AnnulusVolume = np.pi* ((outer*clusterRadius)**2 -(inner*clusterRadius)**2)*(max_T_all-min_T_all)
    
    r_all = np.sqrt(np.square(x_all-centX)+np.square(y_all-centY)) # compute all points radius from the centroid. 
    # Count the number of points within the annulus and find BG density
    r_cut = np.logical_and(r_all>clusterRadius*inner,r_all<clusterRadius*outer)
    idx =  np.where(r_cut==True)[0]
    BGDensity = np.shape(idx)[0]/AnnulusVolume # Density = counts / annulus volume 
    # BG density equal to 
    N_bg = np.pi * clusterRadius**2. * (maxT-minT)* BGDensity # BG count = cluster volume*bgdensity
    N_cl = countIndex # Number of counts in cluster.
    
    print 'clusterRadius: ',clusterRadius, ', N_cl: ', N_cl, ', N_bg: ', N_bg
    
    # Ensure log args are greater than 0.
    if N_cl/(N_cl+N_bg) <= 0 or N_bg/(N_cl+N_bg) <= 0: return 0
    # Compute significance from Li & Ma
    S2 = 2.0*(N_cl*math.log(2.0*N_cl/(N_cl+N_bg)) + N_bg*math.log(2.0*N_bg/(N_cl+N_bg)))
    
    if S2>0.:
        return math.sqrt(S2)   
    else:
        return 0.

    
def Compute_Cluster_Scale(cluster):
    '''
    Computes the mean 2-d pairwise distance matrix and standard deviation 
    Inputs: 
     -cluster: Tuple containing a coordinate pair for each cluster point.
    Returns:
        (mean, std) Mean pairwise dist and std
    '''
    d = distance.pdist(cluster)
    return np.mean(d), np.std(d)
                   
                   
    
    

