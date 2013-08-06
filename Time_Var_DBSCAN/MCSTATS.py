'''
Statistics and plotting tools for Fermi MC tools

Created on Jul 24, 2012
@author: Eric Carlson
'''

import matplotlib.pyplot as plt  #@UnresolvedImport
import matplotlib.image as mpimg #@UnresolvedImport
import matplotlib.cm as cm #@UnresolvedImport
import matplotlib, scipy #@UnresolvedImport
import cPickle as pickle, sys
import numpy as np
import scipy.cluster as cluster
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.mlab as mlab
import math, time
from scipy.cluster.hierarchy import *
from operator import itemgetter
import DBSCAN

import gc
import multiprocessing as mp
from multiprocessing import pool
from functools import partial


def DBSCAN_Compute_Clusters(mcSims, eps, timeScale, min_samples ,nCorePoints = 3, numAnalyze=0, fileout = '',numProcs = 1, indexing = None, plot=False):
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
        dbScanResults: a tuple (clusterReturn, labels) for each simulation
            clusterReturn: For each cluster, a list of points in that cluster
            labels: A list of points with cluster identification number.  -1 corresponds to noise.
                    NOTE: ALL BORDER POINTS ARE CONSIDERED NOISE.  See DBSCAN.py for info if you need
                    Border points.
    '''
        
    # Check number to analyze
    if ((numAnalyze == 0) or (numAnalyze > len(mcSims))):
        numAnalyze =len(mcSims)
    #print 'Analyzing ' + str(numAnalyze) + ' simulations using ' , numProcs, " CPUs..."
    
    

    DBSCAN_PARTIAL = partial(DBSCAN_THREAD,  eps=eps, min_samples=min_samples,timeScale=timeScale,nCorePoints = nCorePoints, indexing = indexing,plot=plot)
    if numProcs>1:
        p = pool.Pool(numProcs)
        # Call mutithreaded map. 
        dbscanResults = p.map(DBSCAN_PARTIAL, mcSims[:numAnalyze])
        p.close()  # Kill pool after jobs complete.  required to free memory.
        p.join()   # wait for jobs to finish.
    else:
        # Serial Version.  Only use for debugging
        dbscanResults = map(DBSCAN_PARTIAL, mcSims[:numAnalyze])
    
    #Single Call Version. Useful for Debugging
    #dbscanResults = DBSCAN_THREAD(mcSims[0],  eps=eps, min_samples=min_samples,nCorePoints = nCorePoints, indexing = indexing)
    
    # Write to file if requested
    if (fileout != ''): pickle.dump(dbscanResults, open(fileout,'wb'))
    
    return dbscanResults


# Define a single input function callable by each thread (async map can only take one argument)
def DBSCAN_THREAD(sim, eps, min_samples,timeScale,nCorePoints,indexing= None,plot=False):
    X = zip(sim[0],sim[1],sim[2])
    return DBSCAN.RunDBScan(X, eps, min_samples,timeScale,nCorePoints = nCorePoints, indexing = indexing,plot=plot)



def Cluster_Sigs_BG(dbscanResults,BGDensity, BGTemplate = 'BGRateMap.pickle',angularSize = 10.,numProcs = 1):
    """
    Compute the cluster significances on results of DBSCAN_Compute_Clusters() using a background model.
    
    Inputs:
        dbscanResults: output from DBSCAN_Compute_Clusters.  Must load from file if using pickled results
        BGTemplate: background template filename.
        angularSize: Size of square in degrees
        BG: The expected percentage of photons that are background
    """
    # Initialize the thread pool to the correct number of threads
    #if (numProcs<=0):numProcs += mp.cpu_count()
    #p = mp.pool.Pool(numProcs)
    
    #===========================================================================
    # Currently assuming isotropic BG
    #===========================================================================
    # Load background template
    #BGTemplate = pickle.load(open(BGTemplate,'r'))

    # Asynchronosly map results 
    BG_PARTIAL = partial(BG_THREAD, BGTemplate= '', angularSize = angularSize, BGDensity = BGDensity)
    return map(BG_PARTIAL,dbscanResults)


def BG_THREAD(sim,BGTemplate, angularSize , BGDensity ):
    clusters,labels = sim 
    return [DBSCAN.Compute_Cluster_Significance(cluster, BGTemplate,BGDensity = BGDensity, totalPhotons=len(labels),angularSize = angularSize) for cluster in clusters]


def Cluster_Sigs_BG_3d(dbscanResults,BGDensity, totalTime):
    """
    Compute the cluster significances on results of DBSCAN_Compute_Clusters() using a background model.
    
    Inputs:
        dbscanResults: output from DBSCAN_Compute_Clusters.  Must load from file if using pickled results
        BGDensity: 2-d number of events density per square degree
        totalTime: Total exposure time in units of scale factor.
    """
    #===========================================================================
    # Currently assuming isotropic BG
    #===========================================================================
    BG_PARTIAL = partial(BG_THREAD_3d, BGDensity = BGDensity, totalTime=totalTime)
    #print dbscanResults[0][0]
    #cluster, sim = dbscanResults[0]
    #return[DBSCAN.Compute_Cluster_Significance_3d(cluster, BGDensity = BGDensity, totalTime=totalTime),]
    
#    res = []
#    for i in dbscanResults:
#        clusters,labels = i
#        sigs = []
#        for cluster in clusters:
#            sigs.append(DBSCAN.Compute_Cluster_Significance_3d(cluster, BGDensity = float(BGDensity), totalTime=float(totalTime)))
#        res.append(sigs)
#    return res
    
    
    return map(BG_PARTIAL,dbscanResults)


def BG_THREAD_3d(sim, BGDensity, totalTime ):
    clusters,labels = sim 
    return [DBSCAN.Compute_Cluster_Significance_3d(cluster, BGDensity = BGDensity, totalTime=totalTime) for cluster in clusters]    


#===============================================================================
# DEPRECATED 
#===============================================================================
def Profile_Clusters(dbscanResults, BGPhotons, BGTemplate = 'BGRateMap.pickle',S_cut=2.0,angularSize = 10., fileout=''):
    """
    Computes properties on the results of DBSCAN_Compute_Clusters()
        input:
            dbscanResults: output from DBSCAN_Compute_Clusters.  Must load from file if using pickled results
        Optional Inputs
            S_cut: Clusters used in statistics must be at least this significance level
            BGTemplate = String with path to the background template file
            BGPhotons: Total number of BG photons expected
            
        
        Returns: (cluster_Scale, cluster_S, cluster_Count, cluster_Members, cluster_out) as described in draft 
            cluster_out is the significance weighted RMS of the clustering scale 
    """
    
    
    cluster_S = []       # Mean Significance weighted by number of cluster members for ALL CLUSTERS    
    cluster_Count = []   # Mean Number of clusters found s>s_cut
    cluster_Scale = []   # Mean Cluster Scale weighted by significance for S>s_cut
    cluster_Members = [] # Mean number of cluster Members s> s_cut
    cluster_Out = []     # for each sim a tuple of (cluster_s, num_members) for each cluster in the simulation.
     
    for sim in dbscanResults: 
        clusters,labels = sim
        numPhotons = len(labels) 
        #===========================================================================
        # Compute Cluster Properties
        #===========================================================================     
        # Determine Cluster Significance from background template
        S = [DBSCAN.Compute_Cluster_Significance(cluster, BGTemplate, BGPhotons, numPhotons,angularSize = angularSize) for cluster in clusters]
        # Number of clusters
        clusterMembersAll = [len(cluster) for cluster in clusters]
        # list of pairs (s,num members) for each cluster
        cluster_Out.append(zip(S, clusterMembersAll))
        # S>S_Cut Cluster Indexes
        sigClustersIDX = np.where((S>=S_cut))[0]
        # S>S_Cut Clusters
        sigClusters = clusters[clusters]
        
        sigs = S[sigClustersIDX]
        # Compute Cluster Scales
        scale = [DBSCAN.Compute_Cluster_Scale(cluster)[0] for cluster in sigClusters]
        # S>S_Cut Cluster Member Counts
        members = clusterMembersAll[sigClustersIDX]
        
        
        #===========================================================================
        # Compute Weighted Means and append to master list 
        #===========================================================================
        # Append All cluster sigs.  Rest of quantities require S>2.0
        cluster_S.append(np.average(S,weights = clusterMembersAll))
        cluster_Count.append(len(sigs))
        if len(sigs)!=0:
            cluster_Scale.append(np.average(scale, weights = sigs))
            cluster_Members.append(np.average(members, weights = sigs))
        
    output = (cluster_Scale, cluster_S, cluster_Count, cluster_Members, cluster_Out)
    # Write results to file
    if fileout != '':
        pickle.dump(output, open(fileout, 'wb'))
    return output



#################################################################################################
# Plotting Tools
#################################################################################################

#TODO: 
def Plot_Cluster_Scales(models, labels,xlabel, fig, subplot, bins = 100, fileout = '', PlotFermiScale = False):
    """
    Generates the results summary plots.  See sample usage in runMC_v2
    """
    
    bins = np.linspace(0,5, 6)
    
    
    fig.add_subplot(4,1,abs(subplot))      
    
    hist = []
    for i in models:
        if (subplot == 1):
            hist.append(np.histogram(i, bins=bins))
        else: hist.append(np.histogram(i,bins = 20))
            
    for i in range(len(hist)):
        plt.step(hist[i][1][:-1], np.array(hist[i][0],'float')/len(models[i]), label=labels[i])
        
    plt.xlabel(xlabel)
    if subplot in [1,3,5,7]:
        plt.ylabel(r'$f$')
    



            
            
            
#===============================================================================
# More plotting tools that are older.
#===============================================================================
def Plot_Rate_Map(mapName,angularSize,fileOut):
    """
    Plot the map of annihilation rate.
        inputs:
            mapName: string with filename to pickle file of rate map.
            
    
    """
    plt.clf()
    map = pickle.load(open(mapName, "r" ))
    img = plt.imshow(map,origin='lower', extent=[-angularSize/2,angularSize/2,-angularSize/2,angularSize/2])
    plt.colorbar(orientation='vertical')
    
    plt.xlabel(r'$l[^\circ]$')
    plt.ylabel(r'$b[^\circ]$')
    plt.title(r'$\rho_{DM}^2$')

    plt.savefig(str(fileOut)+ '.png')
    plt.show()
    

#===============================================================================
# Quick way to plot the monte-carlo or Fermi data, just pass a tuple of coordinate
# pairs for each positions. 
#===============================================================================
def Plot_MC_Positions(MC,fileOut='',angularSize = None):
    """
    Plot results of a Monte Carlo simulation.
        input:
        -MC: One Monte Carlo simulation.  If taking from a set of runs, pass mcSims[i] for the i'th simulation
        -fileOut: if not blank string then saves the file to this name and does not display plot
        -angularSize: If None, auto-scales to the min/max of simulation.  Otherwise pass the width in simulation coordinates
    """
    plt.clf()
    plt.scatter(MC[0], MC[1],color = 'b',s=4,marker='+')
    if angularSize != None:
        plt.xlim(angularSize/2.0,-angularSize/2.0)
        plt.ylim(-angularSize/2.0,angularSize/2.0)
            
        plt.xlabel(r'$l[^\circ]$')
        plt.ylabel(r'$b[^\circ]$')
        plt.title(r'Count Map')
        if fileOut!='':
            pp = PdfPages(fileOut + '.pdf')
            plt.savefig(pp, format='pdf')
            print "Figures saved to ", str(fileOut)+ '.pdf\n',
            pp.close()
            plt.savefig(fileOut + '.png', format='png')
            return
        plt.show()    

