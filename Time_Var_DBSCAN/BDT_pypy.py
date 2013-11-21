# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>


# <codecell>

import cPickle as pickle, numpy as np, matplotlib.pyplot as plt
import psf# Module for gaussian PSF convolution

# Universal Simulation Parameters
sizeX,sizeY,sizeT = 50.,50.,24. # Define box size
dens = 10 #dens/sq deg
N_bg = dens*sizeX*sizeY # Number of background events to be distributed randomly
r_68 = 1.0 # Spatial PSF size in degrees 
numProcs = 12 # How many CPU to use
numSources = 100

def AddRandomPointSource(X,Y,T):
    # Pick random centroid
    cx,cy,ct = np.random.ranf(3)
    cx,cy,ct=(cx-0.5)*sizeX,(cy-0.5)*sizeY,(ct-0.5)*(sizeT-2)
    # pick burst length 
    t_burst = 12*np.random.ranf()
    if t_burst < 1: t_burst =1.
    # pick flux
    while True:
        xs,ys = np.random.ranf(2)
        if xs<.025:
            N_sig_on = 10
            break
        elif 40*ys < 1/xs:
            N_sig_on = int(xs*400)
            break 
    #f.write(str(cx) + ' ' + str(cy) + ' ' + str(ct) + ' ' + str(N_sig_on) + ' ' + str(t_burst))
    x,y,t = np.zeros(N_sig_on), np.zeros(N_sig_on), (np.random.ranf(N_sig_on)-0.5)*t_burst
    x,y,t = x+cx, y+cy, t+ct
    # Shift the signal photons using the PSF
    x,y = psf.ApplyGaussianPSF(XMASTER=x,YMASTER=y,r_68=r_68 )
    X = np.append(X,x)
    Y = np.append(Y,y)
    T = np.append(T,t)
    
    return X,Y,T,(cx,cy,ct,N_sig_on,t_burst)
    
    
# Define a function to run the simulation given a set of input parameters
# Optimized via numpy's vector routines
def RunSim(numSources=25):
    # Generate the signal
    #f = open("src.txt",'wb')
    src = []
    X,Y,T = [],[],[]
    for i in range(numSources):
        ret = AddRandomPointSource(X,Y,T)
        X,Y,T,src2 = ret
        src.append(src2)
    #f.close()
    
    idx = np.where(np.logical_and(T>-.5*sizeT,T<0.5*sizeT))
    X= X[idx]
    Y= Y[idx]
    T= T[idx]
    
    # Generate the background photons-Don't shift these by the PSF
    X = np.append(X,(np.random.ranf(N_bg)-.5)*sizeX)
    Y = np.append(Y,(np.random.ranf(N_bg)-.5)*sizeY)
    T = np.append(T,(np.random.ranf(N_bg)-.5)*sizeT)
    
    return X,Y,T,src
# Run a simulation with 500 photons distributed 
X,Y,T,src = RunSim(numSources=numSources)
X2,Y2,T2,src2 = RunSim(numSources=numSources)

# <codecell>


plt.figure(0,figsize=(18,8))
plt.subplot(121)
p = plt.scatter(X,Y,s=.001)
#p.set_rasterized(True)
plt.xlabel('X [deg]')
plt.ylabel('Y [deg]')
plt.xlim((-.6*sizeX,.6*sizeX))
plt.ylim((-.6*sizeY,.6*sizeY))

plt.subplot(122)
p2 = plt.scatter(X,T,s=.001)
#p2.set_rasterized(True)
plt.xlabel('X [deg]')
plt.ylabel('T [mo]')
plt.xlim((-.6*sizeX,.6*sizeX))
plt.ylim((-.6*sizeT,.6*sizeT))

plt.savefig('sim_patch.png')


#plt.savefig('sim_patch.pdf')
#plt.show()

# <codecell>


import MCSTATS
a = 2.
mcSims =((X,Y,T),)
mcSims2 = ((X2,Y2,T2),)
reload(MCSTATS)
expBG     = dens*np.pi*r_68**2 * (2*a/sizeT) # expected number of events in a cylinder of height 2a
nMin = expBG + 3*np.sqrt(expBG)
if nMin<3.: nMin=3 # If the number of events is less than three, DBSCAN won't work very well....


#from pycallgraph import PyCallGraph
#from pycallgraph.output import GraphvizOutput
#with PyCallGraph(output=GraphvizOutput()):
#    scan = MCSTATS.DBSCAN_Compute_Clusters(mcSims, eps=r_68, timeScale=a, min_samples=nMin, numProcs=numProcs,sigMethod='isotropic',inner=1.25, outer=2.0,plot=False,BGDensity=dens,TotalTime = sizeT)
import datetime
aa = datetime.datetime.now()
scan = MCSTATS.DBSCAN_Compute_Clusters(mcSims, eps=r_68, timeScale=a, min_samples=nMin, numProcs=1,sigMethod='isotropic',inner=1.25, outer=2.0,plot=False,BGDensity=dens,TotalTime = sizeT)
bb = datetime.datetime.now()
print (bb - aa).seconds

scan2 = MCSTATS.DBSCAN_Compute_Clusters(mcSims2, eps=r_68, timeScale=a, min_samples=nMin, numProcs=numProcs,sigMethod='isotropic',inner=1.25, outer=2.0,plot=False,BGDensity=dens,TotalTime = sizeT)
scan = scan[0]
scan2 = scan2[0]

# <codecell>

def Identify_Clusters(scan,src):
    identities = []
    cx,cy,ct,N_sig_on,t_burst = np.transpose(src)
    
    for j in range(len(scan.Members)):
        # Compute radius in units of 3-sigma uncertainty on centroid positions
        r  = np.sqrt(np.square(scan.CentX[j]-cx) + np.square(scan.CentY[j]-cy))/(scan.Size95X[j]/2.)
        dT = np.abs((scan.CentT[j]-ct)/(scan.Size95T[j]/2.))
        idx = np.where(np.logical_and(r<1,dT<1)==True)[0]
        if np.shape(idx)[0] == 0:
            identities.append(-1)
        elif np.shape(idx)[0] == 1:
            identities.append(idx[0])
            #print "Cluster Identified With Original Source: ", idx
        else: 
            identities.append(idx[0])
            #print "Confusion Detected Defaulting to first source"
    return identities

identities = Identify_Clusters(scan,src)
identities2 = Identify_Clusters(scan2,src2)
#print "(cx,cy,ct,N_sig_on,t_burst)"
cnt = 0
cnt2 = 0
for i in range(numSources):
    if i not in identities:
#        print 'ND', np.array(src[i])
        if src[i][3]>20: cnt +=1
        
for i in range(numSources):
    if i not in identities2:
#        print 'ND', np.array(src[i])
        if src2[i][3]>20: cnt2 +=1
        
#print 
#for i in range(numSources):
#    if i in identities:
#        print 'DE', np.array(src[i])    

print "Detected:", np.count_nonzero(np.array(identities)!=-1), ",ND >20 Photons:", cnt, ",Unidentified Detected Clusters:", np.count_nonzero(np.array(identities)==-1)
print "Detected:", np.count_nonzero(np.array(identities2)!=-1), ",ND >20 Photons:", cnt2, ",Unidentified Detected Clusters:", np.count_nonzero(np.array(identities2)==-1)

#print "\nSizeT Comparisons and N_sig_on Comparisons"
#for i in range(len(identities)):
#    ID = identities[i]
#    if ID != -1:
#        print src[ID][4], 2*scan.MedT[i] , '     ', src[ID][3] , scan.Members[i]-np.pi*r_68**2*dens



for i in range(len(scan.Members)):
    x,y,t = np.transpose(scan.Coords[i])
    plt.scatter(x,y,s=.1)
    

# <codecell>

from sklearn.ensemble import GradientBoostingClassifier

X_train = np.transpose((scan.Size95X,scan.Size95Y,scan.Size95T,scan.MedR,scan.MedT, scan.Members, scan.Sigs)) # Nsamples x NFeatures
Y_train = (np.array(identities) !=-1).astype(int) # flag all identified clusters as signal and all others as background
X_train_hold = X_train
clf = GradientBoostingClassifier(n_estimators=500, learning_rate=1.0,max_depth=3, random_state=0).fit(X_train, Y_train)

plt.figure(0,figsize=(10,14))
plt.subplot(421)
N = len(clf.feature_importances_)
ind = np.arange(N)+0.5    # the x locations for the groups
width = 0.35       # the width of the bars: can also be len(x) sequence
plt.bar(ind, clf.feature_importances_,   width, color='r')
plt.ylabel('Scores')
plt.title('BDT Feature Importance')
plt.xticks(ind+width/2., ('Size95X','Size95Y','Size95T','MedR','MedT','Members','Sigs') ,rotation=20)


X_train = np.transpose((scan2.Size95X,scan2.Size95Y,scan2.Size95T,scan2.MedR,scan2.MedT, scan2.Members, scan2.Sigs)) # Nsamples x NFeatures
Y_train = (np.array(identities2) !=-1).astype(int) # flag all identified clusters as signal and all others as background


print "Mean Classification Accuracy:", clf.score(X_train, Y_train)



pred = clf.predict(X_train)
real  = np.where(pred==1)[0]
fake = np.where(pred==0)[0]

def PlotHist(real,fake,f,l,subplot):
    plt.subplot(4,2,subplot)
    plt.xlabel(l)
    plt.hist(f[real],bins=np.linspace(np.min(f),np.max(f),25),color='b',histtype='step',normed=True,label='Real')
    plt.hist(f[fake],bins=np.linspace(np.min(f),np.max(f),25),color='r',histtype='step',normed=True,label='Fake')
PlotHist(real,fake,scan2.Members,'Members',2)
plt.legend()
PlotHist(real,fake,scan2.Size95X,'Size95X',3)
PlotHist(real,fake,scan2.Size95Y,'Size95Y',4)
PlotHist(real,fake,scan2.Size95T,'Size95T',5)
PlotHist(real,fake,scan2.MedR,'MedR',6)
PlotHist(real,fake,scan2.MedT,'MedT',7)
PlotHist(real,fake,scan2.Sigs,'Sigs',8)
    
plt.figure(1,figsize=(10,6))
ax1 = plt.subplot2grid((2,1), (0,0), colspan=2)
plt.hist(np.transpose(clf.predict_proba(X_train[real]))[0],bins=np.linspace(0,1,51),histtype='step',color='b',normed=True)
plt.hist(np.transpose(clf.predict_proba(X_train[fake]))[0],bins=np.linspace(0,1,51),histtype='step',color='r',normed=True)
plt.yscale('log')

# <codecell>

pred = clf.predict(X_train)
real  = np.where(pred==1)[0]
print len(real)
print len(fake)
fake = np.where(pred==0)[0]

def PlotCorrelations(real,fake,f1,f2,l1,l2,subplot):
    plt.subplot(3,2,subplot)
    plt.xlabel(l1)
    plt.ylabel(l2)
    plt.scatter(f1[real],f2[real],s=2,color='b')
    plt.scatter(f1[fake],f2[fake],s=2, color='r')

plt.figure(0,figsize=(8,12))
PlotCorrelations(real,fake,scan2.Members,scan2.Sigs,'Members','Significance',1)
PlotCorrelations(real,fake,scan2.Members,scan2.Size95T,'Members','SizeT [mo]',2)
PlotCorrelations(real,fake,scan2.Members,scan2.MedR,'Members','MedR [deg]',3)
PlotCorrelations(real,fake,scan2.Members,scan2.MedT,'Members','MedT [mo]',4)
PlotCorrelations(real,fake,scan2.Size95X,scan2.MedR,'Size95X [deg]','MedR [deg]',5)
PlotCorrelations(real,fake,scan2.Sigs,scan2.MedR,'Sigs','MedR [deg]',6)

# <codecell>

from sklearn.ensemble.partial_dependence import plot_partial_dependence

plt.figure(0,figsize=(10,10))
features = [0,1,(3,5)]
fig, axs = plot_partial_dependence(clf, X_train_hold, features,figsize=(10,10)) 




