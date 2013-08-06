#===============================================================================
# Generate an isotropic and homogenous event distribution to test sensitivity 
# of dbscan to
#===============================================================================


import numpy as np
import cPickle as pickle
import matplotlib.pyplot as plt

import cPickle as pickle, numpy as np, matplotlib.pyplot as plt
import psf# Module for gaussian PSF convolution

# Universal Simulation Parameters
sizeX,sizeY,sizeT = 10.,10.,10. # Define box size
N_bg = 1000 # Number of background events to be distributed randomly
r_68 = 1.0 # Spatial PSF size in degrees 

# Define a function to run the simulation given a set of input parameters
# Optimized via numpy's vector routines
def RunSim(N_sig_on, t_burst):
    # Generate the signal
    X,Y,T = np.zeros(N_sig_on), np.zeros(N_sig_on), (np.random.ranf(N_sig_on)-.5)*t_burst
    # Shift the signal photons using the PSF
    X,Y = psf.ApplyGaussianPSF(XMASTER=X,YMASTER=X,r_68=r_68 )
    # Generate the background photons-Don't shift these by the PSF
    X = np.append(X,(np.random.ranf(N_bg)-.5)*sizeX)
    Y = np.append(Y,(np.random.ranf(N_bg)-.5)*sizeY)
    T = np.append(T,(np.random.ranf(N_bg)-.5)*sizeT)
    return X,Y,T

X,Y,T = RunSim(500,.1)
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure()
#ax = fig.add_subplot(111, projection='3d')
ax = fig.add_subplot(111)
ax.scatter(X,Y, s=2)
plt.xlabel('x')
plt.ylabel('y')
plt.show()

