import scipy.linalg as la
import numpy as np
from matplotlib import pyplot as plt

def Compute_Centroid(X):
    X,Y = np.transpose(X)
    CentX0,CentY0 = np.mean(X),np.mean(Y)
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
    
    print theta+90
    
    
    POSANG = theta
    CENTX,CENTY =  CentX+CentX0,CentY+CentY0, 
    SIG95X,SIG95Y = 2*SigX/np.sqrt(np.shape(x)[0]),2*SigY/np.sqrt(np.shape(x)[0])
    SIZE95X, SIZE95Y = 2*S/np.sqrt(len(X)) 
    return SIZE95X, SIZE95Y, POSANG,CENTX,CENTY,SIG95X,SIG95Y


def __Rotate(v, theta):
    return  np.dot([[np.cos(theta), -np.sin(theta)],
         [np.sin(theta), np.cos(theta)]],v)

numPoints = 1000
X,Y = np.random.normal(scale=3,size=numPoints), np.random.normal(scale=1,size=numPoints)+3
X,Y = np.transpose([__Rotate([X[i],Y[i]],theta =np.deg2rad(120)) for i in range(len(X))])
SIZEX, SIZEY, PA, CX,CY,SigX,SigY = Compute_Centroid(np.transpose((X,Y)))

    
# Plot 
fig = plt.figure()
ax = fig.add_subplot(111, aspect='equal')
plt.scatter(X, Y, s=2)
plt.axis('equal')
plt.xlim(-10,10)
plt.ylim(-10,10)

from pylab import figure, show, rand
from matplotlib.patches import Ellipse
ell = Ellipse(xy=(CX,CY), width=SigX*2., height=SigY*2., angle = PA, alpha=1, facecolor='r')
ell2 = Ellipse(xy=(CX,CY), width=SIZEX*2., height=SIZEY*2., angle = PA, alpha=.15, facecolor='b')
 
ax.add_artist(ell)
ax.add_artist(ell2)

plt.show()


