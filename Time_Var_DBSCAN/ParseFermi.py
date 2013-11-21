import pyfits
import numpy as np
 

def LoadEvents(fname, elow=30,ehigh=1e6,tlow=0,thigh=1e10):
    hdulist = pyfits.open(fname, mode='update')
    #hdulist.info()
    E = hdulist[1].data['Energy']
    L = hdulist[1].data['L']
    B = hdulist[1].data['B']
    T = hdulist[1].data['Time']
    ecut = np.logical_and(E>elow,E<ehigh)
    tcut = np.logical_and(T>tlow,T<thigh)
    idx = np.where(np.logical_and(ecut,tcut)==True)[0]
    return (E[idx],L[idx],B[idx],T[idx])

#E,L,B,T = LoadEvents('./test_events_0001.fits',elow=1000)

#from matplotlib import pyplot as plt
#from mpl_toolkits.basemap import Basemap
#m = Basemap(projection='hammer',lon_0=0)
#x, y = m(L,B)
#print len(x)/41250.
#m.drawmapboundary(fill_color='#99ffff')
#plt.scatter(x,y,s=2)
#plt.show()
