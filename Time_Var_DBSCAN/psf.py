import numpy as np

def ApplyGaussianPSF(XMASTER, YMASTER, r_68):
        r = []
        num = len(XMASTER)
        # given r68, calculate what sigma should be for a gaussian PSF to meet this (can solve analytically)
        sigma = r_68/1.0674
                
        while (num != 0):
            X,Y = np.random.rand(2,num)
            X = 5.*sigma*X
            # Normalize to max=1 and perform MC sampling
            new = list(X[np.where(Y <=  np.sqrt(2*np.e)/sigma* X*np.exp((-np.square(X)/(sigma*sigma))  ))[0]])
            r+=new
            num-=len(new)
        
        theta = np.random.rand(len(r))*2.*np.pi
        XMASTER += np.cos(theta)*np.array(r)
        YMASTER += np.sin(theta)*np.array(r)
        
        return list(XMASTER),list(YMASTER)