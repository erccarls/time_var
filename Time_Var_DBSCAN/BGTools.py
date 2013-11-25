import pyfits
import numpy as np

class BGTools:
    def __init__(self,Emin,Emax,Time,diff_f, iso_f):
        """Returns a numpy array with a sky map of the number of photons per square degree.
        Inputs:
            E_min : min energy in MeV
            E_max : max energy in MeV
            Time  : Integration time in seconds.
            diff_f: Abosulte path to diffuse BG model (typically '$FERMI_DIR/refdata/fermi/galdiffuse/gll_iem_v05.fits')
            iso_f : Abosulte path to diffuse BG model (typically '$FERMI_DIR/refdata/fermi/galdiffuse/isotrop_4years_P7_v9_repro_clean_v1.txt')
        returns: 2-dim numpy array skymap
        """
        self.Emin   = Emin
        self.Emax   = Emax
        self.Time   = Time
        self.diff_f = diff_f
        self.iso_f  = iso_f
        self.BGMap  = self.Prep_Data(Emin,Emax,Time,diff_f,iso_f)
    
    def Prep_Data(self,E_min,E_max,Time,diff_f, iso_f):
        """Returns a numpy array with a sky map of the number of photons per square degree.
        Inputs:
            E_min: min energy in MeV
            E_max: max energy in MeV
            Time : Integration time in seconds.
        returns: 2-dim numpy array skymap
            """
        
        if E_min<50 or E_max>6e5: print "Seems like your energies are out of range. Should be 50 MeV to 600 GeV in units MeV"
        # Calculate a list of the energies corresponding to the diffuse galactic model
        energies = np.logspace(np.log10(50),np.log10(6e5),31)
        # Calc indicies we care about 
        emin,emax = np.argmax(energies>E_min)-1, np.argmax(energies>E_max)
        if emax==0:emax=30
        # Now compute the average energies in each bin
        energies = np.array([np.mean(energies[i:i+2]) for i in range(len(energies)-1)])
        
        # Interpolate effective area.  Empiracally fudged to agree with  gtobssim simulations.
        EAE =[1000.0, 1579.2081441197518, 2493.8983624541534, 3938.3846045945115, 6219.5290422515081,
              9821.9309161129022, 15510.873293707069, 24494.897427831787, 38682.541507409958, 
              61087.784583752211, 96470.326920894411, 152346.72593937156, 240587.19033343517, 379937.25034545001]
        EA = [0.56474960595679269, 0.61900037984106093, 0.6477423909408575, 0.6494559139904782,
              0.65105871317624175, 0.65634657600811919, 0.64493619600906349, 0.65972804516431405,
              0.70880567499919545, 0.77383792531196494, 0.88505388177027788, 0.84500602665697677, 
              0.93898153929664407, 0.59507693678706786]
        effArea = np.interp(energies,EAE,np.array(EA))
        
        #Determine weights to convert flux to photon counts
        energies = np.logspace(np.log10(50),np.log10(6e5),31)
        if emin==emax:emax+=1
        weights = [(energies[i+1]-energies[i]) for i in range(emin,emax)] 
        # Endpoints need to be reweighted if they don't align with the template endpoints.
        weights[0]*=(energies[emin+1]**-1.5-E_min**-1.5)/(energies[emin+1]**-1.5-energies[emin]**-1.5)
        weights[-1]*=(E_max**-1.5-energies[emax-1]**-1.5)/(energies[emax]**-1.5-energies[emax-1]**-1.5)
        weights = np.multiply(np.array(weights)*Time,effArea[emin:emax])

        # Load the diffuse background model
        hdulist = pyfits.open(diff_f, mode='update')
        scidata = hdulist[0].data[emin:emax]
        scidata = [scidata[i]*weights[i] for i in range(len(scidata))]
        # Load the isotropic model
        energies,N = np.transpose(np.genfromtxt(iso_f, delimiter=None,autostrip=True))
        isotrop = np.multiply(N[emin:emax],weights)
        # Sum the photon counts
        total=np.zeros(shape=np.shape(scidata[0]))
        for i in range(len(scidata)):
            total = np.add(total,scidata[i]+isotrop[i])
        return total

    def GetBG(self,l,b):
        '''Given a lat and long vector, return a vector with the number of photons/sq-deg at that point.
        Inputs:
            l: longitude vector (Nx1)
            b: latitude vector  (Nx1)
        returns:
            evt: Number of photons/deg^2 at the input coordinates. (Nx1)
            '''
        # Find size of BG map
        len_b,len_l = np.shape(self.BGMap)
        # Map the input coords onto the background model
        l_idx = np.divide((np.array(l)+180.)%360,360./float(len_l)).astype(int)
        b_idx = np.divide(np.array(b)+90.,180./float(len_b)).astype(int)
        # Bounds checking on lat.  longitude is handled by modulo operator above
        b_idx[np.where(b_idx==len_b)[0]] = len_b-1
        return self.BGMap[b_idx,l_idx]
        
    def SigsBG(self, CR):
        """Returns Significance vector based on integration of the background template for each cluster."""
        cx,cy = CR.CentX, CR.CentY # Get centroids
        dens  = self.GetBG(cx,cy)  # Evaluate the background density at that location
        N_bg   = dens*np.pi*CR.Size95X*CR.Size95Y# Area of an ellipse times BG density
        N_bg   = N_bg*2.*CR.Size95T/12.*3.15e7/self.Time # Find ratio of cluster length to total integration length
        N_cl   = (0.95*CR.Members) # 95% containment radius
        ######################################################
        # Evaluate significance as defined by Li & Ma (1983).  N_cl corresponds to N_on, N_bg corresponds to N_off
        S2 = np.zeros(len(N_cl))
        idx = np.where(np.logical_and(N_cl/(N_cl+N_bg)>0, N_bg/(N_cl+N_bg)>0))[0]
        N_cl, N_bg = N_cl[idx], N_bg[idx]
        S2[idx] = 2.0*(N_cl*np.log(2.0*N_cl/(N_cl+N_bg)) + N_bg*np.log(2.0*N_bg/(N_cl+N_bg)))
        return np.sqrt(S2)   



