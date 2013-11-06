#==================================================================
# GenSourceXML.py
# In this python module we generate XML data to simulate transient sources for gtobssim.
# Author: Eric Carlson, erccarls@ucsc.edu
#==================================================================
import numpy as np

def GenSourceLists(filename,numSources):
    f = open(filename + '.xml', 'wb')
    f2 = open(filename + '.txt', 'wb')
    f3 = open(filename + '_sources.txt', 'wb')
    f.write('<source_library title="CAP_Simple_Transient">\n\n')    
#     The parameters are:
#     - Flux while in the active state in units of  m^-2 s^-1,
#     - Photon spectral index such that dN/dE~E^-Gamma
#     - Start time of the active state in MET seconds.
#     - Stop time of the active state in MET seconds.
#     - Minimum photon energy in MeV (default = 30)
#     - Maximum photon energy in MeV (default = 1e5)
    mo2sec = 2629743.83
    ra = 360.*(np.random.ranf(numSources)-0.5)
    dec = (np.arccos(2.*np.random.ranf(numSources) - 1)-np.pi/2.) * 180./np.pi
    ra = ra+180.
    
    for i in range(numSources):
        RA,DEC = ra[i],dec[i]
        flux, index, start, stop = 1e-2, 2, 6*mo2sec, 8*mo2sec
        f.write('    <source name="CAPS' + str(i) + '">\n')
        f.write('        <spectrum escale="MeV">\n')
        f.write('            <SpectrumClass name="SimpleTransient"\n' +
                '             params="' + str(flux) + ',' +str(index) + ',' + str(start) +',' + str(stop) + ', 30., 2e5"/>\n')
        f.write('            <celestial_dir ra="' + str(RA) + '" dec="' + str(DEC) + '"/> \n')
        f.write('         </spectrum>\n')
        f.write('    </source>\n\n')
        
        f2.write(str(i) + ' ' + str(RA) + ' ' + str(DEC) + ' ' + str((stop-start)/2.) + ' ' + str(flux) + '\n')
        f3.write("CAPS" + str(i) + '\n')
        
        
    f.write(ExGalBG)
        
    f.write('</source_library>\n')
    f.close()
    f2.close()
    f3.write('GalacticDiffuse_v05\n')
    f3.write('IsotropicDiffuse\n')
    f3.close()



ExGalBG ='''
<source name="GalacticDiffuse_v05">
<!-- This is v05 of the diffuse emission model that is recommended by the Diffuse Group.
Total photon flux from the map (#/m^2/s) = 8.296 over the whole
energy range of the MapCube file (50 MeV-100 GeV) -->
   <spectrum escale="MeV">
     <SpectrumClass name="MapCube" params="8.296,$(FERMI_DIR)/refdata/fermi/galdiffuse/gll_iem_v05.fits"/>
     <use_spectrum frame="galaxy"/>
   </spectrum>
</source>
     <!-- This is the isotropic diffuse spectrum that is recommended by the
          the diffuse group for P6V3 DIFFUSE class FRONT + BACK; integrated
          flux below is 5.01 ph m^-2 s^-1 [Note units]  -->


<source name="IsotropicDiffuse">
   <spectrum escale="MeV">
     <SpectrumClass name="FileSpectrumMap" params="flux=5.01,fitsFile=$(FERMI_DIR)/refdata/fermi/galdiffuse/isotropic_allsky.fits,specFile=$(FERMI_DIR)/refdata/fermi/galdiffuse/iso_p7v6source.txt,emin=39.3884,emax=403761"/>
      <use_spectrum frame="galaxy"/>
   </spectrum>
 </source>
 '''
    
GenSourceLists('test',0)