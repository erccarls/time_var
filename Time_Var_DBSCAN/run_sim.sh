gtobssim infile="test.xml" srclist="test_sources.txt" scfile="none" evroot="test" simtime=125600000 tstart=0 use_ac="no" emin=1000 emax=1e6 irfs=P7REP_SOURCE_V15 seed=$!

#gtselect infile=test_events_0000.fits outfile=events.txt ra=INDEF dec=INDEF rad=INDEF tmin=INDEF tmax=INDEF emin=30 emax=1e6 zmax=180
#gtmktime (DATA_QUAL>0)&&(LAT_CONFIG==1)&&ABS(ROCK_ANGLE)<52
