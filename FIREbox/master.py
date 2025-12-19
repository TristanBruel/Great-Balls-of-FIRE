import numpy as np
#import pandas as pd

from compileClusters import compileClusters
#from sampleBBHs import sampleBBHs
from sampleBBHs_withKremer import sampleBBHs
from compile_dynBBHs import compileBBHs
from GCformrates import compute_form_rates
from mergerRates import compute_merger_rates

import argparse

def main(sample,seed=None):
    print('master:: Random seed is:', seed)
    
    ### Sample massive clusters and BBHs for all the galaxies of interest (Ms > 10^8 Msun) ###
    print('--- Starting sampling in all galaxies ---')
    print('Sampling method for massive clusters:', sample)

    '''
    # Use Elia's galaxies
    Mstar_file = 'Mstar.dat'
    halo_ids = np.loadtxt(Mstar_file, skiprows=1, usecols=0)
    halo_ids = halo_ids.astype(np.int64)
    '''
    # Use all halos in FIREbox
    #Mstar_file = 'Mstar90.dat'
    Mstar_file = 'Mstar90_v2.dat'
    halo_ids, Mstar = np.loadtxt(Mstar_file, skiprows=1, usecols=[0,3]).T
    halo_ids = halo_ids.astype(np.int64)
    '''
    # Get halo ids for galaxies in Elia's file
    gals = pd.read_hdf('/home/tbruel/scratch/snap1200_circularity_M8_sample.hdf5')
    halo_ids = np.array(gals['halo ID'])
    '''
    if sample=='SFR':
        from sampleMassiveClusters import sampleMassiveClusters
        for halo_idx in halo_ids:
            clusters = sampleMassiveClusters(halo_idx,seed=seed)
            bbhs = sampleBBHs(halo_idx, sample=sample)
            print('Halo', halo_idx, 'completed.')
    elif sample=='ElBadry':
        from sampleMassiveClusters_ElBadry import sampleMassiveClustersElBadry
        for halo_idx in halo_ids:
            clusters = sampleMassiveClustersElBadry(halo_idx,seed=seed)
            bbhs = sampleBBHs(halo_idx, sample=sample)
            print('Halo', halo_idx, 'completed.')
    elif sample=='Gamma':
        from sampleClusters import sampleClusters
        for halo_idx in halo_ids:
            clusters = sampleClusters(halo_idx,seed=seed)
            bbhs = sampleBBHs(halo_idx, sample=sample)
            print('Halo', halo_idx, 'completed.')
    else:
        print('Sampling method for massive clusters not indicated, or unknown.')
        print('Try with "Gamma", "SFR" or "ElBadry"')
        return
    
    compileClusters(sample=sample, seed=seed)
    compileBBHs(sample=sample, seed=seed)

    #compute_form_rates(seed=seed)
    #compute_merger_rates(seed=seed)
    print('All done mate !')

##################################################################
### 
### Run it!
###
##################################################################
if __name__ == "__main__":

    #####################################
    # Define command line options
    parser = argparse.ArgumentParser()
    parser.add_argument('--sample', dest='sample', type=str, help='Method to use for the sampling of massive clusters.')
    parser.add_argument('--seed', dest='seed', type=int, help='Random seed', default=None)
    args = parser.parse_args()

    #####################################
    # Run it
    main(sample=args.sample, seed=args.seed)
