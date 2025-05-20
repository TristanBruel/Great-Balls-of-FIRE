import numpy as np
import h5py
import pandas as pd

import os
from glob import glob
import argparse


def compileClusters(sample,seed=None):

    # load galaxy stellar masses
    Mstar_file = 'Mstar90.dat'
    halo_ids, Mvir, Mstar = np.loadtxt(Mstar_file, skiprows=1, usecols=[0,1,3]).T
    #Mstar_file = 'Mstar.dat'
    #halos, Mstar = np.loadtxt(Mstar_file, skiprows=1).T
    halo_ids = halo_ids.astype(np.int64)
    '''
    # Get galaxy stellar masses from Elia's file
    gals = pd.read_hdf('/home/tbruel/scratch/snap1200_circularity_M8_sample.hdf5')
    halos = np.array(gals['halo ID'])
    Mstar = np.array(gals['Mstar_0.1 Rvir']) *1e10 # unit Msun
    ''' 
    formation_times = []
    radii = []
    masses = []
    metallicities = []
    halo_mass = []
    galaxy_mass = []
    for n,halo_idx in enumerate(halo_ids):
        Mv = Mvir[n]
        Ms = Mstar[n]
        cluster_file = './clusters/sampledMassiveClusters_{}_halo{}.hdf5'.format(sample,halo_idx)
        if not os.path.exists(cluster_file):
            print('No massive clusters found for halo {} with cluster sampling method {}'.format(halo_idx,sample))
            continue
        with h5py.File(cluster_file) as f:
            tform = f['FormationTime'][()]
            rhm = f['HalfMassRadius'][()]
            mcl = f['Mass'][()]
            Z = f['Metallicity'][()]

        formation_times.extend(tform)
        radii.extend(rhm)
        masses.extend(mcl)
        metallicities.extend(Z)
        halo_mass.extend([Mv] *len(mcl))
        galaxy_mass.extend([Ms] *len(mcl))

    ## Save everything in one single file ##
    print('compileClusters:: Random seed is:', seed)
    if seed:
        outfile = './clusters/Clusters_{}_compiled_seed{}.h5'.format(sample,seed)
    else:
        outfile = './clusters/Clusters_{}_compiled.h5'.format(sample)
    with h5py.File(outfile, 'w') as out:
        out.create_dataset('FormationTime', data=formation_times)
        out.create_dataset('HalfMassRadius', data=radii)
        out.create_dataset('Mass', data=masses)
        out.create_dataset('Metallicity', data=metallicities)
        out.create_dataset('Mgal', data=galaxy_mass)
        out.create_dataset('Mvir', data=halo_mass)


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
    compileClusters(sample=args.sample, seed=args.seed)
