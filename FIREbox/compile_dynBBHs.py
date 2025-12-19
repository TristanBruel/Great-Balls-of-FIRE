import numpy as np
import h5py
#import pandas as pd

import os
import argparse


def compileBBHs(sample, seed=None):
    print('compileBBHs:: Random seed is', seed)

    # load galaxy stellar masses
    #Mstar_file = 'Mstar90.dat'
    #halo_ids, Mvir, Mstar = np.loadtxt(Mstar_file, skiprows=1, usecols=[0,1,3]).T
    #halo_ids = halo_ids.astype(np.int64)
    Mstar_file = 'Mstar90_v2.dat'
    halo_ids, Mvir, Mstar = np.loadtxt(Mstar_file, skiprows=1, usecols=[0,1,3]).T
    halo_ids = halo_ids.astype(np.int64)
    '''
    # Get galaxy stellar masses from Elia's file
    gals = pd.read_hdf('/home/tbruel/scratch/snap1200_circularity_M8_sample.hdf5')
    halos = np.array(gals['halo ID'])
    Mstar = np.array(gals['Mstar_0.1 Rvir']) *1e10 # unit Msun
    ''' 
    cluster_id = []
    mass_BH1 = []
    mass_BH2 = []
    formation_times = []
    metallicities = []
    merger_times = []
    spin_BH1 = []
    spin_BH2 = []
    halo_mass = []
    galaxy_mass = []
    for n,halo_idx in enumerate(halo_ids):
        BBH_file = './BBHs/sampledBBHs_{}_halo{}_Kremer.dat'.format(sample,halo_idx)
        Mv = Mvir[n]
        Ms = Mstar[n]
        try:
            #Zform, tform, m1, m2, tmerger = np.loadtxt(BBH_file, usecols=[1,3,5,6,8]).T
            Zform, tform, m1, m2, tmerger, a1, a2 = np.loadtxt(BBH_file, usecols=[1,3,5,6,8,9,10], skiprows=1).T
            clid = np.loadtxt(BBH_file, usecols=[0], skiprows=1, dtype='str').T
        except:
            print('No BBH found for halo {} with cluster sampling method {}'.format(halo_idx,sample))
            continue

        if np.isscalar(m1):
            clid = np.array([clid])
            m1 = np.array([m1])
            m2 = np.array([m2])
            tform = np.array([tform])
            Zform = np.array([Zform])
            tmerger = np.array([tmerger])
            a1 = np.array([a1])
            a2 = np.array([a2])
        cluster_id.extend(clid)
        mass_BH1.extend(m1)
        mass_BH2.extend(m2)
        formation_times.extend(tform)
        metallicities.extend(Zform)
        merger_times.extend(tmerger)
        spin_BH1.extend(a1)
        spin_BH2.extend(a2)
        halo_mass.extend([Mv] *len(m1))
        galaxy_mass.extend([Ms] *len(m1))

    ## Save everything in one single file ##
    if seed:
        outfile = './BBHs/BBHs_{}_compiled_withKremer_seed{}.h5'.format(sample,seed)
    else:
        outfile = './BBHs/BBHs_{}_compiled_withKremer.h5'.format(sample)
    with h5py.File(outfile, 'w') as out:
        out.create_dataset('id', data=cluster_id, dtype=h5py.string_dtype())
        out.create_dataset('m1', data=mass_BH1)
        out.create_dataset('m2', data=mass_BH2)
        out.create_dataset('tform', data=formation_times)
        out.create_dataset('Z', data=metallicities)
        out.create_dataset('tmerger', data=merger_times)
        out.create_dataset('a1', data=spin_BH1)
        out.create_dataset('a2', data=spin_BH2)
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
    compileBBHs(sample=args.sample, seed=args.seed)
