import numpy as np
import h5py
import pandas as pd
from scipy.interpolate import interp1d

import os
from glob import glob
import argparse

import gizmo_analysis as gizmo
import utilities as ut
from tool_functions import MassCDF, SampleMassFunc, SampleSizeFunc


def linear(x, a, b):
    return a*x + b


def sampleMassiveClusters(halo_idx, seed=42):
    ### Sample massive clusters in the halo identified by {halo_idx} in snapshot 1200

    print('Sampling massive clusters in halo', halo_idx)
    np.random.seed(seed)
    redshifts, times = np.loadtxt("/data/tbruel/FIREbox/snapshot_times.txt", usecols=[2,3]).T
    tmax = times.max()

    #SFR_file = './SFRs/Mstar90/FB15N1024_halo{}_SFR.txt'.format(halo_idx)
    SFR_file = './SFRs/Rvir/FB15N1024_halo{}_SFR.txt'.format(halo_idx)
    if not os.path.exists(SFR_file):
        print('No SFR file found for this halo.')
        return
    else:
        times, Zmean, Zspread, mform, sfr = np.loadtxt(SFR_file).T

    ## Load fit from my cosmo simulations ##
    massive_fit = np.loadtxt('./fit_Mtot_stellarmass')[:2]
    massive_stdev = np.loadtxt('./fit_Mtot_stellarmass')[2]

    alpha_fit = np.loadtxt('./fit_alpha_stellarmass')[:2]
    alpha_stdev = np.loadtxt('./fit_alpha_stellarmass')[2]

    ## Compute moving SFR over 1 Gyr intervals ##
    sfr_moving = np.zeros_like(sfr)
    for n,t in enumerate(times):
        if t < .5:
            continue
        band = np.abs(times-t) <= .5
        if np.sum(band) > 0:
            sfr_moving[n] = np.mean(sfr[band])
    gamma = np.zeros_like(sfr, dtype='float')
    gamma[sfr_moving>0] = sfr[sfr_moving>0] *(sfr-sfr_moving)[sfr_moving>0] /sfr_moving[sfr_moving>0]
    gamma[gamma < 0] = 0
    gamma = gamma /np.sum(gamma)

    ### Start sampling ###
    # load stellar masses
    Mstar_file = 'Mstar90.dat'
    halo_ids, Mstar = np.loadtxt(Mstar_file, skiprows=1, usecols=[0,3]).T
    #Mstar_file = 'Mstar.dat'
    #halo_ids, Mstar = np.loadtxt(Mstar_file, skiprows=1).T
    halo_ids = halo_ids.astype(np.int64)
    if not halo_idx in halo_ids:
        print('Stellar mass of halo {} not found in {}.'.format(halo_idx, Mstar_file))
        return
    '''
    # Get galaxy stellar masses from Elia's file
    gals = pd.read_hdf('/home/tbruel/scratch/snap1200_circularity_M8_sample.hdf5')
    halo_ids = np.array(gals['halo ID'])
    Mstar = np.array(gals['Mstar_0.1 Rvir']) *1e10 # unit Msun
    '''
    loc = np.where(halo_ids==halo_idx)[0][0]
    Ms = Mstar[loc]
    # Total mass of massive clusters to sample
    Mtot0 = linear(np.log10(Ms), *massive_fit)
    Mtot = np.power(10, np.random.normal(loc=Mtot0, scale=massive_stdev))
    # Slope of the cluster initial mass function
    alpha0 = linear(np.log10(Ms), *alpha_fit)
    alpha_est = np.random.normal(loc=alpha0, scale=alpha_stdev)
    #alpha_est = -2

    Mmin=6e4
    mlim=Mmin
    mcluster_bins = gamma *Mtot
    # Rescale to make sure that we sample clusters only when the mass budget is higher than mlim
    #mcluster_bins[mcluster_bins < mlim] = 0
    #mcluster_bins = mcluster_bins *Mtot /np.sum(mcluster_bins)
    ## Make sure that CFR is never higher than SFR ##
    if np.sum(mcluster_bins > mform) > 0:
        mcluster_bins[mcluster_bins > mform] = mform[mcluster_bins > mform]

    cluster_mass = []
    cluster_radius = []
    cluster_tform = []
    cluster_Z = []
    k=0
    for k in range(1,len(times)):
        seed=(seed*k+k)%(2**32)
        np.random.seed(seed)
        mass_budget = mcluster_bins[k]
        if mass_budget > 0:
            msample = SampleMassFunc(mass_budget, mmin=Mmin, alpha=alpha_est, seed=seed)
            cluster_mass.extend(msample)
            ncl = len(msample)
            random_draws = np.random.rand(ncl)
            tsample = times[k-1] + random_draws *(times[k]-times[k-1])
            cluster_tform.extend(tsample)
            Zsample = (Zmean[k-1] + random_draws *(Zmean[k]-Zmean[k-1])) *10**(np.random.normal(size=msample.shape)*0.1) # additional scatter +/- 0.1 dex
            cluster_Z.extend(Zsample)
            rsample = SampleSizeFunc(msample, Zsample, seed=seed)
            cluster_radius.extend(rsample)
    cluster_mass = np.array(cluster_mass)
    cluster_radius = np.array(cluster_radius)
    cluster_Z = np.array(cluster_Z)
    cluster_tform = np.array(cluster_tform)
    #print('Number of massive clusters sampled: {}'.format(len(cluster_mass)))
    #print('Total mass of massive clusters sampled: {:.2e} Msun'.format(np.sum(cluster_mass)))

    ## Save ##
    cluster_file = './clusters/sampledMassiveClusters_SFR_halo{}.hdf5'.format(halo_idx)
    with h5py.File(cluster_file, 'w') as f:
        f.create_dataset('Mass', data=cluster_mass)
        f.create_dataset('HalfMassRadius', data=cluster_radius)
        f.create_dataset('Metallicity', data=cluster_Z)
        f.create_dataset('FormationTime', data=cluster_tform)

    clusters = {'mass': cluster_mass, 'radius': cluster_radius, 'Z': cluster_Z, 'tform': cluster_tform}
    return clusters


##################################################################
### 
### Run it!
###
##################################################################
if __name__ == "__main__":

    #####################################
    # Define command line options
    parser = argparse.ArgumentParser()
    parser.add_argument('--halo', dest='halo_idx', type=int, help='Halo index at redshift 0.')
    parser.add_argument('--seed', dest='seed', type=int, help='Random seed', default=42)
    args = parser.parse_args()

    #####################################
    clusters = sampleMassiveClusters(halo_idx=args.halo_idx, seed=args.seed)
