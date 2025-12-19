import numpy as np
#import pandas as pd
import h5py 
from scipy.interpolate import interp1d

import os
import argparse

from colossus.cosmology import cosmology
cosmology.setCosmology('planck18');
from colossus.lss import mass_function

def HaloMassFunction():
    dndM = lambda M: mass_function.massFunction(M, 0.0, mdef='vir', model='tinker08', q_out='dndlnM')
    return dndM


def compute_form_rates(seed):
    """
    Compute and save the formation rates associated to the massive clusters in FIREbox
    """
    FIREbox_volume = (22.1)**3 # unit Mpc^3

    redshifts, times = np.loadtxt('./snapshot_times.txt', usecols=[2,3]).T
    tmax = times.max()

    redshifts_to_times = interp1d(redshifts, times, fill_value='extrapolate')
    times_to_redshifts = interp1d(times, redshifts, fill_value='extrapolate')

    deltat = 200 # width of the time bins (in Myr)
    nbins = round(1e3*tmax /deltat) + 1
    tbins = np.linspace(0, tmax, nbins)

    galaxy_file = 'Mstar90.dat'
    Mvir, Mstar = np.loadtxt(galaxy_file, usecols=[1,3], skiprows=1).T

    print('Reweighting galaxies to fit HMF.')
    logMmin=10.
    logMmax=14.
    to_keep = (Mvir>np.power(10,logMmin))&(Mvir<np.power(10,logMmax))
    print('Number of galaxies identified in FIREbox with halo mass in [10^{},10^{}]: {}'.format(logMmin,logMmax,np.sum(to_keep)))
    Mvir = Mvir[to_keep]
    Mstar = Mstar[to_keep]
    obs_HMF = HaloMassFunction() # number density, unit is Mpc^-3
    #logmbins = np.linspace(10., 14., 41)
    nbins = round((logMmax-logMmin)*5 + 1)
    logmbins = np.linspace(logMmin, logMmax, nbins)
    mid_bins = (logmbins[:-1]+logmbins[1:]) /2
    Ngal, _ = np.histogram(np.log10(Mvir), bins=logmbins)
    dngal = Ngal /np.diff(logmbins) /FIREbox_volume
    wgal = np.zeros_like(mid_bins)
    wgal[dngal>0] = obs_HMF(np.power(10,mid_bins))[dngal>0] /dngal[dngal>0]

    for s,sample in enumerate(['Gamma', 'ElBadry', 'SFR']):
        print('\nCluster sampling method:', sample)
        clusterfile = './clusters/Clusters_{}_compiled_seed{}.h5'.format(sample,seed)
        if not os.path.exists(clusterfile):
            print(clusterfile, 'does not exist.\n Try with another sampling method.')
            tform = np.array([])
            Mcl = np.array([])
            Mgal = np.array([])
            Mhalo = np.array([])
        else:
            with h5py.File(clusterfile, 'r') as clusters:
                tform = clusters['FormationTime'][()]
                Mcl = clusters['Mass'][()]
                Mgal = clusters['Mgal'][()]
                Mhalo = clusters['Mvir'][()]

        # Compute weight associated to each galaxy in which there's a massive cluster
        weights = np.array([wgal[k-1] for k in np.digitize(np.log10(Mhalo), logmbins)])

        cluster_nbr, _ = np.histogram(tform, bins=tbins, weights=Mcl*weights)
        formrate = cluster_nbr /np.diff(tbins) /FIREbox_volume # unit Gpc-3 yr-1
        formrate = np.append(0, formrate)

        logMgal = np.log10(Mgal)
        #Mbins = np.append(8, np.linspace(9,11,5))
        #Mbins = np.append(Mbins, 12)
        Mbins = np.linspace(8,12,9)
        all_form_rates = np.zeros((len(tbins),len(Mbins)+1))
        all_form_rates[:,0] = tbins
        all_form_rates[:,-1] = formrate
        for n in range(len(Mbins)-1):
            k = len(Mbins)-2-n
            in_band = (logMgal>=Mbins[k])&(logMgal<Mbins[k+1])
            cluster_nbr, _ = np.histogram(tform[in_band], bins=tbins, weights=Mcl[in_band]*weights[in_band])
            formrate = cluster_nbr /np.diff(tbins) /FIREbox_volume
            formrate = np.append(0, formrate)
            all_form_rates[:,n] = formrate

        output_file = './clusters/formation_rates/cluster_form_rates_{}_seed{}.dat'.format(sample,seed)
        np.savetxt(output_file, all_form_rates)


##################################################################
### 
### Run it!
###
##################################################################
if __name__ == "__main__":
    
    #####################################
    # Define command line options
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', dest='seed', type=int, help='Random seed', default=42)
    args = parser.parse_args()

    #####################################
    # Plot your result
    compute_form_rates(seed=args.seed)
