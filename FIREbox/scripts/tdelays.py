import numpy as np
import h5py
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

import os
import argparse

from colossus.cosmology import cosmology
cosmology.setCosmology('planck18');
from colossus.lss import mass_function

FIREbox_dir = '/home/tristan-bruel/Documents/Science/GBoF/FIREbox'
FIREbox_volume = (22.1)**3 # in unit Mpc^3

def HaloMassFunction():
    dndM = lambda M: mass_function.massFunction(M, 0.0, mdef='vir', model='tinker08', q_out='dndlnM')
    return dndM


def plot_tdelays(seed):

    snapshot_file = os.path.join(FIREbox_dir, 'snapshot_times.txt')
    redshifts, times = np.loadtxt(snapshot_file, usecols=[2,3]).T
    tmax = times.max()
    deltat = 250 # width of the time bins (in Myr)
    nbins = round(1e3*tmax /deltat) + 1
    tbins = np.linspace(0, tmax, nbins)

    times_to_redshifts = interp1d(times, redshifts, fill_value='extrapolate')
    redshifts_to_times = interp1d(redshifts, times, fill_value='extrapolate')
    tz1 = redshifts_to_times(1)
    tz05 = redshifts_to_times(0.5)

    Mstar_file = os.path.join(FIREbox_dir, 'Mstar90.dat')
    haloids, Mvir, Mstar = np.loadtxt(Mstar_file, usecols=[0,1,3], skiprows=1).T
    haloids = haloids.astype(np.int64)

    logMmin=10.
    logMmax=14.
    to_keep = (Mvir>np.power(10,logMmin))&(Mvir<np.power(10,logMmax))
    print('Number of galaxies identified in FIREbox with halo mass in [10^{},10^{}]: {}'.format(logMmin,logMmax,np.sum(to_keep)))
    Mvir = Mvir[to_keep]
    Mstar = Mstar[to_keep]
    haloids = haloids[to_keep]

    nbins = round((logMmax-logMmin)*5 + 1)
    logmbins = np.linspace(logMmin, logMmax, nbins)
    mid_bins = (logmbins[:-1]+logmbins[1:]) /2
    Nhalo, _ = np.histogram(np.log10(Mvir), bins=logmbins)
    dnhalo = Nhalo /np.diff(logmbins) /FIREbox_volume
    wgal = np.zeros_like(dnhalo)
    obs_HMF = HaloMassFunction() # number density, unit is Mpc^-3
    wgal[dnhalo>0] = obs_HMF(np.power(10,mid_bins))[dnhalo>0] /dnhalo[dnhalo>0]
    

    sampling_methods = np.array(['Gamma', 'ElBadry', 'SFR'])
    sampling_names = np.array([r'Gamma', r'EB18', r'SFRpeak'])
    sampling_colors = np.array(['gold', 'darkorange', 'firebrick'])
    hatches = ['/','\\','o']
    for s,sample in enumerate(sampling_methods):
        print('\nCluster sampling method:', sample)
        bbhfile = os.path.join(FIREbox_dir, 'BBHs/BBHs_{}_compiled_seed{}.h5'.format(sample,seed))
        with h5py.File(bbhfile, 'r') as bbhs:
            mBH1 = bbhs['mBH1'][()]
            mBH2 = bbhs['mBH2'][()]
            tmerger = bbhs['tmerger'][()]
            tform = bbhs['tform'][()]
            Mgal = bbhs['Mgal'][()]
            Mhalo = bbhs['Mvir'][()]

        # Compute weight associated to each galaxy in which there's a BBH merger
        weights = np.array([wgal[k-1] for k in np.digitize(np.log10(Mhalo), logmbins)])

        ## Only the BBH mergers at z<0.5
        astro = (tmerger > tz05) & (tmerger < tmax)
        print('Total number of BBH mergers at z<0.5:', np.sum(weights[astro]))
        R = np.sum(weights[astro]) /FIREbox_volume /(tmax-tz05)
        print('Corresponding merger rate density: %.1f Gyr^-1 Mpc^-3' %R)
        early = astro & (tform < tz1)
        print('Number of BBH mergers at z<0.5 from cluster formed at z>1:', np.sum(weights[early]))
        fraction = np.sum(weights[early]) /np.sum(weights[astro]) *100
        print('Fraction: %.1f' %fraction)


    #####################################
    # Plot your result
    fs = 10
    lw = 1.4
    plt.rc('font', family='serif')
    plt.rc('lines', linewidth=lw)
    plt.rc('text', usetex=True)

    CB_color_cycle = ['#377eb8', '#ff7f00', '#4daf4a',
                  '#f781bf', '#a65628', '#984ea3',
                  '#999999', '#e41a1c', '#dede00']

    fig, ax = plt.subplots(1, 1, figsize=(3.,3.))

    tdelay = tmerger-tform
    logMgal = np.log10(Mgal)
    Mgal_bins = np.array([12,11,10,9,8])
    for n in range(len(Mgal_bins)-1):
        inbin = astro & (logMgal<Mgal_bins[n]) & (logMgal>=Mgal_bins[n+1])
        hist, _ = np.histogram(tdelay[inbin], bins=tbins, weights=weights[inbin], density=True)
        ax.stairs(hist, tbins,
                  color=CB_color_cycle[n], 
                  label=r'$%.1f\geq\mathrm{log}(\mathcal{M}_\star/\mathrm{M}_\odot)\geq %.1f$' %(Mgal_bins[n],Mgal_bins[n+1]),
                  )

        #hist, _ = np.histogram(tdelay[astro], bins=tbins, weights=weights[astro], density=True)
        #ax.stairs(hist, tbins,
        #          color=sampling_colors[s], 
        #          fill=True, alpha=0.3,
        #          label=r'$z<0.5$ mergers',
        #          )

    #ax.axvline(x=tz05-tz1, color='k', linestyle='--', linewidth=lw)


    ax.set_xlabel(r'$t_\mathrm{delay}\ [\mathrm{Gyr}]$', fontsize=fs)
    ax.set_xlim(xmin=0, xmax=tmax)
    ax.set_ylabel(r'$p(t_\mathrm{delay})\ [\mathrm{Gyr}^{-1}]$', fontsize=fs)
    ax.set_yscale('log')
    ax.set_ylim(ymin=1e-3, ymax=1e1)
    ax.set_box_aspect(1)
    ax.legend(loc='upper right', fontsize=fs)

    
    # Save
    plot_dir = os.path.join(FIREbox_dir, 'plots')
    #plt.savefig(os.path.join(plot_dir,'tdelays.pdf'), format='pdf', bbox_inches='tight')



##################################################################
### Run it!
###
##################################################################
if __name__ == "__main__":

    #####################################
    # Define command line option for the path to the simulations and the path to save the plots
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", dest='seed', help="Random seed used to sample massive clusters in FIREbox", type=int, default=3)
    args = parser.parse_args()

    plot_tdelays(seed=args.seed)
    plt.show()
