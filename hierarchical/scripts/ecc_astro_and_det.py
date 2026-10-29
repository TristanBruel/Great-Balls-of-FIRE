import numpy as np
import h5py
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

import os
import sys
from glob import glob

import popsummary
from astropy.cosmology import Planck15 
from colossus.cosmology import cosmology
cosmology.setCosmology('planck18');
from colossus.lss import mass_function
from astropy.cosmology import Planck18 as planck


def HaloMassFunction():
    dndM = lambda M: mass_function.massFunction(M, 0.0, mdef='vir', model='tinker08', q_out='dndlnM')
    return dndM


def plot_mBH1_astro_and_det(method='Gamma'):

    snapshot_file = os.path.join(FIREbox_dir, 'snapshot_times.txt')
    redshifts, times = np.loadtxt(snapshot_file, usecols=[2,3]).T
    tmax = times.max()
    deltat = 150 # width of the time bins (in Myr)
    nbins = round(1e3*tmax /deltat) + 1
    tbins = np.linspace(0, tmax, nbins)

    times_to_redshifts = interp1d(times, redshifts, fill_value='extrapolate')
    redshifts_to_times = interp1d(redshifts, times, fill_value='extrapolate')
    tz02 = redshifts_to_times(0.2)
    tz1 = redshifts_to_times(1)
    dt_astro = tmax - tz1

    Mstar_file = os.path.join(FIREbox_dir, 'Mstar90.dat')
    Mvir = np.loadtxt(Mstar_file, usecols=1, skiprows=1).T

    logMmin=10.
    logMmax=14.
    to_keep = (Mvir>np.power(10,logMmin))&(Mvir<np.power(10,logMmax))
    print('Number of galaxies identified in FIREbox with halo mass in [10^{},10^{}]: {}'.format(logMmin,logMmax,np.sum(to_keep)))
    Mvir = Mvir[to_keep]

    nbins = round((logMmax-logMmin)*5 + 1)
    logmbins = np.linspace(logMmin, logMmax, nbins)
    mid_bins = (logmbins[:-1]+logmbins[1:]) /2
    Nhalo, _ = np.histogram(np.log10(Mvir), bins=logmbins)
    dnhalo = Nhalo /np.diff(logmbins) /FIREbox_volume
    wgal = np.zeros_like(dnhalo)
    obs_HMF = HaloMassFunction() # number density, unit is Mpc^-3
    wgal[dnhalo>0] = obs_HMF(np.power(10,mid_bins))[dnhalo>0] /dnhalo[dnhalo>0]

    #####################################
    # Plot your result
    fs = 12
    lw = 1.4
    plt.rcParams['font.size']=fs
    plt.rcParams['font.family']='serif'
    plt.rcParams['font.serif']='cmr10'
    plt.rcParams['mathtext.fontset']='cm'
    plt.rcParams['axes.unicode_minus']=False
    plt.rcParams['axes.formatter.use_mathtext']=True
    plt.rcParams['lines.linewidth']=lw
    plt.rcParams['xtick.labelsize']=fs
    plt.rcParams['ytick.labelsize']=fs
    plt.rcParams['legend.fontsize']=.9*fs

    ### Astro ###
    fig, ax = plt.subplots(1, 1, figsize=(8.,4.))

    ebins = np.linspace(0,1,41)
    bbhfile = os.path.join(bbhs_dir, 'BBHs_{}_seed42.h5'.format(method))
    dRde = np.zeros(len(ebins))
    with h5py.File(bbhfile, 'r') as bbhs:
        ecc = bbhs['eccentricity'][:]
        Mhalo = bbhs['Mhalo'][:]
        zmerger = bbhs['z'][:]
        pdet = bbhs['pdet'][:]

    
    astro = (zmerger < 1) & (zmerger > 0)
    ecc = ecc[astro]
    zmerger = zmerger[astro]
    tmerger = redshifts_to_times(zmerger)
    pdet = pdet[astro]
    pdet = pdet.reshape(len(pdet))
    Mhalo = Mhalo[astro]

    # Compute weight associated to each galaxy in which there's a BBH merger
    weights = np.array([wgal[k-1] for k in np.digitize(np.log10(Mhalo), logmbins)])

    hist, _ = np.histogram(ecc, bins=ebins, weights=weights)
    dRde = hist /FIREbox_volume /dt_astro /np.diff(ebins)
    merger_nbr, _ = np.histogram(tmerger, bins=tbins, weights=weights)
    mergerrate = merger_nbr /np.diff(tbins) /FIREbox_volume # unit Gpc-3 yr-1
    ind_z02 = np.where(tbins>tz02)[0][0]
    Rz02 = mergerrate[ind_z02]
    print('Merger rate density computed at redshift 0.2: %.2f Gpc^-3 yr^-1' %(Rz02))
    dRde *= Rz02 /np.sum(dRde *np.diff(ebins))
    dRde = np.append(0,dRde)

    ax.plot(ebins, dRde, 
            c='forestgreen', 
            label='BBH mergers from GCs (Bruel et al. 2025)',
            )

    ax.set_xlabel(r'$e_\mathrm{10Hz}$')
    ax.set_xlim(xmin=0, xmax=1)
    ax.set_xticks(np.linspace(0,1,11))
    ax.set_ylabel(r'$\mathrm{d}\mathcal{R}/\mathrm{d}e\ [\mathrm{Gpc}^{-3} \mathrm{yr}^{-1}]$')
    ax.set_yscale('log')
    ax.set_ylim(ymin=1e-2, ymax=1e3)
    ax.legend(loc='upper right', frameon=False)
    

    # Save
    plot_dir = '/home/tristan-bruel/Documents/Science/Great-Balls-of-FIRE/hierarchical/plots'
    plt.savefig(os.path.join(plot_dir,'ecc_astro.png'), bbox_inches='tight')


    ### Detectable ###
    fig, ax = plt.subplots(1, 1, figsize=(8.,4.))

    z_list = np.linspace(0.,2.,2500)
    dzdt = interp1d(z_list[:-1], np.diff(z_list)/np.diff(planck.lookback_time(z_list)), fill_value='extrapolate')
    dVcdz = interp1d(z_list, (4*np.pi)*planck.differential_comoving_volume(z_list).value*1e-9, fill_value='extrapolate')
    weights *= 1/FIREbox_volume *Tobs/(1+zmerger) *dVcdz(zmerger) *dzdt(zmerger)
    weights *= pdet

    hist, _ = np.histogram(ecc, bins=ebins, weights=weights)
    dRode = hist /np.diff(ebins)
    dRode = np.append(0,dRode)
    print('Total observable merger rate: %.2f yr^-1' %np.trapezoid(x=ebins,y=dRode))

    ax.plot(ebins, dRode,
            c='forestgreen',
            label='BBH mergers from GCs (Bruel et al. 2025)',
            )

    ax.set_xlabel(r'$e_\mathrm{10Hz}$')
    ax.set_xlim(xmin=0, xmax=1)
    ax.set_xticks(np.linspace(0,1,11))
    ax.set_ylabel(r'$\mathrm{d}\mathcal{R}_O/\mathrm{d}e\ [\mathrm{yr}^{-1}]$')
    ax.set_yscale('log')
    ax.set_ylim(ymin=1e-1, ymax=2e3)
    ax.legend(loc='upper right', frameon=False)


    # Save
    plt.savefig(os.path.join(plot_dir,'e_det.png'), bbox_inches='tight')



##################################################################
### Run it!
###
##################################################################
if __name__ == "__main__":

    FIREbox_dir = '/home/tristan-bruel/Documents/Science/Great-Balls-of-FIRE/FIREbox'
    FIREbox_volume = (22.1)**3 # in unit Mpc^3

    bbhs_dir = '/home/tristan-bruel/Documents/Science/Great-Balls-of-FIRE/hierarchical/BBHs'
    gwtc4_dir = '/home/tristan-bruel/Documents/Science/LVK/o4a-astro'
    sys.path.append(os.path.join(gwtc4_dir,'figure_scripts'))
    import plot_funcs_bbh_mass as pf

    gwtc3_dir = '/home/tristan-bruel/Documents/Science/LVK/obs_space/gwtc3'

    Tobs = 1 # 1 year of observation time
    plot_mBH1_astro_and_det()
    plt.show()
