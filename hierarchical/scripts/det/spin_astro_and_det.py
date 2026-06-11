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


def plot_chi_astro_and_det(method='Gamma'):

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

    file_bptp04 = os.path.join(gwtc4_dir, 'data_release/BBHMassSpinRedshift_BrokenPowerLawTwoPeaks_GaussianComponentSpins_PowerLawRedshift.h5')
    file_bs04 = os.path.join(gwtc4_dir, 'data_release/BBHMassSpinRedshift_BSplineIID.h5')
    bptp_o4 = popsummary.popresult.PopulationResult(file_bptp04)
    bs_o4 = popsummary.popresult.PopulationResult(file_bs04)
    z_list, R_z = bptp_o4.get_rates_on_grids('redshift')
    z_list = z_list[0]
    z02 = np.argmin(np.abs(z_list-0.2))
    bs_z_list, bs_R_z = bs_o4.get_rates_on_grids('rate_vs_redshift')
    bs_z_list = bs_z_list[0]
    bptp_chi1, bptp_chi1_pdfs = bptp_o4.get_rates_on_grids('a_1')
    bptp_chi1 = bptp_chi1[0]
    bptp_chi1_pdfs = bptp_chi1_pdfs /np.trapezoid(x=bptp_chi1,y=bptp_chi1_pdfs)[:,None]
    bs_chi1, bs_chi1_pdfs = bs_o4.get_rates_on_grids('p(a)')
    bs_chi1 = bs_chi1[0]
    color1 = '#648FFF'
    color2 = '#FE6100'

    pf.plot_90CI(ax, bs_chi1, bs_chi1_pdfs, color=color1, label=r'B-Spline, GWTC-4.0', fill_alpha=0.4)
    #ax.plot([],[],color=color1, label=r'B-Spline, GWTC-4.0')
    pf.plot_90CI(ax, bptp_chi1, bptp_chi1_pdfs, color=color2, label=r'Broken Power Law + 2 Peaks, GWTC-4.0', fill_alpha=0.4)
    #ax.plot([],[],color=color2, label=r'Broken Power Law + 2 Peaks, GWTC-4.0')


    a1bins = np.linspace(0,1,41)
    bbhfile = os.path.join(bbhs_dir, 'BBHs_{}_seed42.h5'.format(method))
    with h5py.File(bbhfile, 'r') as bbhs:
        m1 = bbhs['m1'][:]
        m2 = bbhs['m2'][:]
        a1 = bbhs['a1'][:]
        a2 = bbhs['a2'][:]
        Mhalo = bbhs['Mhalo'][:]
        zmerger = bbhs['z'][:]
        pdet = bbhs['pdet'][:]

    a1[m1<m2], a2[m1<m2] = a2[m1<m2], a1[m1<m2]
    astro = (zmerger < 1) & (zmerger > 0)
    a1 = a1[astro]
    zmerger = zmerger[astro]
    pdet = pdet[astro]
    pdet = pdet.reshape(len(pdet))
    Mhalo = Mhalo[astro]

    # Compute weight associated to each galaxy in which there's a BBH merger
    weights = np.array([wgal[k-1] for k in np.digitize(np.log10(Mhalo), logmbins)])

    p_a1, _ = np.histogram(a1, bins=a1bins, weights=weights, density=True)
    p_a1 = np.append(0,p_a1)

    ax.plot(a1bins, p_a1, 
            c='forestgreen', 
            label='BBH mergers from GCs (Bruel et al. 2025)',
            )

    ax.set_xlabel(r'$\chi$')
    ax.set_xlim(xmin=0, xmax=1)
    ax.set_xticks(np.linspace(0,1,11))
    ax.set_ylabel(r'$p(\chi)$')
    #ax.set_yscale('log')
    ax.set_ylim(ymin=0, ymax=5)
    ax.legend(loc='upper left', frameon=False)
    

    # Save
    plot_dir = '/home/tristan-bruel/Documents/Science/Great-Balls-of-FIRE/hierarchical/plots'
    plt.savefig(os.path.join(plot_dir,'spin_astro.png'), bbox_inches='tight')


    ### Detectable ###
    fig, ax = plt.subplots(1, 1, figsize=(8.,4.))

    file_ppds_a1 = os.path.join(gwtc3_dir,'ppds_chi.dat')
    a1_grid, ppd_a1_med, ppd_a1_low, ppd_a1_up = np.loadtxt(file_ppds_a1).T
    color_obs = '#648FFF'
    label_obs = 'Inference on observable population\n (Toubiana et al. 2026)'

    ax.fill_between(a1_grid,ppd_a1_low,ppd_a1_up,
            alpha=0.4,color=color_obs)
    ax.plot(a1_grid,ppd_a1_med,
            color=color_obs,label=label_obs)

    z_list = np.linspace(0.,2.,2500)
    dzdt = interp1d(z_list[:-1], np.diff(z_list)/np.diff(planck.lookback_time(z_list)), fill_value='extrapolate')
    dVcdz = interp1d(z_list, (4*np.pi)*planck.differential_comoving_volume(z_list).value*1e-9, fill_value='extrapolate')
    weights *= 1/FIREbox_volume *Tobs/(1+zmerger) *dVcdz(zmerger) *dzdt(zmerger)
    weights *= pdet

    po_a1, _ = np.histogram(a1, bins=a1bins, weights=weights, density=True)
    po_a1 = np.append(0,po_a1)

    ax.plot(a1bins, po_a1,
            c='forestgreen',
            label='BBH mergers from GCs (Bruel et al. 2025)',
            )

    ax.set_xlabel(r'$\chi$')
    ax.set_xlim(xmin=0, xmax=1)
    ax.set_xticks(np.linspace(0,1,11))
    ax.set_ylabel(r'$p_O(\chi)$')
    #ax.set_yscale('log')
    ax.set_ylim(ymin=0, ymax=5)
    ax.legend(loc='upper left', frameon=False)


    # Save
    plt.savefig(os.path.join(plot_dir,'spin_det.png'), bbox_inches='tight')



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
    plot_chi_astro_and_det()
    plt.show()
