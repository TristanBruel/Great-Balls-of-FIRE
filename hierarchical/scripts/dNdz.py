import numpy as np
import h5py, json
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

import os
import sys
from glob import glob

from colossus.cosmology import cosmology
cosmology.setCosmology('planck18');
from colossus.lss import mass_function
from astropy.cosmology import Planck18 as planck

import popsummary
dir_GWTC4 = '/home/tristan-bruel/Documents/Science/LVK/o4a-astro'
sys.path.append(os.path.join(dir_GWTC4,'figure_scripts'))
import plot_funcs_bbh_mass as pf



def HaloMassFunction():
    dndM = lambda M: mass_function.massFunction(M, 0.0, mdef='vir', model='tinker08', q_out='dndlnM')
    return dndM


def plot_dNdz(sample='SFR'):

    snapshot_file = os.path.join(FIREbox_dir, 'snapshot_times.txt')
    redshifts, times = np.loadtxt(snapshot_file, usecols=[2,3]).T
    zastro = 2.1
    times_to_redshifts = interp1d(times, redshifts, fill_value='extrapolate')
    redshifts_to_times = interp1d(redshifts, times, fill_value='extrapolate')

    ## Re-weight galaxies in FIREbox to match observed HMF ##
    Mstar_file = os.path.join(FIREbox_dir, 'Mstar90_v2.dat')
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

    ## Load GWTC-4 results ##
    file_powerlaw = os.path.join(dir_GWTC4, 'data_release/BBHMassSpinRedshift_BrokenPowerLawTwoPeaks_GaussianComponentSpins_PowerLawRedshift.h5')
    file_bptp04 = os.path.join(dir_GWTC4, 'data_release/BBHMassSpinRedshift_BrokenPowerLawTwoPeaks_GaussianComponentSpins_PowerLawRedshift.h5')
    file_bs04 = os.path.join(dir_GWTC4, 'data_release/BBHMassSpinRedshift_BSplineIID.h5')
    result = popsummary.popresult.PopulationResult(file_powerlaw)
    bptp_o4 = popsummary.popresult.PopulationResult(file_bptp04)
    bs_o4 = popsummary.popresult.PopulationResult(file_bs04)

    color1 = '#648FFF'
    color2 = '#FE6100'

    z_list, R_z = result.get_rates_on_grids('redshift')
    z_list = z_list[0]
    bs_z_list, bs_R_z = bs_o4.get_rates_on_grids('rate_vs_redshift')
    bs_z_list = bs_z_list[0]

    # Extrapolate some cosmology quantities (dz/dt and dVc/dz)
    dzdt = interp1d(z_list[:-1], np.diff(z_list)/np.diff(planck.lookback_time(z_list).value*1e9), fill_value='extrapolate') # in unit yr^-1
    dVcdz = interp1d(z_list, (4*np.pi)*planck.differential_comoving_volume(z_list).value*1e-9, fill_value='extrapolate') # in unit Gpc^3

    
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

    fig, ax = plt.subplots(1, 1, figsize=(4,4))

    zbins = np.linspace(0, 3, 101)

    # Let's go #
    sampling_methods = np.array(['Gamma', 'ElBadry', 'SFR'])
    sampling_names = np.array([r'Gamma', r'EB18', r'SFRpeak'])
    s = np.where(sampling_methods==sample)[0]
    print('\nCluster sampling method:', sample)
    densities = []
    #for seed in np.linspace(1,20,20, dtype='int'):
    #    bbhfile = os.path.join(FIREbox_dir, 'BBHs/BBHs_{}_compiled_seed{}.h5'.format(sample,seed))
    bbhfiles = glob(os.path.join(FIREbox_dir, 'BBHs/BBHs_{}_compiled_seed*.h5'.format(sample)))
    for f in bbhfiles:
        print('\n',f)
        with h5py.File(f, 'r') as bbhs:
            m1 = bbhs['m1'][()]
            m2 = bbhs['m2'][()]
            a1 = bbhs['a1'][()]
            a2 = bbhs['a2'][()]
            tmerger = bbhs['tmerger'][()]
            Mgal = bbhs['Mgal'][()]
            Mhalo = bbhs['Mvir'][()]
        zmerger = times_to_redshifts(tmerger)
        astro = (zmerger>0) & (zmerger<=zastro)
        print('Unweighted number of BBH mergers at z<2: %i' %np.sum(astro))
        m1 = m1[astro]
        m2 = m2[astro]
        a1 = a1[astro]
        a2 = a2[astro]
        Mhalo = Mhalo[astro]
        tmerger = tmerger[astro]
        zmerger = zmerger[astro]

        # Compute weight associated to each galaxy in which there's a BBH merger
        weights = np.array([wgal[k-1] for k in np.digitize(np.log10(Mhalo), logmbins)])

        # Transform merger rate into dN/dz
        weights *= 1/(FIREbox_volume*1e-9) *Tobs/(1+zmerger) *dVcdz(zmerger) *dzdt(zmerger)
        print('Estimated number of BBH mergers up to z=%.1f over %.1f years of observation time at LIGO design sensitivity' %(zastro,Tobs))
        print('%i' %np.sum(weights))
        dNdz, _ = np.histogram(zmerger, bins=zbins, weights=weights)
        dNdz = np.append(0,dNdz/np.diff(zbins))
        densities.append(dNdz)
        #Ntot = np.trapezoid(x=zbins, y=dNdz)
        #print('Integral of dN/dz up to redshift %.1f: %i' %(zastro,Ntot))


    densities = np.asarray(densities)
    R_lower = np.percentile(densities, 10, axis=0)
    R_upper = np.percentile(densities, 90, axis=0)

    ax.plot(zbins, np.percentile(densities, 50, axis=0), 
            c='k', lw=lw+0.5, label='Massive star clusters in FIREbox',
            )
    ax.fill_between(zbins, R_lower, R_upper,
                    alpha=0.4, color='gray',
                    )
    

    ### ADD BBH REDSHIFT DISTRIBUTION FROM GWTC-4 ##
    dNdz = bs_R_z *Tobs/(1+bs_z_list) *dVcdz(bs_z_list)
    ax.plot(bs_z_list, np.percentile(dNdz,50,axis=0),
            c=color1, linestyle='solid', label='\\textsc{B-Spline}, GWTC-4.0', zorder=1,
            )
    ax.fill_between(bs_z_list, np.percentile(dNdz, 5, axis=0), np.percentile(dNdz, 95, axis=0), color=color1, alpha=0.3, zorder=2)
    Ntot = np.trapezoid(x=bs_z_list, y=np.percentile(dNdz,50,axis=0))
    print('Total number of BBH mergers up to redshift %.1f (B-Spline, GWTC-4.0): %i' %(bs_z_list[-1],Ntot))

    dNdz = R_z *Tobs/(1+z_list) *dVcdz(z_list)
    ax.plot(z_list, np.percentile(dNdz,50,axis=0),
            color=color2, label='\\textsc{Power Law Redshift}, GWTC-4.0',
            )
    ax.fill_between(z_list, np.percentile(dNdz, 5, axis=0), np.percentile(dNdz, 95, axis=0), color=color2, alpha=0.3)
    Ntot = np.trapezoid(x=z_list, y=np.percentile(dNdz,50,axis=0))
    print('Total number of BBH mergers up to redshift %.1f (GWTC-4.0): %i' %(z_list[-1],Ntot))

    ax.set_xlabel(r'Redshift', fontsize=fs)
    ax.set_xlim(xmin=0, xmax=2)
    #ax.set_xticks(np.linspace(12,0,7))
    ax.set_ylabel(r'$\mathrm{dN}/\mathrm{d}z$', fontsize=fs)
    ax.set_yscale('log')
    ax.set_ylim(ymin=1e0, ymax=1e6)
    ax.set_yticks(np.logspace(0,6,7))
    ax.set_box_aspect(1)
    ax.legend(loc='lower right', fontsize=fs)
   
    # Save
    plot_dir = '/home/tristan-bruel/Documents/Science/Great-Balls-of-FIRE/hierarchical/plots'
    plt.savefig(os.path.join(plot_dir,'dNdz_%s.png' %sample), bbox_inches='tight')



##################################################################
### Run it!
###
##################################################################
if __name__ == "__main__":

    FIREbox_dir = '/home/tristan-bruel/Documents/Science/Great-Balls-of-FIRE/FIREbox'
    FIREbox_volume = (22.1)**3 # in unit Mpc^3
    Tobs = 1 # LVK observation time in yrs

    plot_dNdz()
    plt.show()
