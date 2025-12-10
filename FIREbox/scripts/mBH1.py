import numpy as np
import h5py
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
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


def plot_mBH1(seed=3):

    snapshot_file = os.path.join(FIREbox_dir, 'snapshot_times.txt')
    redshifts, times = np.loadtxt(snapshot_file, usecols=[2,3]).T
    tmax = times.max()
    deltat = 150 # width of the time bins (in Myr)
    nbins = round(1e3*tmax /deltat) + 1
    tbins = np.linspace(0, tmax, nbins)

    times_to_redshifts = interp1d(times, redshifts, fill_value='extrapolate')
    redshifts_to_times = interp1d(redshifts, times, fill_value='extrapolate')
    tz1 = redshifts_to_times(1)
    dt_astro = tmax - tz1

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

    ### ADD BBH MERGER RATE DENSITY LVK GWTC-3 ##
    LVK_dir = '/home/tristan-bruel/Documents/Science/LVK'
    color_plpeak = "#1f78b4"
    O3_only_PPD = os.path.join(LVK_dir, 'analyses/PowerLawPeak/o1o2o3_mass_c_iid_mag_iid_tilt_powerlaw_redshift_mass_data.h5')
    mass_1 = np.linspace(2, 100, 1000)
    mass_ratio = np.linspace(0.1, 1, 500)
    with h5py.File(O3_only_PPD, "r") as f:
        mass_ppd = f["ppd"]
        mass_lines = f["lines"]
        mass_1_ppd = np.trapezoid(mass_ppd, mass_ratio, axis=0)
        mass_r_ppd = np.trapezoid(mass_ppd, mass_1, axis=-1)
        mass_1_lower = np.percentile(mass_lines["mass_1"], 5, axis=0)
        mass_1_upper = np.percentile(mass_lines["mass_1"], 95, axis=0)
        mass_r_lower = np.percentile(mass_lines["mass_ratio"], 5, axis=0)
        mass_r_upper = np.percentile(mass_lines["mass_ratio"], 95, axis=0)

    ax.plot(
        mass_1, mass_1_ppd, lw=lw, color=color_plpeak, zorder=1, label="GWTC-3"
    )
    ax.plot(mass_1, mass_1_lower, lw=lw-1, color=color_plpeak, zorder=1, alpha=0.3)
    ax.plot(mass_1, mass_1_upper, lw=lw-1, color=color_plpeak, zorder=1, alpha=0.3)
    ax.fill_between(
        mass_1,
        mass_1_lower,
        mass_1_upper,
        alpha=0.14,
        color=color_plpeak,
        zorder=0,
    )
    patch = Patch(color=color_plpeak, alpha=0.14)

    locpeak = np.argmax(mass_1_ppd[mass_1>20])
    xpeak = mass_1[mass_1>20][locpeak]
    ypeak = mass_1_ppd[mass_1>20][locpeak]
    ymin = 5e-4
    ymax = 1e1
    frac_peak = (np.log10(ypeak)-np.log10(ymin)) /(np.log10(ymax)-np.log10(ymin))
    ax.axvline(x=xpeak, ymin=0, ymax=frac_peak, 
               c='k', linestyle='--', lw=lw)


    m1bins = np.linspace(0,200,101)

    sampling_methods = np.array(['Gamma', 'ElBadry', 'SFR'])
    sampling_names = np.array([r'Gamma', r'EB18', r'SFRpeak'])
    sampling_colors = np.array(['gold', 'darkorange', 'firebrick'])
    for s,sample in enumerate(['Gamma', 'ElBadry', 'SFR']):
        print('\nCluster sampling method:', sample)
        bbhfile = os.path.join(FIREbox_dir, 'BBHs/BBHs_{}_compiled_seed{}.h5'.format(sample,seed))
        with h5py.File(bbhfile, 'r') as bbhs:
            mBH1 = bbhs['m1'][()]
            mBH2 = bbhs['m2'][()]
            tmerger = bbhs['tmerger'][()]
            Mgal = bbhs['Mgal'][()]
            Mhalo = bbhs['Mvir'][()]
            cluster_id = bbhs['id'][()]

        m1 = np.maximum(mBH1, mBH2)
        astro = (tmerger > tz1) & (tmerger < tmax)
        m1 = m1[astro]
        tmerger = tmerger[astro]
        Mgal = Mgal[astro]
        Mhalo = Mhalo[astro]
        cluster_id = np.astype(cluster_id[astro], str)

        # Compute weight associated to each galaxy in which there's a BBH merger
        weights = np.array([wgal[k-1] for k in np.digitize(np.log10(Mhalo), logmbins)])

        merger_nbr, _ = np.histogram(tmerger, bins=tbins, weights=weights)
        mergerrate = merger_nbr /np.diff(tbins) /FIREbox_volume # unit Gpc-3 yr-1
        norm = mergerrate[-1]
        print('R(z=0) = %.1f Gpc^-3yr^-1' %norm)
        Nmassive = np.sum(weights[m1>100])
        #Nmassive = np.sum(m1>100)
        print('Number of BBH mergers with M1 > 100 Msun at z<1: {:.0f}'.format(Nmassive))
        Nclusters = len(np.unique(cluster_id[m1>100]))
        print('Unweighted number of BBH mergers with M1 > 100 Msun at z<1: {:.0f}'.format(np.sum(m1>100)))
        print('Number of unique clusters from which these BBHs were sampled: {:.0f}'.format(Nclusters))
        print(np.unique(cluster_id[m1>100]))

        hist, _ = np.histogram(m1, bins=m1bins, weights=weights)
        dRdm1 = hist /FIREbox_volume /dt_astro /np.diff(m1bins)
        dRdm1 *= norm /np.sum(dRdm1 *np.diff(m1bins))
        ax.stairs(dRdm1, m1bins, 
                  color=sampling_colors[s], lw=lw, label=sampling_names[s],
                  )


    ax.set_xlabel(r'$m_1\ [\mathrm{M}_\odot]$', fontsize=fs)
    ax.set_xlim(xmin=0, xmax=170)
    xticks = np.linspace(0,150,7)
    xticks[1] = 34
    ax.set_xticks(xticks)
    ax.set_ylabel(r'$\mathrm{d}\mathcal{R}/\mathrm{d}m_1\ [\mathrm{Gpc}^{-3}\mathrm{yr}^{-1}\mathrm{M}_\odot^{-1}]$', fontsize=fs)
    ax.set_yscale('log')
    ax.set_ylim(ymin=ymin, ymax=ymax)
    ax.set_box_aspect(1)
    handles, labels = ax.get_legend_handles_labels()
    handles[0] = (patch, handles[0])
    ax.legend(handles=handles, labels=labels,
              loc='upper right', fontsize=fs,
              )

    
    # Save
    plot_dir = os.path.join(FIREbox_dir, 'plots')
    plt.savefig(os.path.join(plot_dir,'mBH1_box.pdf'), format='pdf', bbox_inches='tight')



##################################################################
### Run it!
###
##################################################################
if __name__ == "__main__":

    #####################################
    # Define command line option for the path to the simulations and the path to save the plots
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", dest='seed', help="Random seed used to sample massive clusters in FIREbox", type=int, default=42)
    args = parser.parse_args()

    plot_mBH1(seed=args.seed)
    plt.show()
