import numpy as np
import h5py
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from scipy.interpolate import interp1d

import os

from colossus.cosmology import cosmology
cosmology.setCosmology('planck18');
from colossus.lss import mass_function

FIREbox_dir = '/home/tristan-bruel/Documents/Science/GBoF/FIREbox'
FIREbox_volume = (22.1)**3 # in unit Mpc^3

def HaloMassFunction():
    dndM = lambda M: mass_function.massFunction(M, 0.0, mdef='vir', model='tinker08', q_out='dndlnM')
    return dndM


def plot_mergerRates(seed=42):

    snapshot_file = os.path.join(FIREbox_dir, 'snapshot_times.txt')
    redshifts, times = np.loadtxt(snapshot_file, usecols=[2,3]).T

    times_to_redshifts = interp1d(times, redshifts, fill_value='extrapolate')
    redshifts_to_times = interp1d(redshifts, times, fill_value='extrapolate')
    tmax = times.max()
    #deltat = 0.1 # time interval to define plotting bins, in Gyr
    #nbins=round((times.max()-times.min()) /deltat)
    #tbins=np.linspace(times.min(),times.max(),nbins)
    deltat = 200 # width of the time bins (in Myr)
    nbins = round(1e3*tmax /deltat) + 1
    tbins = np.linspace(0, tmax, nbins)

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

    fig, axes = plt.subplots(1, 3, figsize=(10.,3.))

    sampling_methods = np.array(['Gamma', 'ElBadry', 'SFR'])
    sampling_names = np.array([r'Gamma', r'EB18', r'SFRpeak'])
    xtext = np.array([4.1, 3.7, 4.6])
    for s,sample in enumerate(['Gamma', 'ElBadry', 'SFR']):
        print('\nCluster sampling method:', sample)
        ax = axes[s]
        bbhfile = os.path.join(FIREbox_dir, 'BBHs/BBHs_{}_compiled_seed{}.h5'.format(sample,seed))
        with h5py.File(bbhfile, 'r') as bbhs:
                tform = bbhs['tform'][()]
                tmerger = bbhs['tmerger'][()]
                Mgal = bbhs['Mgal'][()]
                Mhalo = bbhs['Mvir'][()]

        # Compute weight associated to each galaxy in which there's a BBH merger
        weights = np.array([wgal[k-1] for k in np.digitize(np.log10(Mhalo), logmbins)])

        merger_nbr, _ = np.histogram(tform, bins=tbins, weights=weights)
        mergerrate = merger_nbr /np.diff(tbins) /FIREbox_volume # unit Gpc-3 yr-1
        mergerrate = np.append(0, mergerrate)
        print(tbins)
        print(mergerrate)
        ax.plot(tmax-tbins, mergerrate, 
                c='k', label=r'Total',
                )

        logMgal = np.log10(Mgal)
        Mgal_bins = np.linspace(8,12,9)[::-1]
        for n in range(len(Mgal_bins)-1):
            inbin = (logMgal<Mgal_bins[n])&(logMgal>=Mgal_bins[n+1])
            merger_nbr, _ = np.histogram(tform[inbin], bins=tbins, weights=weights[inbin])
            mergerrate = merger_nbr /np.diff(tbins) /FIREbox_volume
            mergerrate = np.append(0, mergerrate)
            ax.plot(tmax-tbins, mergerrate, 
                    c=CB_color_cycle[n],
                    label=r'$%.1f\geq\mathrm{log}(\mathcal{M}_\star/\mathrm{M}_\odot)\geq %.1f\ (%i)$' %(Mgal_bins[n],Mgal_bins[n+1],np.sum(inbin)),
                    )

        ax.text(x=xtext[s], y=2e2, s=sampling_names[s], 
                fontsize=fs, c='k',
                bbox=dict(facecolor='none', edgecolor='black', boxstyle='round'),
                )

        ax.set_xlabel(r'Lookback time [Gyr]', fontsize=fs)
        ax.set_xlim(xmin=tmax, xmax=0)
        ax.set_xticks(np.linspace(12,0,7))
        if s==0:
            ax.set_ylabel('BBH merger rate '+r'$[\mathrm{M}_\odot\mathrm{Mpc}^{-3}\mathrm{Gyr}^{-1}]$', fontsize=fs)
        ax.set_yscale('log')
        ax.set_ylim(ymin=1e-3, ymax=1e3)
        ax.set_box_aspect(1)

        # Add time axis
        # convert redshifts to times and convert to Gyr
        ax2 = ax.twiny()
        ax2.set_xlim(ax.get_xlim())
        x2tick_values = np.array([0,0.1,0.5,1,2,3,10])
        x2tick_locations = tmax-redshifts_to_times(x2tick_values)
        ax2.set_xticks(x2tick_locations)
        ax2.set_xticklabels(['0','0.1','0.5','1','2','3','10'])
        ax2.set_xlabel('Redshift', fontsize=fs)
        ax2.set_box_aspect(1)

    axes[1].legend(bbox_to_anchor=(.5,-0.3), loc='upper center', 
                   fontsize=fs, ncol=3,
                   )
    
    # Save
    plot_dir = os.path.join(FIREbox_dir, 'plots')
    plt.savefig(os.path.join(plot_dir,'BBHmergerRates.pdf'), format='pdf', bbox_inches='tight')



##################################################################
### Run it!
###
##################################################################
if __name__ == "__main__":

    plot_mergerRates()
    plt.show()
