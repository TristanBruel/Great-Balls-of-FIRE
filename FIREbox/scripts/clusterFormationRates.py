import numpy as np
import h5py
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

import os
from glob import glob

from colossus.cosmology import cosmology
cosmology.setCosmology('planck18');
from colossus.lss import mass_function

FIREbox_dir = '/home/tristan-bruel/Documents/Science/GBoF/FIREbox'
FIREbox_volume = (22.1)**3 # in unit Mpc^3

def HaloMassFunction():
    dndM = lambda M: mass_function.massFunction(M, 0.0, mdef='vir', model='tinker08', q_out='dndlnM')
    return dndM


def plot_formationRates():

    snapshot_file = os.path.join(FIREbox_dir, 'snapshot_times.txt')
    redshifts, times = np.loadtxt(snapshot_file, usecols=[2,3]).T

    times_to_redshifts = interp1d(times, redshifts, fill_value='extrapolate')
    redshifts_to_times = interp1d(redshifts, times, fill_value='extrapolate')
    tmax = times.max()
    deltat = 150 # width of the time bins (in Myr)
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

    fig, axes = plt.subplots(1, 3, figsize=(8.25,2.35))

    sampling_methods = np.array(['Gamma', 'ElBadry', 'SFR'])
    sampling_names = np.array([r'Gamma', r'EB18', r'SFRpeak'])
    xtext = np.array([4.2, 3.1, 4.7])
    for s,sample in enumerate(sampling_methods):
        ax = axes[s]
        print('\nCluster sampling method:', sample)
        formRates = []
        #for seed in np.linspace(1,20,20, dtype='int'):
        #    clusterfile = os.path.join(FIREbox_dir, 'clusters/Clusters_{}_compiled_seed{}.h5'.format(sample,seed))
        for clusterfile in glob(os.path.join(FIREbox_dir, 'clusters/Clusters_{}_compiled_seed*.h5'.format(sample))):
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
            formRates.append(formrate)

        formRates = np.asarray(formRates)
        R_lower = np.percentile(formRates, 10, axis=0)
        R_upper = np.percentile(formRates, 90, axis=0)

        ax.plot(tmax-tbins, np.mean(formRates, axis=0), 
                c='k', lw=lw+0.5, label=r'Total',
                )
        ax.fill_between(tmax-tbins, R_lower, R_upper,
                        alpha=0.4, color='gray',
                        )

        logMgal = np.log10(Mgal)
        Mgal_bins = np.linspace(8,12,9)[::-1]
        for n in range(len(Mgal_bins)-1):
            inbin = (logMgal<Mgal_bins[n])&(logMgal>=Mgal_bins[n+1])
            cluster_nbr, _ = np.histogram(tform[inbin], bins=tbins, weights=Mcl[inbin]*weights[inbin])
            formrate = cluster_nbr /np.diff(tbins) /FIREbox_volume
            formrate = np.append(0, formrate)
            ax.plot(tmax-tbins, formrate, 
                    c=CB_color_cycle[n],
                    label=r'$%.1f\geq\mathrm{log}(\mathcal{M}_\star/\mathrm{M}_\odot)\geq %.1f$' %(Mgal_bins[n],Mgal_bins[n+1]),
                    )

        ax.text(x=xtext[s], y=2e7, s=sampling_names[s], 
                fontsize=fs, c='k',
                bbox=dict(facecolor='none', edgecolor='black', boxstyle='round'),
                )

        ax.set_xlabel(r'Lookback time [Gyr]', fontsize=fs)
        ax.set_xlim(xmin=tmax, xmax=0)
        ax.set_xticks(np.linspace(12,0,7))
        if s==0:
            ax.set_ylabel('Massive cluster formation rate\n'+r'$[\mathrm{M}_\odot\mathrm{Mpc}^{-3}\mathrm{Gyr}^{-1}]$', fontsize=fs)
        ax.set_yscale('log')
        ax.set_ylim(ymin=1e2, ymax=1e8)
        ax.set_yticks(np.logspace(2,8,7))
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

    axes[1].legend(bbox_to_anchor=(.4,-0.3), loc='upper center', 
                   fontsize=fs, ncol=3,
                   )
    
    # Save
    plot_dir = os.path.join(FIREbox_dir, 'plots')
    plt.savefig(os.path.join(plot_dir,'clusterFormationRates.pdf'), format='pdf', bbox_inches='tight')



##################################################################
### Run it!
###
##################################################################
if __name__ == "__main__":

    plot_formationRates()
    plt.show()
