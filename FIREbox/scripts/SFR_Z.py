import numpy as np
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

def Madau(z, a=0.015, b=2.7, c=2.9, d=5.6):
    sfr = a * ((1 + z)**b) / (1 + ((1 + z) / c)**d)
    return sfr


def plot_SFR_Zdist(Zsun=0.017):

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
    weights = np.array([wgal[k-1] for k in np.digitize(np.log10(Mvir), logmbins)])

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

    fig, axes = plt.subplots(1, 2, figsize=(8.,3.))

    Mforms = np.zeros((len(haloids),len(times)))
    Zmeans = np.zeros((len(haloids),len(times)))
    for k,h in enumerate(haloids):
        SFR_file = os.path.join(FIREbox_dir, 'SFRs/FB15N1024_halo{}_SFR.txt'.format(h))
        t, Z, sigmaZ, mform, sfr = np.loadtxt(SFR_file).T
        Mforms[k,:] = mform *weights[k]
        Zmeans[k,:] = Z *mform *weights[k]

    ## Left-hand plot: SFRs per bins of galaxy mass ##
    ax = axes[0]

    Mform_tot = np.sum(Mforms, axis=0)
    Mform_plot, _ = np.histogram(times, weights=Mform_tot, bins=tbins)
    SFR_tot = Mform_plot /np.diff(tbins*1e9) /FIREbox_volume
    ax.plot(tmax-tbins[:-1], SFR_tot,
            c='k', lw=lw+0.5, label=r'Total SFRD in $\texttt{FIREbox}$',
            )
    ax.plot(tmax-times, Madau(redshifts),
            c='gray', lw=lw+0.5, linestyle='--', label=r'Madau \& Dickinson 2014',
            )
    top_leg = ax.legend(loc='upper left', fontsize=fs)
    ax.add_artist(top_leg)

    Mgal_bins = np.linspace(8,12,9)[::-1]
    handles = []
    for n in range(len(Mgal_bins)-1):
        inbin = (np.log10(Mstar)<Mgal_bins[n]) & (np.log10(Mstar)>=Mgal_bins[n+1])
        Mform_inbin = np.sum(Mforms[inbin,:], axis=0)
        Mform_plot, _ = np.histogram(times, weights=Mform_inbin, bins=tbins)
        SFR_plot = Mform_plot /np.diff(tbins*1e9) /FIREbox_volume
        line, = ax.plot(tmax-tbins[:-1], SFR_plot,  
                        c=CB_color_cycle[n], 
                        label=r'$%.1f\geq\mathrm{log}(\mathcal{M}_\star/\mathrm{M}_\odot)\geq %.1f\ (%i)$' %(Mgal_bins[n],Mgal_bins[n+1],np.sum(inbin)),
                        )
        handles.append(line)

    ax.set_xlabel(r'Lookback time [Gyr]', fontsize=fs)
    ax.set_xlim(xmin=tmax, xmax=0)
    ax.set_xticks(np.linspace(12,0,7))
    ax.set_ylabel(r'$\mathrm{SFRD}\ [\mathrm{M}_{\odot}\mathrm{Mpc}^{-3}\mathrm{yr}^{-1}]$', fontsize=fs)
    ax.set_yscale('log')
    ax.set_ylim(ymin=1e-4, ymax=1e0)
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

    ax.legend(bbox_to_anchor=(1.05,-0.5), loc='upper center',
              handles=handles, fontsize=fs, ncol=2,
              )

    ## Right-hand plot: SMF ##
    ax = axes[1]

    Mform_plot, _ = np.histogram(times, weights=Mform_tot, bins=tbins)
    Zmean_tot = np.sum(Zmeans, axis=0)
    Zmean_plot, _ = np.histogram(times, weights=Zmean_tot, bins=tbins)
    Zmean_plot[Mform_plot>0] /= Mform_plot[Mform_plot>0]
    ax.plot(tmax-tbins[:-1], Zmean_plot /Zsun,
            c='k', lw=lw+0.5, label=r'Mean Z in $\texttt{FIREbox}$',
            )

    Mgal_bins = np.linspace(8,12,9)[::-1]
    handles = []
    for n in range(len(Mgal_bins)-1):
        inbin = (np.log10(Mstar)<Mgal_bins[n]) & (np.log10(Mstar)>=Mgal_bins[n+1])
        Mform_inbin = np.sum(Mforms[inbin,:], axis=0)
        Mform_plot, _ = np.histogram(times, weights=Mform_inbin, bins=tbins)
        Zmean_inbin = np.sum(Zmeans[inbin,:], axis=0)
        Zmean_plot, _ = np.histogram(times, weights=Zmean_inbin, bins=tbins)
        Zmean_plot[Mform_plot>0] /= Mform_plot[Mform_plot>0]
        line, = ax.plot(tmax-tbins[:-1], Zmean_plot /Zsun,
                        c=CB_color_cycle[n],
                        )
        handles.append(line)

    ax.set_xlabel(r'Lookback time [Gyr]', fontsize=fs)
    ax.set_xlim(xmin=tmax, xmax=0)
    ax.set_xticks(np.linspace(12,0,7))
    ax.set_ylabel('Average metallicity\n'+r'of newly formed stars $[\mathrm{Z}_\odot]$', fontsize=fs)
    ax.set_yscale('log')
    ax.set_ylim(ymin=1e-3, ymax=1e1)
    ax.set_box_aspect(1)
    ax.legend(loc='lower right', fontsize=fs)

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

    
    # Save
    plot_dir = os.path.join(FIREbox_dir, 'plots')
    plt.savefig(os.path.join(plot_dir,'SFRs_Zdist_FIREbox.pdf'), format='pdf', bbox_inches='tight')



##################################################################
### Run it!
###
##################################################################
if __name__ == "__main__":

    plot_SFR_Zdist()
    plt.show()
