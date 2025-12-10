import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

import os


def plot_SFR_Zdist(Zsun=0.017):

    #####################################
    # Plot your result
    fs = 10
    lw = 1.4
    plt.rc('font', family='serif')
    plt.rc('lines', linewidth=lw)
    plt.rc('text', usetex=True)

    CB_color_cycle = ['#377eb8', '#ff7f00', '#4daf4a',
                  '#f781bf', '#a65628', '#e41a1c']

    fig, axes = plt.subplots(1, 2, figsize=(8.,3.))

    for n,simu in enumerate(['m12i','m11d','m11h','m11e','m11q','m11i']):
        sim_dir = os.path.join(FIRE_dir,simu+'_res7100')
        snapshot_file = os.path.join(sim_dir, 'snapshot_times.txt')
        redshifts, times = np.loadtxt(snapshot_file, usecols=[2,3]).T

        tmax = times.max()
        deltat = 200 # width of the time bins (in Myr)
        nbins = round(1e3*tmax /deltat) + 1
        tbins = np.linspace(0, tmax, nbins)

        SFR_file = os.path.join(sim_dir, 'SFR_{}.txt'.format(simu))
        t, Z, sigmaZ, mform, sfr = np.loadtxt(SFR_file).T

        ## Left-hand plot: SFRs per bins of galaxy mass ##
        ax = axes[0]

        mform_plot, _ = np.histogram(times, weights=mform, bins=tbins)
        SFR = mform_plot /np.diff(tbins*1e9)
        ax.plot(tmax-tbins[:-1], SFR,
                c=CB_color_cycle[n], lw=lw, label=simu,
                )

        ## Right-hand plot: SMF ##
        ax = axes[1]

        Zmean_plot, _ = np.histogram(times, weights=Z*mform, bins=tbins)
        Zmean_plot[mform_plot>0] /= mform_plot[mform_plot>0]
        ax.plot(tmax-tbins[:-1], Zmean_plot /Zsun,
                c=CB_color_cycle[n], lw=lw, 
                )


    times_to_redshifts = interp1d(times, redshifts, fill_value='extrapolate')
    redshifts_to_times = interp1d(redshifts, times, fill_value='extrapolate')

    ## Left-hand plot: SFRs per bins of galaxy mass ##
    ax = axes[0]

    ax.set_xlabel(r'Lookback time [Gyr]', fontsize=fs)
    ax.set_xlim(xmin=tmax, xmax=0)
    ax.set_xticks(np.linspace(12,0,7))
    ax.set_ylabel('Galactic star formation rate\n' + r'$[\mathrm{M}_{\odot}\mathrm{Mpc}^{-3}\mathrm{yr}^{-1}]$', fontsize=fs)
    ax.set_yscale('log')
    ax.set_ylim(ymin=1e-4, ymax=5e1)
    ax.set_box_aspect(1)
    ax.legend(loc='lower right', fontsize=fs, ncol=2)

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


    ## Right-hand plot: SMF ##
    ax = axes[1]

    ax.set_xlabel(r'Lookback time [Gyr]', fontsize=fs)
    ax.set_xlim(xmin=tmax, xmax=0)
    ax.set_xticks(np.linspace(12,0,7))
    ax.set_ylabel('Average metallicity\n'+r'of newly formed stars $[\mathrm{Z}_\odot]$', fontsize=fs)
    ax.set_yscale('log')
    ax.set_ylim(ymin=1e-3, ymax=5e0)
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

    
    # Save
    plot_dir = os.path.join(FIRE_dir, 'plots')
    plt.savefig(os.path.join(plot_dir,'SFRs_Zdist.pdf'), format='pdf', bbox_inches='tight')



##################################################################
### Run it!
###
##################################################################
if __name__ == "__main__":

    FIRE_dir = '/home/tristan-bruel/Documents/Science/GBoF/FIRE-2'

    plot_SFR_Zdist()
    plt.show()
