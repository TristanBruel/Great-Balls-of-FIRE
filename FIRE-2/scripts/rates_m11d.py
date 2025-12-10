import numpy as np
import h5py
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

import os


def plot_rates(simu='m11d'):

    #####################################
    # Plot your result
    fs = 10
    lw = 1.4
    plt.rc('font', family='serif')
    plt.rc('lines', linewidth=lw)
    plt.rc('text', usetex=True)

    CB_color_cycle = ['#377eb8', '#ff7f00', '#4daf4a',
                  '#f781bf', '#a65628', '#e41a1c']

    fig, ax = plt.subplots(1, 1, figsize=(3.5,3.5))

    sim_dir = os.path.join(FIRE_dir,simu+'_res7100')
    snapshot_file = os.path.join(sim_dir, 'snapshot_times.txt')
    redshifts, times = np.loadtxt(snapshot_file, usecols=[2,3]).T
    redshifts_to_times = interp1d(redshifts, times, fill_value='extrapolate')

    tmax = times.max()
    deltat = 250 # width of the time bins (in Myr)
    nbins = round(1e3*tmax /deltat) + 1
    tbins = np.linspace(0, tmax, nbins)

    SFR_file = os.path.join(sim_dir, 'SFR_{}.txt'.format(simu))
    t, Z, sigmaZ, mform, sfr = np.loadtxt(SFR_file).T

    mform_plot, _ = np.histogram(tmax-times, weights=mform, bins=tbins)
    SFR = mform_plot /np.diff(tbins*1e9)
    ax.plot(tbins[:-1], SFR/1e5,
            c='k', lw=lw, 
            label=r'$\mathrm{SFR}/10^5\mathrm{M}_\odot$',
            )

    cluster_file = os.path.join(sim_dir, 'AllMassiveClusters.hdf5')
    with h5py.File(cluster_file) as c:
        mass = c['Mass'][()]
        tform = c['FormationTime'][()]
    mcluster_plot, _ = np.histogram(tmax-tform, weights=mass, bins=tbins)
    form_rate = mcluster_plot /np.diff(tbins*1e9)
    ax.stairs(form_rate/1e5, tbins, 
              color=CB_color_cycle[1], fill=True, alpha=.5,
              label='Massive cluster formation\n'+r'$(\Sigma M_\mathrm{cl}/10^5\mathrm{M}_\odot)$',
              )

    all_bbh_file = os.path.join(FIRE_dir, 'grid/allBBHs.dat')
    bbh_ids = np.loadtxt(all_bbh_file, dtype='str', skiprows=1, usecols=0)
    tmerge = np.loadtxt(all_bbh_file, skiprows=1, usecols=7)
    insim = np.array([i.split('_')[0]==simu for i in bbh_ids])
    Nbbhs, _ = np.histogram(tmax-tmerge[insim], bins=tbins)
    merger_rate = Nbbhs /np.diff(tbins*1e9)
    ax.plot(tbins[:-1], merger_rate,
              c=CB_color_cycle[1], lw=lw,
              label='BBH mergers',
              )
    
    ## Plot galaxy SFR surface density ##
    ax_right = ax.twinx()
    gal_file = os.path.join(sim_dir, 'properties_{}.txt'.format(simu))
    t, Mstar, Rstar, mform = np.loadtxt(gal_file, skiprows=1).T
    mbin, _ = np.histogram(tmax-t, weights=mform, bins=tbins)
    sfr = mbin /np.diff(tbins*1e9)
    ax.plot(tbins[:-1], sfr/1e5,
            c='k', lw=lw, ls='--',
            )
    Rbin, _ = np.histogram(tmax-t, weights=Rstar, bins=tbins)
    Nbin, _ = np.histogram(tmax-t, bins=tbins)
    Rstar_mean = Rbin /Nbin
    SigmaSFR = sfr /(np.pi*Rstar_mean**2)
    ax_right.plot(tbins[:-1], SigmaSFR,
                  c='red', lw=lw,
                  )
    ax_right.set_ylabel(r'$\Sigma_\mathrm{SFR}\ [\mathrm{M}_\odot \mathrm{yr}^{-1} \mathrm{kpc}^{-1}]$', fontsize=fs)
    ax_right.set_yscale('log')
    ax_right.set_ylim(ymin=1e-4, ymax=1e-1)


    ax.set_xlabel(r'Lookback time [Gyr]', fontsize=fs)
    ax.set_xlim(xmin=tmax, xmax=0)
    ax.set_xticks(np.linspace(12,0,7))
    ax.set_ylabel('Galactic event rate' + r'$[\mathrm{yr}^{-1}]$', fontsize=fs)
    ax.set_yscale('log')
    ax.set_ylim(ymin=1e-9, ymax=1e-3)
    ax.set_box_aspect(1)
    ax.legend(loc='upper left', fontsize=fs)

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

    plot_rates()
    plt.show()
