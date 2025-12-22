import numpy as np
import h5py
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

import os


def plot_rates():

    #####################################
    # Plot your result
    fs = 10
    lw = 1.4
    plt.rc('font', family='serif')
    plt.rc('lines', linewidth=lw)
    #plt.rc('text', usetex=True)

    CB_color_cycle = ['#377eb8', '#ff7f00', '#4daf4a',
                  '#f781bf', '#a65628', '#e41a1c']

    fig, axes = plt.subplots(1, 4, figsize=(10,3.5))

    ## Use the m12i MHD version of GBoF I & II ##
    ax = axes[-1]
    sim_dir = os.path.join(FIRE_dir,'m12i_MHD_res7100')
    snapshot_file = os.path.join(sim_dir, 'snapshot_times.txt')
    redshifts, times = np.loadtxt(snapshot_file, usecols=[2,3]).T
    redshifts_to_times = interp1d(redshifts, times, fill_value='extrapolate')

    tmax = times.max()
    deltat = 250 # width of the time bins (in Myr)
    nbins = round(1e3*tmax /deltat) + 1
    tbins = np.linspace(0, tmax, nbins)

    SFR_file = os.path.join(sim_dir, 'SFR_m12i_MHD.txt')
    t, Z, sigmaZ, mform, sfr = np.loadtxt(SFR_file).T

    ## SFR ##
    mform_plot, _ = np.histogram(tmax-times, weights=mform, bins=tbins)
    SFR = mform_plot /np.diff(tbins*1e9)
    SFR = np.append(SFR,0)
    ax.plot(tbins, SFR/1e5,
            c='k', lw=lw,
            )

    ## Massive cluster formation rate ##
    #cluster_file = os.path.join(sim_dir, 'AllMassiveClusters_{}.hdf5'.format(simu))
    cluster_file = os.path.join(sim_dir, 'ClusterPopulation_1_reduced.hdf5')
    with h5py.File(cluster_file) as c:
        mass = c['Mass'][()]
        tform = c['FormationTime'][()]
    tokeep = mass>=6e4
    mass = mass[tokeep]
    tform = tform[tokeep]
    mcluster_plot, _ = np.histogram(tmax-tform, weights=mass, bins=tbins)
    form_rate = mcluster_plot /np.diff(tbins*1e9)
    ax.stairs(form_rate/1e5, tbins,
              color=CB_color_cycle[0], fill=True, alpha=.5,
              )

    ## BBH mergers ##
    bbhfile = os.path.join(FIRE_dir, 'grid/allBBHs.dat')
    cluster_id = np.loadtxt(bbhfile, dtype='str', skiprows=1, usecols=0)
    m1, m2, tmerge = np.loadtxt(bbhfile, skiprows=1, usecols=[4,5,7]).T
    m1 = np.maximum(m1,m2)
    in_sim = np.array([i.split('_')[0]=='m12i' for i in cluster_id])
    Nbbhs, _ = np.histogram(tmax-tmerge[in_sim], bins=tbins)
    merger_rate = Nbbhs /np.diff(tbins*1e9)
    merger_rate = np.append(merger_rate,0)
    ax.plot(tbins, merger_rate,
              c=CB_color_cycle[0], lw=lw,
              )

    ## Massive BBH mergers ##
    massive = in_sim & (m1 >= 50)
    Nmassive, _ = np.histogram(tmax-tmerge[massive], bins=tbins)
    merger_rate = Nmassive /np.diff(tbins*1e9)
    merger_rate = np.append(merger_rate,0)
    ax.plot(tbins, merger_rate,
              c=CB_color_cycle[0], lw=lw, linestyle='--',
              )

    ax.text(13, 2e-4, 'm12i', fontsize=fs,
            bbox=dict(boxstyle='round',facecolor='none',edgecolor='k'),
            )

    ax.set_xlabel(r'Lookback time [Gyr]', fontsize=fs)
    ax.set_xlim(xmin=tmax, xmax=0)
    ax.set_xticks(np.linspace(10,0,3))
    ax.set_yscale('log')
    ax.set_ylim(ymin=1e-9, ymax=1e-3)
    ax.set_yticks(np.logspace(-9,-3,4))
    ax.set_yticklabels([])
    ax.tick_params(axis='y', which='minor', left=False)
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

    for n,simu in enumerate(['m11d','m11h','m11e']):
        ax = axes[2-n]
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

        ## SFR ##
        mform_plot, _ = np.histogram(tmax-times, weights=mform, bins=tbins)
        SFR = mform_plot /np.diff(tbins*1e9)
        SFR = np.append(SFR,0)
        ax.plot(tbins, SFR/1e5,
                c='k', lw=lw, 
                )
    
        ## Massive cluster formation rate ##
        cluster_file = os.path.join(sim_dir, 'AllMassiveClusters_{}.hdf5'.format(simu))
        with h5py.File(cluster_file) as c:
            mass = c['Mass'][()]
            tform = c['FormationTime'][()]
        mcluster_plot, _ = np.histogram(tmax-tform, weights=mass, bins=tbins)
        form_rate = mcluster_plot /np.diff(tbins*1e9)
        ax.stairs(form_rate/1e5, tbins, 
                  color=CB_color_cycle[n+1], fill=True, alpha=.5,
                  )

        ## BBH mergers ##
        bbhfile = os.path.join(FIRE_dir, 'grid/allBBHs.dat')
        cluster_id = np.loadtxt(bbhfile, dtype='str', skiprows=1, usecols=0)
        m1, m2, tmerge = np.loadtxt(bbhfile, skiprows=1, usecols=[4,5,7]).T
        m1 = np.maximum(m1,m2)
        in_sim = np.array([i.split('_')[0]==simu for i in cluster_id])
        Nbbhs, _ = np.histogram(tmax-tmerge[in_sim], bins=tbins)
        merger_rate = Nbbhs /np.diff(tbins*1e9)
        merger_rate = np.append(merger_rate,0)
        ax.plot(tbins, merger_rate,
                  c=CB_color_cycle[n+1], lw=lw,
                  )

        ## Massive BBH mergers ##
        massive = in_sim & (m1 >= 50)
        Nmassive, _ = np.histogram(tmax-tmerge[massive], bins=tbins)
        merger_rate = Nmassive /np.diff(tbins*1e9)
        merger_rate = np.append(merger_rate,0)
        ax.plot(tbins, merger_rate,
                  c=CB_color_cycle[n+1], lw=lw, linestyle='--',
                  )
        
        ax.text(13, 2e-4, simu, fontsize=fs, 
                bbox=dict(boxstyle='round',facecolor='none',edgecolor='k'),
                )

        ax.set_xlabel(r'Lookback time [Gyr]', fontsize=fs)
        ax.set_xlim(xmin=tmax, xmax=0)
        ax.set_xticks(np.linspace(10,0,3))
        ax.set_yscale('log')
        ax.set_ylim(ymin=1e-9, ymax=1e-3)
        ax.set_yticks(np.logspace(-9,-3,4))
        ax.set_yticklabels([])
        ax.tick_params(axis='y', which='minor', left=False)
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

    ## Add y label and legend ##
    axes[0].set_ylabel('Galactic event rate ' + r'$[\mathrm{yr}^{-1}]$', fontsize=fs)
    axes[0].set_yticklabels([r'$10^{-9}$',r'$10^{-7}$',r'$10^{-5}$',r'$10^{-3}$'])

    axes[0].plot([], [], c='k', lw=lw,
                 label=r'$\mathrm{SFR}/10^5\mathrm{M}_\odot$',
                 )
    axes[0].plot([], [], c='gray', lw=lw,
                 label='BBH mergers',
                 )
    axes[0].plot([], [], c='gray', lw=lw, linestyle='--',
                 label='Massive BBH mergers\n'+r'$(M_\mathrm{BH1}>50\ \mathrm{M}_\odot)$',
                 )
    axes[0].stairs([1e-10], [0,1], color='gray', fill=True, alpha=.5,
                   label='Massive cluster formation\n'+r'$(\Sigma M_\mathrm{cl}/10^5\mathrm{M}_\odot)$',
                   )
    axes[0].legend(bbox_to_anchor=(2,-0.25), loc='upper center', fontsize=fs, ncols=4)


    # Save
    plot_dir = os.path.join(FIRE_dir, 'plots')
    plt.savefig(os.path.join(plot_dir,'rates.pdf'), format='pdf', bbox_inches='tight')



##################################################################
### Run it!
###
##################################################################
if __name__ == "__main__":

    #FIRE_dir = '/home/tristan-bruel/Documents/Science/GBoF/FIRE-2'
    FIRE_dir = '/home/tbruel/Great-Balls-of-FIRE/FIRE-2'

    plot_rates()
    plt.show()
