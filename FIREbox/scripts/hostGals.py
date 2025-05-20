import numpy as np
import h5py
import matplotlib.pyplot as plt
import matplotlib as mpl
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


def plot_hist2D(sample='Gamma', seed=1):

    snapshot_file = os.path.join(FIREbox_dir, 'snapshot_times.txt')
    redshifts, times = np.loadtxt(snapshot_file, usecols=[2,3]).T
    tmax = times.max()
    deltat = 50 # width of the time bins (in Myr)
    nbins = round(1e3*tmax /deltat) + 1
    tbins = np.linspace(0, tmax, nbins)

    times_to_redshifts = interp1d(times, redshifts, fill_value='extrapolate')
    redshifts_to_times = interp1d(redshifts, times, fill_value='extrapolate')
    tz1 = redshifts_to_times(1)

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

    fig, ax = plt.subplots(1, 1, figsize=(3.5,3.5))

    print('Cluster sampling method:', sample)
    bbhfile = os.path.join(FIREbox_dir, 'BBHs/BBHs_{}_compiled_seed{}.h5'.format(sample,seed))
    with h5py.File(bbhfile, 'r') as bbhs:
        tmerger = bbhs['tmerger'][()]
        Mgal = bbhs['Mgal'][()]
        Mhalo = bbhs['Mvir'][()]

    print('Total number of BBH mergers:', len(tmerger))

    # Compute weight associated to each galaxy in which there's a BBH merger
    weights = np.array([wgal[k-1] for k in np.digitize(np.log10(Mhalo), logmbins)])

    Mgal_bins = np.linspace(7, 12, 26)
    H, xedges, yedges = np.histogram2d(x=tmerger, y=np.log10(Mgal), bins=[tbins,Mgal_bins], weights=weights)
    H /= FIREbox_volume *np.diff(tbins)[:,None] *np.diff(Mgal_bins) 
    print(H[H>0].min(), H.max())
    H[H==0] = 1e-32

    cmap = plt.get_cmap('plasma')
    norm = mpl.colors.LogNorm(vmin=8e-3, vmax=3e1)

    X, Y = np.meshgrid(tmax-xedges[1:], yedges[:-1])
    im = ax.contourf(X, Y, H.T,
                     levels=np.logspace(-2,2,20),
                     cmap=cmap,
                     norm=norm,
                    )

    ax_top = ax.inset_axes([0, 1.05, 1, 0.4])
    ax_left = ax.inset_axes([-.45, 0, 0.4, 1], sharey=ax)
    
    sampling_methods = np.array(['Gamma', 'ElBadry', 'SFR'])
    sampling_names = np.array([r'Gamma', r'EB18', r'SFRpeak'])
    sampling_colors = np.array(['gold', 'darkorange', 'firebrick'])
    for s,sample in enumerate(sampling_methods):
    #for s,sample in enumerate(['Gamma']):

        print('\nCluster sampling method:', sample)
        bbhfile = os.path.join(FIREbox_dir, 'BBHs/BBHs_{}_compiled_seed{}.h5'.format(sample,seed))
        with h5py.File(bbhfile, 'r') as bbhs:
            tmerger = bbhs['tmerger'][()]
            Mgal = bbhs['Mgal'][()]
            Mhalo = bbhs['Mvir'][()]

        # Compute weight associated to each galaxy in which there's a BBH merger
        weights = np.array([wgal[k-1] for k in np.digitize(np.log10(Mhalo), logmbins)])

        H, xedges, yedges = np.histogram2d(x=tmerger, y=np.log10(Mgal), bins=[tbins,Mgal_bins], weights=weights)
        #H /= FIREbox_volume *np.diff(tbins)[:,None] *np.diff(Mgal_bins)
        R = np.sum(H /FIREbox_volume /np.diff(Mgal_bins), axis=1)
        ax_top.plot(tmax-tbins[1:], R,
                    c=sampling_colors[s], label=sampling_names[s],
                    )

        #MF = np.sum(H*np.diff(tbins)[:,None], axis=0)
        MF = np.sum(H /FIREbox_volume /np.diff(tbins)[:,None], axis=0)
        ax_left.plot(MF, Mgal_bins[1:], c=sampling_colors[s])

    ## Add equivalent plots for star formation ##
    weights = np.array([wgal[k-1] for k in np.digitize(np.log10(Mvir), logmbins)])
    Mforms = np.zeros((len(haloids),len(times)))
    for k,h in enumerate(haloids):
        SFR_file = os.path.join(FIREbox_dir, 'SFRs/FB15N1024_halo{}_SFR.txt'.format(h))
        t, Z, sigmaZ, mform, sfr = np.loadtxt(SFR_file).T
        Mforms[k,:] = mform *weights[k]
    hist, _ = np.histogram(times, bins=tbins, weights=np.sum(Mforms*weights[:,None], axis=0))
    R = hist /FIREbox_volume
    R = R /R[-1] *1e0
    ax_top.plot(tmax-tbins[1:], R, c='k', label='Star form.')
    hist, _ = np.histogram(np.log10(Mstar), bins=Mgal_bins, weights=np.sum(Mforms*weights[:,None], axis=1))
    MF = hist /FIREbox_volume
    MF = MF /MF[0] *1e-1
    ax_left.plot(MF, Mgal_bins[1:], c='k')


    ax_left.set_xlabel(r'$\mathrm{Host\ MF}\ [\mathrm{Gpc}^{-3}]$')
    ax_left.set_xscale('log')
    ax_left.set_xlim(xmin=1e-2, xmax=1e4)
    ax_left.set_xticks(np.logspace(-3,3,3))
    ax_left.set_ylabel(r'$\mathrm{log}(\mathcal{M}_\star/\mathrm{M}_\odot)$', fontsize=fs)
    ax_left.set_ylim(ymin=8, ymax=12)
    ax_left.set_yticks(np.linspace(8,12,5))
    
    # Add redshift axis on top
    ax_top.set_xlim(xmin=tmax, xmax=0)
    ax_top.set_xticks([])
    # convert redshifts to times and convert to Gyr
    ax2 = ax_top.twiny()
    ax2.set_xlim(ax_top.get_xlim())
    x2tick_values = np.array([0,0.1,0.5,1,2,3,10])
    x2tick_locations = tmax-redshifts_to_times(x2tick_values)
    ax2.set_xticks(x2tick_locations)
    ax2.set_xticklabels(['0','0.1','0.5','1','2','3','10'])
    ax2.set_xlabel('Redshift', fontsize=fs)

    ax_top.set_yscale('log')
    ax_top.set_ylim(ymin=1e-2, ymax=1e2)
    ax_top.set_yticks([])
    ax_top.set_yticks([], minor=True)
    ax3 = ax_top.twinx()
    ax3.set_ylabel('Rate density\n'+r'$[\mathrm{Gpc}^{-3}\mathrm{yr}^{-1}]$')
    ax3.set_yscale('log')
    ax3.set_ylim(ymin=1e-1, ymax=1e3)
    ax3.set_yticks(np.logspace(-1,3,3))

    ax_top.legend(bbox_to_anchor=(-0.05,0), loc='lower right', fontsize=fs)

    ax.set_xlabel(r'Lookback time [Gyr]', fontsize=fs)
    ax.set_xlim(xmin=tmax, xmax=0)
    ax.set_xticks(np.linspace(12,0,4))
    ax.set_ylim(ymin=8, ymax=12)
    ax.tick_params(axis='y', which='both', left=False, labelleft=False)
    ax.set_box_aspect(1)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    cbar = fig.colorbar(sm, ax=ax, fraction=0.045, pad=0.06)
    cbar.set_label(r'$\mathrm{d}\mathcal{R}/\mathrm{dlog}(\mathcal{M}_\star/\mathrm{M}_\odot)\ [\mathrm{Gpc}^{-3}\mathrm{yr}-1]$', fontsize=fs)
    cbar.ax.tick_params(labelsize=0.8*fs)
  
    # Save
    plot_dir = os.path.join(FIREbox_dir, 'plots')
    plt.savefig(os.path.join(plot_dir,'hostGals.pdf'), format='pdf', bbox_inches='tight')



##################################################################
### Run it!
###
##################################################################
if __name__ == "__main__":

    plot_hist2D()
    plt.show()
