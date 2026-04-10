import numpy as np
import h5py
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.patches import Patch
from scipy.interpolate import interp1d

import os
import argparse

from colossus.cosmology import cosmology
cosmology.setCosmology('planck18');
from colossus.lss import mass_function


def HaloMassFunction():
    dndM = lambda M: mass_function.massFunction(M, 0.0, mdef='vir', model='tinker08', q_out='dndlnM')
    return dndM


def plot_mBH_spin(sample, seed=1):
    snapshot_file = os.path.join(FIREbox_dir, 'snapshot_times.txt')
    redshifts, times = np.loadtxt(snapshot_file, usecols=[2,3]).T
    tmax = times.max()
    deltat = 50 # width of the time bins (in Myr)
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


    m1bins = np.linspace(0,100,51)
    qbins = np.linspace(0,1,101)

    bbhfile = os.path.join(FIREbox_dir, 'BBHs/BBHs_{}_compiled_seed{}.h5'.format(sample,seed))
    with h5py.File(bbhfile, 'r') as bbhs:
        m1 = bbhs['m1'][()]
        m2 = bbhs['m2'][()]
        a1 = bbhs['a1'][()]
        a2 = bbhs['a2'][()]
        tmerger = bbhs['tmerger'][()]
        Mgal = bbhs['Mgal'][()]
        Mhalo = bbhs['Mvir'][()]

    a1[m1<m2], a2[m1<m2] = a2[m1<m2], a1[m1<m2]
    m1, m2 = np.maximum(m1,m2), np.minimum(m1,m2)
    
    astro = (tmerger > tz1) & (tmerger < tmax)
    m1 = m1[astro]
    m2 = m2[astro]
    a1 = a1[astro]
    a2 = a2[astro]
    tmerger = tmerger[astro]
    Mgal = Mgal[astro]
    Mhalo = Mhalo[astro]
    # Compute weight associated to each galaxy in which there's a BBH merger
    weights = np.array([wgal[k-1] for k in np.digitize(np.log10(Mhalo), logmbins)])

    m = np.append(m1,m2)
    a = np.append(a1,a2)
    q = np.minimum(m1, m2) /np.maximum(m1,m2)
    q = np.append(q, q)
    weights = np.append(weights, weights) /2
    tmerger = np.append(tmerger, tmerger)

    ### First-generation mergers ###
    m_1g = m[a==0]
    q_1g = q[a==0]
    weights_1g = weights[a==0]
    print('Number of 1g BHs in mergers inside FIREbox volume up to z=1: {:.2e}'.format(np.sum(weights_1g)))
    R = np.sum(weights_1g) /FIREbox_volume /dt_astro
    print('Corresponding merger rate: {:.2e} Gpc^-3yr^-1'.format(R))

    merger_nbr, _ = np.histogram(tmerger[a==0], bins=tbins, weights=weights_1g)
    mergerrate = merger_nbr /np.diff(tbins) /FIREbox_volume # unit Gpc-3 yr-1
    R0 = mergerrate[-1]
    print('R(z=0) = %.1f Gpc^-3yr^-1' %R0)

    H, xedges, yedges = np.histogram2d(x=m_1g, y=q_1g, bins=[m1bins,qbins], weights=weights_1g)
    H /= FIREbox_volume *dt_astro *np.diff(m1bins)[:,None] *np.diff(qbins)
    #print(H[H>0].min(), H.max())
    Hmin=H[H>0].min()
    Hmax=H[H>0].max()
    if Hmin>1:
        vmin = int(Hmin /10**int(np.log10(Hmin)))*10**int(np.log10(Hmin))
    else:
        vmin = int(Hmin /10**int(np.log10(Hmin)-1))*10**(int(np.log10(Hmin))-1)
    if Hmax>1:
        vmax = int(Hmax /10**int(np.log10(Hmax))+1)*10**int(np.log10(Hmax))
    else:
        vmax = int(Hmax /10**int(np.log10(Hmax)-1)+1)*10**(int(np.log10(Hmax))-1)
    #print(vmin, vmax)
    H[H==0] = 1e-32

    cmap = plt.get_cmap('viridis')
    norm = mpl.colors.LogNorm(vmin=vmin, vmax=vmax)

    """
    X, Y = np.meshgrid(xedges[:-1], yedges[:-1])
    im = ax.contourf(X, Y, H.T,
                     levels=np.logspace(np.log10(vmin),np.log10(vmax),5),
                     cmap=cmap,
                     norm=norm,
                    )
    """
    ax_top = ax.inset_axes([0, 1.05, 1, 0.4])
    mhist = np.sum(H *np.diff(qbins), axis=1)
    mhist *= R0 /np.sum(mhist*np.diff(m1bins))
    ax_top.plot(m1bins[:-1], mhist,
                c='forestgreen', label='1g')

    ax_left = ax.inset_axes([-.45, 0., 0.4, 1.], sharey=ax)
    qhist = np.sum(H *np.diff(m1bins)[:,None], axis=0)
    qhist *= R0 /np.sum(qhist*np.diff(qbins))
    ax_left.plot(qhist, qbins[:-1], c='forestgreen')


    ### Hierarchical mergers ###
    m_hierar = m[a>0]
    q_hierar = q[a>0]
    weights_hierar = weights[a>0]
    print('Number of ng BHs in mergers inside FIREbox volume up to z=1: {:.2e}'.format(np.sum(weights_hierar)))
    R = np.sum(weights_hierar) /FIREbox_volume /dt_astro
    print('Corresponding merger rate: {:.2e} Gpc^-3yr^-1'.format(R))

    merger_nbr, _ = np.histogram(tmerger[a>0], bins=tbins, weights=weights_hierar)
    mergerrate = merger_nbr /np.diff(tbins) /FIREbox_volume # unit Gpc-3 yr-1
    R0 = mergerrate[-1]
    print('R(z=0) = %.1f Gpc^-3yr^-1' %R0)

    H, xedges, yedges = np.histogram2d(x=m_hierar, y=q_hierar, bins=[m1bins,qbins], 
                                       weights=weights_hierar,
                                       )
    H /= FIREbox_volume *dt_astro *np.diff(m1bins)[:,None] *np.diff(qbins)
    #print(H[H>0].min(), H.max())
    Hmin=H[H>0].min()
    Hmax=H[H>0].max()
    if Hmin>1:
        vmin = int(Hmin /10**int(np.log10(Hmin)))*10**int(np.log10(Hmin))
    else:
        vmin = int(Hmin /10**int(np.log10(Hmin)-1))*10**(int(np.log10(Hmin))-1)
    if Hmax>1:
        vmax = int(Hmax /10**int(np.log10(Hmax))+1)*10**int(np.log10(Hmax))
    else:
        vmax = int(Hmax /10**int(np.log10(Hmax)-1)+1)*10**(int(np.log10(Hmax))-1)
    #print(vmin, vmax)
    H[H==0] = 1e-32

    cmap2 = plt.get_cmap('plasma')
    norm2 = mpl.colors.LogNorm(vmin=vmin, vmax=vmax)

    X, Y = np.meshgrid(xedges[:-1], yedges[:-1])
    im2 = ax.contourf(X, Y, H.T,
                     levels=np.logspace(np.log10(vmin),np.log10(vmax),10),
                     cmap=cmap2,
                     norm=norm2,
                    )

    mhist = np.sum(H *np.diff(qbins), axis=1)
    mhist *= R0 /np.sum(mhist*np.diff(m1bins))
    ax_top.plot(m1bins[:-1], mhist, c='orangered', label='ng')
    ymin = 1e-3
    ymax = 1e1

    qhist = np.sum(H *np.diff(m1bins)[:,None], axis=0)
    qhist *= R0 /np.sum(qhist*np.diff(qbins))
    ax_left.plot(qhist, qbins[:-1], c='orangered')

    ### Add LVK primary mass and mass ratio distributions (GWTC-3) ##
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

    ax_top.plot(
            mass_1, mass_1_ppd, lw=lw, color=color_plpeak, zorder=1, label="GWTC-3"
    )
    ax_top.plot(mass_1, mass_1_lower, lw=lw-1, color=color_plpeak, zorder=1, alpha=0.3)
    ax_top.plot(mass_1, mass_1_upper, lw=lw-1, color=color_plpeak, zorder=1, alpha=0.3)
    ax_top.fill_between(
        mass_1,
        mass_1_lower,
        mass_1_upper,
        alpha=0.14,
        color=color_plpeak,
        zorder=0,
    )

    ax_left.plot(mass_r_ppd, mass_ratio, lw=lw, color=color_plpeak, zorder=1)
    ax_left.plot(mass_r_lower, mass_ratio, lw=lw-1, color=color_plpeak, zorder=1, alpha=0.3)
    ax_left.plot(mass_r_upper, mass_ratio, lw=lw-1, color=color_plpeak, zorder=1, alpha=0.3)
    ax_left.fill_betweenx(
        mass_ratio,
        mass_r_lower,
        mass_r_upper,
        alpha=0.14,
        color=color_plpeak,
        zorder=0,
    )

    """
    locpeak = np.argmax(mass_1_ppd[mass_1>20])
    xpeak = mass_1[mass_1>20][locpeak]
    ypeak = mass_1_ppd[mass_1>20][locpeak]
    frac_peak = (np.log10(ypeak)-np.log10(ymin)) /(np.log10(ymax)-np.log10(ymin))
    ax_top.axvline(x=xpeak, ymin=0, ymax=frac_peak,
               c='gray', linestyle='--', lw=lw)
    """

    ax_top.set_xlim(xmin=-5., xmax=100)
    ax_top.set_xticks([])
    ax_top.set_yscale('log')
    ax_top.set_ylim(ymin=ymin, ymax=ymax)
    ax_top.set_yticks([])
    ax_top.set_yticks([], minor=True)
    ax3 = ax_top.twinx()
    ax3.set_ylabel(r'$\mathrm{d}\mathcal{R}/\mathrm{d}m_\mathrm{BH}$', fontsize=fs, y=0.5)
    ax3.set_yscale('log')
    ax3.set_ylim(ymin=ymin, ymax=ymax)

    ax_left.set_ylabel(r'$q$', fontsize=fs)
    ax_left.set_ylim(ymin=0.1, ymax=1.0)
    ax_left.set_xlabel(r'$\mathrm{d}\mathcal{R}/\mathrm{d}q$', fontsize=fs)
    ax_left.set_xscale('log')
    ax_left.set_xlim(xmin=1e-2, xmax=2e2)
    ax_left.set_xticks(np.logspace(-2,2,3))

    handles, labels = ax_top.get_legend_handles_labels()
    patch = Patch(color=color_plpeak, alpha=0.14)
    handles[2] = (handles[2],patch)
    ax_top.legend(handles, labels,
                  bbox_to_anchor=(0.,-0.1), loc='lower right', fontsize=fs,
                  )

    ax.set_xlabel(r'$m_\mathrm{BH}\ [\mathrm{M}_\odot]$', fontsize=fs)
    ax.set_xlim(xmin=-5., xmax=100)
    ax.set_xticks(np.linspace(0,100,5))
    ax.tick_params(axis='y', which='both', left=False, labelleft=False)
    ax.set_ylim(ymin=0.1, ymax=1.0)
    ax.set_box_aspect(1)

    sm2 = plt.cm.ScalarMappable(cmap=cmap2, norm=norm2)
    cbar2 = fig.colorbar(sm2, ax=ax, fraction=0.045)
    cbar2.set_label(r'$\mathrm{d}^2\mathcal{R}/\mathrm{d}m_\mathrm{BH}\ \mathrm{d}q$',
                   fontsize=fs)
    cbar2.ax.tick_params(labelsize=0.8*fs)

    # Save
    plot_dir = '/home/tristan-bruel/Documents/Science/Great-Balls-of-FIRE/hierarchical/plots'
    plt.savefig(os.path.join(plot_dir,'mBH_q_hierarchical_box.png'), bbox_inches='tight')



##################################################################
### Run it!
###
##################################################################
if __name__ == "__main__":

    FIREbox_dir = '/home/tristan-bruel/Documents/Science/Great-Balls-of-FIRE/FIREbox'
    FIREbox_volume = (22.1)**3 # in unit Mpc^3

    #####################################
    # Define command line option for the path to the simulations and the path to save the plots
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", dest='sample', help="Cluster formation model", type=str, default='SFR')
    parser.add_argument("--seed", dest='seed', help="Random seed used to sample massive clusters in FIREbox", type=int, default=42)
    args = parser.parse_args()

    #####################################
    plot_mBH_spin(sample=args.sample, seed=args.seed)
    plt.show()
