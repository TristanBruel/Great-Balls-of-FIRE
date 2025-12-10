import numpy as np
import h5py
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from scipy.interpolate import interp1d

import os
import sys
import argparse

from colossus.cosmology import cosmology
cosmology.setCosmology('planck18');
from colossus.lss import mass_function

import popsummary 
dir_GWTC4 = '/home/tristan-bruel/Documents/Science/o4a-astro'
sys.path.append(os.path.join(dir_GWTC4,'figure_scripts'))
import plot_funcs_bbh_mass as pf

FIREbox_dir = '/home/tristan-bruel/Documents/Science/GBoF/FIREbox'
FIREbox_volume = (22.1)**3 # in unit Mpc^3

def HaloMassFunction():
    dndM = lambda M: mass_function.massFunction(M, 0.0, mdef='vir', model='tinker08', q_out='dndlnM')
    return dndM


def plot_mBH1(sample='SFR',seed=42):

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

    fig, ax = plt.subplots(1, 1, figsize=(11.,4.5))

    ### ADD BBH MERGER RATE DENSITY LVK GWTC-4 ##
    file_bptp04 = os.path.join(dir_GWTC4, 'data_release/BBHMassSpinRedshift_BrokenPowerLawTwoPeaks_GaussianComponentSpins_PowerLawRedshift.h5')
    file_bs04 = os.path.join(dir_GWTC4, 'data_release/BBHMassSpinRedshift_BSplineIID.h5')
    bptp_o4 = popsummary.popresult.PopulationResult(file_bptp04)
    bs_o4 = popsummary.popresult.PopulationResult(file_bs04)

    color1 = '#648FFF'
    color2 = '#FE6100'

    bptp_m1, bptp_m1_pdfs = pf.get_params(bptp_o4, 'mass_1')
    bs_m1, bs_m1_pdfs = pf.get_params(bs_o4, 'rate_vs_mass_1_at_z0-2', rate=False)
    #plp_m1, plplow, plppd, plphi = pf.get_03b_plp_ppds(os.path.join(dir_GWTC4,'gwtc3_data/analyses/PowerLawPeak/'))

    pf.plot_90CI(ax, bs_m1, bs_m1_pdfs, color=color1, label=r'\textsc{B-Spline}, \textsc{GWTC-4.0}', fill_alpha=0.4)
    pf.plot_90CI(ax, bptp_m1, bptp_m1_pdfs, color=color2, label=r'\textsc{Broken Power Law + 2 Peaks}, \textsc{GWTC-4.0}', fill_alpha=0.4)

    #locpeak = np.argmax(bptp_m1_pdfs[bptp_m1>20])
    #xpeak = bptp_m1[bptp_m1>20][locpeak]
    #ypeak = bptp_m1[bptp_m1>20][locpeak]
    ymin = 1e-3
    ymax = 4e1
    #frac_peak = (np.log10(ypeak)-np.log10(ymin)) /(np.log10(ymax)-np.log10(ymin))
    #ax.axvline(x=xpeak, ymin=0, ymax=frac_peak,
    #           c='k', linestyle='--', lw=lw)


    # GW241110 posterior #
    gw241110_qm = np.loadtxt(os.path.join(w_dir,'gw241110_qm.txt'))
    q_bins_241110 = gw241110_qm[1:,0]
    log_m1_bins_241110 = gw241110_qm[0,1:]
    heights_241110_q = gw241110_qm[1:,1:]
    c90_241110_q = [gw241110_qm[0,0]]
    gw241110_am = np.loadtxt(os.path.join(w_dir,'gw241110_am.txt'))
    s1_bins_241110 = gw241110_am[1:,0]
    heights_241110_a = gw241110_am[1:,1:]
    c90_241110_a = [gw241110_am[0,0]]


    m1bins = np.linspace(0,200,101)

    sampling_methods = np.array(['Gamma', 'ElBadry', 'SFR'])
    sampling_names = np.array([r'Gamma', r'EB18', r'SFRpeak'])
    sampling_colors = np.array(['gold', 'darkorange', 'firebrick'])
    print('\nCluster sampling method:', sample)
    s = np.where(sampling_methods==sample)[0][0]
    bbhfile = os.path.join(FIREbox_dir, 'BBHs/BBHs_{}_compiled_seed{}.h5'.format(sample,seed))
    with h5py.File(bbhfile, 'r') as bbhs:
        m1 = bbhs['m1'][()]
        m2 = bbhs['m2'][()]
        a1 = bbhs['a1'][()]
        a2 = bbhs['a2'][()]
        tmerger = bbhs['tmerger'][()]
        Mgal = bbhs['Mgal'][()]
        Mhalo = bbhs['Mvir'][()]
        cluster_id = bbhs['id'][()]

    a1[m1<m2], a2[m1<m2] = a2[m1<m2], a1[m1<m2]
    m1, m2 = np.maximum(m1,m2), np.minimum(m1,m2)
    q = m2 /m1

    astro = (tmerger > tz1) & (tmerger < tmax)
    m1 = m1[astro]
    q = q[astro]
    a1 = a1[astro]
    tmerger = tmerger[astro]
    Mgal = Mgal[astro]
    Mhalo = Mhalo[astro]
    cluster_id = np.astype(cluster_id[astro], str)
    print('Unweighted number of BBH mergers at z<1:', len(m1))

    # Compute weight associated to each galaxy in which there's a BBH merger
    weights = np.array([wgal[k-1] for k in np.digitize(np.log10(Mhalo), logmbins)])

    merger_nbr, _ = np.histogram(tmerger, bins=tbins, weights=weights)
    mergerrate = merger_nbr /np.diff(tbins) /FIREbox_volume # unit Gpc-3 yr-1
    norm = mergerrate[-1]
    print('R(z=0) = %.1f Gpc^-3yr^-1' %norm)
    Nmassive = np.sum(weights[m1>100])

    # GW241011-like BBH mergers #
    fig2, ax2 = plt.subplots(1, 1, figsize=(3.,3.))
    contour_q = ax2.contour(log_m1_bins_241110, q_bins_241110, heights_241110_q,
                           levels=[*c90_241110_q, np.inf], colors=gw241110_color)
    p = contour_q.get_paths()[0]
    mask_q = p.contains_points(np.hstack((np.log10(m1)[:,np.newaxis],q[:,np.newaxis])))
    contour_a = ax2.contour(log_m1_bins_241110, s1_bins_241110, heights_241110_a,
                           levels=[*c90_241110_a, np.inf], colors=gw241110_color)
    p = contour_a.get_paths()[0]
    mask_a = p.contains_points(np.hstack((np.log10(m1)[:,np.newaxis],a1[:,np.newaxis])))
    GW241110 = mask_q & mask_a
    plt.close(fig2)
    print('Number of GW241110-like BBH mergers: %i' %np.sum(weights[GW241110]))
    unique, indexes = np.unique(cluster_id[GW241110], return_index=True)
    print('Number of clusters hosting these events:', len(unique))
    mBH = np.array([])
    w = np.array([])
    for i in range(len(unique)):
        mBH = np.append(mBH,m1[cluster_id==unique[i]])
        w = np.append(w,weights[cluster_id==unique[i]])
    print('Total number of BBH mergers from these clusters: %i' %np.sum(w),'\n')

    hist, _ = np.histogram(m1, bins=m1bins, weights=weights)
    dRdm1 = hist /FIREbox_volume /dt_astro /np.diff(m1bins)
    dRdm1 *= norm /np.sum(dRdm1 *np.diff(m1bins))
    ax.stairs(dRdm1, m1bins, 
              color=sampling_colors[s], lw=lw, 
              label='BBH mergers from massive star clusters',
              )

    hist, _ = np.histogram(m1[GW241110], bins=m1bins, weights=weights[GW241110])
    dRdm1_GW241110 = hist /FIREbox_volume /dt_astro /np.diff(m1bins)
    dRdm1_GW241110 *= norm /np.sum(dRdm1 *np.diff(m1bins))
    ax.stairs(dRdm1_GW241110, m1bins,
              color=sampling_colors[s], fill=True,
              label='GW241110-like BBH mergers',
              )

    ax.set_xlabel(r'$m_1\ [\mathrm{M}_\odot]$', fontsize=fs)
    ax.set_xlim(xmin=0, xmax=170)
    xticks = np.linspace(0,150,7)
    xticks[1] = 34
    ax.set_xticks(xticks)
    ax.set_ylabel(r'$\mathrm{d}\mathcal{R}/\mathrm{d}m_1\ [\mathrm{Gpc}^{-3}\mathrm{yr}^{-1}\mathrm{M}_\odot^{-1}]$', fontsize=fs)
    ax.set_yscale('log')
    ax.set_ylim(ymin=ymin, ymax=ymax)
    #ax.set_box_aspect(1)
    ax.legend(loc='upper right', fontsize=fs)

    
    # Save
    plot_dir = '/home/tristan-bruel/Documents/Science/GBoF/hierarchical/plots'
    plt.savefig(os.path.join(plot_dir,'mBH1_box.png'), bbox_inches='tight')



##################################################################
### Run it!
###
##################################################################
if __name__ == "__main__":

    #####################################
    # Define command line option for the path to the simulations and the path to save the plots
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", dest='sample', help="Method used to sample massive clusters in FIREbox", type=str, default="SFR")
    parser.add_argument("--seed", dest='seed', help="Random seed used to sample massive clusters in FIREbox", type=int, default=42)
    args = parser.parse_args()

    w_dir = '/home/tristan-bruel/Documents/Science/GBoF/hierarchical'
    gw241011_color = '#1f78b4'
    gw241110_color = '#33a02c'

    plot_mBH1(sample=args.sample,seed=args.seed)
    plt.show()
