import numpy as np
import h5py
import matplotlib.pyplot as plt
from matplotlib import gridspec
from scipy.interpolate import interp1d
import json
from popsummary.popresult import PopulationResult
from kaydee import KDE

import os
import sys
import argparse
from glob import glob
from tqdm import tqdm

from astropy.cosmology import Planck18 as planck
from colossus.cosmology import cosmology
cosmology.setCosmology('planck18');
from colossus.lss import mass_function


def HaloMassFunction():
    dndM = lambda M: mass_function.massFunction(M, 0.0, mdef='vir', model='tinker08', q_out='dndlnM')
    return dndM


def plot_zevol_mBH1(method='Gamma'):

    snapshot_file = os.path.join(FIREbox_dir, 'snapshot_times.txt')
    redshifts, times = np.loadtxt(snapshot_file, usecols=[2,3]).T
    tmax = times.max()
    deltat = 150 # width of the time bins (in Myr)
    nbins = round(1e3*tmax /deltat) + 1
    tbins = np.linspace(0, tmax, nbins)

    times_to_redshifts = interp1d(times, redshifts, fill_value='extrapolate')
    redshifts_to_times = interp1d(redshifts, times, fill_value='extrapolate')
    tz02 = redshifts_to_times(0.2)
    tz1 = redshifts_to_times(1.0)
    ind_z02 = np.where(tbins>tz02)[0][0]
    ind_z1 = np.where(tbins>tz1)[0][0]

    #####################################
    # Plot your result
    fs = 10
    lw = 1.4
    plt.rcParams['font.size']=fs
    plt.rcParams['font.family']='serif'
    plt.rcParams['font.serif']='cmr10'
    #plt.rcParams['text.usetex'] = True
    plt.rcParams['mathtext.fontset']='cm'
    plt.rcParams['axes.unicode_minus']=False
    plt.rcParams['axes.formatter.use_mathtext']=True
    plt.rcParams['lines.linewidth']=lw
    plt.rcParams['xtick.labelsize']=fs
    plt.rcParams['ytick.labelsize']=fs
    plt.rcParams['legend.fontsize']=.9*fs

    spm_color = '#CD5C5C'
    wpm_color = '#2C5AA0'

    ### Astro ###
    fig, axes = plt.subplots(1, 2, figsize=(8.,3.), 
            gridspec_kw={'width_ratios': [1.5, 1]},
            )
    fig.subplots_adjust(wspace=0.1)

    ## MY CLUSTER SIMULATIONS ##
    m1bins = np.linspace(0,100,41)
    midbins = (m1bins[:-1]+m1bins[1:]) /2
    qbins = np.linspace(0,1,41)
    midqbins = (qbins[:-1]+qbins[1:]) /2
    bbhfiles = glob(os.path.join(bbhs_dir, 'BBHs_{}_seed*.h5'.format(method)))
    #bbhfiles = glob(os.path.join(bbhs_dir, 'BBHs_{}_withKremer_seed*.h5'.format(method)))
    dRdm1_local = np.zeros((len(bbhfiles),len(m1bins)-1))
    dRdq_local = np.zeros((len(bbhfiles),len(qbins)-1))
    dRdm1_astro = np.zeros((len(bbhfiles),len(m1bins)-1))
    dRdq_astro = np.zeros((len(bbhfiles),len(qbins)-1))
    for n,bbhfile in tqdm(enumerate(bbhfiles)):
        with h5py.File(bbhfile, 'r') as bbhs:
            m1 = bbhs['m1'][:]
            m2 = bbhs['m2'][:]
            a1 = bbhs['a1'][:]
            a2 = bbhs['a2'][:]
            zmerger = bbhs['zmerger'][:]
            weights = bbhs['weights'][:]
        
        a1[m1<m2], a2[m1<m2] = a2[m1<m2], a1[m1<m2]
        m1 = np.maximum(m1, m2)
        m2 = np.minimum(m1, m2)
        tmerger = redshifts_to_times(zmerger)
        merger_nbr, _ = np.histogram(tmerger, bins=tbins, weights=weights)
        mergerrate = merger_nbr /np.diff(tbins) /FIREbox_volume # unit Gpc-3 yr-1
        Rz02 = mergerrate[ind_z02]
        ## Mergers up to z=0.2 ##
        local = (zmerger < 0.2) & (zmerger > 0.)
        m1_local = m1[local]
        m2_local = m2[local]
        a1_local = a1[local]
        a2_local = a2[local]
        zmerger_local = zmerger[local]
        weights_local = weights[local]
        hist, _ = np.histogram(m1_local, bins=m1bins, weights=weights_local)
        norm = Rz02 /np.sum(hist *np.diff(m1bins))
        dRdm1_local[n,:] = hist * norm
        hist, _ = np.histogram(m2_local /m1_local, bins=qbins, weights=weights_local)
        normq = Rz02 /np.sum(hist *np.diff(qbins))
        dRdq_local[n,:] = hist * normq
        ## Between redshift 0.2 and 1.0 ##
        astro = (zmerger < 1.) & (zmerger > 0.2)
        m1_astro = m1[astro]
        m2_astro = m2[astro]
        a1_astro = a1[astro]
        a2_astro = a2[astro]
        zmerger_astro = zmerger[astro]
        weights_astro = weights[astro]
        hist, _ = np.histogram(m1_astro, bins=m1bins, weights=weights_astro)
        norm = Rz02 /np.sum(hist *np.diff(m1bins))
        dRdm1_astro[n,:] = hist * norm
        hist, _ = np.histogram(m2_astro /m1_astro, bins=qbins, weights=weights_astro)
        normq = Rz02 /np.sum(hist *np.diff(qbins))
        dRdq_astro[n,:] = hist * normq

    ## Primary masses ##
    dRdm1_local = np.asarray(dRdm1_local)
    R_low = np.percentile(dRdm1_local, 10, axis=0)
    R_up = np.percentile(dRdm1_local, 90, axis=0)
    R_med = np.percentile(dRdm1_local, 50, axis=0)
    axes[0].plot(midbins, R_med, 
            c='royalblue', 
            label=r'$z\leq0.2$',
            zorder=4,
            )
    axes[0].fill_between(midbins, R_low, R_up,
            alpha=0.4, color='royalblue',
            zorder=4,
            )

    dRdm1_astro = np.asarray(dRdm1_astro)
    R_low = np.percentile(dRdm1_astro, 10, axis=0)
    R_up = np.percentile(dRdm1_astro, 90, axis=0)
    R_med = np.percentile(dRdm1_astro, 50, axis=0)
    axes[0].plot(midbins, R_med, 
            c='firebrick', 
            label=r'$0.2\leq z\leq1.0$',
            zorder=4,
            )
    axes[0].fill_between(midbins, R_low, R_up,
            alpha=0.4, color='firebrick',
            zorder=4,
            )

    ## Mass ratios ##
    dRdq_local = np.asarray(dRdq_local)
    R_low = np.percentile(dRdq_local, 10, axis=0)
    R_up = np.percentile(dRdq_local, 90, axis=0)
    R_med = np.percentile(dRdq_local, 50, axis=0)
    axes[1].plot(midqbins, R_med,
            c='royalblue',
            zorder=4,
            )
    axes[1].fill_between(midqbins, R_low, R_up,
            alpha=0.4, color='royalblue',
            zorder=4,
            )

    dRdq_astro = np.asarray(dRdq_astro)
    R_low = np.percentile(dRdq_astro, 10, axis=0)
    R_up = np.percentile(dRdq_astro, 90, axis=0)
    R_med = np.percentile(dRdq_astro, 50, axis=0)
    axes[1].plot(midqbins, R_med,
            c='firebrick',
            zorder=4,
            )
    axes[1].fill_between(midqbins, R_low, R_up,
            alpha=0.4, color='firebrick',
            zorder=4,
            )


    axes[0].set_xlabel('$m_1\\ [\\rm{M}_\\odot]$')
    axes[0].set_xscale('log')
    axes[0].set_xlim(xmin=4, xmax=2e2)
    #axes[0].set_xticks(np.linspace(0,100,11))
    axes[0].set_ylabel('$\\left. \\frac{\\rm{d} \\mathcal{R}}{\\rm{d} m_1} \\right|_{z=0.2} [ {\\rm Gpc}^{-3} {\\rm yr}^{-1} \\rm{M}_{\\odot}^{-1}]$')
    axes[0].set_yscale('log')
    axes[0].set_ylim(ymin=3e-3, ymax=5e1)
    axes[0].legend(loc='upper left', bbox_to_anchor=(0.3,1.0), frameon=False)

    axes[1].set_xlabel('$q$')
    axes[1].set_xlim(xmin=0, xmax=1)
    axes[1].set_xticks(np.linspace(0,1,5))
    axes[1].tick_params(direction='in', which='both', right=True, left=False, labelleft=False, labelright=True)
    axes[1].set_ylabel('$\\left. \\frac{\\rm{d} \\mathcal{R}}{\\rm{d} q} \\right|_{z=0.2} \\, [{\\rm Gpc}^{-3} {\\rm yr }^{-1}]$')
    axes[1].yaxis.set_label_position("right") # Move the label to the right
    axes[1].set_yscale('log')
    axes[1].set_ylim(ymin=1e0, ymax=2e2)
    axes[1].set_yticks(np.array([1e1,1e2]))
    
    # Save
    plot_dir = '/home/tristan.bruel/Great-Balls-of-FIRE/hierarchical/plots/paper'
    plt.savefig(os.path.join(plot_dir,'zevol_m1_q.pdf'), format='pdf', bbox_inches='tight')



##################################################################
### Run it!
###
##################################################################
if __name__ == "__main__":

    FIREbox_dir = '/home/tristan.bruel/Great-Balls-of-FIRE/FIREbox'
    FIREbox_volume = (22.1)**3 # in unit Mpc^3
    bbhs_dir = '/home/tristan.bruel/Great-Balls-of-FIRE/hierarchical/BBHs'

    ### Astro ###
    #gwtc4_dir = '/common/tristan.bruel/LVK/o4a-astro'
    #sys.path.append(os.path.join(gwtc4_dir,'figure_scripts'))
    #import plot_funcs_bbh_mass as pf
    gwtc5_dir = '/common/tristan.bruel/LVK/gwtc5/'
    O4b_run_dir = os.path.join(gwtc5_dir,'popsummary_files')
    pixelpop_run_dir = os.path.join(gwtc5_dir,'popsummary_files/may26_datarelease')

    ### Detectable ###
    gwtc3_dir = '/common/tristan.bruel/LVK/gwtc3'

    #####################################
    # Define command line option for the path to the simulations and the path to save the plots
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", dest='method', help="Cluster formation model", type=str, default='Gamma')
    args = parser.parse_args()

    plot_zevol_mBH1(method=args.method)
    #plt.show()
