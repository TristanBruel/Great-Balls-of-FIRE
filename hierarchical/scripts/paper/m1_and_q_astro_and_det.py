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


def plot_mBH1_astro_and_det(zeval=0.2, Tobs=1, method='Gamma'):

    snapshot_file = os.path.join(FIREbox_dir, 'snapshot_times.txt')
    redshifts, times = np.loadtxt(snapshot_file, usecols=[2,3]).T
    tmax = times.max()
    deltat = 150 # width of the time bins (in Myr)
    nbins = round(1e3*tmax /deltat) + 1
    tbins = np.linspace(0, tmax, nbins)

    times_to_redshifts = interp1d(times, redshifts, fill_value='extrapolate')
    redshifts_to_times = interp1d(redshifts, times, fill_value='extrapolate')
    tz02 = redshifts_to_times(zeval)
    ind_z02 = np.where(tbins>tz02)[0][0]

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

    with open(os.path.join(gwtc5_dir,"figure_scripts/one_percent_pe_limits.json")) as f:
        bounds = json.load(f)
    m1_ll_1percent, m1_ul_1percent = (bounds["mass_1_source"]["1st"], bounds["mass_1_source"]["99th"])
    q_ll_1percent, q_ul_1percent = (bounds["mass_ratio"]["1st"], bounds["mass_ratio"]["99th"])

    ### DEFAULT ###
    strong_bbh_O4b = PopulationResult(fname=f'{O4b_run_dir}/gwtc5_updated_default_mmax_mass_TwoPeakBrokenPowerLawSmoothedMassDistribution_redshift_PowerLawRedshift_magnitude_iid_spin_magnitude_gaussian_tilt_iid_spin_orientation_popsummary_result.h5')
    strong_bbh_O4b_m1_grid, strong_bbh_O4b_Rm1 = strong_bbh_O4b.get_rates_on_grids('mass_1')
    strong_bbh_O4b_m1_grid = strong_bbh_O4b_m1_grid.reshape((strong_bbh_O4b_m1_grid.size, ))
    strong_bbh_O4b_q_grid, strong_bbh_O4b_Rq = strong_bbh_O4b.get_rates_on_grids('mass_ratio')
    strong_bbh_O4b_q_grid = strong_bbh_O4b_q_grid.reshape((strong_bbh_O4b_q_grid.size, ))
    redshift_factor_strong_bbh_o4b = (1 + zeval)**(strong_bbh_O4b.get_hyperparameter_samples(hyperparameters=['lamb']))
    int_rates = np.trapezoid(x=strong_bbh_O4b_m1_grid,
                             y=redshift_factor_strong_bbh_o4b*strong_bbh_O4b_Rm1,
                             )
    Rlvk_default = np.quantile(int_rates,0.5)
    print('Median inferred merger rate density at redshift %.2f: %.2f Gpc^-3 yr^-1 (DEFAULT)\n' %(zeval,Rlvk_default))
    axes[0].plot(strong_bbh_O4b_m1_grid, 
            np.quantile(redshift_factor_strong_bbh_o4b * strong_bbh_O4b_Rm1, 0.5, axis=0),
            color=spm_color, label='Default BBH, GWTC-5.0', lw=1.5, zorder=3)

    axes[0].fill_between(strong_bbh_O4b_m1_grid,
            np.quantile(redshift_factor_strong_bbh_o4b * strong_bbh_O4b_Rm1, 0.05, axis=0),
            np.quantile(redshift_factor_strong_bbh_o4b * strong_bbh_O4b_Rm1, 0.95, axis=0), 
            color=spm_color, edgecolor=None, alpha=0.5, zorder=3)

    axes[1].semilogy(strong_bbh_O4b_q_grid, 
            np.quantile(redshift_factor_strong_bbh_o4b * strong_bbh_O4b_Rq , 0.5, axis=0),
            color=spm_color, label='pl in q, O4b', lw=1.5, zorder=3)
    axes[1].fill_between(strong_bbh_O4b_q_grid, 
            np.quantile(redshift_factor_strong_bbh_o4b * strong_bbh_O4b_Rq, 0.05, axis = 0), 
            np.quantile(redshift_factor_strong_bbh_o4b * strong_bbh_O4b_Rq, 0.95, axis = 0), 
            color=spm_color, edgecolor=None, alpha=0.5, zorder=3)

    ### PIXEL POP ###
    m1q_pixelpop = PopulationResult(fname=f'{pixelpop_run_dir}/bbh_m1q_varcut1/bbh_m1q_varcut1_popsummary.h5')
    m1m2_pixelpop = PopulationResult(fname=f'{pixelpop_run_dir}/m1m2_varcut1/m1m2_varcut1_popsummary.h5')
    (pixelpop_m1_grid, pixelpop_m2_grid), pixelpop_R = m1m2_pixelpop.get_rates_on_grids('joint_pixelpop_rate')
    pixelpop_m1_grid, pixelpop_Rm1 = m1m2_pixelpop.get_rates_on_grids('log_mass_1')
    pixelpop_m1_grid = pixelpop_m1_grid.reshape((pixelpop_m1_grid.size, ))
    pixelpop_q_grid, pixelpop_Rq = m1q_pixelpop.get_rates_on_grids('mass_ratio')
    pixelpop_q_grid = pixelpop_q_grid.reshape((pixelpop_q_grid.size, ))
    redshift_factor_pixelpop_m1 = (1 + zeval)**(m1m2_pixelpop.get_hyperparameter_samples(hyperparameters=['lamb']))
    redshift_factor_pixelpop_q = (1 + zeval)**(m1q_pixelpop.get_hyperparameter_samples(hyperparameters=['lamb']))
    int_rates_pixelpop = np.trapezoid(x=np.exp(pixelpop_m1_grid), 
                                      y=redshift_factor_pixelpop_m1*pixelpop_Rm1/np.exp(pixelpop_m1_grid),
                                      )
    Rlvk_pixelpop = np.quantile(int_rates_pixelpop,0.5)
    print('Median inferred merger rate density at redshift %.2f: %.2f Gpc^-3 yr^-1 (PIXELPOP)\n' %(zeval,Rlvk_pixelpop))

    axes[0].step(np.exp(pixelpop_m1_grid), 
            np.quantile(redshift_factor_pixelpop_m1 * pixelpop_Rm1 /np.exp(pixelpop_m1_grid) , 0.5, axis = 0),
            color=wpm_color, label='PixelPop, GWTC-5.0', lw=2.0, zorder=2)

    axes[0].fill_between(np.exp(pixelpop_m1_grid),
            np.quantile(redshift_factor_pixelpop_m1 * pixelpop_Rm1 / np.exp(pixelpop_m1_grid) , 0.05, axis = 0),
            np.quantile(redshift_factor_pixelpop_m1 * pixelpop_Rm1 / np.exp(pixelpop_m1_grid), 0.95, axis = 0),
            color=wpm_color, edgecolor=None, alpha=0.4, step='pre', zorder=2)

    axes[1].step(pixelpop_q_grid, 
            np.quantile(redshift_factor_pixelpop_q * pixelpop_Rq, 0.5, axis = 0), 
            color=wpm_color, label='$\\textsc{pixelpop}$', lw=2.0, zorder=2)
    axes[1].fill_between(pixelpop_q_grid, 
            np.quantile(redshift_factor_pixelpop_q * pixelpop_Rq, 0.05, axis = 0), 
            np.quantile(redshift_factor_pixelpop_q * pixelpop_Rq, 0.95, axis = 0), 
            color=wpm_color, edgecolor=None, alpha=0.4, step='pre', zorder=2)

    ## MY CLUSTER SIMULATIONS ##
    m1bins = np.linspace(0,100,41)
    midbins = (m1bins[:-1]+m1bins[1:]) /2
    qbins = np.linspace(0,1,41)
    midqbins = (qbins[:-1]+qbins[1:]) /2
    bbhfiles = glob(os.path.join(bbhs_dir, 'BBHs_{}_seed*.h5'.format(method)))
    #bbhfiles = glob(os.path.join(bbhs_dir, 'BBHs_{}_withKremer_seed*.h5'.format(method)))
    dRdm1 = np.zeros((len(bbhfiles),len(m1bins)-1))
    dRdq = np.zeros((len(bbhfiles),len(qbins)-1))
    dRdm1_hierar = np.zeros((len(bbhfiles),len(m1bins)-1))
    dRdq_hierar = np.zeros((len(bbhfiles),len(qbins)-1))
    fractions = np.zeros(len(bbhfiles))
    for n,bbhfile in tqdm(enumerate(bbhfiles)):
        with h5py.File(bbhfile, 'r') as bbhs:
            m1 = bbhs['m1'][:]
            m2 = bbhs['m2'][:]
            a1 = bbhs['a1'][:]
            a2 = bbhs['a2'][:]
            zmerger = bbhs['zmerger'][:]
            weights = bbhs['weights'][:]
        
        a1[m1<m2], a2[m1<m2] = a2[m1<m2], a1[m1<m2]
        astro = (zmerger < 1.) & (zmerger > 0.)
        m1 = np.maximum(m1, m2)
        m2 = np.minimum(m1, m2)
        m1 = m1[astro]
        m2 = m2[astro]
        a1 = a1[astro]
        a2 = a2[astro]
        zmerger = zmerger[astro]
        tmerger = redshifts_to_times(zmerger)
        weights = weights[astro]

        merger_nbr, _ = np.histogram(tmerger, bins=tbins, weights=weights)
        mergerrate = merger_nbr /np.diff(tbins) /FIREbox_volume # unit Gpc-3 yr-1
        Rz02 = mergerrate[ind_z02]
        fractions[n] = Rz02 /Rlvk_default
        hist, _ = np.histogram(m1, bins=m1bins, weights=weights)
        norm = Rz02 /np.sum(hist *np.diff(m1bins))
        dRdm1[n,:] = hist * norm
        hist, _ = np.histogram(m2 /m1, bins=qbins, weights=weights)
        normq = Rz02 /np.sum(hist *np.diff(qbins))
        dRdq[n,:] = hist * normq
        ## Hierarchical mergers ##
        hierar = a1 + a2 > 0
        hist, _ = np.histogram(m1[hierar], bins=m1bins, weights=weights[hierar])
        dRdm1_hierar[n,:] = hist * norm
        hist, _ = np.histogram(m2[hierar] /m1[hierar], bins=qbins, weights=weights[hierar])
        dRdq_hierar[n,:] = hist * normq

    dRdm1 = np.asarray(dRdm1)
    R = np.median(np.sum(dRdm1 *np.diff(m1bins),axis=1))
    print('Median BBH merger rate at redshift %.2f over %i realisations: %.2f Gpc^-3 yr^-1' %(zeval,len(bbhfiles),R))
    print('Median fraction of the total rate at redshift %.2f over %i realisations: %.2f' %(zeval,len(bbhfiles),np.median(fractions)))
    R_low = np.percentile(dRdm1, 10, axis=0)
    R_up = np.percentile(dRdm1, 90, axis=0)
    R_med = np.percentile(dRdm1, 50, axis=0)
    axes[0].plot(midbins, R_med, 
            c='forestgreen', 
            label='BBH mergers from MSCs',
            zorder=4,
            )
    axes[0].fill_between(midbins, R_low, R_up,
            alpha=0.4, color='forestgreen',
            zorder=4,
            )

    dRdq = np.asarray(dRdq)
    R_low = np.percentile(dRdq, 10, axis=0)
    R_up = np.percentile(dRdq, 90, axis=0)
    R_med = np.percentile(dRdq, 50, axis=0)
    axes[1].plot(midqbins, R_med,
            c='forestgreen',
            zorder=4,
            )
    axes[1].fill_between(midqbins, R_low, R_up,
            alpha=0.4, color='forestgreen',
            zorder=4,
            )

    dRdm1_hierar = np.asarray(dRdm1_hierar)
    Rhierar = np.median(np.sum(dRdm1_hierar *np.diff(m1bins),axis=1))
    print('Median hierarchical BBH merger rate at redshift %.2f over %i realisations: %.2f Gpc^-3 yr^-1' %(zeval,len(bbhfiles),Rhierar))
    Rhierar_low = np.percentile(dRdm1_hierar, 10, axis=0)
    Rhierar_up = np.percentile(dRdm1_hierar, 90, axis=0)
    Rhierar_med = np.percentile(dRdm1_hierar, 50, axis=0)
    axes[0].plot(midbins, Rhierar_med,
            c='darkorange',
            label='Hierarchical mergers',
            zorder=5,
            )
    axes[0].fill_between(midbins, Rhierar_low, Rhierar_up,
            alpha=0.4, color='darkorange',
            zorder=5,
            )

    axes[0].fill_betweenx(np.linspace(1e-3,1e2,100), 3, m1_ll_1percent, 
            alpha=0.15, facecolor='none', lw=0, edgecolor='lightgray', hatch='//////', zorder=1,
            )
    axes[0].fill_betweenx(np.linspace(1e-3,1e2,100), 3, m1_ll_1percent,
            alpha=0.75, ls='-.', lw=1.0, facecolor='none', edgecolor='k', zorder=1,
            )
    axes[0].fill_betweenx(np.linspace(1e-3,1e2,100), m1_ul_1percent, 300,
            alpha=0.15, facecolor='none', lw=0, edgecolor='lightgray', hatch='//////', zorder=1,
            )
    axes[0].fill_betweenx(np.linspace(1e-3,1e2,100), m1_ul_1percent, 300,
            alpha=0.75, ls='-.', lw=1.0, facecolor='none', edgecolor='k', zorder=1,
            )

    dRdq_hierar = np.asarray(dRdq_hierar)
    Rhierar_low = np.percentile(dRdq_hierar, 10, axis=0)
    Rhierar_up = np.percentile(dRdq_hierar, 90, axis=0)
    Rhierar_med = np.percentile(dRdq_hierar, 50, axis=0)
    axes[1].plot(midqbins, Rhierar_med,
            c='darkorange',
            label='Hierarchical mergers',
            zorder=5,
            )
    axes[1].fill_between(midqbins, Rhierar_low, Rhierar_up,
            alpha=0.4, color='darkorange',
            zorder=5,
            )

    axes[1].fill_betweenx(np.linspace(1e0,2e2,100), 0, q_ll_1percent,
            alpha=0.15, facecolor='none', lw=0, edgecolor='lightgray', hatch='//////', zorder=1,
            )
    axes[1].fill_betweenx(np.linspace(1e0,2e2,100), 0, q_ll_1percent,
            alpha=0.75, ls='-.', lw=1.0, facecolor='none', edgecolor='k', zorder=1,
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
    #plt.savefig(os.path.join(plot_dir,'m1_and_q_astro.png'), bbox_inches='tight')
    plt.savefig(os.path.join(plot_dir,'m1_and_q_astro.pdf'), format='pdf', bbox_inches='tight')


    ### Detectable ###
    print('\nComputing detectable population')
    fig, axes = plt.subplots(1, 2, figsize=(8.,3.), 
            gridspec_kw={'width_ratios': [1.5, 1]},
            )
    fig.subplots_adjust(wspace=0.1)

    file_ppds_m1 = os.path.join(gwtc3_dir,'obs_space/Ro_m1.dat')
    m1_grid, ppd_m1_med, ppd_m1_low, ppd_m1_up = np.loadtxt(file_ppds_m1).T
    file_ppds_q = os.path.join(gwtc3_dir,'obs_space/Ro_q.dat')
    q_grid, ppd_q_med, ppd_q_low, ppd_q_up = np.loadtxt(file_ppds_q).T
    #color_obs = '#648FFF'
    color_obs = 'gray'
    label_obs = 'Inference on observable population\n(Toubiana et al. 2025)'

    axes[0].fill_between(m1_grid,ppd_m1_low,ppd_m1_up,
            color=color_obs,edgecolor=None, alpha=0.4, step='pre', zorder=1)
    axes[0].plot(m1_grid,ppd_m1_med,
            color=color_obs,label=label_obs,lw=1.5,zorder=1)
    Ro_alex = np.trapezoid(x=m1_grid,y=ppd_m1_med)
    print('Median estimate observable merger rate: %.2f yr^-1' %Ro_alex)

    axes[1].fill_between(q_grid,ppd_q_low,ppd_q_up,
            color=color_obs,edgecolor=None, alpha=0.4, step='pre', zorder=1)
    axes[1].plot(q_grid,ppd_q_med,
            color=color_obs,label=label_obs,lw=1.5,zorder=1)

    z_list = np.linspace(0.,2.,2500)
    dzdt = interp1d(z_list[:-1], np.diff(z_list)/np.diff(planck.lookback_time(z_list)), fill_value='extrapolate')
    dVcdz = interp1d(z_list, (4*np.pi)*planck.differential_comoving_volume(z_list).value*1e-9, fill_value='extrapolate')

    dRodm1 = np.zeros((len(bbhfiles),len(m1bins)-1))
    dRodq = np.zeros((len(bbhfiles),len(qbins)-1))
    dRodm1_hierar = np.zeros((len(bbhfiles),len(m1bins)-1))
    dRodq_hierar = np.zeros((len(bbhfiles),len(qbins)-1))
    fractions = np.zeros(len(bbhfiles))
    for n,bbhfile in tqdm(enumerate(bbhfiles)):
        with h5py.File(bbhfile, 'r') as bbhs:
            m1 = bbhs['m1'][:]
            m2 = bbhs['m2'][:]
            a1 = bbhs['a1'][:]
            a2 = bbhs['a2'][:]
            zmerger = bbhs['zmerger'][:]
            pdet = bbhs['pdet'][:]
            weights = bbhs['weights'][:]

        a1[m1<m2], a2[m1<m2] = a2[m1<m2], a1[m1<m2]
        astro = (zmerger < 1.4) & (zmerger > 0)
        m1 = np.maximum(m1, m2)
        m2 = np.minimum(m1, m2)
        m1 = m1[astro]
        m2 = m2[astro]
        a1 = a1[astro]
        a2 = a2[astro]
        zmerger = zmerger[astro]
        tmerger = redshifts_to_times(zmerger)
        pdet = pdet[astro]
        pdet = pdet.reshape(len(pdet))
        weights = weights[astro]

        weights *= 1/FIREbox_volume *Tobs/(1+zmerger) *dVcdz(zmerger) *dzdt(zmerger)
        weights *= pdet

        hist, _ = np.histogram(m1, bins=m1bins, weights=weights)
        dRodm1[n,:] = hist /np.diff(m1bins)
        fractions[n] = np.sum(hist) /Ro_alex
        hist, _ = np.histogram(m2 /m1, bins=qbins, weights=weights)
        dRodq[n,:] = hist /np.diff(qbins)

        ## Hierarchical mergers ##
        hierar = a1 + a2 > 0
        hist, _ = np.histogram(m1[hierar], bins=m1bins, weights=weights[hierar])
        dRodm1_hierar[n,:] = hist /np.diff(m1bins)
        hist, _ = np.histogram(m2[hierar] /m1[hierar], bins=qbins, weights=weights[hierar])
        dRodq_hierar[n,:] = hist /np.diff(qbins)

    dRodm1 = np.asarray(dRodm1)
    Ro = np.median(np.sum(dRodm1 *np.diff(m1bins),axis=1))
    print('Median observable BBH merger rate over %i realisations: %.2f yr^-1' %(len(bbhfiles),Ro))
    print('Median fraction of the total rate over %i realisations: %.2f' %(len(bbhfiles),np.median(fractions)))
    R_low = np.percentile(dRodm1, 10, axis=0)
    R_up = np.percentile(dRodm1, 90, axis=0)
    R_med = np.percentile(dRodm1, 50, axis=0)
    axes[0].plot(midbins, R_med,
            c='forestgreen',
            label='Detectable BBH mergers from MSCs',
            zorder=2,
            )
    axes[0].fill_between(midbins, R_low, R_up,
            alpha=0.4, color='forestgreen',
            zorder=2,
            )

    dRodq = np.asarray(dRodq)
    R_low = np.percentile(dRodq, 10, axis=0)
    R_up = np.percentile(dRodq, 90, axis=0)
    R_med = np.percentile(dRodq, 50, axis=0)
    axes[1].plot(midqbins, R_med,
            c='forestgreen',
            label='Detectable BBH mergers from GCs',
            zorder=2,
            )
    axes[1].fill_between(midqbins, R_low, R_up,
            alpha=0.4, color='forestgreen',
            zorder=2,
            )

    dRodm1_hiera = np.asarray(dRodm1_hierar)
    Rohierar = np.median(np.sum(dRodm1_hierar *np.diff(m1bins),axis=1))
    print('Median observable hierarchical BBH merger rate over %i realisations: %.2f yr^-1' %(len(bbhfiles),Rohierar))
    Rhierar_low = np.percentile(dRodm1_hierar, 10, axis=0)
    Rhierar_up = np.percentile(dRodm1_hierar, 90, axis=0)
    Rhierar_med = np.percentile(dRodm1_hierar, 50, axis=0)
    axes[0].plot(midbins, Rhierar_med,
            c='darkorange',
            label='Detectable hierarchical mergers',
            zorder=3,
            )
    axes[0].fill_between(midbins, Rhierar_low, Rhierar_up,
            alpha=0.4, color='darkorange',
            zorder=3,
            )
    """
    axes[0].fill_betweenx(np.linspace(1e-3,1e2,100), 3, m1_ll_1percent,
            alpha=0.15, facecolor='none', lw=0, edgecolor='lightgray', hatch='//////', zorder=1,
            )
    axes[0].fill_betweenx(np.linspace(1e-3,1e2,100), 3, m1_ll_1percent,
            alpha=0.75, ls='-.', lw=1.0, facecolor='none', edgecolor='k', zorder=1,
            )
    axes[0].fill_betweenx(np.linspace(1e-3,1e2,100), m1_ul_1percent, 300,
            alpha=0.15, facecolor='none', lw=0, edgecolor='lightgray', hatch='//////', zorder=1,
            )
    axes[0].fill_betweenx(np.linspace(1e-3,1e2,100), m1_ul_1percent, 300,
            alpha=0.75, ls='-.', lw=1.0, facecolor='none', edgecolor='k', zorder=1,
            )
    """
    dRodq_hiera = np.asarray(dRodq_hierar)
    Rhierar_low = np.percentile(dRodq_hierar, 10, axis=0)
    Rhierar_up = np.percentile(dRodq_hierar, 90, axis=0)
    Rhierar_med = np.percentile(dRodq_hierar, 50, axis=0)
    axes[1].plot(midqbins, Rhierar_med,
            c='darkorange',
            label='Detectable hierarchical mergers',
            zorder=3,
            )
    axes[1].fill_between(midqbins, Rhierar_low, Rhierar_up,
            alpha=0.4, color='darkorange',
            zorder=3,
            )
    """
    axes[1].fill_betweenx(np.linspace(1e0,2e2,100), 0, q_ll_1percent,
            alpha=0.15, facecolor='none', lw=0, edgecolor='lightgray', hatch='//////', zorder=1,
            )
    axes[1].fill_betweenx(np.linspace(1e0,2e2,100), 0, q_ll_1percent,
            alpha=0.75, ls='-.', lw=1.0, facecolor='none', edgecolor='k', zorder=1,
            )
    """
    axes[0].set_xlabel('$m_1\\ [\\rm{M}_\\odot]$')
    axes[0].set_xscale('log')
    axes[0].set_xlim(xmin=4, xmax=2e2)
    #axes[0].set_xticks(np.linspace(0,100,11))
    axes[0].set_ylabel('$\\left. \\frac{\\rm{d} \\mathcal{R}_O}{\\rm{d} m_1} \\right [{\\rm yr}^{-1} \\rm{M}_{\\odot}^{-1}]$')
    axes[0].set_yscale('log')
    axes[0].set_ylim(ymin=1e-2, ymax=1e2)
    axes[0].legend(loc='upper right', frameon=False)

    axes[1].set_xlabel('$q$')
    axes[1].set_xlim(xmin=0, xmax=1)
    axes[1].set_xticks(np.linspace(0,1,5))
    axes[1].tick_params(direction='in', which='both', right=True, left=False, labelleft=False, labelright=True)
    axes[1].set_ylabel('$\\left. \\frac{\\rm{d} \\mathcal{R}_O}{\\rm{d} q} \\right [{\\rm yr}^{-1}]$')
    axes[1].yaxis.set_label_position("right") # Move the label to the right
    axes[1].set_yscale('log')
    axes[1].set_ylim(ymin=5e-1, ymax=5e2)
    axes[1].set_yticks(np.array([1e0,1e1,1e2]))

    # Save
    #plt.savefig(os.path.join(plot_dir,'m1_and_q_det.png'), bbox_inches='tight')
    plt.savefig(os.path.join(plot_dir,'m1_and_q_det.pdf'), format='pdf', bbox_inches='tight')



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
    parser.add_argument("--z", dest='zeval', help="Redshift at which the differential merger rate is evaluated", 
            type=float, default=0.2)
    parser.add_argument("--Tobs", dest='Tobs', help="Observation time for the detectable population", 
            type=float, default=1)
    parser.add_argument("--method", dest='method', help="Cluster formation model", type=str, default='Gamma')
    args = parser.parse_args()

    plot_mBH1_astro_and_det(zeval=args.zeval, Tobs=args.Tobs, method=args.method)
    #plt.show()
