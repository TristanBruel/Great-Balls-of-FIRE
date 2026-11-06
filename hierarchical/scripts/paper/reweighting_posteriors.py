import numpy as np
import h5py
import pandas as pd
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.lines import Line2D
from matplotlib import colors
import bilby

from astropy.cosmology import Planck18, z_at_value
import astropy.units as u

import os
import sys
import argparse



def plot_posteriors_reweight(event, plot_dir=None):

    #####################################
    # Plot your result
    fs = 10
    lw = 1.4
    plt.rcParams['font.size']=fs
    plt.rcParams['font.family']='serif'
    plt.rcParams['font.serif']='cmr10'
    plt.rcParams['mathtext.fontset']='cm'
    plt.rcParams['axes.unicode_minus']=False
    plt.rcParams['axes.formatter.use_mathtext']=True
    plt.rcParams['lines.linewidth']=lw
    plt.rcParams['xtick.labelsize']=fs
    plt.rcParams['ytick.labelsize']=fs
    plt.rcParams['legend.fontsize']=.9*fs
    plt.rcParams['axes.titlesize']=fs

    CB_color_cycle = ['#377eb8', '#ff7f00', '#4daf4a',
                  '#f781bf', '#a65628', '#984ea3',
                  '#999999', '#e41a1c', '#dede00']
    gw_events = np.array(['GW241011', 'GW241110'])

    if event not in gw_events:
        raise ValueError('Events must be part of %s' %gw_events)
    n = np.where(gw_events==event)[0][0]

    PE_dir = '/home/tristan-bruel/Documents/Science/LVK/PE/%s' %event

    ## Load PE samples from standard LVK priors ##
    file = os.path.join(PE_dir,'standard/%s_merged_result.hdf5' %event)
    with h5py.File(file, 'r') as h:
        pos = h['posterior']
        m1_std = pos['mass_1_source'][:]
        q_std = pos['mass_ratio'][:]
        Mc_std = pos['chirp_mass'][:]
        a1_std = pos['a_1'][:]
        a2_std = pos['a_2'][:]
        tilt1_std = pos['tilt_1'][:]
        tilt2_std = pos['tilt_2'][:]
        Dl_std = pos['luminosity_distance'][:]
        z_std = pos['redshift'][:]
        log_prior_std = pos['log_prior'][:]
    print('\tNumber of PE samples in my analysis with standard LVK priors:', len(m1_std))

    std_prior_file = '/home/tristan-bruel/Documents/Science/LVK/PE/%s/standard/%s_prior.json' %(event,event)
    std_prior = bilby.gw.prior.BBHPriorDict.from_json(std_prior_file)
    std_samples = std_prior.sample(100_000)
    ## convert luminosity distance to redshift ##
    z_grid = np.linspace(0.01, 2.5, 1000)
    dL = Planck18.luminosity_distance(z_grid).value
    dL_to_z = interp1d(x=dL, y=z_grid, fill_value='extrapolate')
    std_samples_z = dL_to_z(std_samples['luminosity_distance'])
    #std_samples_z = z_at_value(Planck18.luminosity_distance, std_samples['luminosity_distance'] *u.Mpc).value
    std_samples_m1s = std_samples['chirp_mass'] *(1+std_samples['mass_ratio'])**(1/5) /std_samples['mass_ratio']**(3/5) /(1+std_samples_z)

    ## Load PE samples from astro priors ##
    file = os.path.join(PE_dir,'astro_pop_prior/%s_astro_merged_result.hdf5' %event)
    with h5py.File(file, 'r') as h:
        pos = h['posterior']
        m1_astro = pos['mass_1_source'][:]
        q_astro = pos['mass_ratio'][:]
        Mc_astro = pos['chirp_mass'][:]
        a1_astro = pos['a_1'][:]
        a2_astro = pos['a_2'][:]
        Dl_astro = pos['luminosity_distance'][:]
        z_astro = pos['redshift'][:]
        #log_prior_astro = pos['log_prior'][:]
    print('\tNumber of PE samples in my analysis with astro priors:', len(m1_astro))

    astro_prior_file = '/home/tristan-bruel/Documents/Science/LVK/PE/%s/astro_pop_prior/%s_prior.json' %(event,event)
    astro_prior = bilby.gw.prior.BBHPriorDict.from_json(astro_prior_file)
    log_prior_astro = np.array([
        astro_prior.ln_prob(
            {
                'mass_1_source': m1_std[i],
                'mass_ratio': q_std[i],
                'a_1': a1_std[i],
                'a_2': a2_std[i],
                'tilt_1': tilt1_std[i],
                'tilt_2': tilt2_std[i],
                'redshift': z_std[i],
            },
            normalized=False,
        )
        for i in range(len(m1_std))
        ])
    reweight_std_to_astro = np.exp(log_prior_astro) /np.exp(log_prior_std)
    reweight_std_to_astro /= np.sum(reweight_std_to_astro)

    #astro_samples = astro_prior.sample(1000)
    #astro_samples_m1s = astro_samples['mass_1_source']
    #astro_samples_q = astro_samples['mass_ratio']
    #astro_samples_a1 = astro_samples['a_1']

    ### Reweight source-frame primary mass samples ###
    fig, ax = plt.subplots(1, 1, figsize=(3.5,3.5))
    m1bins = np.linspace(5.,35.,51)
    m1_grid = np.linspace(5.,35.,1000)

    legend_elements = []

    hist, _ = np.histogram(m1_std, bins=m1bins, density=True)
    st = ax.stairs(hist, m1bins, color=CB_color_cycle[2*n], 
                   #fill=True,
                   label='std',
                   )
    legend_elements.append(st)

    hist, _ = np.histogram(m1_astro, bins=m1bins, density=True)
    st = ax.stairs(hist, m1bins, color=CB_color_cycle[2*n+1], 
                   #fill=True,
                   label='astro',
                   )
    legend_elements.append(st)

    hist, _ = np.histogram(m1_std, bins=m1bins, weights=reweight_std_to_astro, density=True)
    st = ax.stairs(hist, m1bins, 
                   color='k', zorder=10,
                   label='std re-weighted',
                   )
    legend_elements.append(st)
    prim_legend = plt.legend(handles=legend_elements, loc='upper right', 
                             title='PE samples',
                             )
    ax.add_artist(prim_legend)

    ### Induced priors ###
    prior_legend = []
    hist, _ = np.histogram(std_samples_m1s, bins=m1bins, density=True)
    mid_bins = (m1bins[1:]+m1bins[:-1])/2
    p = ax.plot(mid_bins, hist, 
                   color=CB_color_cycle[2*n], linestyle='--',
                   label=r'$\pi_\mathrm{U}(m_1)$',
                   )
    prior_legend.append(p[0])

    p = ax.plot(m1_grid, astro_prior['mass_1_source'].prob(m1_grid),
                color=CB_color_cycle[2*n+1], linestyle='--',
                label=r'$\pi_\mathrm{A}(m_1)$',
                )
    prior_legend.append(p[0])
    alt_legend = plt.legend(handles=prior_legend, loc='center right', bbox_to_anchor=(1.0,0.54),
                            title='Priors',
                            )
    ax.add_artist(alt_legend)

    ax.set_xlabel(r'$m_1\ [\mathrm{M}_\odot]$')
    ax.set_xlim(5.,35.)
    ax.set_xticks(np.linspace(5,35,7))
    ax.set_ylabel(r'$p(m_1)\ [\mathrm{M}_\odot^{-1}]$')
    ax.set_title(event)
    ax.set_box_aspect(1)

    # Save
    filename = 'PE_m1_reweight_%s'%event+'.pdf'
    plt.savefig(os.path.join(plot_dir,filename), format='pdf', bbox_inches='tight')


    ### Reweight mass ratio samples ###
    """
    fig, ax = plt.subplots(1, 1, figsize=(3.5,3.5))
    qbins = np.linspace(0.,1.,51)
    legend_elements = []

    hist, _ = np.histogram(q_std, bins=qbins, density=True)
    st = ax.stairs(hist, qbins, color=CB_color_cycle[2*n], 
                   label='std',
                   )
    legend_elements.append(st)

    hist, _ = np.histogram(q_astro, bins=qbins, density=True)
    st = ax.stairs(hist, qbins, color=CB_color_cycle[2*n+1], 
                   label='astro',
                   )
    legend_elements.append(st)

    hist, _ = np.histogram(q_std, bins=qbins, weights=reweight_std_to_astro, density=True)
    st = ax.stairs(hist, qbins, 
                   color='k', zorder=10,
                   label='std re-weighted',
                   )
    legend_elements.append(st)
    prim_legend = plt.legend(handles=legend_elements, loc='upper left', title=event)
    ax.add_artist(prim_legend)

    ### Induced priors ###
    prior_legend = []
    hist, _ = np.histogram(std_samples_q, bins=qbins, density=True)
    st = ax.stairs(hist, qbins, 
                   color='gray',
                   hatch='//', alpha=0.7,
                   label=r'$\pi_\mathrm{U}(q)$',
                   )
    st._hatch_color = colors.to_rgba(CB_color_cycle[2*n])
    prior_legend.append(st)

    hist, _ = np.histogram(astro_samples_q, bins=qbins, density=True)
    st = ax.stairs(hist, qbins,
                   color='gray',
                   hatch=r'\\', alpha=0.7,
                   label=r'$\pi_\mathrm{A}(q)$',
                   )
    st._hatch_color = colors.to_rgba(CB_color_cycle[2*n+1])
    prior_legend.append(st)
    alt_legend = plt.legend(handles=prior_legend, loc='lower left')
    ax.add_artist(alt_legend)


    ax.set_xlabel(r'$q$')
    ax.set_xticks(np.linspace(0,1,5))
    ax.set_ylabel(r'$p(q)$')
    #ax.legend(loc='upper left', title=event)
    ax.set_box_aspect(1)

    # Save
    filename = 'PE_q_reweight_%s'%event+'.pdf'
    plt.savefig(os.path.join(plot_dir,filename), format='pdf', bbox_inches='tight')
    """


    ### Reweight primary spin magnitude samples ###
    fig, ax = plt.subplots(1, 1, figsize=(3.5,3.5))
    a1bins = np.linspace(0.,1.,51)
    a1_grid = np.linspace(0.,1.,1000)
    legend_elements = []

    hist, _ = np.histogram(a1_std, bins=a1bins, density=True)
    st = ax.stairs(hist, a1bins, color=CB_color_cycle[2*n], 
                   label='std',
                   )
    legend_elements.append(st)

    hist, _ = np.histogram(a1_astro, bins=a1bins, density=True)
    st = ax.stairs(hist, a1bins, color=CB_color_cycle[2*n+1], 
                   label='astro',
                   )
    legend_elements.append(st)

    hist, _ = np.histogram(a1_std, bins=a1bins, weights=reweight_std_to_astro, density=True)
    st = ax.stairs(hist, a1bins, 
                   color='k', zorder=10,
                   label='std re-weighted',
                   )
    legend_elements.append(st)
    prim_legend = plt.legend(handles=legend_elements, loc='upper left', 
                             title='PE samples',
                             )
    ax.add_artist(prim_legend)

    ### Induced priors ###
    prior_legend = []
    p = ax.plot(a1_grid, std_prior['a_1'].prob(a1_grid),
                   color=CB_color_cycle[2*n], linestyle='--',
                   label=r'$\pi_\mathrm{U}(\chi_1)$',
                   )
    prior_legend.append(p[0])

    p = ax.plot(a1_grid, astro_prior['a_1'].prob(a1_grid),
                   color=CB_color_cycle[2*n+1], linestyle='--',
                   label=r'$\pi_\mathrm{A}(\chi_1)$',
                   )
    prior_legend.append(p[0])
    if n==0:
        alt_legend = plt.legend(handles=prior_legend, loc='center left', bbox_to_anchor=(0.0,.54),
                                title='Priors',
                                )
    else:
        alt_legend = plt.legend(handles=prior_legend, loc='upper left', bbox_to_anchor=(0.5,1.0),
                                title='Priors',
                                )
    ax.add_artist(alt_legend)


    ax.set_xlabel(r'$\chi_1$')
    ax.set_xlim(0.,1.)
    ax.set_xticks(np.linspace(0,1,6))
    ax.set_ylabel(r'$p(\chi_1)$')
    ax.set_title(event)
    ax.set_box_aspect(1)

    # Save
    filename = 'PE_a1_reweight_%s'%event+'.pdf'
    plt.savefig(os.path.join(plot_dir,filename), format='pdf', bbox_inches='tight')



##################################################################
### Run it!
###
##################################################################
if __name__ == "__main__":
    
    plot_dir = '/home/tristan-bruel/Documents/Science/Great-Balls-of-FIRE/hierarchical/paper/Figures/PE'
    #####################################
    # Define command line options
    parser = argparse.ArgumentParser()
    parser.add_argument("--events", help="GW event to consider", nargs='+', default=['GW241011','GW241110'])
    args = parser.parse_args()

    for event in args.events:
        print('\nEvent:', event)
        plot_posteriors_reweight(event=event, plot_dir=plot_dir)
    plt.show()
