import numpy as np
import h5py
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.lines import Line2D
import bilby

import os
import sys
import argparse


def plot_posteriors_reweight(events):

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

    for event in events:
        if event not in gw_events:
            raise ValueError('Events must be part of %s' %gw_events)
        n = np.where(gw_events==event)[0][0]

        fig, ax = plt.subplots(1, 1, figsize=(3.5,3.5))
        m1bins = np.linspace(5.,35.,101)

        PE_dir = '/home/tristan-bruel/Documents/Science/LVK/PE/%s' %event

        ## Load PE samples from standard LVK priors ##
        '''
        file = os.path.join(PE_dir,'standard/web/samples/posterior_samples.h5')
        with h5py.File(file, 'r') as h:
            pos = h['C00:IMRPhenomXPHM']['posterior_samples']
            m1_std = pos['mass_1_source']
            q_std = pos['mass_ratio']
            Mc_std = pos['chirp_mass']
            a1_std = pos['a_1']
            a2_std = pos['a_2']
            tilt1_std = pos['tilt_1']
            tilt2_std = pos['tilt_2']
            Dl_std = pos['luminosity_distance']
            z_std = pos['redshift']
            log_prior_std = pos['log_prior']
        print('Number of PE samples in my standard analysis:', len(m1_std))
        '''
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
        print('Number of PE samples in my analysis with standard LVK priors:', len(m1_std))
    
        ## Load PE samples from astro priors ##
        '''
        file = os.path.join(PE_dir,'astro_pop_prior/web/samples/posterior_samples.h5')
        with h5py.File(file, 'r') as h:
            pos = h['C00:IMRPhenomXPHM']['posterior_samples']
            m1_astro = pos['mass_1_source']
            q_astro = pos['mass_ratio']
            Mc_astro = pos['chirp_mass']
            a1_astro = pos['a_1']
            a2_astro = pos['a_2']
            Dl_astro = pos['luminosity_distance']
            z_astro = pos['redshift']
            log_prior_astro = pos['log_prior']
        print('Number of PE samples in my standard analysis:', len(m1_astro))
        '''
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
            log_prior_astro = pos['log_prior'][:]
        print('Number of PE samples in my analysis with astro priors:', len(m1_astro))

        astro_prior_file = '/home/tristan-bruel/Documents/Science/LVK/PE/%s/astro_pop_prior/%s_prior.json' %(event,event)
        astro_prior = bilby.gw.prior.BBHPriorDict.from_json(astro_prior_file)
        reweight_std_to_astro = np.exp([
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
            ]) / np.exp(log_prior_std)
        reweight_std_to_astro /= np.sum(reweight_std_to_astro)

        hist, _ = np.histogram(m1_std, bins=m1bins, density=True)
        ax.stairs(hist, m1bins, color=CB_color_cycle[2*n], label='std LVK prior')

        hist, _ = np.histogram(m1_astro, bins=m1bins, density=True)
        ax.stairs(hist, m1bins, color=CB_color_cycle[2*n+1], label='astro prior')

        hist, _ = np.histogram(m1_std, bins=m1bins, weights=reweight_std_to_astro, density=True)
        ax.stairs(hist, m1bins, color='k', label='std LVK prior\nre-weighted')

        ax.set_xlabel(r'$m_1\ [\mathrm{M}_\odot]$')
        ax.set_xticks(np.linspace(5,35,7))
        ax.set_ylabel(r'$p(m_1)\ [\mathrm{M}_\odot^{-1}]$')
        ax.legend(loc='upper right', title=event)
        ax.set_box_aspect(1)

        # Save
        plot_dir = '/home/tristan-bruel/Documents/Science/Great-Balls-of-FIRE/hierarchical/plots/paper'
        filename = 'PE_reweight_%s'%event+'.pdf'
        plt.savefig(os.path.join(plot_dir,filename), format='pdf', bbox_inches='tight')



##################################################################
### Run it!
###
##################################################################
if __name__ == "__main__":
    

    #####################################
    # Define command line options
    parser = argparse.ArgumentParser()
    parser.add_argument("--events", help="GW event to consider", nargs='+', default=['GW241011','GW241110'])
    args = parser.parse_args()

    plot_posteriors_reweight(events=args.events)
    plt.show()
