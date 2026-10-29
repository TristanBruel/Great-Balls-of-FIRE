import numpy as np
import h5py
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import corner

import os
import argparse


def plot_posteriors(events):

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
    durations = [32., 16.]
    start_times = [1412725102.970, 1415277687.667]
    trigger_times = [1412725132.9554443, 1415277701.675049]
    ylim = [7,4]
    post_trigger_duration = 2  # Time between trigger time and end of segment

    for event in events:
        if event not in gw_events:
            raise ValueError('Events must be part of %s' %gw_events)
        n = np.where(gw_events==event)[0][0]

        fig = plt.figure(figsize=(7.,7.))

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
            chi_eff_std = pos['chi_eff']
            Dl_std = pos['luminosity_distance']
            z_std = pos['redshift']
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
            chi_eff_std = pos['chi_eff'][:]
            Dl_std = pos['luminosity_distance'][:]
            z_std = pos['redshift'][:]
        print('Number of PE samples in my analysis with standard LVK priors:', len(m1_std))
        data_std = np.array([Mc_std,q_std,m1_std,chi_eff_std,a1_std,z_std]).T
        labels = [r'$\mathcal{M}_\mathrm{c}$',r'$q$',r'$m_1$',r'$\chi_\mathrm{eff}$',r'$\chi_1$',r'$z$']
    
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
            chi_eff_astro = pos['chi_eff']
            Dl_astro = pos['luminosity_distance']
            z_astro = pos['redshift']
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
            chi_eff_astro = pos['chi_eff'][:]
            Dl_astro = pos['luminosity_distance'][:]
            z_astro = pos['redshift'][:]
        print('Number of PE samples in my analysis with astro priors:', len(m1_astro))
        data_astro = np.array([Mc_astro,q_astro,m1_astro,chi_eff_astro,a1_astro,z_astro]).T

        corner.corner(data_std, hist_kwargs={'density':True},
                      labels=labels,  color=CB_color_cycle[2*n], 
                      #quantiles=[0.05,0.5,0.95], 
                      show_titles=True, labelpad=0.05,
                      fig=fig,
                     )
        for k in range(len(labels)):
            ax = fig.axes[k*(len(labels)+1)]
            title = ax.get_title()
            ax.text(x=0.5,y=1.25,
                    verticalalignment='baseline', horizontalalignment='center', transform=ax.transAxes,
                    s=title,
                    color=CB_color_cycle[2*n],
                    )
            ax.set_title(None)

        corner.corner(data_astro, hist_kwargs={'density':True},
                      labels=labels,  color=CB_color_cycle[2*n+1], 
                      #quantiles=[0.05,0.5,0.95], 
                      show_titles=True, labelpad=0.05,
                      fig=fig,
                     )
        for k in range(len(labels)):
            ax = fig.axes[k*(len(labels)+1)]
            title = ax.get_title()
            ax.text(x=0.5,y=1.05,
                    verticalalignment='baseline', horizontalalignment='center', transform=ax.transAxes,
                    s=title,
                    color=CB_color_cycle[2*n+1],
                    )
            ax.set_title(None)


        ax_inset = fig.add_axes([0.572,0.725,0.4,0.245])
        det='H1'
        trigger_time = trigger_times[n]
        xmin = - 0.03
        xmax = + 0.02
        strain_filename = './waveforms/strain_%s_%s.txt' %(event,det)
        times_det, strain_det = np.loadtxt(strain_filename).T
        l, = ax_inset.plot(times_det - trigger_time, strain_det,
                color="#ff7f0e", alpha=0.3,
                )
        alt_legend = plt.legend([l], ['Whitened strain (%s)' %det], frameon=False, loc='upper left')
        ax_inset.add_artist(alt_legend)

        wvf_filename = './waveforms/waveforms_std_%s_%s.txt' %(event,det)
        if not os.path.exists(wvf_filename):
            raise ValueError('Waveforms for event %s in detector %s have not been computed.' %(event,det))
        dat = np.loadtxt(wvf_filename).T
        times_std = dat[0]
        wvf_low = dat[1]
        wvf_mean = dat[2]
        wvf_high = dat[3]
        ax_inset.plot(times_std-trigger_time, wvf_mean,
                color=CB_color_cycle[2*n], 
                label='std LVK',
                )
        ax_inset.fill_between(times_std-trigger_time,
                        wvf_low,
                        wvf_high,
                        color=CB_color_cycle[2*n], alpha=0.3,
                       )
        wvf_filename = './waveforms/waveforms_astro_%s_%s.txt' %(event,det)
        if not os.path.exists(wvf_filename):
            raise ValueError('Waveforms for event %s in detector %s have not been computed.' %(event,det))
        dat = np.loadtxt(wvf_filename).T
        times_astro = dat[0]
        wvf_low = dat[1]
        wvf_mean = dat[2]
        wvf_high = dat[3]
        ax_inset.plot(times_astro-trigger_time, wvf_mean,
                color=CB_color_cycle[2*n+1], 
                label='astro',
                )
        ax_inset.fill_between(times_astro-trigger_time,
                        wvf_low,
                        wvf_high,
                        color=CB_color_cycle[2*n+1], alpha=0.3,
                       )
        ax_inset.set_xlabel(r"$t - t_\mathrm{trigger}\ [\mathrm{s}]$")
        ax_inset.set_xlim(xmin, xmax)
        ax_inset.set_ylabel(r"h")
        ax_inset.set_ylim(-ylim[n], ylim[n])
        #ax_inset.legend(loc='upper left')

        line_std = Line2D([], [], color=CB_color_cycle[2*n], label='std LVK prior')
        line_astro = Line2D([], [], color=CB_color_cycle[2*n+1], label='astro prior')
        plt.legend(handles=[line_std,line_astro],
                   loc='center right',
                   bbox_to_anchor=(.9815,0.5625),
                   bbox_transform=fig.transFigure,
                   title=event,
                  )


        # Save
        plot_dir = '/home/tristan-bruel/Documents/Science/Great-Balls-of-FIRE/hierarchical/plots/paper'
        filename = 'PE_std_astro_%s'%event+'.pdf'
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

    plot_posteriors(events=args.events)
    plt.show()
