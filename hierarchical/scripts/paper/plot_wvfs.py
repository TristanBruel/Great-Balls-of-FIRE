import numpy as np
import h5py
import pandas as pd
import matplotlib.pyplot as plt

import os
import sys
import argparse



def plot_waveforms(event, det):
    CB_color_cycle = ['#377eb8', '#ff7f00', '#4daf4a',
                  '#f781bf', '#a65628', '#984ea3',
                  '#999999', '#e41a1c', '#dede00']
    gw_events = np.array(['GW241011', 'GW241110'])
    durations = [32., 16.]
    start_times = [1412725102.970, 1415277687.667]
    trigger_times = [1412725132.9554443, 1415277701.675049]
    post_trigger_duration = 2  # Time between trigger time and end of segment

    if event not in ['GW241011','GW241110']:
        raise ValueError('Events must be part of %s' %gw_events)
    n = np.where(gw_events==event)[0][0]
    trigger_time = trigger_times[n]
    xmin = - 0.03
    xmax = + 0.02

    fig, ax = plt.subplots(1, 1, figsize=(7., 3.))

    strain_filename = './waveforms/strain_%s_%s.txt' %(event,det)
    times_det, strain_det = np.loadtxt(strain_filename).T
    l, = ax.plot(times_det - trigger_time, strain_det,
            color="#ff7f0e", alpha=0.3,
            )
    alt_legend = plt.legend([l], ['Whitened strain (%s)' %det], frameon=False, loc='upper left')
    ax.add_artist(alt_legend)

    wvf_filename = './waveforms/waveforms_std_%s_%s.txt' %(event,det)
    if not os.path.exists(wvf_filename):
        raise ValueError('Waveforms for event %s in detector %s have not been computed.' %(event,det))
    dat = np.loadtxt(wvf_filename).T
    times_std = dat[0]
    wvf_low = dat[1]
    wvf_mean = dat[2]
    wvf_high = dat[3]

    ax.plot(times_std-trigger_time, wvf_mean,
            color=CB_color_cycle[2*n], label='std LVK',
            )
    ax.fill_between(times_std-trigger_time,
                    wvf_low,
                    wvf_high,
                    color=CB_color_cycle[2*n], alpha=0.3,
                   )

    ## Astro priors ##
    wvf_filename = './waveforms/waveforms_astro_%s_%s.txt' %(event,det)
    if not os.path.exists(wvf_filename):
        raise ValueError('Waveforms for event %s in detector %s have not been computed.' %(event,det))
    dat = np.loadtxt(wvf_filename).T
    times_astro = dat[0]
    wvf_low = dat[1]
    wvf_mean = dat[2]
    wvf_high = dat[3]

    ax.plot(times_astro-trigger_time, wvf_mean,
            color=CB_color_cycle[2*n+1], label='astro',
            )
    ax.fill_between(times_astro-trigger_time,
                    wvf_low,
                    wvf_high,
                    color=CB_color_cycle[2*n+1], alpha=0.3,
                   )
    
    ax.set_xlabel(r"$t - t_\mathrm{trigger}\ [\mathrm{s}]$")
    ax.set_xlim(xmin, xmax)
    ax.set_ylabel(r"h")
    ax.set_ylim(-5, 5)
    ax.legend(loc='upper right', title='Waveforms')


##################################################################
### Run it!
###
##################################################################
if __name__ == "__main__":
    

    #####################################
    # Define command line options
    parser = argparse.ArgumentParser()
    parser.add_argument("--event", help="GW event to consider", default='GW241011')
    parser.add_argument("--det", help="Detector to use for the gw strain", default='H1')
    args = parser.parse_args()

    #times_det, strain_det, ifo = load_strain(event=args.event, det=args.det)
    plot_waveforms(event=args.event, det=args.det)
    plt.show()
