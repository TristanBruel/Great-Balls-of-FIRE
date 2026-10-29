import numpy as np
import h5py
import matplotlib.pyplot as plt
import matplotlib as mpl

import os
from glob import glob


def plot_mBH1_detectable(method='SFR'):

    #####################################
    # Plot your result
    fs = 12
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

    fig, ax = plt.subplots(1, 1, figsize=(8.,4.))

    file_ppds_m1 = os.path.join(gwtc3_dir,'Ro_m1.dat')
    m1_grid, ppd_m1_med, ppd_m1_low, ppd_m1_up = np.loadtxt(file_ppds_m1).T
    color_obs = '#648FFF'
    label_obs = 'Inference on observable population\n (Toubiana et al. 2026)'

    ax.fill_between(m1_grid,ppd_m1_low,ppd_m1_up,
            alpha=0.4,color=color_obs)
    ax.plot(m1_grid,ppd_m1_med,
            color=color_obs,label=label_obs)

    m1bins = np.linspace(0,100,41)
    bbhfiles = glob(os.path.join(bbhs_dir, 'detectableBBHs_{}_seed*.h5'.format(method)))
    #bbhfiles = glob(os.path.join(bbhs_dir, 'detectableBBHs_{}_seed4.h5'.format(method)))
    dRodm1 = np.zeros((len(bbhfiles),len(m1bins)))
    for n,bbhfile in enumerate(bbhfiles):
        with h5py.File(bbhfile, 'r') as bbhs:
            m1 = bbhs['m1'][:]
            m2 = bbhs['m2'][:]
            a1 = bbhs['a1'][:]
            pdet = bbhs['pdet'][:,0]
        merger_nbr, _ = np.histogram(m1, bins=m1bins, weights=pdet)
        dRodm1[n,:] = np.append(0,merger_nbr)

    dRodm1 = np.asarray(dRodm1)
    R_low = np.percentile(dRodm1, 10, axis=0)
    R_up = np.percentile(dRodm1, 90, axis=0)
    R_med = np.percentile(dRodm1, 50, axis=0)
    ax.plot(m1bins, R_med, 
            c='forestgreen', 
            label='Detectable BBH mergers from GCs\n (Bruel et al. 2025)',
            )
    ax.fill_between(m1bins, R_low, R_up, 
                    alpha=0.4, color='forestgreen',
                    )

    ax.set_xlabel(r'$m_1\ [\mathrm{M}_\odot]$')
    ax.set_xlim(xmin=0, xmax=100)
    ax.set_xticks(np.linspace(0,100,11))
    ax.set_ylabel(r'$\mathrm{d}\mathcal{R}_O/\mathrm{d}m_1\ [\mathrm{yr}^{-1} \mathrm{M}_\odot^{-1}]$')
    ax.set_yscale('log')
    ax.set_ylim(ymin=1e-2, ymax=1e2)
    ax.legend(loc='upper right', frameon=False)
    

    # Save
    plot_dir = '/home/tristan-bruel/Documents/Science/Great-Balls-of-FIRE/hierarchical/plots'
    plt.savefig(os.path.join(plot_dir,'mBH1_detectable.png'), bbox_inches='tight')



##################################################################
### Run it!
###
##################################################################
if __name__ == "__main__":

    bbhs_dir = '/home/tristan-bruel/Documents/Science/Great-Balls-of-FIRE/hierarchical/BBHs'
    gwtc3_dir = '/home/tristan-bruel/Documents/Science/LVK/obs_space/gwtc3'

    plot_mBH1_detectable()
    plt.show()
