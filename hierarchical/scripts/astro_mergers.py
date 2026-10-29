import numpy as np
import h5py
import matplotlib.pyplot as plt

import os
import sys
from glob import glob

import popsummary
from astropy.cosmology import Planck15 



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

    file_bptp04 = os.path.join(gwtc4_dir, 'data_release/BBHMassSpinRedshift_BrokenPowerLawTwoPeaks_GaussianComponentSpins_PowerLawRedshift.h5')
    file_bs04 = os.path.join(gwtc4_dir, 'data_release/BBHMassSpinRedshift_BSplineIID.h5')
    bptp_o4 = popsummary.popresult.PopulationResult(file_bptp04)
    bs_o4 = popsummary.popresult.PopulationResult(file_bs04)
    z_list, R_z = bptp_o4.get_rates_on_grids('redshift')
    z_list = z_list[0]
    z02 = np.argmin(np.abs(z_list-0.2))
    bs_z_list, bs_R_z = bs_o4.get_rates_on_grids('rate_vs_redshift')
    bs_z_list = bs_z_list[0]
    bptp_m1, bptp_m1_pdfs = pf.get_params(bptp_o4, 'mass_1')
    bs_m1, bs_m1_pdfs = pf.get_params(bs_o4, 'rate_vs_mass_1_at_z0-2', rate=False)
    color1 = '#648FFF'
    color2 = '#FE6100'

    pf.plot_90CI(ax, bs_m1, bs_m1_pdfs, color=color1, label=r'B-Spline, GWTC-4.0', fill_alpha=0.4)
    #ax.plot([],[],color=color1, label=r'B-Spline, GWTC-4.0')
    pf.plot_90CI(ax, bptp_m1, bptp_m1_pdfs, color=color2, label=r'Broken Power Law + 2 Peaks, GWTC-4.0', fill_alpha=0.4)
    #ax.plot([],[],color=color2, label=r'Broken Power Law + 2 Peaks, GWTC-4.0')

    dVcdz02 = (4*np.pi)*Planck15.differential_comoving_volume(z_list[z02]).value*1e-9 # in unit Gpc^3

    m1bins = np.linspace(0,100,41)
    zbins=np.linspace(0,2,21)
    bbhfiles = glob(os.path.join(bbhs_dir, 'detectableBBHs_{}_seed*.h5'.format(method)))
    #bbhfiles = glob(os.path.join(bbhs_dir, 'detectableBBHs_{}_seed4.h5'.format(method)))
    dRdm1 = np.zeros((len(bbhfiles),len(m1bins)))
    for n,bbhfile in enumerate(bbhfiles):
        with h5py.File(bbhfile, 'r') as bbhs:
            m1 = bbhs['m1'][:]
            m2 = bbhs['m2'][:]
            z = bbhs['z'][:]
        pm1, _ = np.histogram(m1, bins=m1bins, density=True)
        dNdz, _ = np.histogram(z, bins=zbins)
        dNdz = dNdz /np.diff(zbins)
        Rz02 = dNdz[np.argmin(np.abs(zbins-0.2))] *(1+z_list[z02]) /(Tobs*dVcdz02)
        dRdm1[n,:] = np.append(0,Rz02*pm1)

    dRdm1 = np.asarray(dRdm1)
    R_low = np.percentile(dRdm1, 10, axis=0)
    R_up = np.percentile(dRdm1, 90, axis=0)
    R_med = np.percentile(dRdm1, 50, axis=0)
    ax.plot(m1bins, R_med, 
            c='forestgreen', 
            label='BBH mergers from GCs (Bruel et al. 2025)',
            )
    ax.fill_between(m1bins, R_low, R_up, 
                    alpha=0.4, color='forestgreen',
                    )

    ax.set_xlabel(r'$m_1\ [\mathrm{M}_\odot]$')
    ax.set_xlim(xmin=0, xmax=100)
    ax.set_xticks(np.linspace(0,100,11))
    ax.set_ylabel(r'$\mathrm{d}\mathcal{R}/\mathrm{d}m_1\ [\mathrm{Gpc}^{-3} \mathrm{yr}^{-1} \mathrm{M}_\odot^{-1}]$')
    ax.set_yscale('log')
    ax.set_ylim(ymin=1e-3, ymax=1e2)
    ax.legend(loc='upper right', frameon=False)
    

    # Save
    plot_dir = '/home/tristan-bruel/Documents/Science/Great-Balls-of-FIRE/hierarchical/plots'
    plt.savefig(os.path.join(plot_dir,'mBH1_astro.png'), bbox_inches='tight')



##################################################################
### Run it!
###
##################################################################
if __name__ == "__main__":

    bbhs_dir = '/home/tristan-bruel/Documents/Science/Great-Balls-of-FIRE/hierarchical/BBHs'
    gwtc4_dir = '/home/tristan-bruel/Documents/Science/LVK/o4a-astro'
    sys.path.append(os.path.join(gwtc4_dir,'figure_scripts'))
    import plot_funcs_bbh_mass as pf

    Tobs = 1 # 1 year of observation time
    plot_mBH1_detectable()
    plt.show()
