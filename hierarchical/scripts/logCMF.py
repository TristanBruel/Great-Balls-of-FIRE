import numpy as np
import h5py
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.optimize import curve_fit

import os

def func_linear(x, A, B):
    return A + B*x

def func_powerlaw(x, A, B, alpha):
    return A + B*x**alpha


def plot_CMF():

    cluster_file = os.path.join(grid_dir, 'gridClusters_withKremer.hdf5')
    with h5py.File(cluster_file) as c:
        cluster_mass = c['Mass'][()]
        cluster_nBBHs = c['nBBHs'][()]
        cluster_id = c['id'][()]
    cluster_id = np.array(cluster_id, dtype='str')
    isKremer = np.array([f.split('_')[0]=='Kremer' for f in cluster_id])
    print('Number of clusters in the grid:', len(cluster_id))
    print('Number of clusters from Kyle Kremer simulations:', np.sum(isKremer))
    print('Number of clusters having at least 1 BBH merger:', np.sum(cluster_nBBHs>0))

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

    CB_color_cycle = ['#377eb8', '#ff7f00', '#4daf4a',
                      '#f781bf', '#a65628', '#e41a1c',
                      '#984ea3', '#999999', '#dede00']

    fig, ax = plt.subplots(1, 1, figsize=(5,5))

    logmbins = np.linspace(4.75,6.75,17)
    Ncl, _ = np.histogram(np.log10(cluster_mass), bins=logmbins)
    midbins = (logmbins[1:]+logmbins[:-1])/2
    Ncl = Ncl /np.diff(logmbins)
    ax.stairs(np.log10(Ncl), logmbins,
            color=CB_color_cycle[0], lw=lw, label='All massive star clusters',
            )

    # Best-fit power-law #
    popt, pcov = curve_fit(func_linear, midbins[Ncl>0], np.log10(Ncl[Ncl>0]))
    ax.plot(logmbins, func_linear(logmbins, *popt),
            color=CB_color_cycle[0], ls='--',
            label=r'$\mathrm{dN}/\mathrm{d}M_\mathrm{cl}\propto M_\mathrm{cl}^{%.2f}$'%(popt[1]-1),
            )
    print('Best-fit powerlaw dN/dMcl = A * Mcl^alpha is: alpha=%.2f' %(popt[1]-1))

    # Clusters that produce at least 1 BBH merger
    active = cluster_nBBHs>0
    Nactive, _ = np.histogram(np.log10(cluster_mass[active]), bins=logmbins)
    midbins = (logmbins[1:]+logmbins[:-1])/2
    Nactive = Nactive /np.diff(logmbins)
    ax.stairs(np.log10(Nactive), logmbins,
            color=CB_color_cycle[1], lw=lw, label='BBH-forming star clusters',
            )

    bbh_file = os.path.join(grid_dir, 'allBBHs_withKremer.dat')
    bbh_cl_id = np.loadtxt(bbh_file, usecols=0, skiprows=1, dtype=str)
    Z, m1, m2, tmerge, a1, a2 = np.loadtxt(bbh_file, usecols=[1,4,5,7,8,9]).T
    a1[m1<m2], a2[m1<m2] = a2[m1<m2], a1[m1<m2]
    m1, m2 = np.maximum(m1,m2), np.minimum(m1,m2)
    q = m2 /m1
    print('Total number of BBH mergers in CMC clusters: {}'.format(len(m1)))
    print('Number of hierarchical BBH mergers: {}\n'.format(np.sum(a1+a2>0)))

    # BBH mergers with stricly positive primary spin, m1 < 30 Msun, and unequal masses
    GW24x = (m1 < 30) & (q < 0.85) & (a1 > 0.05)
    unique, indexes = np.unique(bbh_cl_id[GW24x], return_index=True)
    lochost = np.array([np.where(cluster_id==u)[0][0] for u in unique])
    print('Looking for BBH mergers with stricly positive primary spin, m1 < 30 Msun, and unequal masses (q<0.85)')
    print('Number of events found:', np.sum(GW24x))
    print('Number of clusters hosting these events:', len(lochost))
    print('Total number of BBH mergers from these clusters:',np.sum(cluster_nBBHs[lochost]),'\n')
    Nhost, _ = np.histogram(np.log10(cluster_mass[lochost]), bins=logmbins)
    midbins = (logmbins[1:]+logmbins[:-1])/2
    Nhost = Nhost /np.diff(logmbins)
    ax.stairs(np.log10(Nhost), logmbins,
            color=CB_color_cycle[2], lw=lw, label='Star clusters with\nGW24x-like mergers',
            )

    ax.set_xlabel(r'$\mathrm{log}_{10}(M_\mathrm{cl}/\mathrm{M}_\odot)$')
    ax.set_xlim(xmin=4.5, xmax=7)
    ax.set_xticks(np.linspace(4.5,7,6))
    ax.set_ylabel(r'$\mathrm{log}_{10}(\mathrm{dN}/\mathrm{dlog}_{10}(M_\mathrm{cl}/\mathrm{M}_\odot))$')
    ax.set_ylim(ymin=.7, ymax=5)
    ax.set_box_aspect(1)
    ax.legend(loc='upper right')

    
    # Save
    plot_dir = '../plots'
    plt.savefig(os.path.join(plot_dir,'logCMF.pdf'), format='pdf', bbox_inches='tight')



##################################################################
### Run it!
###
##################################################################
if __name__ == "__main__":

    FIRE_dir = '/home/tristan-bruel/Documents/Science/Great-Balls-of-FIRE/FIRE-2'
    grid_dir = os.path.join(FIRE_dir,'grid')

    plot_CMF()
    plt.show()
