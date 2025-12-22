import numpy as np
import h5py
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.optimize import curve_fit
from uncertainties import ufloat


import os


def func_powerlaw(x, A, alpha):
    return A*x**alpha



def plot_nBBHs_Mcl():

    #####################################
    # Plot your result
    fs = 10
    lw = 1.4
    plt.rc('font', family='serif')
    plt.rc('lines', linewidth=lw)
    plt.rc('text', usetex=True)

    CB_color_cycle = ['#377eb8', '#ff7f00', '#4daf4a',
                  '#f781bf', '#a65628', '#e41a1c']

    fig, ax = plt.subplots(1, 1, figsize=(3.5,3.5))

    mbins = np.logspace(np.log10(5e4), np.log10(6.5e6), 15)
    mid_bins = 10**((np.log10(mbins[:-1])+np.log10(mbins[1:]))/2)
    width_bins = np.diff(mbins) /1.5

    data = []
    ## Clusters that produce merging BBHs ##
    grid_cluster_file = os.path.join(FIRE_dir, 'grid/gridClusters.hdf5')
    with h5py.File(grid_cluster_file) as c:
        mass = c['Mass'][()]
        nbbhs = c['nBBHs'][()]

    for n in range(len(mbins)-1):
        in_bin = (mass>=mbins[n])&(mass<mbins[n+1])
        data.append(nbbhs[in_bin] /mass[in_bin])

    medianprops={"color": "firebrick", "linewidth": lw+.5}
    flierprops={"markerfacecolor": "gray", "markeredgecolor": "k", "markersize": 5}
    ax.boxplot(data,
               positions = mid_bins,
               widths = width_bins,
               medianprops = medianprops,
               flierprops = flierprops,
               )

    ax.scatter(np.max(mass), 
               nbbhs[np.argmax(mass)] /np.max(mass),
               facecolor="gray", edgecolor='k', s=5**2,
               )

    # Best-fit power-law #
    popt, pcov = curve_fit(func_powerlaw, mass, nbbhs/mass)
    sigma = np.sqrt(np.diagonal(pcov))
    A = ufloat(popt[0], sigma[0])
    alpha = ufloat(popt[1], sigma[1])
    print('Best-fit powerlaw nBBHs/Mcl = A * Mcl^alpha is:')
    print('A:{}, alpha={}'.format(A,alpha))

    mplot = np.logspace(4,7,3001)
    ax.plot(mplot, func_powerlaw(mplot, *popt),
            color='k', ls='-',
            label='Power-law fit',
            )
    bound_lower = func_powerlaw(mplot, *(popt-sigma))
    bound_upper = func_powerlaw(mplot, *(popt+sigma))
    ax.fill_between(mplot, bound_lower, bound_upper,
                    color='black', alpha = 0.15)


    ax.set_xlabel(r'$M_\mathrm{cl}\ [\mathrm{M}_\odot]$', fontsize=fs)
    ax.set_xscale('log')
    ax.set_xlim(xmin=5e4, xmax=1e7)
    ax.set_ylabel(r'$n_\mathrm{BBHs}/M_\mathrm{cl}\ [\mathrm{M}_\odot^{-1}]$', fontsize=fs)
    ax.ticklabel_format(axis='y', style='sci', scilimits=[1,3])
    #ax.set_ylim(ymin=-1.5e-5, ymax=3.5e-4)
    #ax.set_yticks(np.linspace(0., 3.5e-4, 8))
    #ax.yaxis.get_major_formatter().set_useOffset(True)
    ax.set_box_aspect(1)

    ax.legend(loc='upper left', fontsize=fs)
        
    # Save
    plot_dir = os.path.join(FIRE_dir, 'plots')
    plt.savefig(os.path.join(plot_dir,'nBBHs_Mcl.pdf'), format='pdf', bbox_inches='tight')



##################################################################
### Run it!
###
##################################################################
if __name__ == "__main__":

    FIRE_dir = '/home/tristan-bruel/Documents/Science/GBoF/FIRE-2'

    plot_nBBHs_Mcl()
    plt.show()
