import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from scipy.interpolate import interp1d

import os

from colossus.cosmology import cosmology
cosmology.setCosmology('planck18');
from colossus.lss import mass_function

FIREbox_dir = '/home/tristan-bruel/Documents/Science/GBoF/FIREbox'
FIREbox_volume = (22.1)**3 # in unit Mpc^3

def HaloMassFunction():
    dndM = lambda M: mass_function.massFunction(M, 0.0, mdef='vir', model='tinker08', q_out='dndlnM')
    return dndM

def schechter(logm, logphi, logmstar, alpha, m_lower=None):
    """ Generate a Schechter function (in dlogm) """
    phi=(10**logphi *np.log(10) *10**((logm-logmstar) *(alpha+1)) *np.exp(-10**(logm-logmstar)))
    return phi

# GSMF from Tomczak et al. 2014
def Tomczak2014(logm):
    pars = {'logphi1': -2.44,
            'logphi2': -3.69,
            'logmstar': 10.79,
            'alpha1': -0.74,
            'alpha2': -1.75,
            }
    phi1 = schechter(logm, pars['logphi1'], pars['logmstar'], pars['alpha1']) 
    phi2 = schechter(logm, pars['logphi2'], pars['logmstar'], pars['alpha2'])
    phi = phi1 + phi2
    return phi


# GSMF extrapolation from Leja et al. 2020
def parameter_at_z0(y,z0,z1=0.2,z2=1.6,z3=3.0):
    """ Compute parameter at redshift ‘z0‘ as a function of the polynomial parameters ‘y‘ and the redshift anchor points ‘z1‘, ‘z2‘, and ‘z3‘.
    """
    y1, y2, y3 = y
    a = (((y3-y1) + (y2-y1) / (z2-z1) * (z1-z3)) /(z3**2-z1**2 + (z2**2-z1**2) / (z2-z1) *(z1-z3)))
    b = ((y2-y1) - a * (z2**2-z1**2)) / (z2-z1)
    c = y1-a * z1**2-b * z1
    return a * z0**2 + b*z0 + c

def Leja2020(logm, z0):
    # Draw samples from posterior assuming independent Gaussian uncertainties.
    # Then convert to mass function at ‘z=z0‘.
    pars = {'logphi1': [-2.44, -3.08, -4.14],
            'logphi1_err': [0.02, 0.03, 0.1],
            'logphi2': [-2.89, -3.29, -3.51],
            'logphi2_err': [0.04, 0.03, 0.03],
            'logmstar': [10.79,10.88,10.84],
            'logmstar_err': [0.02, 0.02, 0.04],
            'alpha1': [-0.28],
            'alpha1_err': [0.07],
            'alpha2': [-1.48],
            'alpha2_err': [0.1]}
    draws = {}
    ndraw = 1000
    for par in ['logphi1', 'logphi2', 'logmstar', 'alpha1', 'alpha2']:
        samp = np.array([np.random.normal(median,scale=err,size=ndraw) for median, err in zip(pars[par], pars[par+'_err'])])
        if par in ['logphi1', 'logphi2', 'logmstar']:
            draws[par]=parameter_at_z0(samp,z0)
        else:
            draws[par]=samp.squeeze()
    # Generate Schechter functions.
    phi1 = schechter(logm, draws['logphi1'], draws['logmstar'], draws['alpha1']) 
    phi2 = schechter(logm, draws['logphi2'], draws['logmstar'], draws['alpha2'])
    phi = phi1 + phi2
    # Compute median and 1-sigma uncertainties as a function of mass.
    phi_50, phi_84, phi_16 = np.percentile(phi, [50, 84, 16], axis=1)
    return phi_50, phi_84, phi_16


def plot_HMF_SMF():

    snapshot_file = os.path.join(FIREbox_dir, 'snapshot_times.txt')
    redshifts, times = np.loadtxt(snapshot_file, usecols=[2,3]).T

    times_to_redshifts = interp1d(times, redshifts, fill_value='extrapolate')
    redshifts_to_times = interp1d(redshifts, times, fill_value='extrapolate')
    tmax = times.max()

    Mstar_file = os.path.join(FIREbox_dir, 'Mstar90.dat')
    Mvir, Mstar = np.loadtxt(Mstar_file, usecols=[1,3], skiprows=1).T

    logMmin=10.
    logMmax=14.
    to_keep = (Mvir>np.power(10,logMmin))&(Mvir<np.power(10,logMmax))
    print('Number of galaxies identified in FIREbox with halo mass in [10^{},10^{}]: {}'.format(logMmin,logMmax,np.sum(to_keep)))
    Mvir = Mvir[to_keep]
    Mstar = Mstar[to_keep]

    #####################################
    # Plot your result
    fs = 10
    lw = 1.4
    plt.rc('font', family='serif')
    plt.rc('lines', linewidth=lw)
    plt.rc('text', usetex=True)

    fig, axes = plt.subplots(1, 2, figsize=(8.,3.))

    ## Left-hand plot: HMF ##
    ax = axes[0]

    nbins = round((logMmax-logMmin)*5 + 1)
    logmbins = np.linspace(logMmin, logMmax, nbins)
    mid_bins = (logmbins[:-1]+logmbins[1:]) /2
    Nhalo, _ = np.histogram(np.log10(Mvir), bins=logmbins)
    dnhalo = Nhalo /np.diff(logmbins) /FIREbox_volume

    obs_HMF = HaloMassFunction() # number density, unit is Mpc^-3
    Mvir_plot = np.logspace(9,15,1000)
    dnobs = obs_HMF(Mvir_plot)

    ax.plot(np.log10(Mvir_plot), np.log10(dnobs),
            c='k', label=r'Tinker et al. 2018',
            )

    toplot = dnhalo>0
    ax.plot(mid_bins[toplot], np.log10(dnhalo[toplot]),
            c='firebrick', marker='o', markersize=2**2, 
            label=r'\texttt{FIREbox} halos',
            )

    ax.axvline(x=logMmin, 
               c='k', linestyle='--', lw=lw,
               )
    ax.axvline(x=logMmax, 
               c='k', linestyle='--', lw=lw,
               )

    ax.set_xlabel(r'$\mathrm{log}(M_\mathrm{vir}/\mathrm{M}_\odot)$', fontsize=fs)
    ax.set_xlim(9.2, 14.8)
    ax.set_ylabel(r'$\mathrm{log}\left(\mathrm{dn}/\mathrm{dlog}(M_\mathrm{vir}/\mathrm{M}_\odot)/\mathrm{Mpc}^{-3} \right)$', fontsize=fs)
    ax.set_ylim(-5, 0)

    ax.legend(bbox_to_anchor=(0.5,1.01), loc='lower center', 
              fontsize=fs, title=r'HMF $(z\sim0)$')
    ax.set_box_aspect(1)


    ## Right-hand plot: SMF ##
    ax = axes[1]

    wgal = np.zeros_like(dnhalo)
    wgal[dnhalo>0] = obs_HMF(np.power(10,mid_bins))[dnhalo>0] /dnhalo[dnhalo>0]
    weights = np.array([wgal[k-1] for k in np.digitize(np.log10(Mvir), logmbins)])
    # New mass bins for the SMF
    logmbins = np.linspace(8., 12., nbins)
    mid_bins = (logmbins[:-1]+logmbins[1:]) /2
    Ngal, _ = np.histogram(np.log10(Mstar), bins=logmbins)
    dngal = Ngal /np.diff(logmbins) /FIREbox_volume
    Ngal_w, _ = np.histogram(np.log10(Mstar), bins=logmbins, weights=weights)
    dngal_w = Ngal_w /np.diff(logmbins) /FIREbox_volume

    handles = []
    legends = []
    logMstar_plot = np.linspace(7,12,1000)
    gsmf = Tomczak2014(logMstar_plot)
    line, = ax.plot(logMstar_plot, np.log10(gsmf), 
                    c='k', 
                    )
    handles.append(line)
    legends.append(r'Tomczak et al. 2014')
    phi_50, phi_84, phi_16 = Leja2020(logm=logMstar_plot[:,None], z0=0.0)
    ax.fill_between(logMstar_plot, np.log10(phi_16), np.log10(phi_84), 
                    color='gray', alpha=.4,
                    )
    line, = ax.plot(logMstar_plot, np.log10(phi_50), 
                    c='k', linestyle='--', 
                    )
    patch = Patch(color='gray', alpha=.4)
    handles.append((patch,line))
    legends.append(r'Leja et al. 2020')

    toplot = dngal>0
    line, = ax.plot(mid_bins[toplot], np.log10(dngal[toplot]), 
                    c='firebrick', linestyle='--', 
                    )
    handles.append(line)
    legends.append(r'\texttt{FIREbox} galaxies')

    toplot = dngal_w>0
    line, = ax.plot(mid_bins[toplot], np.log10(dngal_w[toplot]),
                    c='firebrick', marker='o', markersize=2**2, 
                    )
    handles.append(line)
    legends.append(r'\texttt{FIREbox} galaxies - re-weighted')

    ax.set_xlabel(r'$\mathrm{log}(\mathcal{M}_\star/\mathrm{M}_\odot)$', fontsize=fs)
    ax.set_xlim(7.5, 12)
    ax.set_ylabel(r'$\mathrm{log}(\mathrm{dn}/\mathrm{dlog}(\mathcal{M}_\star/\mathrm{M}_\odot)/\mathrm{Mpc}^{-3})$', fontsize=fs)
    ax.set_ylim(-5, 0)
    ax.legend(handles=handles, labels=legends, 
              bbox_to_anchor=(0.5,1.01), loc='lower center', 
              fontsize=fs, title=r'GSMF $(z\sim0)$')
    ax.set_box_aspect(1)

    # Save
    plot_dir = os.path.join(FIREbox_dir, 'plots')
    plt.savefig(os.path.join(plot_dir,'HMF_SMF.pdf'), format='pdf', bbox_inches='tight')



##################################################################
### Run it!
###
##################################################################
if __name__ == "__main__":

    plot_HMF_SMF()
    plt.show()
