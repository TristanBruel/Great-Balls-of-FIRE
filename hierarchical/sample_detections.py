import numpy as np
import h5py 
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from tqdm import tqdm

import os
import sys
from glob import glob

from colossus.cosmology import cosmology
cosmology.setCosmology('planck18');
from colossus.lss import mass_function
from astropy.cosmology import Planck18 as planck
sys.path.append('/shared/tbruel/envs/fire/pdet')
from pdet.emulator import pdet_O3


def HaloMassFunction():
    dndM = lambda M: mass_function.massFunction(M, 0.0, mdef='vir', model='tinker08', q_out='dndlnM')
    return dndM


def sample_detections(method='SFR'):

    snapshot_file = os.path.join(FIREbox_dir, 'snapshot_times.txt')
    redshifts, times = np.loadtxt(snapshot_file, usecols=[2,3]).T

    times_to_redshifts = interp1d(times, redshifts, fill_value='extrapolate')
    redshifts_to_times = interp1d(redshifts, times, fill_value='extrapolate')

    Mstar_file = os.path.join(FIREbox_dir, 'Mstar90_v2.dat')
    haloids, Mvir, Mstar = np.loadtxt(Mstar_file, usecols=[0,1,3], skiprows=1).T
    haloids = haloids.astype(np.int64)

    logMmin=10.
    logMmax=14.
    to_keep = (Mvir>np.power(10,logMmin))&(Mvir<np.power(10,logMmax))
    print('Number of galaxies identified in FIREbox with halo mass in [10^{},10^{}]: {}'.format(logMmin,logMmax,np.sum(to_keep)))
    Mvir = Mvir[to_keep]
    Mstar = Mstar[to_keep]
    haloids = haloids[to_keep]

    nbins = round((logMmax-logMmin)*5 + 1)
    logmbins = np.linspace(logMmin, logMmax, nbins)
    mid_bins = (logmbins[:-1]+logmbins[1:]) /2
    Nhalo, _ = np.histogram(np.log10(Mvir), bins=logmbins)
    dnhalo = Nhalo /np.diff(logmbins) /FIREbox_volume
    wgal = np.zeros_like(dnhalo)
    obs_HMF = HaloMassFunction() # number density, unit is Mpc^-3
    wgal[dnhalo>0] = obs_HMF(np.power(10,mid_bins))[dnhalo>0] /dnhalo[dnhalo>0]

    # Extrapolate some cosmology quantities (dz/dt and dVc/dz)
    z_list = np.linspace(0.,2.,2500)
    dzdt = interp1d(z_list[:-1], np.diff(z_list)/np.diff(planck.lookback_time(z_list)), fill_value='extrapolate')
    dVcdz = interp1d(z_list, (4*np.pi)*planck.differential_comoving_volume(z_list).value*1e-9, fill_value='extrapolate')

    bbhfiles = glob(os.path.join(FIREbox_dir, 'BBHs/BBHs_{}_compiled_withKremer_seed*.h5'.format(method)))
    nobs = np.zeros(len(bbhfiles))
    nhier = np.zeros(len(bbhfiles))
    ngw24 = np.zeros(len(bbhfiles))
    indent = 0
    for f in bbhfiles:
        print(f)
        with h5py.File(f, 'r') as bbhs:
            m1 = bbhs['m1'][()]
            m2 = bbhs['m2'][()]
            a1 = bbhs['a1'][()]
            a2 = bbhs['a2'][()]
            tmerger = bbhs['tmerger'][()]
            Mgal = bbhs['Mgal'][()]
            Mhalo = bbhs['Mvir'][()]
            clid = bbhs['id'][()]

        a1[m1<m2], a2[m1<m2] = a2[m1<m2], a1[m1<m2]
        m1, m2 = np.maximum(m1,m2), np.minimum(m1,m2)

        zmerger = times_to_redshifts(tmerger)
        astro = (zmerger>0) & (zmerger<=2)
        m1 = m1[astro]
        m2 = m2[astro]
        a1 = a1[astro]
        a2 = a2[astro]
        Mhalo = Mhalo[astro]
        tmerger = tmerger[astro]
        zmerger = zmerger[astro]
        #print('Unweighted number of BBH mergers in FIREbox at z<2: %i' %len(m1))

        # Compute weight associated to each galaxy in which there's a BBH merger
        weights = np.array([wgal[k-1] for k in np.digitize(np.log10(Mhalo), logmbins)])
        weights *= 1/FIREbox_volume *Tobs/(1+zmerger) *dVcdz(zmerger) *dzdt(zmerger)
        #print('Estimated number of BBH mergers up to z=2 over 1 year of observation time: %i' %np.sum(weights))

        ## Sample this number of BBHs and compute their pdets ##
        p = weights /np.sum(weights)
        sample = np.random.choice(np.arange(0,len(m1)),round(np.sum(weights)),p=p)
        m1s = m1[sample]
        m2s = m2[sample]
        qs = m2s/m1s
        a1s = a1[sample]
        a2s = a2[sample]
        zs = zmerger[sample]
        clids = clid[sample]
        Mgals = Mgal[sample]

        params = {'mass_1': m1s,
                  'mass_2': m2s,
                  'a_1': a1s,
                  'a_2': a2s,
                  'redshift': zs,
                  }
        pdet = pdet_O3().predict(params)
        print('Detectable rate of BBH mergers at LVK O3 sensitivity: %i yr^-1' %np.sum(pdet))
        hierar = a1s+a2s > 0
        #print('Number of hierarchical mergers: %i' %np.sum(hierar))
        print('Detectable rate of hierarchical BBH mergers at LVK O3 sensitivity: %i yr^-1' %np.sum(pdet[hierar]))
        GW24x = (m1s < 30) & (qs < 0.85) & (a1s > 0.05)
        #print('Number of BBH mergers with stricly positive primary spin, m1<30 Msun, and unequal masses: %i' %np.sum(GW24x))
        print('Detectable rate of GW241011-like BBH mergers at LVK O3 sensitivity: %i yr^-1\n' %np.sum(pdet[GW24x]))

        seed = int(f.split('seed')[1].split('.h5')[0])
        outfile = './BBHs/detectableBBHs_{}_seed{}.h5'.format(method,seed)
        with h5py.File(outfile, 'w') as out:
            out.create_dataset('cluster_id', data=clids, dtype=h5py.string_dtype())
            out.create_dataset('m1', data=m1s)
            out.create_dataset('m2', data=m2s)
            out.create_dataset('a1', data=a1s)
            out.create_dataset('a2', data=a2s)
            out.create_dataset('z', data=zs)
            out.create_dataset('Mgal', data=Mgals)
            out.create_dataset('pdet', data=pdet)



##################################################################
### Run it!
###
##################################################################
if __name__ == "__main__":

    FIREbox_dir = '/home/tbruel/Great-Balls-of-FIRE/FIREbox'
    FIREbox_volume = (22.1)**3 # in unit Mpc^3
    Tobs = 1 # LVK observation time in yrs

    sample_detections()
