import numpy as np
import h5py
from scipy.interpolate import interp1d

import os

FIREbox_dir = '/home/tristan-bruel/Documents/Science/GBoF/FIREbox'
FIREbox_volume = (22.1)**3 # in unit Mpc^3


def compileBBHs(sample, seed):
    snapshot_file = os.path.join(FIREbox_dir, 'snapshot_times.txt')
    redshifts, times = np.loadtxt(snapshot_file, usecols=[2,3]).T
    tmax = times.max()

    times_to_redshifts = interp1d(times, redshifts, fill_value='extrapolate')
    redshifts_to_times = interp1d(redshifts, times, fill_value='extrapolate')

    outputfile = './BBHs_compiled.h5'
    ## Isolated BBHs ##
    bbhfile = os.path.join(FIREbox_dir, 'BBHs/isolated/BBHs_iso_compiled.h5')
    with h5py.File(bbhfile, 'r') as bbhs:
        mBH1_iso = bbhs['mBH1'][()]
        mBH2_iso = bbhs['mBH2'][()]
        tmerger_iso = bbhs['tmerger'][()]
    tokeep = tmerger_iso < tmax
    m1_iso = np.maximum(mBH1_iso[tokeep], mBH2_iso[tokeep])
    m2_iso = np.minimum(mBH1_iso[tokeep], mBH2_iso[tokeep])
    zmerger_iso = times_to_redshifts(tmerger_iso[tokeep])
    a1_iso = np.zeros_like(m1_iso)
    a2_iso = np.zeros_like(m1_iso)
    print('Number of BBH mergers from isolated evolution:', len(zmerger_iso))


    ## BBHs in massive clusters ##
    bbhfile = os.path.join(FIREbox_dir, 'BBHs/BBHs_{}_compiled_seed{}.h5'.format(sample,seed))
    with h5py.File(bbhfile, 'r') as bbhs:
        mBH1_dyn = bbhs['mBH1'][()]
        mBH2_dyn = bbhs['mBH2'][()]
        tmerger_dyn = bbhs['tmerger'][()]
        spin1_dyn = bbhs['spin1'][()]
        spin2_dyn = bbhs['spin2'][()]
    tokeep = tmerger_dyn < tmax
    m1_dyn = np.maximum(mBH1_dyn[tokeep], mBH2_dyn[tokeep])
    m2_dyn = np.minimum(mBH1_dyn[tokeep], mBH2_dyn[tokeep])
    zmerger_dyn = times_to_redshifts(tmerger_dyn[tokeep])
    a1_dyn = spin1_dyn[tokeep]
    a2_dyn = spin2_dyn[tokeep]
    print('Number of BBH mergers from massive clusters:', len(zmerger_dyn))

    m1 = np.append(m1_iso, m1_dyn)
    m2 = np.append(m2_iso, m2_dyn)
    a1 = np.append(a1_iso, a1_dyn)
    a2 = np.append(a2_iso, a2_dyn)
    zmerger = np.append(zmerger_iso, zmerger_dyn)
    channel = np.append(['iso']*len(m1_iso), ['dyn']*len(m1_dyn))
    channel = np.astype(channel, 'S6')
    with h5py.File(outputfile, 'w') as f:
        f.create_dataset('m1', data=m1)
        f.create_dataset('m2', data=m2)
        f.create_dataset('z', data=zmerger)
        f.create_dataset('a1', data=a1)
        f.create_dataset('a2', data=a2)
        f.create_dataset('channel', data=channel, dtype='S6')


##################################################################
### Run it!
###
##################################################################
if __name__ == "__main__":

    compileBBHs(sample='Gamma', seed=13)
