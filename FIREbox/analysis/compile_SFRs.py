import numpy as np
import h5py
import pandas as pd

import os
from glob import glob
import argparse


def compileSFRs():

    redshifts, times = np.loadtxt("/data/tbruel/FIREbox/snapshot_times.txt", usecols=[2,3]).T

    # load galaxy stellar masses
    Mstar_file = 'Mstar90.dat'
    halo_ids, Mvir, Mstar = np.loadtxt(Mstar_file, skiprows=1, usecols=[0,1,3]).T
    halo_ids = halo_ids.astype(np.int64)

    header = 'Halo ids:'+str(halo_ids)
    SFRs = np.zeros((len(times),len(halo_ids)), dtype='float')
    metals = np.zeros((len(times),len(halo_ids)), dtype='float')
    Mform = np.zeros((len(times),len(halo_ids)), dtype='float')
    for n,halo_idx in enumerate(halo_ids):
        SFR_file = './SFRs/Rvir/FB15N1024_halo{}_SFR.txt'.format(halo_idx)
        times, Zmean, Zspread, mbin, sfr = np.loadtxt(SFR_file).T
        SFRs[:,n] = sfr
        metals[:,n] = Zmean
        Mform[:,n] = mbin

    np.savetxt('./SFRs/Rvir/SFRs_allhalos_Rvir.txt', SFRs, header=header)
    np.savetxt('./SFRs/Rvir/Metals_allhalos_Rvir.txt', metals, header=header)
    np.savetxt('./SFRs/Rvir/Mform_allhalos_Rvir.txt', Mform, header=header)


##################################################################
### 
### Run it!
###
##################################################################
if __name__ == "__main__":

    #####################################
    # Run it
    compileSFRs()
