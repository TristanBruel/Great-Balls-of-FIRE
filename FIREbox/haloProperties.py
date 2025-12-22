import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

import os
from glob import glob
import argparse

import gizmo_analysis as gizmo
import utilities as ut
import auxiliary_SFR as aux_SFR


def getProperties():
    ### Compute the stellar mass (M90), corresponding radius, 
    # and Star Formation Rate for all the halos identified in snapshot 1200
    outputfile = './Mstar90.dat'
    
    redshifts, times = np.loadtxt("./snapshot_times.txt", usecols=[2,3]).T
    ## Load halos at z=0 ##
    halopath = '/net/cephfs/shares/feldmann.ics.mnf.uzh/data/FIREbox/analysis/AHF/FB15N1024/halo/1200'
    halofile = glob(halopath+'/*.AHF_halos')[0]
    halos = pd.read_csv(halofile, sep='\t')
    '''
    # Use Elia's galaxies
    halofile = '/scratch/tbruel/snap1200_circularity_M8_sample.hdf5'
    galaxies = pd.read_hdf(halofile)
    halo_ids = np.array(galaxies['halo ID'])
    '''
    # Load star particles and properties
    snap_dir = '/net/cephfs/shares/feldmann.ics.mnf.uzh/data/FIREbox/production_runs/FB15N1024'
    part = gizmo.io.Read.read_snapshots(['star'], 'redshift', 0,
            simulation_directory='/home/tbruel/Great-Balls-of-FIRE/FIREbox',
            snapshot_directory=snap_dir,
            )
    h = part.Cosmology['hubble']
    t0 = part.snapshot['time']
    star_particles = part['star']
    masses = star_particles['mass']
    positions = star_particles['position']
    x = positions[:,0]
    y = positions[:,1]
    z = positions[:,2]

    # Select halos that are not subhalos and that are more massive than ~ 10^7 Msun
    to_keep = (halos['hostHalo(2)']==-1)&(halos['M_star(65)']/h>1e7)
    halo_ids = np.array(halos['#ID(1)'][to_keep])

    Mvir = np.zeros_like(halo_ids, dtype='float')
    Rvir = np.zeros_like(halo_ids, dtype='float')
    Ms = np.zeros_like(halo_ids, dtype='float')
    Rs = np.zeros_like(halo_ids, dtype='float')
    for n,halo_idx in enumerate(halo_ids):
        print('Getting star particles in halo', halo_idx)
        galaxy = halos.loc[halo_idx]
        Xhalo, Yhalo, Zhalo = galaxy[['Xc(6)', 'Yc(7)', 'Zc(8)']] /h
        Mvir[n] = galaxy['Mvir(4)'] /h
        Rvir[n] = galaxy['Rvir(12)'] /h

        R = 20
        kpc_20 = np.sqrt((x-Xhalo)**2+(y-Yhalo)**2+(z-Zhalo)**2) < R
        M20kpc = sum(masses[kpc_20])
        print('Total mass of star particles within 20kpc: {:.2e} Msun'.format(M20kpc))
        dr = 0.1
        x_gal = x[kpc_20]
        y_gal = y[kpc_20]
        z_gal = z[kpc_20]
        masses_gal = masses[kpc_20]
        while sum(masses_gal) > 0.9*M20kpc:
            R = R - dr
            to_keep = np.sqrt((x_gal-Xhalo)**2+(y_gal-Yhalo)**2+(z_gal-Zhalo)**2) < R
            masses_gal = masses_gal[to_keep]
            x_gal = x_gal[to_keep]
            y_gal = y_gal[to_keep]
            z_gal = z_gal[to_keep]
        print('Radius that encloses 90% of the stellar mass: {:.2f} kpc'.format(R))
        print('Stellar mass in that sphere: {:.2e} Msun'.format(sum(masses_gal)))
        Ms[n] = sum(masses_gal)
        Rs[n] = R
        print('Halo {} done !'.format(halo_idx))

    tosave = np.array([halo_ids, Mvir, Rvir, Ms, Rs]).T
    header = 'HaloID Mvir[Msun] Rvir[kpc] Mstar_90[Msun] R_90[kpc]'
    np.savetxt(outputfile, tosave, header=header, fmt="%i"+3*" %.8e"+" %.1f")


##################################################################
### 
### Run it!
###
##################################################################
if __name__ == "__main__":

    #####################################
    # Run it
    getProperties()
