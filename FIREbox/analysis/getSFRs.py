import pandas as pd
import numpy as np

import os
from glob import glob
import argparse

import gizmo_analysis as gizmo
import utilities as ut
import auxiliary_SFR as aux_SFR


def getProperties():
    ### Compute the stellar mass (M90), corresponding radius, 
    # and Star Formation Rate for all the halos identified in snapshot 1200
    
    redshifts, times = np.loadtxt("/data/tbruel/FIREbox/snapshot_times.txt", usecols=[2,3]).T
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
            simulation_directory='/data/tbruel/FIREbox',
            snapshot_directory=snap_dir,
            )
    h = part.Cosmology['hubble']
    t0 = part.snapshot['time']
    star_particles = part['star']
    masses = star_particles['mass']
    positions = star_particles['position']
    formation_times = star_particles.prop('form.time')
    metallicities = part['star'].prop('massfraction.metals')
    x = positions[:,0]
    y = positions[:,1]
    z = positions[:,2]

    # Select halos that are not subhalos and that are more massive than ~ 10^7 Msun
    to_keep = (halos['hostHalo(2)']==-1)&(halos['M_star(65)']/h>1e7)
    halo_ids = np.array(halos['#ID(1)'][to_keep])
    # reverse order to go from the smallest halos to the more massive ones
    halo_ids = halo_ids[::-1]

    Mvir = np.zeros_like(halo_ids, dtype='float')
    Rvir = np.zeros_like(halo_ids, dtype='float')
    Ms = np.zeros_like(halo_ids, dtype='float')
    Rs = np.zeros_like(halo_ids, dtype='float')
    for n,halo_idx in enumerate(halo_ids):
        print('Getting star particles in halo', halo_idx)
        SFR_file = 'FB15N1024_halo{}_SFR.txt'.format(halo_idx)
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

        #in_gal = np.sqrt((x-Xhalo)**2+(y-Yhalo)**2+(z-Zhalo)**2) < R
        in_gal = np.sqrt((x-Xhalo)**2+(y-Yhalo)**2+(z-Zhalo)**2) < Rvir[n]
        tform = formation_times[in_gal] # time of formation in unit Gyr
        Z = metallicities[in_gal]
        #mform = star_particles.prop('form.mass')[in_gal]
        age = (t0 - tform) *1e9 # age in yr
        mform = masses[in_gal] *aux_SFR.mass_ratio(age, Z)

        # Compute SFR #
        mbin, t = np.histogram(tform, weights=mform, bins=times)
        sfr = mbin /np.diff(t*1e9)
        sfr = np.append(0, sfr)
        # Compute metallicity of newly formed stars #
        Zbin, t = np.histogram(tform, weights=mform*Z, bins=times)
        Zmean = np.zeros_like(Zbin)
        Zmean[mbin>0] = Zbin[mbin>0] /mbin[mbin>0]
        Zmean = np.append(0, Zmean)
        # metallicity spread #
        Zspread = np.zeros_like(Zbin)
        for k in range(len(times)-1):
            new_stars = (tform>times[k])&(tform<=times[k+1])
            Zspread[k] = np.std(Z[new_stars])
        Zspread = np.append(0, Zspread)
        mbin = np.append(0, mbin)
        tosave = np.array([times, Zmean, Zspread, mbin, sfr]).T
        np.savetxt(os.path.join('SFRs/Rvir',SFR_file), tosave, header='Time[Gyr] Metallicity[] sigmaZ[] Mform[Msun] SFR[Msun.yr-1]')

        print('Halo {} done !'.format(halo_idx))

        ### Remove the star particles identified in this halo to avoid double counting ###
        in_gal = np.sqrt((x-Xhalo)**2+(y-Yhalo)**2+(z-Zhalo)**2) < R
        x = np.delete(x, in_gal)
        y = np.delete(y, in_gal)
        z = np.delete(z, in_gal)
        masses = np.delete(masses, in_gal)
        formation_times = np.delete(formation_times, in_gal)
        metallicities = np.delete(metallicities, in_gal)

    outputfile = './Mstar90.dat'
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
