import pandas as pd
import numpy as np

import os
from glob import glob
import argparse

import gizmo_analysis as gizmo
import utilities as ut


def getProperties(simu):
    simu = simu.split('_res')[0]
    sim_dir = os.path.join(FIRE_dir,simu+'_res7100')
    ### Compute the stellar mass (M90), corresponding radius, and SFR for that galaxy ###
    snapshot_file = os.path.join(sim_dir, 'snapshot_times.txt')
    redshifts, times = np.loadtxt(snapshot_file, usecols=[2,3]).T

    SFR_file = os.path.join(sim_dir, 'SFR_{}.txt'.format(simu))
    if not os.path.exists(SFR_file):
        # Load star particles and properties
        print('Getting star particles in', simu)
        snap_dir = os.path.join(sim_dir,'output')
        part = gizmo.io.Read.read_snapshots(['star'], 'redshift', 0,
                simulation_directory=sim_dir,
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

        Xc, Yc, Zc = star_particles.host['position'][0]

        R = 20
        kpc_20 = np.sqrt((x-Xc)**2+(y-Yc)**2+(z-Zc)**2) < R
        M20kpc = sum(masses[kpc_20])
        print('Total mass of star particles within 20kpc: {:.2e} Msun'.format(M20kpc))
        dr = 0.1
        x_gal = x[kpc_20]
        y_gal = y[kpc_20]
        z_gal = z[kpc_20]
        masses_gal = masses[kpc_20]
        while sum(masses_gal) > 0.9*M20kpc:
            R = R - dr
            to_keep = np.sqrt((x_gal-Xc)**2+(y_gal-Yc)**2+(z_gal-Zc)**2) < R
            masses_gal = masses_gal[to_keep]
            x_gal = x_gal[to_keep]
            y_gal = y_gal[to_keep]
            z_gal = z_gal[to_keep]
        print('Radius that encloses 90% of the stellar mass: {:.2f} kpc'.format(R))
        print('Stellar mass in that sphere: {:.2e} Msun'.format(sum(masses_gal)))

        in_halo = np.sqrt((x-Xc)**2+(y-Yc)**2+(z-Zc)**2) < 300 
        tform = formation_times[in_halo] # time of formation in unit Gyr
        Z = metallicities[in_halo]
        mform = star_particles.prop('form.mass')[in_halo]

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
        np.savetxt(SFR_file, tosave, header='Time[Gyr] Metallicity[] sigmaZ[] Mform[Msun] SFR[Msun.yr-1]')


##################################################################
### 
### Run it!
###
##################################################################
if __name__ == "__main__":

    FIRE_dir = '/home/tristan-bruel/Documents/Science/GBoF/FIRE-2'

    #####################################
    # Define command line options
    parser = argparse.ArgumentParser()
    parser.add_argument('--simu', dest='simu', type=str, help='Name of the individual FIRE-2 galaxy')
    args = parser.parse_args()

    #####################################
    # Run it
    getProperties(simu=args.simu)
