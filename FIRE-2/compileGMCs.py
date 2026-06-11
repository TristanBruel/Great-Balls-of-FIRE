import os
import h5py
import numpy as np

import argparse


def compute_cloud_metallicity(simu):

    sim_dir = os.path.join(FIRE_dir,simu+'_res7100')
    snapshot_file = os.path.join(sim_dir, 'snapshot_times.txt')
    times = np.loadtxt(snapshot_file, usecols=[3]).T

    cloud_nparts = []
    cloud_formtime = []
    cloud_snaps = []
    cloud_mass = []
    cloud_rhm = []
    cloud_reff = []
    cloud_coord = []
    cloud_Z = []

    for snapnum in range(601):
        filename = os.path.join(sim_dir, 'clouds/Clouds_{}_n1_alpha2.hdf5'.format(snapnum))
        if not os.path.exists(filename):
            continue
        with h5py.File(filename) as cloud_table:
            for n,c in enumerate(cloud_table):
                cld = cloud_table[c]['PartType0']
                cloud_nparts.append(len(cld['Masses']))
                cloud_formtime.append(times[snapnum])
                cloud_snaps.append(snapnum)
                cloud_mass.append(np.sum(cld['Masses']) *1e10)
                center = np.average(cld['Coordinates'][()] *1e3, weights=cld['Masses'], axis=0)
                cloud_coord.append(center)
                dx = cld['Coordinates'][()]*1e3 - center
                ## find half mass radius, assumes all particles have same mass
                r = np.sum(dx**2, axis=1) ** 0.5
                cloud_rhm.append(np.median(r))
                cloud_reff.append(np.sqrt(5.0 / 3 * np.average(r**2, weights=cld['Masses'])))
                if cld['Metallicity'].ndim >1:
                    cloud_Z.append(np.average(cld['Metallicity'][:,0], weights = cld['Masses']))
                else:
                    cloud_Z.append(np.average(cld['Metallicity'], weights = cld['Masses']))
        print(snapnum) 
        
    file_out = os.path.join(sim_dir, 'AllGMCs_%s.hdf5' %simu)
    with h5py.File(file_out, 'w') as f:
        #f.create_dataset('GMC_ID', data=cloud_id, dtype='i8')
        f.create_dataset('FormationTime', data=cloud_formtime)
        f.create_dataset('Snapshot', data=cloud_snaps, dtype='i8')
        f.create_dataset('Mass', data=cloud_mass)
        f.create_dataset('HalfMassRadius', data=cloud_rhm)
        f.create_dataset('Reff', data=cloud_reff)
        f.create_dataset('Coordinates', data=cloud_coord)
        f.create_dataset('Metallicity', data=cloud_Z)


##################################################################
### 
### Run it!
###
##################################################################
if __name__ == "__main__":

    FIRE_dir = '/home/tristan-bruel/Documents/Science/Great-Balls-of-FIRE/FIRE-2'

    #####################################
    # Define command line options
    parser = argparse.ArgumentParser()
    parser.add_argument('--simu', dest='simu', type=str, help='Name of the individual FIRE-2 galaxy')
    args = parser.parse_args()

    #####################################
    compute_cloud_metallicity(simu=args.simu)
