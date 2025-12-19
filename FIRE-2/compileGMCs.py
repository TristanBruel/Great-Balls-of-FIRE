import h5py
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.stats import linregress

import os
import argparse


def compileGMCs(simu):
    ### Compile GMCs from all snapshots of the FIRE simulation $simu in one single file
    nbr_clouds = int(1e9)
    cloud_coordinates = np.zeros((nbr_clouds,3))
    cloud_masses = np.zeros(nbr_clouds)
    cloud_radii = np.zeros(nbr_clouds)
    cloud_formtimes = np.zeros(nbr_clouds)
    cloud_metallicities = np.zeros(nbr_clouds)
    cloud_ids = np.zeros(nbr_clouds)
    cloud_snapshots = np.zeros(nbr_clouds)
    count = 0

    cloud_path = os.path.join(simu_dir, simu, 'clouds')
    print(cloud_path)
    for snapnum in range(600):
        filename = 'GMCPopulation_{}_42.hdf5'.format(snapnum)
        file_path = os.path.join(cloud_path, filename)
        if not os.path.isfile(file_path):
            continue
        with h5py.File(file_path) as clouds:
            end = count+len(clouds['Mass'])
            cloud_masses[count:end] = np.array(clouds['Mass'])
            cloud_radii[count:end] = np.array(clouds['HalfMassRadius'])
            cloud_coordinates[count:end,] = np.array(clouds['ClusterPos'])
            cloud_formtimes[count:end] = np.array(clouds['FormationTime'])
            cloud_metallicities[count:end] = np.array(clouds['Metallicity'])
            cloud_ids[count:end] = np.array(clouds['TracerID'])
            cloud_snapshots[count:end] = np.repeat(snapnum, end-count)
            count = end
        print(snapnum)

    print('Number of clouds:', len(cloud_masses))

    file_out = os.path.join(simu, 'AllGMCs.hdf5')
    with h5py.File(file_out, 'w') as f:
        f.create_dataset('Mass', data=cloud_masses)
        f.create_dataset('HalfMassRadius', data=cloud_radii)
        f.create_dataset('Coordinates', data=cloud_coordinates)
        f.create_dataset('Metallicity', data=cloud_metallicities)
        f.create_dataset('FormationTime', data=cloud_formtimes)
        f.create_dataset('GMC_ID', data=cloud_GMCs, dtype='i8')
        f.create_dataset('Snapshot', data=cloud_snapshots, dtype='i8')



##################################################################
### 
### Run it!
###
##################################################################
if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--simu', dest='simu',  help="Name of the FIRE simulation", type=str, default=None)
    args = parser.parse_args()

    simu_dir = "/shares/feldmann.ics.mnf.uzh/Tristan/dwarfs/metal_diff"

    #####################################
    # Run
    compileGMCs(simu=args.simu)

