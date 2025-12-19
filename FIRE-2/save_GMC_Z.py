import pandas as pd
import os
import h5py
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt


def compute_cloud_metallicity():
    
    for snapnum in range(601):
        filename = '/scratch/tbruel/dwarves/metal_diff/m11d_res7100/clouds/Clouds_{}_n1_alpha2.hdf5'.format(snapnum)
        if not os.path.exists(filename):
            continue
        with h5py.File(filename) as cloud_table:
            n_clouds = len(cloud_table)
            metallicities = np.zeros(n_clouds)
            count = 0
            for c in cloud_table:
                cld = cloud_table[c]['PartType0']
                if cld['Metallicity'].ndim >1:
                    metallicities[count] = np.average(cld['Metallicity'][:,0], weights = cld['Masses'])
                else:
                    metallicities[count] = np.average(cld['Metallicity'], weights = cld['Masses'])
                count += 1
        
        # Add the cloud metallicities in the bound*.dat file
        filename = '/scratch/tbruel/dwarves/metal_diff/m11d_res7100/clouds/bound_{}_n1_alpha2.dat'.format(snapnum)
        colnames = np.array(['Mass', 'xc', 'yc', 'zc', 'Lx', 'Ly', 'Lz', 'Reff', 'HalfMassRadius', 'NumParticles', 'VirialParam', 'e1x', 'e1y', 'e1z', 'e2x', 'e2y', 'e2z', 'e3x', 'e3y', 'e3z'])
        clouds = pd.read_table(filename, header=10, names=colnames)
        clouds['Metallicity'] = metallicities
        clouds.to_csv(filename, sep='\t', index=False)
        print(snapnum) 


##################################################################
### 
### Run it!
###
##################################################################
if __name__ == "__main__":

    #####################################
    compute_cloud_metallicity()
