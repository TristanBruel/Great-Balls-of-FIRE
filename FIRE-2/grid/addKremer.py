import numpy as np
import os
import h5py


def addKremerClusters():

    bbh_file = os.path.join(grid_dir, 'allBBHs.dat')
    bbh_file_Kremer = os.path.join(GW24_dir, 'external_cluster_simulations/BBHmergers.dat')
    cluster_id, rv, Z, N, tmerge, m1, m2, a1, a2, vkick, vesc, ecc = np.loadtxt(bbh_file_Kremer, usecols=(0,1,3,4,5,8,9,10,11,12,13,20)).T
    cluster_id = cluster_id.astype(int)
    Mcl = N *1e5 *0.6 # mean mass for Kroupa 2001 IMF
    rhm = rv *0.833 # conversion from virial to half-mass radius for king profile (w_0=5)
    tmerge *= 1e-3 # convert to Gyr
    save_file = os.path.join(grid_dir, 'allBBHs_withKremer.dat')
    if not os.path.exists(save_file):
        with open(save_file, 'w') as s:
            with open(bbh_file, 'r') as f:
                header = f.readline()
                s.write(header)
                for line in f.readlines():
                    s.write(line)
            for i in range(len(cluster_id)):
                s.write('Kremer_%s %.5e %.5e %.5e %.5e %.5e %.5e %.5e %.5e %.5e %.5e %.5e\n' %(cluster_id[i],Z[i],0,0,m1[i],m2[i],ecc[i],tmerge[i],a1[i],a2[i],vkick[i],vesc[i]))

    cluster_file = os.path.join(grid_dir, 'gridClusters.hdf5')
    with h5py.File(cluster_file,'r') as c:
        r = c['HalfMassRadius'][()]
        m = c['Mass'][()]
        met = c['Metallicity'][()]
        cid = c['id'][()]
        nBBHs = c['nBBHs'][()]

    unique_ids, indexes, nbbhs_Kremer = np.unique(cluster_id,return_index=True,return_counts=True)
    cluster_file_Kremer = os.path.join(grid_dir, 'gridClusters_withKremer.hdf5')
    cid = np.array(cid, dtype=h5py.string_dtype())
    id_Kremer = np.array(['Kremer_%s' %u for u in unique_ids], dtype=h5py.string_dtype())
    with h5py.File(cluster_file_Kremer,'w') as d:
        d.create_dataset('Mass', data=np.append(m, Mcl[indexes]))
        d.create_dataset('HalfMassRadius', data=np.append(r, rhm[indexes]))
        d.create_dataset('Metallicity', data=np.append(met, Z[indexes]))
        d.create_dataset('id', data=np.append(cid, id_Kremer), dtype=h5py.string_dtype())
        d.create_dataset('nBBHs', data=np.append(nBBHs, nbbhs_Kremer))



##################################################################
### 
### Run it!
###
##################################################################
if __name__ == "__main__":

    grid_dir = '/home/tristan-bruel/Documents/Science/GBoF/FIRE-2/grid'
    GW24_dir = '/home/tristan-bruel/Documents/Science/o4a-astro/GW241011_GW241110_data_release/data'

    #####################################
    # Run it
    addKremerClusters()
