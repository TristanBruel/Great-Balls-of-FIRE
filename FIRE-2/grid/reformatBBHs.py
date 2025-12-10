import numpy as np
import os


def identifyMergerGen():

    saved_file = os.path.join(grid_dir, 'tosave.dat')
    bbh_file = os.path.join(grid_dir, 'allBBHs.dat')
    ids = np.loadtxt(saved_file, usecols=0, dtype=str).T
    Z, tform, tej, m1, m2, e, tmerge, a1, a2, vkick, vesc = np.loadtxt(saved_file, usecols=[1,2,3,4,5,6,7,8,9,10,11]).T
    with open(saved_file, 'r') as f:
        with open(bbh_file, 'w') as s:
            header = f.readline()
            s.write(header)
            for i,line in enumerate(f.readlines()):
                s.write(ids[i] + ' {:.5e} {:.5e} {:.5e} {:.5e} {:.5e} {:.5e} {:.5e} {:.5e} {:.5e} {:.5e} {:.5e}\n'.format(Z[i],tform[i],tej[i],m1[i],m2[i],e[i],tmerge[i],a1[i],a2[i],vkick[i],vesc[i]))


##################################################################
### 
### Run it!
###
##################################################################
if __name__ == "__main__":

    grid_dir = '/home/tristan-bruel/Documents/Science/GBoF/FIRE-2/grid'

    #####################################
    # Run it
    identifyMergerGen()
