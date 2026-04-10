import numpy as np
import os


def identifyMergerGen():

    bbh_file = os.path.join(grid_dir, 'allBBHs.dat')
    save_file = os.path.join(grid_dir, 'allBBHs_withGen.dat')
    if not os.path.exists(save_file):
        m1, m2, a1, a2 = np.loadtxt(bbh_file, usecols=[4,5,8,9]).T
        ids = np.loadtxt('allBBHs.dat', usecols=0, dtype=str).T
        gen1 = np.ones_like(m1, dtype=int)
        gen2 = np.ones_like(m2, dtype=int)
        q = np.minimum(m1,m2) /np.maximum(m1,m2)
        chi1 = np.zeros_like(m1)
        chi2 = np.zeros_like(m1)
        chi1[m1>m2] = a1[m1>m2]
        chi2[m1>m2] = a2[m1>m2]
        chi1[m1<m2] = a2[m1<m2]
        chi2[m1<m2] = a1[m1<m2]
        eta = m1*m2 /np.power(m1+m2,2)
        chi_z = (q*q*chi2 + chi1) /np.power(1+q,2) # ignore spin orientation here, only minor impact on final mass
        z1 = 1 + np.power(1-chi_z*chi_z,1/3)*(np.power(1+chi_z,1/3)+np.power(1-chi_z,1/3))
        z2 = np.sqrt(3*chi_z*chi_z + z1*z1)
        rISCO = 3 + z2 - np.sign(np.sqrt((3-z1)*(3+z1+2*z2)),chi_z)
        eISCO = np.sqrt(1 - 2. / 3. / rISCO)
        mass_frac = 1 - eta*(1-4*eta)*(1-eISCO) - 16*eta*eta*(0.04827 + 4*0.01707*chi_z*(chi_z + 1))
        mfinal = mass_frac *(m1+m2)

        for i in range(len(m1)):
            cluster_id = ids[i]
            incluster = np.where(ids==cluster_id)[0]
            if (a1[i]>0) & (np.sum(np.abs(mfinal[incluster]-m1[i]) < 1e-2)>0):
                prog = np.argmin(np.abs(mfinal[incluster]-m1[i]))
                gen1[i] = np.maximum(gen1[incluster][prog], gen2[incluster][prog]) + 1
            if (a2[i]>0) & (np.sum(np.abs(mfinal[incluster]-m2[i]) < 1e-2)>0):
                prog = np.argmin(np.abs(mfinal[incluster]-m2[i]))
                gen2[i] = np.maximum(gen1[incluster][prog], gen2[incluster][prog]) + 1
        
        with open(save_file, 'w') as s:
            with open(bbh_file, 'r') as f:
                header = f.readline()
                s.write(header.split('\n')[0] + ' #13:Mremnant #14:gen1 #15:gen2\n')
                for i,line in enumerate(f.readlines()):
                    s.write(line.split('\n')[0] + ' {:.5e} {} {}\n'.format(mfinal[i],gen1[i],gen2[i]))


##################################################################
### 
### Run it!
###
##################################################################
if __name__ == "__main__":

    grid_dir = '/home/tristan-bruel/Documents/Science/Great-Balls-of-FIRE/FIRE-2/grid'

    #####################################
    # Run it
    identifyMergerGen()
