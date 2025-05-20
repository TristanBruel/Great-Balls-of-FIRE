import numpy as np

import warnings
warnings.filterwarnings('error')

def MassCDF(m, alpha, k=0):
    return m**(alpha + 1) *np.exp(-k /(1-m))

def SampleMassFunc(mass_budget, mmin, alpha, k=0, seed=42):
    """Returns a stochastically-sampled list of masses from a power-law"""
    if mass_budget < mmin:
        return np.array([])
    np.random.seed(seed)
    masses = []
    mtot = 0.

    Nmax = MassCDF(mmin/mass_budget, alpha, k)
    mgrid = np.logspace(np.log10(mmin/mass_budget), -1e-7, 1000)
    Nm = MassCDF(mgrid, alpha, k)
    Nm[-1] = 0.
    Nm = Nm.max() - Nm
    while mtot < 1: #mass_budget:
        x = np.random.rand(100) * Nmax
        m = np.interp(x, Nm, mgrid)
        mtot += m.sum()
        masses.append(m)

    masses = np.concatenate(masses)
    idx = np.arange(len(masses))[masses.cumsum() > 1][0]
    mtot = masses[:idx+1].sum()
    masses = masses[:idx+1]
    mtot *= mass_budget
    masses = np.array(masses) * mass_budget
    mass_error = mtot - mass_budget
    #if abs(mass_error - masses[-1]) < abs(mass_error): # if rejecting the last one gets us closer to budget, do it
    #    masses = masses[:-1]
    try:
        test=abs(mass_error - masses[-1])
    except:
        print('Problem here')
        print(mass_budget)
        print(Nmax)
        print(mgrid)
        print(Nm)
        print(idx)
        print(mtot)
        print(masses)
        print(mass_error)
        return
    if test<abs(mass_error):
        masses = masses[:-1]
    return np.array(masses)


def SampleSizeFunc(masses, metallicity, Zsun=0.017, seed=42):
    np.random.seed(seed)
    #M_GMC = 1e7
    #R_GMC = 1e2
    #return 3 * (M_GMC/1e6)**(1./5) * (masses/1e4)**(1./3) * (M_GMC / (np.pi*R_GMC**2)/100)**-1. * (metallicity/Zsun)**.1 * 10**(np.random.normal(size=masses.shape)*0.5)
    return 3 * (masses/1e4)**(1./3) * (metallicity/Zsun)**.1 * 10**(np.random.normal(size=masses.shape)*0.4)
