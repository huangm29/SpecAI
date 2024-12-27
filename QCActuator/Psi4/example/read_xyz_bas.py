# This is a python script to read a .xyz file
# into a multi-line string

import psi4
import numpy as np
import os

def read_xyz(path_to_file):
    with open(path_to_file) as f:
        f.readline()       # strip the first line
        f.readline()       # strip the second line
        data=''.join(line for line in f)
    return data

def read_bas(path_to_file):
    with open(path_to_file) as f:
        data=''.join(line for line in f)
    return data

os.environ['KMP_DUPLICATE_LIB_OK']='True' # to fix the omp running issue

PATH = 'test.xyz'
geom = read_xyz(PATH)

PATH = 'test.bas'
bas = read_bas(PATH)

C3N4 = psi4.geometry(geom)
psi4.basis_helper(bas)

# set charge to +1, multiplicity to 1
psi4.core.Molecule.set_molecular_charge(C3N4, 1)
psi4.core.Molecule.set_multiplicity(C3N4, 1)


# scf_e, scf_wfn = psi4.energy('scf/sto-3g', return_wfn=True)
scf_e, scf_wfn = psi4.energy('b3lyp', return_wfn=True)
