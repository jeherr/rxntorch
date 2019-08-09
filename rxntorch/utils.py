from __future__ import print_function

import random
import numpy as np

import rdkit.Chem as Chem

elem_list = ['C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na', 'Ca', 'Fe', 'As', 'Al', 'I', 'B', 'V', 'K', 'Tl', 'Yb', 'Sb', 'Sn', 'Ag', 'Pd', 'Co', 'Se', 'Ti', 'Zn', 'H', 'Li', 'Ge', 'Cu', 'Au', 'Ni', 'Cd', 'In', 'Mn', 'Zr', 'Cr', 'Pt', 'Hg', 'Pb', 'W', 'Ru', 'Nb', 'Re', 'Te', 'Rh', 'Tc', 'Ba', 'Bi', 'Hf', 'Mo', 'U', 'Sm', 'Os', 'Ir', 'Ce','Gd','Ga','Cs', 'unknown']
atom_fdim = len(elem_list) + 6 + 6 + 6 + 1
bond_fdim = 6
max_nb = 10

def rxn_smiles_reader(txt_file):
    """Loads txt from a files containing reaction SMILES.

    Files should be in the format of one reaction string per line. Additional
    data can be added onto the end of the file in comma-seperated values. The
    order of data will need to be standardized.

    Args:
        txt_file (str): Path to the csv file containing the data

    Returns:
        bins (dict): Dictionary of binned reaction strings by size. Keys
            are the bin size and values are lists of the reaction strings.
    """
    rxns = []
    with open(txt_file, "r") as datafile:
        for i, line in enumerate(datafile):
            r = line.strip("\n").split()[0]
            rxns.append(r)
    return rxns

def count(s):
    """Counts the number of heavy atoms in a reaction string."""
    c = 0
    for i in range(len(s)):
        if s[i] == ':':
            c += 1
    return c

def get_mol_features(mol):
    n_atoms = mol.GetNumAtoms()
    n_bonds = max(mol.GetNumBonds(), 1)

    fatoms = np.zeros((n_atoms, atom_fdim))
    fbonds = np.zeros((n_bonds, bond_fdim))
    atom_nb = np.zeros((n_atoms, max_nb), dtype=np.int32)
    bond_nb = np.zeros((n_atoms, max_nb), dtype=np.int32)
    num_nbs = np.zeros((n_atoms,), dtype=np.int32)

    for atom in mol.GetAtoms():
        idx = atom.GetIdx()
        if idx >= n_atoms:
            raise Exception("idx >= n_atoms")
        fatoms[idx] = atom_features(atom)

    for bond in mol.GetBonds():
        a1 = bond.GetBeginAtom().GetIdx()
        a2 = bond.GetEndAtom().GetIdx()
        idx = bond.GetIdx()
        if num_nbs[a1] == max_nb or num_nbs[a2] == max_nb:
            raise Exception(smiles)
        atom_nb[a1,num_nbs[a1]] = a2
        atom_nb[a2,num_nbs[a2]] = a1
        bond_nb[a1,num_nbs[a1]] = idx
        bond_nb[a2,num_nbs[a2]] = idx
        num_nbs[a1] += 1
        num_nbs[a2] += 1
        fbonds[idx] = bond_features(bond)
    return fatoms, fbonds, atom_nb, bond_nb, num_nbs

def atom_features(atom):
    return np.array(onek_encoding_unk(atom.GetSymbol(), elem_list)
            + onek_encoding_unk(atom.GetDegree(), [0,1,2,3,4,5])
            + onek_encoding_unk(atom.GetExplicitValence(), [1,2,3,4,5,6])
            + onek_encoding_unk(atom.GetImplicitValence(), [0,1,2,3,4,5])
            + [atom.GetIsAromatic()], dtype=np.float32)

def onek_encoding_unk(x, allowable_set):
    if x not in allowable_set:
        x = allowable_set[-1]
    return list(map(lambda s: x == s, allowable_set))

def bond_features(bond):
    bt = bond.GetBondType()
    return np.array([bt == Chem.rdchem.BondType.SINGLE,
                bt == Chem.rdchem.BondType.DOUBLE,
                bt == Chem.rdchem.BondType.TRIPLE,
                bt == Chem.rdchem.BondType.AROMATIC,
                bond.GetIsConjugated(),
                bond.IsInRing()], dtype=np.float32)
