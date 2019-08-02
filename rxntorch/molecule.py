from __future__ import print_function

import rdkit.Chem as Chem

elem_list = ['C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na', 'Ca', 'Fe', 'As', 'Al', 'I', 'B', 'V', 'K', 'Tl', 'Yb', 'Sb', 'Sn', 'Ag', 'Pd', 'Co', 'Se', 'Ti', 'Zn', 'H', 'Li', 'Ge', 'Cu', 'Au', 'Ni', 'Cd', 'In', 'Mn', 'Zr', 'Cr', 'Pt', 'Hg', 'Pb', 'W', 'Ru', 'Nb', 'Re', 'Te', 'Rh', 'Tc', 'Ba', 'Bi', 'Hf', 'Mo', 'U', 'Sm', 'Os', 'Ir', 'Ce','Gd','Ga','Cs', 'unknown']
atom_fdim = len(elem_list) + 6 + 6 + 6 + 1
bond_fdim = 6
max_nb = 10

class Mol(object):
    """
    """
    def __init__(self, smiles):
        self.smiles = smiles
        self.rdkit_mol = Chem.MolFromSmiles(self.smiles)
        if not mol:
            raise ValueError("Could not parse smiles string:", smiles)

    def get_atom_feats(self):
        self.n_atoms = self.rdkit_mol.GetNumAtoms()
        fatoms = np.zeros((self.n_atoms, atom_fdim))

        for atom in self.rdkit_mol.GetAtoms():
            idx = atom.GetIdx()
            if idx >= n_atoms:
                raise Exception(smiles)
            fatoms[idx] = atom_features(atom)
        return fatoms

    def get_bond_feats(self):
        self.n_bonds = max(self.rdkit_mol.GetNumBonds(), 1)
        fbonds = np.zeros((self.n_bonds, bond_fdim))
        atom_nb = np.zeros((self.n_atoms, max_nb), dtype=np.int32)
        bond_nb = np.zeros((self.n_atoms, max_nb), dtype=np.int32)
        num_nbs = np.zeros((n_atoms,), dtype=np.int32)

        for bond in self.rdkit_mol.GetBonds():
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
        return fbonds, atom_nb, bond_nb, num_nbs

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
