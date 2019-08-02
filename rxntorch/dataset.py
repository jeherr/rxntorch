from __future__ import print_function

from torch.utils.data import Dataset
import rdkit.Chem as Chem

from rxntorch.utils import rxn_smiles_reader


class RxnSmilesData(Dataset):
    """Object for containing large sets of reaction SMILES strings.

    Attributes:
        file  (str): location of the file rxns were loaded from
        rxns (dict): a list of reaction strings
    """
    def __init__(self, file_):
        self.file = file_
        self.rxns = rxn_smiles_reader(self.file)

    def __len__(self):
        return len(self.rxns)

    def __getitem__(self, idx):
        return sample

    def smiles2graph(smiles, idxfunc=lambda x:x.GetIdx()):
        """
        """
        mol = Chem.MolFromSmiles(smiles)
        if not mol:
            raise ValueError("Could not parse smiles string:", smiles)

        n_atoms = mol.GetNumAtoms()
        n_bonds = max(mol.GetNumBonds(), 1)
        fatoms = np.zeros((n_atoms, atom_fdim))
        fbonds = np.zeros((n_bonds, bond_fdim))
        atom_nb = np.zeros((n_atoms, max_nb), dtype=np.int32)
        bond_nb = np.zeros((n_atoms, max_nb), dtype=np.int32)
        num_nbs = np.zeros((n_atoms,), dtype=np.int32)

        for atom in mol.GetAtoms():
            idx = idxfunc(atom)
            if idx >= n_atoms:
                raise Exception(smiles)
            fatoms[idx] = atom_features(atom)

        for bond in mol.GetBonds():
            a1 = idxfunc(bond.GetBeginAtom())
            a2 = idxfunc(bond.GetEndAtom())
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
