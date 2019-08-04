from __future__ import print_function

from torch.utils.data import Dataset
import rdkit.Chem as Chem

from rxntorch.utils import rxn_smiles_reader, get_mol_features


class RxnDataset(Dataset):
    """Object for containing large sets of reaction SMILES strings.

    Attributes:
        file      (str): location of the file rxns are loaded from.
        rxn_strs  (set): reaction strings of the dataset and the bonding changes.
        bins     (dict): contains lists of indices to map rxn_strs to bin sizes
                for mapping data into efficient batches.

    """
    def __init__(self, file_):
        super(RxnDataset, self).__init__()
        self.file = file_
        self.rxn_strs, self.bins = rxn_smiles_reader(self.file)

    def __len__(self):
        return len(self.rxn_strs)

    def __getitem__(self, idx):
        reactants, products = [Chem.MolFromSmiles(smiles) for smiles in self.rxn_strs[idx][0].split('>>')]
        atom_features, bond_features, atom_nbs, bond_nbs, num_nbs = get_mol_features(reactants)

        return {'atom_features': torch.from_numpy(atom_features),
                'bond_features': torch.from_numpy(bond_features),
                'atom_nbs': torch.from_numpy(atom_nbs),
                'bond_nbs': torch.from_numpy(bond_nbs),
                'num_nbs': torch.from_numpy(num_nbs)}
