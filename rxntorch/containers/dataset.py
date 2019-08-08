from __future__ import print_function

from torch.utils.data import Dataset
import rdkit.Chem as Chem
import rdkit.Chem.AllChem as AllChem

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
        self.rxn_smiles = rxn_smiles_reader(self.file)

    def __len__(self):
        return len(self.rxn_smiles)

    def __getitem__(self, idx):
        return self.rxn_smiles[idx]

    def remove_rxn_mappings(self):
        for i, rxn_smile in enumerate(self.rxn_smiles):
            rxn = AllChem.ReactionFromSmarts(rxn_smile, useSmiles=True)
            AllChem.RemoveMappingNumbersFromReactions(rxn)
            self.rxn_smiles[i] = AllChem.ReactionToSmiles(rxn)

    def split_rxn(self):
        self.reactants, self.reagents, self.products = [], [], []
        for i, rxn_smile in enumerate(self.rxn_smiles):
            reactant, reagent, product = rxn_smile.split('>')
            self.reactants.append(reactant)
            self.reagents.append(reagent)
            self.products.append(product)

    #def canonicalize_smiles(self):

