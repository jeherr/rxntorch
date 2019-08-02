from __future__ import print_function

from torch.utils.data import Dataset
from rdkit.Chem.rdChemReactions import ReactionFromSmarts

from rxntorch.utils import rxn_smiles_reader


class RxnDataset(Dataset):
    """Object for containing large sets of reaction SMILES strings.

    Attributes:
        file      (str): location of the file rxns were loaded from
        rxn_strs (dict): a dictionary of reaction strings binned by the number
                of heavy atoms.
        rxn      (dict): a dictionary of rdkit ChemicalReactions objects,
                binned by the number of heavy atoms.
    """
    def __init__(self, file_):
        self.file = file_
        self.rxn_strs = rxn_smiles_reader(self.file)
        self.build_rxns()

    def __len__(self):
        return len(self.rxns)

    def __getitem__(self, idx):
        return sample

    def build_rxns(self):
        self.rxns = {bin_size: [] for bin_size in self.rxn_strs.keys()}
        for bin_size, ibin in self.rxn_strs.items():
            for rxn in ibin:
                self.rxns[bin_size].append(ReactionFromSmarts(rxn[0]))
