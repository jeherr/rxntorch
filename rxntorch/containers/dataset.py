from __future__ import print_function

from torch.utils.data import Dataset
import rdkit.Chem as Chem
import rdkit.Chem.AllChem as AllChem

from rxntorch.utils import rxn_smiles_reader, get_mol_features


class RxnDataset(Dataset):
    """Object for containing sets of reaction SMILES strings.

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
        self.reactants, self.reagents, self.products = [], [], []

    def __len__(self):
        return len(self.rxn_smiles)

    def __getitem__(self, idx):
        return self.rxn_smiles[idx]

    def save_to_file(self, filename):
        with open(filename, "w") as f:
            for rxn in self.rxn_smiles:
                f.write(rxn+"\n")

    def remove_rxn_mappings(self):
        for i, rxn_smile in enumerate(self.rxn_smiles):
            rxn = AllChem.ReactionFromSmarts(rxn_smile, useSmiles=True)
            AllChem.RemoveMappingNumbersFromReactions(rxn)
            self.rxn_smiles[i] = AllChem.ReactionToSmiles(rxn)

    def split_rxns(self):
        for i, rxn_smile in enumerate(self.rxn_smiles):
            reactant, reagent, product = rxn_smile.split('>')
            self.reactants.append(reactant)
            self.reagents.append(reagent)
            self.products.append(product)

    def canonicalize_smiles(self):
        new_rxn_smiles = []
        new_reactants = []
        new_reagents = []
        new_products = []
        for i in range(len(self.rxn_smiles)):
            try:
                new_reactants.append(Chem.MolToSmiles(Chem.MolFromSmiles(self.reactants[i])))
                new_reagents.append(Chem.MolToSmiles(Chem.MolFromSmiles(self.reagents[i])))
                new_products.append(Chem.MolToSmiles(Chem.MolFromSmiles(self.products[i])))
                new_rxn_smiles.append(new_reactants[-1]+'>'+new_reagents[-1]+'>'+new_products[-1])
            except:
                pass
        # TODO This function needs to make sure reactants, reagents, and products all match up with rxn_smiles
        (self.rxn_smiles, self.reactants, self.reagents, self.products) = (
            new_rxn_smiles, new_reactants, new_reagents, new_products)
