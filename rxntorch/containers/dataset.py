from __future__ import print_function

import gzip
import os
import _pickle as pickle

import torch
from torch.utils.data import Dataset

from .reaction import Rxn
from .vocabulary import SmilesVocab
from rxntorch.utils import rxn_smiles_reader
import rxntorch.smiles_parser as sp


class RxnDataset(Dataset):
    """Object for containing sets of reactions SMILES strings.

    Attributes:
        file      (str): location of the file rxns are loaded from.
        rxn_strs  (set): reaction strings of the dataset and the bonding changes.
        bins     (dict): contains lists of indices to map rxn_strs to bin sizes
                for mapping data into efficient batches.

    """
    def __init__(self, file_name, path="data/", vocab=None):
        super(RxnDataset, self).__init__()
        self.file_name = file_name
        self.path = path
        self.vocab = vocab

    def __len__(self):
        return len(self.rxns)

    def __getitem__(self, idx):
        rxn = self.rxns[idx]
        reactant_list, product_list = self.vocab.split(rxn.smile)
        reactant_list, product_list = self.vocab.to_seq(reactant_list, seq_len=150), self.vocab.to_seq(product_list, seq_len=150)
        reactant_list.insert(0, self.vocab.sos_index)
        reactant_list.append(self.vocab.eos_index)
        product_list.insert(0, self.vocab.sos_index)
        product_list.append(self.vocab.eos_index)

        output = {"input": reactant_list,
                  "label": product_list}

        return {key: torch.tensor(value) for key, value in output.items()}

    def __setitem__(self, idx, value):
        self.rxns[idx] = value

    @property
    def rxn_smiles(self):
        return [rxn.smile for rxn in self.rxns]

    @rxn_smiles.setter
    def rxn_smiles(self, rxn_smiles):
        self.rxns = [Rxn(rxn_smile) for rxn_smile in rxn_smiles]

    @classmethod
    def from_list(cls, rxns):
        new_dataset = cls('')
        new_dataset.rxns = rxns
        return new_dataset

    def load_from_file(self):
        rxn_smiles = rxn_smiles_reader(os.path.join(self.path, self.file_name))
        self.rxns = [Rxn(rxn_smile) for rxn_smile in rxn_smiles]

    def save_to_file(self, file_name=None, path=None):
        if file_name == None:
            file_name = self.file_name
        if path == None:
            path = self.path
        with open(os.path.join(path, file_name), "w") as f:
            for rxn in self.rxn_smiles:
                f.write(rxn+"\n")

    def canonicalize(self):
        for rxn in self.rxns:
            rxn.canonicalize()

    def remove_rxn_mappings(self):
        for rxn in self.rxns:
            rxn.remove_rxn_mapping()

    def get_indices_bins(self):
        #TODO This method needs finished to collect indices for each reaction into bins to separate batches by size
        lengths = [len(rxn.reactants_smile) for rxn in self.rxns]
        print(max(lengths))
        print(min(lengths))

    def remove_max_reactants(self, max_reactants):
        keep_rxns = [rxn for rxn in self.rxns if len(rxn.reactants) <= max_reactants]
        self.rxns = keep_rxns

    def remove_max_products(self, max_products):
        keep_rxns = [rxn for rxn in self.rxns if len(rxn.products) <= max_products]
        self.rxns = keep_rxns

