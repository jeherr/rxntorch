from __future__ import print_function

import random
import gzip
import os
import _pickle as pickle

from torch.utils.data import Dataset
from torch.utils.data.sampler import Sampler

from .reaction import Rxn
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
    def __init__(self, file_name, path="data/"):
        super(RxnDataset, self).__init__()
        self.file_name = file_name
        self.path = path
        rxn_smiles = rxn_smiles_reader(os.path.join(self.path, self.file_name))
        self.rxns = [Rxn(rxn_smile) for rxn_smile in rxn_smiles]
        self.vocab_reactants = {}
        self.vocab_products = {}

    def __len__(self):
        return len(self.rxns)

    def __getitem__(self, idx):
        return self.rxns[idx]

    @property
    def rxn_smiles(self):
        return [rxn.smile for rxn in self.rxns]

    @rxn_smiles.setter
    def rxn_smiles(self, rxn_smiles):
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

    def create_vocab(self):
        _PAD = "_PAD"
        _GO = "_GO"
        _EOS = "_EOS"
        _START_VOCAB = [_PAD, _GO, _EOS]

        PAD_ID = 0
        GO_ID = 1
        EOS_ID = 2

        error_rsmi = {}

        for i in range(len(self.rxns)):
            try:
                reactants = self.reactants[i].split('.')
                reagents = self.reagents[i].split('.')
                products = self.products[i].split('.')

                reactant_list = []
                reagent_list = []
                product_list = []

                for reactant in reactants:
                    reactant_list += sp.parser_list(reactant)
                    reactant_list += '.'
                for reagent in reagents:
                    reagent_list += sp.parser_list(reagent)
                    reagent_list += '.'
                for product in products:
                    product_list += sp.parser_list(product)
                    product_list += '.'
                reactant_list.pop()
                reagent_list.pop()
                product_list.pop()
                reactant_list += '>'
                reactant_list += reagent_list

                for reactant_token in reactant_list:
                    if reactant_token in self.vocab_reactants:
                        self.vocab_reactants[reactant_token] += 1
                    else:
                        self.vocab_reactants[reactant_token] = 1

                for product_token in product_list:
                    if product_token in self.vocab_products:
                        self.vocab_products[product_token] += 1
                    else:
                        self.vocab_products[product_token] = 1
            except:
                error_rsmi.update({i: self.rxn_smiles[i]})

        self.reactants_token_list = _START_VOCAB \
                               + sorted(self.vocab_reactants, key=self.vocab_reactants.get, reverse=True)

        self.products_token_list = _START_VOCAB \
                              + sorted(self.vocab_products, key=self.vocab_products.get, reverse=True)

    def save_vocab_to_file(self, dict_file=None, list_file=None, path=None):
        if dict_file == None:
            dict_file = "vocab_dict.pkl.gz"
        if list_file == None:
            list_file = "vocab_list.pkl.gz"
        if path == None:
            path = self.path
        with gzip.open(os.path.join(path, dict_file), 'wb') as dict_f:
            pickle.dump((self.vocab_reactants, self.vocab_products), dict_f)

        with gzip.open(os.path.join(path, list_file), 'wb') as list_f:
            pickle.dump((self.reactants_token_list, self.products_token_list), list_f)

    def load_vocab_from_file(self, dict_file=None, list_file=None, path=None):
        if dict_file == None:
            dict_file = "vocab_dict.pkl.gz"
        if list_file == None:
            list_file = "vocab_list.pkl.gz"
        if path == None:
            path = self.path
        with gzip.open(os.path.join(path, dict_file), 'rb') as dict_f:
            self.vocab_reactants, self.vocab_products = pickle.load(dict_f)

        with gzip.open(os.path.join(path, list_file), 'rb') as list_f:
            self.reactants_token_list, self.products_token_list = pickle.load(list_f)


#TODO Dataset class needs a method to bin reactions based on length of string


#TODO This should be moved to the model class
class BinRandomSampler(Sampler):
    """Samples elements randomly from a dictionary of bin sizes and lists
    of indices corresponding to the master list of reactions for the dataset.

    Arguments:
        indices (dict): a sequence of indices
    """

    def __init__(self, indices, batch_size):
        super(BinRandomSampler, self).__init__()
        self.indices = indices
        self.batch_size = batch_size

    def __iter__(self):
        # Get the random permutations of the binned indices.
        rand_bins = [torch.randperm(len(idx_bin)) for idx_bin in self.indices.values()]
        # Trim the permuted indices so that each set is divisible by the batch size.
        rand_bins = [idx_bin[:-1 * (len(idx_bin) % self.batch_size)] for idx_bin in rand_bins]
        # Now collect batch size chunks of indices from each bin into a master list.
        batches = []
        for idx_bin in rand_bins:
            for i in range(len(idx_bin) / self.batch_size):
                batches.append([idx_bin[i*self.batch_size:(i+1)*self.batch_size]])
        # Shuffle to keep batches together but order of batches randomized.
        random.shuffle(batches)
        # Then merge into one list of indices for the current epoch.
        return [idx for batch in batches for idx in batch]


        return (self.indices[i] for i in torch.randperm(len(self.indices)))

    def __len__(self):
        return sum([len(bin) for bin in self.indices.values()])
