from __future__ import print_function

import gzip
import _pickle as pickle

from torch.utils.data import Dataset

from rxntorch import Rxn
import rxntorch.smiles_parser as sp
from rxntorch.utils import rxn_smiles_reader


class RxnDataset(Dataset):
    """Object for containing sets of reaction SMILES strings.

    Attributes:
        file      (str): location of the file rxns are loaded from.
        rxn_strs  (set): reaction strings of the dataset and the bonding changes.
        bins     (dict): contains lists of indices to map rxn_strs to bin sizes
                for mapping data into efficient batches.

    """
    def __init__(self, file_name):
        super(RxnDataset, self).__init__()
        self.file_name = file_name
        rxn_smiles = rxn_smiles_reader(self.file_name)
        self.rxns = [Rxn(rxn_smile) for rxn_smile in rxn_smiles]
        self.reactants, self.reagents, self.products = [], [], []

    def __len__(self):
        return len(self.rxn_smiles)

    def __getitem__(self, idx):
        return self.rxn_smiles[idx]

    @property
    def rxn_smiles(self):
        return [rxn.smile for rxn in self.rxns]

    @rxn_smiles.setter
    def rxn_smiles(self, rxn_smiles):
        self.rxns = [Rxn(rxn_smile) for rxn_smile in rxn_smiles]

    def save_to_file(self, file_name):
        with open(file_name, "w") as f:
            for rxn in self.rxn_smiles:
                f.write(rxn+"\n")

    def canonicalize(self):
        for rxn in self.rxns:
            rxn.canonicalize()

    def remove_rxn_mappings(self):
        for rxn in self.rxns:
            rxn.remove_rxn_mapping()

    def create_vocab(self):
        _PAD = "_PAD"
        _GO = "_GO"
        _EOS = "_EOS"
        _START_VOCAB = [_PAD, _GO, _EOS]

        PAD_ID = 0
        GO_ID = 1
        EOS_ID = 2

        vocab_reactants = {}
        vocab_products = {}
        error_rsmi = {}

        for i in range(len(self.rxn_smiles)):
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
                    if reactant_token in vocab_reactants:
                        vocab_reactants[reactant_token] += 1
                    else:
                        vocab_reactants[reactant_token] = 1

                for product_token in product_list:
                    if product_token in vocab_products:
                        vocab_products[product_token] += 1
                    else:
                        vocab_products[product_token] = 1
            except:
                error_rsmi.update({i: self.rxn_smiles[i]})

        reactants_token_list = _START_VOCAB \
                               + sorted(vocab_reactants, key=vocab_reactants.get, reverse=True)

        products_token_list = _START_VOCAB \
                              + sorted(vocab_products, key=vocab_products.get, reverse=True)

        with gzip.open('data/vocab_dict.pkl.gz', 'wb') as dict_file:
            pickle.dump((vocab_reactants, vocab_products), dict_file, 2)

        with gzip.open('data/vocab_list.pkl.gz', 'wb') as list_file:
            pickle.dump((reactants_token_list, products_token_list), list_file, 2)

        #with gzip.open('data/vocab_dict.pkl.gz', 'rb') as dict_file:
        #    vocab_reactants, vocab_products = pickle.load(dict_file)

        #with gzip.open('data/vocab_list.pkl.gz', 'rb') as list_file:
        #    reactants_token_list, products_token_list = pickle.load(list_file)

        #print(len(reactants_token_list))
        #print(reactants_token_list[:100])
        #print(reactants_token_list[-15:])

        #print('--------')

        #print(len(products_token_list))
        #print(products_token_list[:100])
        #print(products_token_list[-15:])

        #print('--------')
        #print('--------')

        #for token in reactants_token_list[3:20]:
        #    print(token, vocab_reactants.get(token))

        #print('--------')

        #for token in products_token_list[3:20]:
        #    print(token, vocab_products.get(token))

        #print('--------')
        #print('--------')

        #print(sum(vocab_reactants.values()))
        #print(sum(vocab_products.values()))
