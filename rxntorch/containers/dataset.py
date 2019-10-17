from __future__ import print_function
from __future__ import division

import os
import random

import torch
from torch.utils.data import Dataset

import rdkit.Chem as Chem

from sklearn.preprocessing import LabelEncoder

from .reaction import Rxn
from .molecule import Mol
from rxntorch.utils import rxn_smiles_reader, mol_smiles_reader


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

    def remove_max_reactants(self, max_reactants):
        keep_rxns = [rxn for rxn in self.rxns if len(rxn.reactants) <= max_reactants]
        self.rxns = keep_rxns

    def remove_max_products(self, max_products):
        keep_rxns = [rxn for rxn in self.rxns if len(rxn.products) <= max_products]
        self.rxns = keep_rxns


class RxnGraphDataset(RxnDataset):
    """Object for containing sets of reactions SMILES strings.

    Attributes:
        file      (str): location of the file rxns are loaded from.
        rxn_strs  (set): reaction strings of the dataset and the bonding changes.
        bins     (dict): contains lists of indices to map rxn_strs to bin sizes
                for mapping data into efficient batches.

    """
    def __init__(self, file_name, path="data/"):
        super(RxnGraphDataset, self).__init__(file_name, path)
        self.file_name = file_name
        self.path = path
        self.rxns = []
        self.degree_codec = LabelEncoder()
        self.symbol_codec = LabelEncoder()
        self.expl_val_codec = LabelEncoder()
        self.bond_type_codec = LabelEncoder()
        self._init_dataset()
        self.max_nb = 10

    def __getitem__(self, idx):
        rxn, edits, heavy_count = self.rxns[idx]
        mol = Chem.MolFromSmiles('.'.join(filter(None, (rxn.reactants_smile, rxn.reagents_smile)))) 

        n_atoms = mol.GetNumAtoms()
        fatoms = self.get_atom_features(mol)
        fbonds = self.get_bond_features(mol)
        atom_nb = torch.zeros((n_atoms, self.max_nb), dtype=torch.long)
        bond_nb = torch.zeros((n_atoms, self.max_nb), dtype=torch.long)
        num_nbs = torch.zeros((n_atoms,), dtype=torch.int)

        for bond in mol.GetBonds():
            a1 = bond.GetBeginAtom().GetIdx()
            a2 = bond.GetEndAtom().GetIdx()
            idx = bond.GetIdx()
            if num_nbs[a1] == self.max_nb or num_nbs[a2] == self.max_nb:
                raise Exception(rxn.reactants_smile)
            atom_nb[a1, num_nbs[a1]] = a2
            atom_nb[a2, num_nbs[a2]] = a1
            bond_nb[a1, num_nbs[a1]] = idx
            bond_nb[a2, num_nbs[a2]] = idx
            num_nbs[a1] += 1
            num_nbs[a2] += 1

        blabels = self.get_bond_labels(mol, edits, n_atoms)
        return fatoms, fbonds, atom_nb, bond_nb, num_nbs, n_atoms

    def _init_dataset(self):
        symbols = set()
        degrees = set()
        explicit_valences = set()
        bond_types = set()

        with open(os.path.join(self.path, self.file_name), "r") as datafile:
            for line in datafile:
                rxn_smile, edits = line.strip("\r\n ").split()
                count = rxn_smile.count(":")
                rxn = Rxn(rxn_smile)
                self.rxns.append((rxn, edits, count))
                mol = Chem.MolFromSmiles('.'.join(filter(None, (rxn.reactants_smile, rxn.reagents_smile))))
                for atom in mol.GetAtoms():
                    symbols.add(atom.GetSymbol())
                    degrees.add(atom.GetDegree())
                    explicit_valences.add(atom.GetExplicitValence())
                for bond in mol.GetBonds():
                    bond_types.add(bond.GetBondType())
        symbols.add("unknown")

        self.degree_codec.fit(list(degrees))
        self.symbol_codec.fit(list(symbols))
        self.expl_val_codec.fit(list(explicit_valences))
        self.bond_type_codec.fit(list(bond_types))

    def get_atom_features(self, mol):
        symbols = [atom.GetSymbol() for atom in mol.GetAtoms()]
        degrees = [atom.GetDegree() for atom in mol.GetAtoms()]
        expl_vals = [atom.GetExplicitValence() for atom in mol.GetAtoms()]

        t_symbol = self.to_one_hot(self.symbol_codec, symbols)
        t_degree = self.to_one_hot(self.degree_codec, degrees)
        t_expl_val = self.to_one_hot(self.expl_val_codec, expl_vals)
        t_aromatic = torch.tensor([atom.GetIsAromatic() for atom in mol.GetAtoms()]).float().unsqueeze(1)
        return torch.cat((t_symbol, t_degree, t_expl_val, t_aromatic), dim=1)

    def get_bond_features(self, mol):
        bond_types = [bond.GetBondType() for bond in mol.GetBonds()]
        t_bond_types = self.to_one_hot(self.bond_type_codec, bond_types)
        t_conjugated = torch.tensor([bond.GetIsConjugated() for bond in mol.GetBonds()]).float().unsqueeze(1)
        t_in_ring = torch.tensor([bond.IsInRing() for bond in mol.GetBonds()]).float().unsqueeze(1)
        return torch.cat((t_bond_types, t_conjugated, t_in_ring), dim=1)

    def to_one_hot(self, codec, values):
        value_idxs = codec.transform(values)
        return torch.eye(len(codec.classes_), dtype=torch.float)[value_idxs]

    def get_bond_labels(self, mol, edits, n_atoms):
        bo_to_index = {0.0: 0, 1: 1, 2: 2, 3: 3, 1.5: 4}
        edits = edits.split(";")
        bond_labels = torch.zeros((n_atoms, n_atoms, len(bo_to_index)))
        for edit in edits:
            atom1, atom2, bond_order = edit.split("-")
            bo_index = bo_to_index(float(bond_order))
            bond_labels[int(atom1)-1, int(atom2)-1, bo_index] = bond_labels[int(atom2)-1, int(atom1)-1, bo_index] = 1
        for i in range(n_atoms):
            bond_labels[i,i,:] = -1
        return bond_labels



    def get_indices_bins(self):
        #TODO This method needs finished to collect indices for each reaction into bins to separate batches by size
        lengths = [len(rxn.reactants_smile) for rxn in self.rxns]
        print(max(lengths))
        print(min(lengths))


class RxnTransformerDataset(RxnDataset):
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

    def __getitem__(self, idx):
        rxn = self.rxns[idx]
        reactant_list, product_list = self.vocab.split(rxn.smile)
        reactant_list, product_list = self.vocab.to_seq(reactant_list, seq_len=150), self.vocab.to_seq(product_list, seq_len=150)

        output_label = []
        for i, token in enumerate(reactant_list):
            prob = random.random()
            if prob < 0.15:
                prob /= 0.15

                # 80% randomly change token to mask token
                if prob < 0.8:
                    reactant_list[i] = self.vocab.mask_index

                # 10% randomly change token to random token
                elif prob < 0.9:
                    reactant_list[i] = random.randrange(len(self.vocab))

                # 10% randomly change token to current token
                else:
                    reactant_list[i] = self.vocab.stoi.get(token, self.vocab.unk_index)

                output_label.append(self.vocab.stoi.get(token, self.vocab.unk_index))

            else:
                reactant_list[i] = self.vocab.stoi.get(token, self.vocab.unk_index)
                output_label.append(0)

        reactant_list.insert(0, self.vocab.sos_index)
        reactant_list.append(self.vocab.eos_index)
        output_label.insert(0, self.vocab.sos_index)
        output_label.append(self.vocab.eos_index)
        #product_list.insert(0, self.vocab.sos_index)
        #product_list.append(self.vocab.eos_index)

        output = {"input": reactant_list,
                  "label": output_label}

        return {key: torch.tensor(value) for key, value in output.items()}

    def get_indices_bins(self):
        #TODO This method needs finished to collect indices for each reaction into bins to separate batches by size
        lengths = [len(rxn.reactants_smile) for rxn in self.rxns]
        print(max(lengths))
        print(min(lengths))


class MolDataset(Dataset):
    """Object for containing sets of reactions SMILES strings.

    Attributes:
        file      (str): location of the file rxns are loaded from.
        rxn_strs  (set): reaction strings of the dataset and the bonding changes.
        bins     (dict): contains lists of indices to map rxn_strs to bin sizes
                for mapping data into efficient batches.

    """
    def __init__(self, file_name, path="data/", vocab=None):
        super(MolDataset, self).__init__()
        self.file_name = file_name
        self.path = path
        self.vocab = vocab

    def __len__(self):
        return len(self.mols)

    def __getitem__(self, idx):
        mol1 = self.mols[idx]
        mol2 = self.mols[random.randint(0, len(self.mols))]
        length = len(mol1.smile)+len(mol2.smile)+1
        input_list = self.vocab.to_seq('.'.join([mol1.smile, mol2.smile]), seq_len=150)
        output_label = input_list
        mask_idxs = random.sample(range(length), int(length * 0.15))
        for idx in mask_idxs:
            prob = random.random()

            # 80% randomly change token to mask token
            if prob < 0.8:
                input_list[idx] = self.vocab.mask_index

            # 10% randomly change token to random token
            elif prob < 0.9:
                input_list[idx] = random.randrange(len(self.vocab))

            # 10% randomly change token to current token
            #else:
                #input_list[i] = self.vocab.stoi.get(token, self.vocab.unk_index)

        input_list.insert(0, self.vocab.sos_index)
        input_list.append(self.vocab.eos_index)
        output_label.insert(0, self.vocab.sos_index)
        output_label.append(self.vocab.eos_index)

        output = {"input": input_list,
                  "label": output_label}

        return {key: torch.tensor(value) for key, value in output.items()}

    def __setitem__(self, idx, value):
        self.mols[idx] = value

    @classmethod
    def from_list(cls, mols):
        new_dataset = cls('')
        new_dataset.mols = mols
        return new_dataset

    @classmethod
    def from_rxn_dataset(cls, rxn_dataset):
        new_dataset = cls('')
        mols = []
        for rxn in rxn_dataset.rxns:
            for mol in rxn.reactants:
                mols.append(mol)
            for mol in rxn.reagents:
                mols.append(mol)
            for mol in rxn.products:
                mols.append(mol)
        new_dataset.mols = mols
        return new_dataset

    @property
    def mol_smiles(self):
        return [mol.smile for mol in self.mols]

    @mol_smiles.setter
    def mol_smiles(self, mol_smiles):
        self.mols = [Mol(mol_smile) for mol_smile in mol_smiles]

    def load_from_file(self):
        mol_smiles = mol_smiles_reader(os.path.join(self.path, self.file_name))
        self.mols = [Mol(mol_smile) for mol_smile in mol_smiles]

    def save_to_file(self, file_name=None, path=None):
        if file_name == None:
            file_name = self.file_name
        if path == None:
            path = self.path
        with open(os.path.join(path, file_name), "w") as f:
            for mol in self.mols:
                f.write(mol.smile+"\n")

    def canonicalize(self):
        for mol in self.mols:
            mol.canonicalize()

    def strip_repeats(self):
        unique_mols
        return