from __future__ import print_function
from __future__ import division

import os
import random
import logging

import rdkit.Chem as Chem
from sklearn.preprocessing import LabelEncoder

import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence

from .reaction import Rxn


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
        self.max_nbonds = 10

    def _init_dataset(self):
        logging.info("Loading Dataset {dataset} in {datapath}".format(
            dataset=self.file_name, datapath=self.path))
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
        logging.info("Dataset contains {:d} total samples".format(len(self.rxns)))

        self.degree_codec.fit(list(degrees))
        self.symbol_codec.fit(list(symbols))
        self.expl_val_codec.fit(list(explicit_valences))
        self.bond_type_codec.fit(list(bond_types))

    def __getitem__(self, idx):
        rxn, edits, heavy_count = self.rxns[idx]
        reactants = rxn.reactants
        reagents = rxn.reagents
        all_mols = reactants + reagents
        idxs = [i for i in range(len(all_mols))]
        reactant_labels = [0 for _ in range(len(reactants))]
        reagent_labels = [1 for _ in range(len(reagents))]
        all_labels = reactant_labels + reagent_labels
        random.shuffle(idxs)
        all_mols = [all_mols[idx_] for idx_ in idxs]
        all_labels = torch.tensor([all_labels[idx_] for idx_ in idxs])

        n_atoms = []
        atom_idxs = []
        atom_feats = []
        bond_feats = []
        atom_graphs = []
        bond_graphs = []
        n_bonds = []
        n_mols = len(all_mols)
        for i, mol in enumerate(all_mols):
            rdmol = Chem.MolFromSmiles(mol.smile)
            n_atoms.append(rdmol.GetNumAtoms())
            atom_idx = [atom.GetIdx() for atom in rdmol.GetAtoms()]
            atom_idxs.append(atom_idx)
            atom_feats.append(self.get_atom_features(rdmol, atom_idx))
            bond_feats.append(self.get_bond_features(rdmol))
            atom_graph = torch.zeros((n_atoms[-1], self.max_nbonds), dtype=torch.int64)
            bond_graph = torch.zeros((n_atoms[-1], self.max_nbonds), dtype=torch.int64)
            n_bond = torch.zeros((n_atoms[-1],), dtype=torch.int32)

            for bond in rdmol.GetBonds():
                a1 = bond.GetBeginAtom().GetIdx()
                a2 = bond.GetEndAtom().GetIdx()
                idx = bond.GetIdx()
                if n_bond[a1] == self.max_nbonds or n_bond[a2] == self.max_nbonds:
                    raise Exception(rxn.reactants_smile)
                atom_graph[a1, n_bond[a1]] = a2
                atom_graph[a2, n_bond[a2]] = a1
                bond_graph[a1, n_bond[a1]] = idx
                bond_graph[a2, n_bond[a2]] = idx
                n_bond[a1] += 1
                n_bond[a2] += 1

            atom_graphs.append(atom_graph)
            bond_graphs.append(bond_graph)
            n_bonds.append(n_bond)

        print([atom_feat.shape for atom_feat in atom_feats])
        print([bond_feat.shape for bond_feat in bond_feats])
        print([bond_graph.shape for bond_graph in bond_graphs])
        atom_mol_idx = torch.cat([torch.full((atom_feats[i].shape[0],), i) for i in range(n_mols)], dim=0)
        bond_mol_idx = torch.cat([torch.full((bond_feats[i].shape[0],), i) for i in range(n_mols)], dim=0)
        #atom_feats = pad_sequence(atom_feats, batch_first=True)
        #bond_feats = pad_sequence(bond_feats, batch_first=True)
        #atom_graphs = pad_sequence(atom_graphs, batch_first=True)
        #bond_graphs = pad_sequence(bond_graphs, batch_first=True)
        #n_bonds = pad_sequence(n_bonds, batch_first=True)
        #n_atoms = torch.tensor(n_atoms, dtype=torch.int32)

        output = {"atom_feats": atom_feats,
                  "bond_feats": bond_feats,
                  "atom_graphs": atom_graphs,
                  "bond_graphs": bond_graphs,
                  "n_mols": n_mols,
                  "atom_mol_idx": atom_mol_idx,
                  "bond_mol_idx": bond_mol_idx,
                  "n_bonds": n_bonds,
                  "n_atoms": n_atoms,
                  "reagent_labels": all_labels}
        return output

    def __getitem__(self, idx):
        rxn, edits, heavy_count = self.rxns[idx]
        reactants = rxn.reactants
        reagents = rxn.reagents
        all_mols = reactants + reagents
        idxs = [i for i in range(len(all_mols))]
        reactant_labels = [0 for _ in range(len(reactants))]
        reagent_labels = [1 for _ in range(len(reagents))]
        all_labels = reactant_labels + reagent_labels
        random.shuffle(idxs)
        all_mols = [all_mols[idx_] for idx_ in idxs]
        atom_counts = [mol.smile.count(":") for mol in all_mols]
        mol_map = torch.cat([torch.full((atom_count,), i) for i, atom_count in enumerate(atom_counts)], dim=0)
        print(mol_map)
        all_labels = torch.tensor([all_labels[idx_] for idx_ in idxs])

        react_smiles = '.'.join(filter(None, [mol.smile for mol in all_mols]))
        mol = Chem.MolFromSmiles(react_smiles)
        atom_idx = torch.tensor([atom.GetIdx() for atom in mol.GetAtoms()], dtype=torch.int64)

        n_mols = len(all_mols)
        n_atoms = mol.GetNumAtoms()
        sparse_idx = []
        for i in range(n_atoms):
            for j in range(i+1, n_atoms):
                sparse_idx.append([atom_idx[i],atom_idx[j]])
        sparse_idx = torch.tensor(sparse_idx, dtype=torch.int64)
        atom_feats = self.get_atom_features(mol, atom_idx)
        bond_feats = self.get_bond_features(mol)
        atom_graph = torch.zeros((n_atoms, self.max_nbonds), dtype=torch.int64)
        bond_graph = torch.zeros((n_atoms, self.max_nbonds), dtype=torch.int64)
        n_bonds = torch.zeros((n_atoms,), dtype=torch.int32)

        for bond in mol.GetBonds():
            a1 = bond.GetBeginAtom().GetIdx()
            a2 = bond.GetEndAtom().GetIdx()
            idx = bond.GetIdx()
            if n_bonds[a1] == self.max_nbonds or n_bonds[a2] == self.max_nbonds:
                raise Exception(rxn.reactants_smile)
            atom_graph[a1, n_bonds[a1]] = a2
            atom_graph[a2, n_bonds[a2]] = a1
            bond_graph[a1, n_bonds[a1]] = idx
            bond_graph[a2, n_bonds[a2]] = idx
            n_bonds[a1] += 1
            n_bonds[a2] += 1

        output = {"atom_feats": atom_feats,
                  "bond_feats": bond_feats,
                  "atom_graph": atom_graph,
                  "bond_graph": bond_graph,
                  "n_mols": n_mols,
                  "n_atoms": n_atoms,
                  "n_bonds": n_bonds,
                  "reagent_labels": all_labels,
                  "sparse_idx": sparse_idx}
        return output

    def get_atom_features(self, mol, atom_idx):
        symbols = [atom.GetSymbol() for atom in mol.GetAtoms()]
        degrees = [atom.GetDegree() for atom in mol.GetAtoms()]
        expl_vals = [atom.GetExplicitValence() for atom in mol.GetAtoms()]

        t_symbol = self.to_one_hot(self.symbol_codec, symbols)
        t_degree = self.to_one_hot(self.degree_codec, degrees)
        t_expl_val = self.to_one_hot(self.expl_val_codec, expl_vals)
        t_aromatic = torch.tensor([atom.GetIsAromatic() for atom in mol.GetAtoms()]).float().unsqueeze(1)
        return torch.cat((t_symbol, t_degree, t_expl_val, t_aromatic), dim=1)[atom_idx]

    def get_bond_features(self, mol):
        bond_types = [bond.GetBondType() for bond in mol.GetBonds()]
        t_bond_types = self.to_one_hot(self.bond_type_codec, bond_types)
        t_conjugated = torch.tensor([bond.GetIsConjugated() for bond in mol.GetBonds()]).float().unsqueeze(1)
        t_in_ring = torch.tensor([bond.IsInRing() for bond in mol.GetBonds()]).float().unsqueeze(1)
        return torch.cat((t_bond_types, t_conjugated, t_in_ring), dim=1)

    @staticmethod
    def to_one_hot(codec, values):
        value_idxs = codec.transform(values)
        return torch.eye(len(codec.classes_), dtype=torch.float)[value_idxs]

    @staticmethod
    def get_bond_labels(edits, n_atoms, sparse_idx, atom_map_idx):
        bo_index_map = {0.0: 0,
                       1: 1,
                       2: 2,
                       3: 3,
                       1.5: 4}
        edits = edits.split(";")
        bond_labels = torch.zeros((n_atoms, n_atoms, len(bo_index_map)))
        for edit in edits:
            atom1, atom2, bond_order = edit.split("-")
            atom1, atom2 = atom_map_idx[int(atom1)], atom_map_idx[int(atom2)]
            bo_index = bo_index_map[float(bond_order)]
            bond_labels[int(atom1), int(atom2), bo_index] = bond_labels[int(atom2), int(atom1), bo_index] = 1
        bond_labels = bond_labels[sparse_idx[:,0],sparse_idx[:,1]]
        return bond_labels

    def get_binary_features(self, smiles, n_atoms):
        comp = {}
        for i, s in enumerate(smiles.split('.')):
            mol = Chem.MolFromSmiles(s)
            for atom in mol.GetAtoms():
                comp[atom.GetIntProp('molAtomMapNumber')] = i
        n_comp = len(smiles.split('.'))
        rmol = Chem.MolFromSmiles(smiles)
        bond_map = {}
        for bond in rmol.GetBonds():
            a1 = bond.GetBeginAtom().GetIdx()
            a2 = bond.GetEndAtom().GetIdx()
            bond_map[(a1, a2)] = bond

        binary_feats = torch.zeros((n_atoms, n_atoms, 10))
        for i in range(n_atoms):
            for j in range(i+1, n_atoms):
                if (i,j) in bond_map:
                    bond = bond_map[(i,j)]
                    binary_feats[i,j,1:1+6] = binary_feats[j,i,1:1+6] = self.bond_features(bond)
                elif (j,i) in bond_map:
                    bond = bond_map[(j,i)]
                    binary_feats[i,j,1:1+6] = binary_feats[j,i,1:1+6] = self.bond_features(bond)
                else:
                    binary_feats[i,j,0] = binary_feats[j,i,0] = 1.0
                atom1, atom2 = rmol.GetAtomWithIdx(i), rmol.GetAtomWithIdx(j)
                map_i, map_j = atom1.GetIntProp('molAtomMapNumber'), atom2.GetIntProp('molAtomMapNumber')
                binary_feats[i,j,-4] = binary_feats[j,i,-4] = 1.0 if comp[map_i] != comp[map_j] else 0.0
                binary_feats[i,j,-3] = binary_feats[j,i,-3] = 1.0 if comp[map_i] == comp[map_j] else 0.0
                binary_feats[i,j,-2] = binary_feats[j,i,-2] = 1.0 if n_comp == 1 else 0.0
                binary_feats[i,j,-1] = binary_feats[j,i,-1] = 1.0 if n_comp > 1 else 0.0
        return binary_feats

    @staticmethod
    def bond_features(bond):
        bt = bond.GetBondType()
        return torch.Tensor(
            [bt == Chem.rdchem.BondType.SINGLE, bt == Chem.rdchem.BondType.DOUBLE, bt == Chem.rdchem.BondType.TRIPLE,
             bt == Chem.rdchem.BondType.AROMATIC, bond.GetIsConjugated(), bond.IsInRing()])
