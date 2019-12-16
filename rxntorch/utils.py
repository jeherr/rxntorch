from __future__ import print_function

import torch

def collate_fn(batch_data):
    """
    Takes a batch of data for the graph network model and collates it into
    torch tensors which are padded to accommodate different numbers of atoms
    """
    n_atoms = torch.tensor([sample['n_atoms'] for sample in batch_data], dtype=torch.int32)
    batch_size = len(batch_data)
    max_atoms = n_atoms.max()
    max_bonds = max([sample['bond_feats'].shape[-2] for sample in batch_data])
    afeatures_size = batch_data[0]['atom_feats'].shape[-1]
    bfeatures_size = batch_data[0]['bond_feats'].shape[-1]
    binfeatures_size = batch_data[0]['binary_feats'].shape[-1]
    blabel_size = batch_data[0]['bond_labels'].shape[-1]

    # Create torch tensors for inputs which need padding. Every tensor is padded with
    # 0s except for the bond labels which are padded with -1 to catch invalid bonds
    # which correspond to either padded atoms or atoms bonding to itself
    atom_feats = torch.zeros((batch_size, max_atoms, afeatures_size))
    bond_feats = torch.zeros((batch_size, max_bonds, bfeatures_size))
    binary_feats = torch.zeros((batch_size, max_atoms, max_atoms, binfeatures_size))
    bond_labels = torch.full((batch_size, max_atoms, max_atoms, blabel_size), -1)
    atom_graph = torch.zeros((batch_size, max_atoms, 10, 2), dtype=torch.int64)
    bond_graph = torch.zeros((batch_size, max_atoms, 10, 2), dtype=torch.int64)
    n_bonds = torch.zeros((batch_size, max_atoms), dtype=torch.int32)

    for i, sample in enumerate(batch_data):
        n_atom = n_atoms[i]
        n_bond = sample['bond_feats'].shape[-2]
        atom_feats[i,:n_atom,:] = sample['atom_feats']
        bond_feats[i,:n_bond,:] = sample['bond_feats']
        binary_feats[i,:n_atom,:n_atom,:] = sample['binary_feats']
        bond_labels[i,:n_atom,:n_atom,:] = sample['bond_labels']
        atom_graph[i,:n_atom,:,0] = i
        atom_graph[i,:n_atom,:,1] = sample['atom_graph']
        bond_graph[i,:n_atom,:,0] = i
        bond_graph[i,:n_atom,:,1] = sample['bond_graph']
        n_bonds[i,:n_atom] = sample['n_bonds']

    output = {"atom_feats": atom_feats,
              "bond_feats": bond_feats,
              "atom_graph": atom_graph,
              "bond_graph": bond_graph,
              "n_bonds": n_bonds,
              "n_atoms": n_atoms,
              "bond_labels": bond_labels,
              "binary_feats": binary_feats}
    return output

