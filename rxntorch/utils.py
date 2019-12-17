from __future__ import print_function

import torch
from torch.nn.utils.rnn import pad_sequence


def collate_fn(batch_data):
    """
    Takes a batch of data for the graph network model and collates it into
    torch tensors which are padded to accommodate different numbers of atoms
    """
    output = {}
    for key in batch_data[0].keys():
        if (key == "bond_labels") or (key == "binary_feats"):
            to_stack = [sample[key] for sample in batch_data]
            n_atoms = [label.shape[0] for label in to_stack]
            max_atoms = max(n_atoms)
            if key == "bond_labels":
                values = torch.full((len(to_stack), max_atoms, max_atoms, to_stack[0].shape[-1]), -1)
            else:
                values = torch.zeros((len(to_stack), max_atoms, max_atoms, to_stack[0].shape[-1]))
            for i, (label, n_atom) in enumerate(zip(to_stack, n_atoms)):
                values[i,:n_atom,:n_atom] = label
            #values = pad_sequence([sample[key] for sample in batch_data], batch_first=True, padding_value=-1.0)
        elif key == "n_atoms":
            values = torch.tensor([sample[key] for sample in batch_data], dtype=torch.int32)
        else:
            values = pad_sequence([sample[key] for sample in batch_data], batch_first=True)
        output[key] = values


    #for i, sample in enumerate(batch_data):
    #    n_atom = n_atoms[i]
    #    n_bond = sample['bond_feats'].shape[-2]
    #    atom_feats[i,:n_atom,:] = sample['atom_feats']
    #    bond_feats[i,:n_bond,:] = sample['bond_feats']
    #    binary_feats[i,:n_atom,:n_atom,:] = sample['binary_feats']
    #    bond_labels[i,:n_atom,:n_atom,:] = sample['bond_labels']
    #    atom_graph[i,:n_atom,:,0] = i
    #    atom_graph[i,:n_atom,:,1] = sample['atom_graph']
    #    bond_graph[i,:n_atom,:,0] = i
    #    bond_graph[i,:n_atom,:,1] = sample['bond_graph']
    #    n_bonds[i,:n_atom] = sample['n_bonds']

    #output = {"atom_feats": atom_feats,
    #          "bond_feats": bond_feats,
    #          "atom_graph": atom_graph,
    #          "bond_graph": bond_graph,
    #          "n_bonds": n_bonds,
    #          "n_atoms": n_atoms,
    #          "bond_labels": bond_labels,
    #          "binary_feats": binary_feats}
    return output

