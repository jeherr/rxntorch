import torch
import torch.nn as nn
import torch.nn.functional as F
from .layers import Linear

class WLNet(nn.Module):
    def __init__(self, depth, afeats_size, bfeats_size, hidden_size):
        super(WLNet, self).__init__()
        self.depth = depth
        self.fc1a = Linear(afeats_size, hidden_size, bias=False)
        self.fc1b = Linear(bfeats_size, hidden_size, bias=False)
        self.graph_conv_nei = Linear(hidden_size * 2, hidden_size)
        self.graph_conv_atom = Linear(hidden_size * 2, hidden_size)
        self.graph_conv_bond = Linear(hidden_size * 2, hidden_size)
        self.fc2a_atom = Linear(hidden_size, hidden_size, bias=False)
        self.fc2a_bond = Linear(hidden_size, hidden_size, bias=False)
        self.fc2b_atom = Linear(hidden_size, hidden_size, bias=False)
        self.fc2a = Linear(hidden_size, hidden_size, bias=False)
        self.fc2b = Linear(hidden_size, hidden_size, bias=False)

    def forward(self, atom_feats, bond_feats, atom_graph, bond_graph, rev_atom_graph, mask_neis, mask_atoms):
        atom_feats = F.relu(self.fc1a(atom_feats))
        bond_feats = F.relu(self.fc1b(bond_feats))
        for i in range(self.depth):
            atom_neighs = torch.stack([atom_feats[i,atom_graph[i],:] for i in range(atom_feats.shape[0])])
            bond_neighs = torch.stack([bond_feats[i,bond_graph[i],:] for i in range(atom_feats.shape[0])])
            bond_neigh_feats = torch.stack([atom_feats[i, rev_atom_graph[i], :] for i in range(bond_feats.shape[0])])
            if (i + 1) == self.depth:
                atomnei_feats = self.fc2a_atom(atom_neighs)
                bondnei_feats = self.fc2a_bond(bond_neighs)
                atom_neigh_feats = atomnei_feats * bondnei_feats
                atom_neigh_feats = torch.where(mask_neis, atom_neigh_feats, torch.zeros_like(atom_neigh_feats)).sum(-2)
                self_feats = self.fc2a(atom_feats)
                atom_local_feats = self_feats * atom_neigh_feats
                atom_local_feats = torch.where(mask_atoms, atom_local_feats, torch.zeros_like(atom_local_feats))

                bond_neigh_feats = self.fc2b_atom(bond_neigh_feats).prod(-2)
                self_bond_feats = self.fc2b(bond_feats)
                bond_local_feats = bond_neigh_feats * self_bond_feats
            else:
                atom_neigh_feats = F.relu(self.graph_conv_nei(torch.cat([atom_neighs, bond_neighs], dim=-1)))
                atom_neigh_feats = torch.where(mask_neis, atom_neigh_feats, torch.zeros_like(atom_neigh_feats)).sum(-2)
                atom_updates = torch.cat([atom_feats, atom_neigh_feats], dim=-1)
                atom_feats = F.relu(self.graph_conv_atom(atom_updates))

                bond_updates = torch.cat([bond_feats, bond_neigh_feats.sum(-2)], dim=-1)
                bond_feats = F.relu(self.graph_conv_bond(bond_updates))
        return atom_local_feats, bond_local_feats
