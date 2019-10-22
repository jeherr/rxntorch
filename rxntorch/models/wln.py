import torch
import torch.nn as nn
import torch.nn.functional as F

class WLNet(nn.Module):
    def __init__(self, depth, afeats_size, bfeats_size, hidden_size):
        super(WLNet, self).__init__()
        self.depth = depth
        self.fc1 = nn.Linear(afeats_size, hidden_size)
        self.graph_conv_nei = nn.Linear(afeats_size + bfeats_size, hidden_size)
        self.graph_conv_atom = nn.Linear(hidden_size * 2, hidden_size)
        self.fc2atom_nei = nn.Linear(hidden_size, hidden_size)
        self.fc2bond_nei = nn.Linear(bfeats_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)

    def forward(self, atom_feats, atom_graph, bond_feats, bond_graph, num_nbs, n_atoms):
        atom_feats = F.relu(self.fc1(atom_feats))
        bondnei_feats = bond_feats[:, bond_graph, :]
        # Creates tensors to mask padded neighbors and atoms
        mask_neis = torch.unsqueeze(num_nbs.unsqueeze(-1) > torch.arange(0, 10, dtype=torch.int).view(1, 1, -1), -1)
        max_n_atoms = n_atoms.max()
        mask_atoms = n_atoms.unsqueeze(-1) > torch.arange(0, max_n_atoms, dtype=torch.int).view(1, -1, 1)
        for i in range(self.depth):
            atomnei_feats = atom_feats[:,atom_graph,:]
            if (i + 1) == self.depth:
                atomnei_feats = self.fc2atom_nei(atomnei_feats)
                bondnei_feats = self.fc2bond_nei(bondnei_feats)
                nei_feats = atomnei_feats * bondnei_feats
                nei_feats = torch.where(mask_neis, nei_feats, torch.zeros_like(nei_feats)).sum(-2)
                self_feats = self.fc2(atom_feats)
                local_feats = self_feats * nei_feats
                local_feats = torch.where(mask_atoms, local_feats, torch.zeros_like(local_feats))
            else:
                nei_feats = F.relu(self.graph_conv_nei(torch.concat([atomnei_feats, bondnei_feats], dim=-1)))
                nei_feats = torch.where(mask_neis, nei_feats, torch.zeros_like(nei_feats)).sum(-2)
                update_feats = torch.concat([atom_feats, nei_feats], dim=-1)
                atom_feats = F.relu(self.graph_conv_atom(update_feats))
        return local_feats



