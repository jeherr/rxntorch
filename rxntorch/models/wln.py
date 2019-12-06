import torch
import torch.nn as nn
import torch.nn.functional as F
from .layers import Linear

class WLNet(nn.Module):
    def __init__(self, depth, afeats_size, bfeats_size, hidden_size):
        super(WLNet, self).__init__()
        self.depth = depth
        self.fc1 = Linear(afeats_size, hidden_size, bias=False)
        self.graph_conv_nei = Linear(hidden_size + bfeats_size, hidden_size)
        self.graph_conv_atom = Linear(hidden_size * 2, hidden_size)
        self.fc2atom_nei = Linear(hidden_size, hidden_size, bias=False)
        self.fc2bond_nei = Linear(bfeats_size, hidden_size, bias=False)
        self.fc2 = Linear(hidden_size, hidden_size, bias=False)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def forward(self, atom_feats, bond_feats, atom_graph, bond_graph, num_nbs, n_atoms):
        atom_feats = F.relu(self.fc1(atom_feats))
        bondnei_feats = bond_feats[bond_graph[:,:,:,0],bond_graph[:,:,:,1],:]
        # Creates tensors to mask padded neighbors and atoms
        mask_neis = torch.unsqueeze(num_nbs.unsqueeze(-1) > torch.arange(0, 10, dtype=torch.int32, device=self.device).view(1,1,-1), -1)
        max_n_atoms = n_atoms.max()
        mask_atoms = torch.unsqueeze(n_atoms.unsqueeze(-1) > torch.arange(0, max_n_atoms, dtype=torch.int32, device=self.device).view(1,-1), -1)
        for i in range(self.depth):
            atomnei_feats = atom_feats[atom_graph[:,:,:,0],atom_graph[:,:,:,1],:]
            if (i + 1) == self.depth:
                atomnei_feats = self.fc2atom_nei(atomnei_feats)
                bondnei_feats = self.fc2bond_nei(bondnei_feats)
                nei_feats = atomnei_feats * bondnei_feats
                nei_feats = torch.where(mask_neis, nei_feats, torch.zeros_like(nei_feats)).sum(-2)
                self_feats = self.fc2(atom_feats)
                local_feats = self_feats * nei_feats
                local_feats = torch.where(mask_atoms, local_feats, torch.zeros_like(local_feats))
            else:
                nei_feats = F.relu(self.graph_conv_nei(torch.cat([atomnei_feats, bondnei_feats], dim=-1)))
                nei_feats = torch.where(mask_neis, nei_feats, torch.zeros_like(nei_feats)).sum(-2)
                update_feats = torch.cat([atom_feats, nei_feats], dim=-1)
                atom_feats = F.relu(self.graph_conv_atom(update_feats))
        return local_feats



