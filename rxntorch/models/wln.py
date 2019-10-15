import torch
import torch.nn as nn
import torch.nn.functional as F

class WeisLehmanNet(nn.Module):
    def __init__(self, depth, afeats_size, bfeats_size, hidden_size):
        super(WeisLehmanNet, self).__init__()
        self.fc1 = nn.Linear(afeats_size, hidden_size)
        self.fc2a = nn.Linear(afeats_size, hidden_size)
        self.fc2b = nn.Linear(afeats_size, hidden_size)

    def forward(self, atom_feats, atom_graph, bond_feats, bond_graph, num_nbs, node_mask):
        atom_feats = F.relu(self.fc1(atom_feats))
        layers = []
        for i in range(depth):
            atomnei_feats =


