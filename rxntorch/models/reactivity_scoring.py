import torch
import torch.nn as nn
import torch.nn.functional as F
from .layers import Linear


class ReactivityScoring(nn.Module):
    def __init__(self, hidden_size, binary_size):
        super(ReactivityScoring, self).__init__()
        self.fcall = Linear(hidden_size * 2 + binary_size, hidden_size)
        self.fcscore = Linear(hidden_size, 5)

    def forward(self, local_pair, global_pair, binary_feats, sparse_idx):
        binary_feats = binary_feats[sparse_idx[:,0],sparse_idx[:,1],sparse_idx[:,2]]
        all_feats = torch.cat([local_pair, global_pair, binary_feats], dim=-1)
        pair_feats = F.relu(self.fcall(all_feats))
        score = self.fcscore(pair_feats)
        return score



