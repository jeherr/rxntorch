import torch
import torch.nn as nn
import torch.nn.functional as F
from .layers import Linear


class Attention(nn.Module):
    def __init__(self, hidden_size, binary_size):
        super(Attention, self).__init__()
        self.fc_attention1 = Linear(hidden_size + binary_size, hidden_size)
        self.fc_attention = Linear(hidden_size, 1)

    def forward(self, local_feats, binary_feats, sparse_idx):
        local_pair = local_feats.unsqueeze(1) + local_feats.unsqueeze(2)
        all_feats = torch.cat([local_pair, binary_feats],dim=-1)
        attention_features = F.relu(self.fc_attention1(all_feats))
        attention_score = torch.sigmoid(self.fc_attention(attention_features))
        global_feats = torch.sum(local_feats.unsqueeze(1) * attention_score, dim=2)
        global_pair1 = global_feats[sparse_idx[:,0],sparse_idx[:,1]]
        global_pair2 = global_feats[sparse_idx[:,0],sparse_idx[:,2]]
        global_pair = global_pair1 + global_pair2
        local_pair = local_pair[sparse_idx[:,0],sparse_idx[:,1],sparse_idx[:,2]]
        return local_pair, global_pair
