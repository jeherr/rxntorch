import torch
import torch.nn as nn
import torch.nn.functional as F

class Attention(nn.Module):
    def __init__(self, hidden_size, binary_size):
        super(Attention, self).__init__()
        self.fcapair = nn.Linear(hidden_size, hidden_size)
        self.fcbinary = nn.Linear(binary_size, hidden_size)
        self.fcattention = nn.Linear(hidden_size, 1)

    def forward(self, local_feats, binary_feats):
        local_pair = local_feats.unsqueeze(1) + local_feats.unsqueeze(2)
        attention_features = F.relu(self.fcapair(local_pair) + self.fcbinary(binary_feats))
        attention_score = F.sigmoid(self.fcattention(attention_features))
        global_features = torch.sum(local_feats.unsqueeze(1) * attention_score, dim=2)
        global_pair = global_features.unsqueeze(1) + global_features.unsqueeze(2)
        return local_pair, global_pair
