import torch
import torch.nn as nn
import torch.nn.functional as F

class Attention(nn.Module):
    def __init__(self, hidden_size):
        super(Attention, self).__init__()
        self.fcapair = nn.Linear(hidden_size, hidden_size)

    def forward(self, local_feats):
        atompair_feats = local_feats.unsqueeze(1) + local_feats.unsqueeze(2)
        atompair_feats = self.fcapair(atompair_feats)