import torch
import torch.nn as nn
import torch.nn.functional as F

class ReactivityScoring(nn.Module):
    def __init__(self, hidden_size, binary_size):
        super(ReactivityScoring, self).__init__()
        self.fclocal = nn.Linear(hidden_size, hidden_size)
        self.fcglobal = nn.Linear(hidden_size, hidden_size)
        self.fcbinary = nn.Linear(binary_size, hidden_size)
        self.fcscore = nn.Linear(hidden_size, 5)

    def forward(self, local_pair, global_pair, binary):
        pair_feats = F.relu(self.fclocal(local_pair) + self.fcglobal(global_pair) + self.fcbinary(binary))
        score = self.fcscore(pair_feats)
        return score



