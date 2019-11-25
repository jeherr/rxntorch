import torch.nn as nn

class Linear(nn.Linear):
    def reset_parameters(self):
        nn.init.xavier_normal_(self.weight)
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            nn.init.constant_(self.bias, 0.0)
