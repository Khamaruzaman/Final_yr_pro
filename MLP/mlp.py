import torch
import torch.nn as nn
import torch.optim as optim

class FusionMLP(nn.Module):
    def __init__(self):
        super(FusionMLP, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(2, 4),  # 2 inputs (S & L predictions), 4 hidden neurons
            nn.ReLU(),
            nn.Linear(4, 1),  # Output = 1 (real/fake probability)
            nn.Sigmoid()  # Output range [0,1]
        )

    def forward(self, x):
        return self.fc(x)
