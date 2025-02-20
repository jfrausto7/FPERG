import torch.nn as nn
import torch.nn.functional as F

class DQN(nn.Module):
    def __init__(self, input_size=6):
        super(DQN, self).__init__()
        # Smaller network architecture - still powerful enough for our task
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, 64)
        self.head = nn.Linear(64, 7)

    def forward(self, x):
        # Simpler forward pass with fewer layers
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.head(x)