import torch
import torch.nn as nn
import torch.nn.functional as F

class DQNModel(nn.Module):
    def __init__(self, state_dim, action_dim=1):
        super(DQNModel, self).__init__()
        # state_dim is typically max_features (text) + queue_size + depth
        self.fc1 = nn.Linear(state_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        # We predict a scalar Q-value for a specific state-action pair
        # In this crawling context, an action is choosing a URL, which has its own text features.
        # But for simplicity, we predict the Q-value of moving to a specific next state (representing the URL)
        # So action_dim=1 (evaluating a candidate URL)
        self.fc3 = nn.Linear(64, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)
