import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn


class model(nn.Module):
    def __init__(self, state_size, action_size, seed):
        super(model, self).__init__()
        self.seed = torch.manual_seed(seed)
   
        self.fc1 = nn.Linear(state_size, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, action_size)
        
    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        
        return self.fc3(x)
    

