import torch
import torch.nn as nn
import torch.nn.functional as F
from model.base_model import BaseModel

class FCNModel(BaseModel, nn.Module):
    def __init__(self, env):
        BaseModel.__init__(self, env)
        nn.Module.__init__(self)
        self.trainable = True
        self.nn_model = True
        self.fc1 = nn.Linear(self.n*3, self.n)
        self.fc_pi = nn.Linear(self.n, self.n)
        self.fc_v = nn.Linear(self.n, 1)
    
    def resize(self, obs):
        if len(obs.shape) == 2:
            return obs.view(self.n*3)
        else:
            return obs.view(obs.shape[0], self.n*3)
    
    def forward(self, obs):
        x = self.resize(obs)
        x = F.relu(self.fc1(x))
        x = torch.argmax(x, dim=-1)
        return x
    
    def pi(self, obs):
        x = self.resize(obs)
        x = F.relu(self.fc1(x))
        x = self.fc_pi(x)
        x = F.softmax(x, -1)
        return x
    
    def v(self, obs):
        x = self.resize(obs)
        x = F.relu(self.fc1(x))
        x = self.fc_v(x)
        return x