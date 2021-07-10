import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.F as F

from tqdm import tqdm

import gym

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
dtype = torch.float

class ReplayBuffer:
    def __init__(self, size, batch_size = 30, use_torch = True):
        self.size = size
        self.batch_size = batch_size
        self.buffer = np.ndarray((self.size, 1), dtype=Transition)
        self.use_torch = use_torch
        self.i = 0
        
    def add(self, transition):
        self.i = self.i % self.size
        self.buffer[self.i][0] = transition
        self.i += 1
    
    def sample_(self, batch_size = None):
        if batch_size is None:
            batch_size = self.batch_size
        idx = np.random.choice(range(self.size), self.batch_size)
        sample = []
        for idx_ in idx:
            sample.append(self.buffer[idx_][0])
        return sample
    
    def sample(self, batch_size = None):
        if self.use_torch:
            # Return the transitions split up as arrays
            sample = self.sample_(batch_size=batch_size)
            sample = Transition(*[torch.cat([torch.tensor(i).to(dtype).to(device)]) for i in [*zip(*sample)]])
            return sample
        else:
            return self.sample(batch_size)

# Agents
class QNetwork(nn.Module):
    def __init__(self, input_dims, output_dims):
        super(QNetwork, self).__init__()
        
        self.model = nn.Sequential(
            nn.Linear(input_dims, 128),
            nn.ReLU()
            nn.Linear(128, 64),
            nn.ReLU()
            nn.Relu(64, 32),
            nn.ReLU()
            nn.Linear(32, output_dims)
        )

        self.model.to(dtype).to(device)
    def forward(self, X):
        return self.model(torch.tensor(X).to(dtype).to(device)).cpu().detach()

class Agent:
    def __init__(self, env: gym.Env):

        self.env = env
        self.observation_space = self.env.observation_space.size[0]
        self.action_space = self.env.action_space.size[0]

        self.Q = QNetwork(self.observation_space, self.action_space)
        self.target_Q = QNetwork(self.observation_space, self.action_space)
        self.target_Q.load_state_dict(self.Q.state_dict())
        for param in self.target_Q.parameters():
            param.requires_grad = False

        self.Q_optimiser = optim.Adam(self.Q.parameters(), lr=3e-3)

        self.buffer = ReplayBuffer()

        # Use epsilon greedy strategy
        self.eps_start = 1
        self.eps_end = 0.05
        self.eps_decay = 2000
        self.eps = self.eps_start

        self.polyak = 0.99

    def update_params(self):
        # Sample 