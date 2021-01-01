import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from collections import namedtuple
import random
import gym
from tqdm import tqdm

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
dtype = torch.float
env_name = "CartPole-v0"

# Define a transition
Transition = namedtuple("Transition", ["state", "action", "reward", "done", "next_state"])

# Class for Experience replay
class ExperienceBuffer:
    def __init__(self, memory_len, env):
        self.capacity = memory_len
        self.memory = []
        self.position = 0
        self.env = env

    def push(self, *args):
        # used to push the transition into memory
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        sample = random.sample(self.memory, batch_size)
        state = torch.cat([torch.tensor(trans.state) for trans in sample]).to(device).to(dtype)
        action = torch.cat([torch.LongTensor([trans.action]) for trans in sample]).to(device)
        reward = torch.cat([torch.tensor([trans.reward]).to(dtype) for trans in sample]).to(device).to(dtype)
        done = torch.cat([torch.IntTensor([trans.done]) for trans in sample]).to(device)
        next_state = torch.cat([torch.tensor(trans.next_state) for trans in sample]).to(device).to(dtype)

        return state.view(-1, self.env.observation_space.shape[0]), action.view(-1, 1), reward.view(-1, 1), done.view(-1, 1), next_state.view(-1, self.env.observation_space.shape[0])


    def __len__(self):
        return len(self.memory)


class Network(nn.Module):
    def __init__(self, input_dims, output_dims, lr=1e-2):
        super(Network, self).__init__()
        self.model = self.__create__(input_dims, output_dims).to(device).to(dtype)
        self.optimiser = Adam(self.model.parameters(), lr=lr)
    
    def __create__(self, input_dims, output_dims):
        model = nn.Sequential(
            nn.Linear(input_dims, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, output_dims)
        )
        return model

    def forward(self, X):
        return self.model(X)


class DQNAgent:
    
    def __init__(self, env):
        self.env = env
        self.QNetwork = Network(self.env.observation_space.shape[0], self.env.action_space.n)
        self.experience = ExperienceBuffer(1_000, self.env)

        # Hyper Parameters
        self.EPISODES = 500

        self.eps_begin = 1.0
        self.eps_end = 0.01
        self.eps_decay = 5000

        self.gamma = 0.99

        self.rewards = []
        self.losses = []
        self.epsilons = []

        self.min_len = 100
    
    def epsilon(self, t):
        epsilon = self.eps_end + (self.eps_begin - self.eps_end) * np.exp(-1 * t / self.eps_decay)
        self.epsilons.append(epsilon)
        return epsilon
    
    def select_action(self,state,t):
        prob = np.random.uniform(0, 1)
        if prob < self.epsilon(t):
            return self.env.action_space.sample()
        else:
            return torch.argmax(self.QNetwork(torch.tensor(state, dtype=dtype, device=device))).item()
    
    def learn(self):
        t = 0
        for episode in range(self.EPISODES):
            observation = self.env.reset()
            cum_reward = 0
            done = False
            while not done:
                action = self.select_action(observation,t)
                observation_, reward, done, _ = self.env.step(action)
                self.experience.push(observation, action ,reward, done, observation_)
                observation = observation_
                t += 1
                cum_reward += reward
            if episode % 5 == 0 and len(self.experience) > self.min_len:
                self.optimise()
                # Save model
                torch.save(self.QNetwork.state_dict(), f"./{env_name}.pt")
            print(f"Episode: {episode}, Cumulative Reward: {cum_reward}")
            self.rewards.append(cum_reward)
    
    def optimise(self):
        self.QNetwork.optimiser.zero_grad()
        # sample a batch from the memory and optimise
        state, action, reward, done, next_state = self.experience.sample(100)
        q_values = self.QNetwork(state).gather(1, action)
        q_next = self.QNetwork(next_state).max(1)[0].view(-1, 1)
        target = reward + self.gamma * q_next * (~done)
        # Find Huber Loss between target and the q_values
        # Q(s, a) - Reward + gamma * max(Q(s', a))
        loss = F.smooth_l1_loss(q_values, target)
        loss.backward()
        self.QNetwork.optimiser.step()
        self.losses.append(loss)

    def plot_stats(self):
        fig , (ax1, ax2, ax3) = plt.subplots(3, 1)
        fig.suptitle("DQN Statistics")

        ax1.plot(self.losses)
        ax1.set_xlabel("Loss Update")
        ax1.set_ylabel("Loss")

        ax2.plot(self.rewards)
        ax2.set_xlabel("Episodes")
        ax2.set_ylabel("Cumulative Rewards")

        ax3.plot(self.epsilons)
        ax3.set_xlabel("Time steps")
        ax3.set_ylabel("Epsilon")

        plt.savefig(f"./{env_name}.png")
if __name__ == "__main__":
    env = gym.make(env_name)
    agent = DQNAgent(env)
    agent.learn()
    agent.plot_stats()
    


