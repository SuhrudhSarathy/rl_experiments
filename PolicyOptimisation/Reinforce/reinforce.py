import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.distributions import Categorical as CTDis
import matplotlib.pyplot as plt
import gym
from tqdm import tqdm
import time

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
dtype = torch.float
env_name = "CartPole-v0"

class Network(nn.Module):
    def __init__(self, input_dims, output_dims, lr = 1e-4):
        super(Network, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dims, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 32),
            nn.ReLU(),
            nn.Linear(32, output_dims),
            nn.Softmax(dim=0)
        )
        # Hyperparameters
        self.lr = lr
        # Optimiser
        self.optimiser = Adam(self.model.parameters(), self.lr)
        

    def forward(self, X):
        return self.model(X)

class Agent():
    def __init__(self, env):
        self.env = env
        self.policy = Network(env.observation_space.shape[0], env.action_space.n, lr=3e-4).to(device=device).to(dtype)

        # hyperparameters
        self.gamma = 0.99

        # memory
        self.log_probs = []
        self.rewards = []

        # plots
        self.cum_rewards = []

    def reset_memory(self):
        self.log_probs = []
        self.rewards = []
    
    def run_an_episode(self):
        cum_reward = 0
        done = False
        observation = self.env.reset()
        while not done:
            action_dis = CTDis(self.policy(torch.tensor(observation, dtype=dtype, device=device)))
            action = action_dis.sample()
            observation, reward, done, _ = self.env.step(action.item())
            self.log_probs.append(action_dis.log_prob(action))
            self.rewards.append(reward)
            cum_reward += reward
        self.cum_rewards.append(cum_reward)
        self.learn()
    
    def obtain_Gt(self, rewards):
        GTs = []
        for t in range(len(rewards)):
            GT = 0
            gamma = 1
            for k in range(len(rewards[t:])):
                GT += gamma * rewards[t+k]
                gamma *= self.gamma
            GTs.append(GT)
        GTs = np.asarray(GTs, dtype=np.float64)
        mean = GTs.mean()
        std = GTs.std() if GTs.std() != 0 else 1
        GTs = (GTs-mean)/std 
        return GTs
    
    def learn(self):
        self.policy.optimiser.zero_grad()
        GTs = self.obtain_Gt(self.rewards)
        for gt, log_prob in zip(GTs, self.log_probs):
            loss = -log_prob * gt
            loss.backward()
        self.policy.optimiser.step()
        self.reset_memory()

    def plot_stats(self, save=True):
        plt.plot(self.cum_rewards, color='blue')
        plt.xlabel("Episodes")
        plt.ylabel("Cumulative Rewards")
        if save:
            plt.savefig(f"./{env_name}.png")

    def train(self, episodes):
        for episode in range(episodes):
            self.run_an_episode()
            print(f"Episode : {episode}, Cumulative Reward : {self.cum_rewards[-1]}")
            if episode % 50 == 0:
                torch.save(self.policy.state_dict(),f"./{env_name}.pt")

        print(f"Saved model to {env_name}.pt")
        torch.save(self.policy.state_dict(),f"./{env_name}.pt")
        print("Saving stats")
        self.plot_stats()

    def eval(self, render=True):
        # Load model from that saved in the training
        self.policy.load_state_dict(torch.load(f"./{env_name}.pt"))
        self.policy.eval()

        done = False
        cum_reward = 0
        if render:
            self.env.render()
        observation = self.env.reset()
        while not done:
            if render:
                self.env.render()
            action_dis = CTDis(self.policy(torch.tensor(observation, dtype=dtype, device=device)))
            action = action_dis.sample()
            observation, reward, done, _ = self.env.step(action.item())
            cum_reward += reward
            time.sleep(0.01)
        print(f"Cumulative Reward: {cum_reward}")         

if __name__ == "__main__":
    env = gym.make(env_name)
    agent = Agent(env)
    agent.train(400)
    agent.eval(True)
    
