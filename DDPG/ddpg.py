import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from collections import namedtuple
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import gym
from gym import wrappers
import logging

logging.basicConfig(filename= "__main__.log", filemode="w", format="%(asctime)s - %(message)s", level=logging.INFO)

dtype = torch.float
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

Transition = namedtuple("Transition", ["state", "action", "reward", "done", "next_state"])

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


# Critic Network
class Critic(nn.Module):
    def __init__(self, s_dim, a_dim):
        super(Critic, self).__init__()
        # Crtitc is the network based on DQN
        self.s_dim = s_dim
        self.a_dim = a_dim
        self.model = nn.Sequential(
            nn.Linear(s_dim + a_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
        self.to(dtype).to(device)

    def forward(self, s, a):
        s = torch.tensor(s).to(dtype).to(device)
        x = torch.cat([s.view(-1, self.s_dim), a.view(-1, self.a_dim)], dim=1).to(dtype).to(device)
        return self.model(x)

class Actor(nn.Module):
    def __init__(self, s_dim, a_dim):
        # Actor network works on policy gradient
        super(Actor, self).__init__()
        self.s_dim = s_dim
        self.a_dim = a_dim
        self.model = nn.Sequential(
            nn.Linear(s_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, a_dim),
            nn.Tanh() # To squish the output between [-1, 1]
        )
        self.to(dtype).to(device)

    def forward(self, s):
        s = torch.tensor(s).to(dtype).to(device)
        return self.model(s)

class Noise:
    # Using zero mean custom standard deviation gaussian normal distribution
    def __init__(self, std_dev):
        self.std_dev = std_dev
    
    def __call__(self):
        return np.random.randn()*self.std_dev

class Agent:
    def __init__(self, env: gym.Env):
        self.env = env
        self.a_dim = self.env.action_space.shape[0]
        self.s_dim = self.env.observation_space.shape[0]
        
        # Replay buffer
        self.buffer = ReplayBuffer(1000, 128)

        # Noise
        self.noise = Noise(0.2)
        # networks
        self.Q = Critic(self.s_dim, self.a_dim)
        self.Q_target = Critic(self.s_dim, self.a_dim)
        self.Q_target.load_state_dict(self.Q.state_dict())
        for param in self.Q_target.parameters():
            param.requires_grad = False

        self.policy = Actor(self.s_dim, self.a_dim)
        self.policy_target = Actor(self.s_dim, self.a_dim)
        self.policy_target.load_state_dict(self.policy.state_dict())
        for param in self.policy_target.parameters():
            param.requires_grad = False
        
        # Optimisers
        self.Q_optimiser = optim.Adam(self.Q.parameters(), lr=1e-3)
        self.policy_optimiser = optim.Adam(self.policy.parameters(), lr=1e-3)

        # constants
        self.gamma = 0.99
        self.polyak = 0.995

        # other parameters
        self.exploration = 1000
        self.step_count = 0

        # containers
        self.Q_losses = []
        self.policy_losses = []
        self.rewards = []

        self.updatable = False

    def select_action(self, state):
        self.step_count += 1
        if self.step_count - 1 < self.exploration:
            # Select a random action
            return self.env.action_space.sample()
        else:
            # choose the deterministic action from the policy
            with torch.no_grad():
                action = self.policy(state).detach().cpu().numpy() + self.noise() 
            return np.clip(action, -1, 1)

    def update_networks(self):
        # Step 1: Sample a batch from the replay buffer
        sample = self.buffer.sample()

        # Step2: Calculate the target value and update the parameters of Critic
        with torch.no_grad():
            # The target networks haver required grad set to false
            # but still doing this to double check
            policy_values = self.policy_target(sample.next_state)
            target_q = sample.reward.view(-1, 1) + self.gamma * (1 - sample.done).view(-1, 1) * self.Q_target(sample.next_state, policy_values)
            target_q = target_q.view(-1, 1)
        # compute the loss
        self.Q_optimiser.zero_grad()
        q_loss = F.smooth_l1_loss(self.Q(sample.state, sample.action), target_q)
        q_loss.backward()
        self.Q_losses.append(q_loss.item())
        self.Q_optimiser.step()

        # Step3: Calculate the loss for policy network
        self.policy_optimiser.zero_grad()
        policy_loss = -torch.mean(self.Q(sample.state, self.policy(sample.state)))
        policy_loss.backward()
        self.policy_losses.append(policy_loss.item())
        self.policy_optimiser.step()

    def soft_target_update(self):
        # Update the target networks using polyak parameter
        for p, p_ in zip(self.Q_target.parameters(), self.Q.parameters()):
            p.data = self.polyak*p.data + (1-self.polyak)*p_.data

        for p, p_ in zip(self.policy_target.parameters(), self.policy.parameters()):
            p.data = self.polyak*p.data + (1-self.polyak)*p_.data

    def train(self, EPOCHS, MAX_ITER):
        self.Q.train()
        self.policy.train()

        for epoch in tqdm(range(EPOCHS)):
            cumm_reward = 0
            observation = self.env.reset()
            for i in range(MAX_ITER):
                action = self.select_action(observation)
                observation_, reward, done, _ = self.env.step(action)
                self.buffer.add(Transition(observation, action, reward, done, observation_))
                observation = observation_
                cumm_reward += reward

                if done:
                    break

            self.rewards.append(cumm_reward)

            if VERBOSE:
                logging.info(f"Epochs: {epoch}, Reward: {cumm_reward}")
            if SAVE:
                if epoch % 5 == 1:
                    self.save()
            if self.updatable:
                self.update_networks()
                self.soft_target_update()

            # Decide whether to update or not
            if epoch % 2 == 0 and self.step_count > 1000:
                self.updatable = True

    def eval(self, EPOCHS):
        self.Q.eval()
        self.policy.eval()

        eval_rewards = []
        self.env = wrappers.Monitor(self.env, "./extras/", force=True)
        for epoch in range(EPOCHS):
            cumm_reward = 0
            observation = self.env.reset()
            while True:
                if RENDER:
                    self.env.render()
                action = self.select_action(observation)
                observation_, reward, done, _ = self.env.step(action)
                observation = observation_
                if done:
                    break

                cumm_reward += reward
            eval_rewards.append(cumm_reward)
        
        return eval_rewards

    def save(self):
        torch.save(self.Q.state_dict(), "./models/QNetwork.pt")
        torch.save(self.policy, "./models/PolicyNetwork.pt")
    
    def plot(self):
        fig , (ax1, ax2, ax3) = plt.subplots(3, 1)
        fig.suptitle("DDPG Statistics")

        ax1.plot(self.Q_losses)
        ax1.set_xlabel("Q Network Loss Update")
        ax1.set_ylabel("Q Network Loss")

        ax2.plot(self.policy_losses)
        ax2.set_xlabel("Policy Network Loss Update")
        ax2.set_ylabel("Policy Network Loss")

        ax3.plot(self.rewards)
        ax3.set_xlabel("Episodes")
        ax3.set_ylabel("Cumulative Rewards")

        plt.savefig(f"./{ENV}.png")

if __name__ == "__main__":
    ENV = "LunarLanderContinuous-v2"
    VERBOSE = True
    SAVE = True
    RENDER = True
    EPOCHS = 2500
    MAX_ITER = 10000

    env = gym.make(ENV)
    agent = Agent(env)
    agent.train(EPOCHS, MAX_ITER)
    agent.eval(10)
    agent.plot()        
            



