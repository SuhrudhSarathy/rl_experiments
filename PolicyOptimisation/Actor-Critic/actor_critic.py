import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
import numpy as np
import matplotlib.pyplot as plt
import gym

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
dtype = torch.float32

# Actor Critic model
class Actor(nn.Module):
	"""Actor spits out the probability value for each of the action"""
	def __init__(self, input_dims, output_dims):
		super(Actor, self).__init__()
		self.input_dims = input_dims
		self.output_dims = output_dims

		self.model = nn.Sequential(
			nn.Linear(self.input_dims, 128),
			nn.ReLU(),
			nn.Linear(128, 128),
			nn.ReLU(),
			nn.Linear(128, self.output_dims),
			nn.Softmax(dim=0)
			)
		self.to(dtype).to(device)

	def forward(self, X):
		return self.model(X)

class Critic(nn.Module):
	"""Critic spits out the Q values of the (s, a) pairs"""
	def __init__(self, input_dims):
		super(Critic, self).__init__()
		self.input_dims = input_dims
		self.model = nn.Sequential(
			nn.Linear(self.input_dims, 128),
			nn.ReLU(),
			nn.Linear(128, 1)
			)
		self.to(dtype).to(device)

	def forward(self, X):
		return self.model(X)

class Agent:
	def __init__(self, env):
		self.env = env
		self.input_dims = self.env.observation_space.shape[0]
		self.output_dims = self.env.action_space.n

		# Actor and Critic
		self.actor = Actor(self.input_dims, self.output_dims)
		self.critic = Critic(self.input_dims)

		# Hyperparameters
		self.lr_actor = 1e-4
		self.lr_critic = 1e-3

		# optimisers
		self.actor_optimiser = optim.Adam(self.actor.parameters(), lr=self.lr_actor)
		self.critic_optimiser = optim.Adam(self.critic.parameters(), lr=self.lr_critic)

		# Transition [state, log_prob, next_state, reward, done]
		self.transition = []

		# Gamma
		self.gamma = 0.99

	def select_action(self, state):
		state = torch.from_numpy(state).to(dtype).to(device)
		dist = Categorical(self.actor(state))
		action = dist.sample()
		self.transition = [state, dist.log_prob(action)]

		return action

	def step(self, action, t):
		action = action.cpu().detach().numpy()
		state, reward, done, _ = self.env.step(action)
		if not done:
			self.transition.extend([state, reward, done])
			return state, reward, done
		else:
			self.transition.extend([state, reward, done])
			return state, reward, done

	def update_networks(self):
		"""Calculate the gradients"""
		state, log_prob, next_state, reward, done = self.transition

		mask = 1-done
		q_val_prime = self.critic(torch.from_numpy(next_state).to(dtype).to(device))
		q_val = self.critic(state)

		# Update Critic Loss
		# Critic Grad = r + gamma * q(s`) * (1-done) - q(s)
		target_value = reward + self.gamma * q_val_prime
		critic_loss = F.smooth_l1_loss(target_value, q_val)

		self.critic_optimiser.zero_grad()
		critic_loss.backward()
		self.critic_optimiser.step()

		# Update actor's loss
		# Actor grad = -log(PI(a|s))*Q(s) == -log_prob * q_val
		# Use of an advantage function leads to better policy updates
		
		advantage_func = (target_value - q_val).detach() # Detach is to avoid calculation of grads again
		actor_loss = -log_prob * advantage_func 

		self.actor_optimiser.zero_grad()		
		actor_loss.backward()
		self.actor_optimiser.step()

		return actor_loss, critic_loss

	def train(self, EPISODES=100):
		self.is_train = True
		# set both the models to train
		self.actor.train()
		self.critic.train()

		self.actor_losses = []
		self.critic_losses = []
		self.cumm_rewards = []
		self.moving_avs = [[], []]
		self.stds = []

		for episode in range(EPISODES):
			state = self.env.reset()
			done = False
			cumm_reward = 0
			t = 0
			while not done:
				action = self.select_action(state)
				next_state, reward, done = self.step(action, t)

				actor_loss, critic_loss = self.update_networks()

				self.actor_losses.append(actor_loss.item())
				self.critic_losses.append(critic_loss.item())

				cumm_reward += reward
				state = next_state

			self.cumm_rewards.append(cumm_reward)

			if episode % 15 == 0 and episode != 0:
				arr = np.array(self.cumm_rewards[episode-10: episode])
				self.moving_avs[0].append(episode)
				self.moving_avs[1].append(arr.mean())
				self.stds.append(arr.std())
				print(f"Episode: {episode}, Moving Av: {arr.mean()}, Moving Std: {arr.std()}")

	def plot_results(self):
		if self.is_train:
			# Plot the training data
			plt.plot(self.cumm_rewards, alpha=0.5)
			plt.plot(self.moving_avs[0], self.moving_avs[1], color="orange")
			plt.errorbar(self.moving_avs[0], self.moving_avs[1], self.stds, linestyle="None", marker="^", color="green", alpha=0.75)
			plt.savefig("Acrobot_ActorCritic.png")

if __name__ == "__main__":
	env = gym.make("Acrobot-v1")
	agent = Agent(env)
	agent.train(250)
	agent.plot_results()

		





