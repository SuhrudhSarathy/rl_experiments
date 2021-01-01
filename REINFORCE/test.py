import gym
import torch
from reinforce import Agent
import time
env_name = "CartPole-v0"
if __name__ == "__main__":
    env = gym.make(env_name)
    # load the learned pytorch model
    test_agent = Agent(env)
    test_agent.policy.load_state_dict(torch.load(f"./{env_name}.pt"))

    for episode in range(5):
        test_agent.eval()

