import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from env import Environ
from maddpg import MaDDPG

agent_state_dim = 5
agent_action_dim = 1
max_edge= 1

num_agents = 3
maddpg = MaDDPG(num_agents,agent_state_dim, agent_action_dim)

Env = Environ(num_agents, render=False, savefig=False)

max_episode = 1000000

max_epoch = 1000

for episode in range(max_episode):
    print('episode',episode)
    agents_state, prey_state = Env.reset()

    for epoch in range(max_epoch):

        agents_action = maddpg.noise_action(agents_state)
        prey_action = np.random.uniform(-1, 1)

        agents_next_state, prey_next_state, agents_reward, prey_reward, done = Env.step(agents_action, prey_action)

        maddpg.perceive(agents_state, agents_action, agents_reward, agents_next_state,done)
        agents_state = agents_next_state
        if done:
            print('Done!!!!!!!!!!!! at epoch{} , reward:{}'.format(epoch,agents_reward))
            # add summary for each episode
            maddpg.summary(episode)
            break
    if epoch ==max_epoch-1:
        print('Time up >>>>>>>>>>>>>>')

