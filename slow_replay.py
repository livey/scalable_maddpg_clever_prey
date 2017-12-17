import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from env import Environ
from maddpg import MaDDPG
from preyddpg import PreyDDPG
import tensorflow as tf
from network_manual import NetworkManual
import time


agent_state_dim = 5
agent_action_dim = 1
num_agents = 3
Save_Step = 4000
max_episode = 100
max_epoch = 1000

sess = tf.InteractiveSession()

maddpg = MaDDPG(sess,num_agents,agent_state_dim)
preyddpg = PreyDDPG(sess)
networks = NetworkManual(sess)

networks.load_network()
Env = Environ(num_agents, render=True, savefig=False)


for episode in range(max_episode):
    print('episode',episode)
    agents_state, prey_state = Env.reset()

    for epoch in range(max_epoch):

        agents_action = maddpg.action(agents_state)
        #prey_action = np.random.uniform(-1, 1)
        prey_action = preyddpg.action(prey_state)

        agents_next_state, prey_next_state, agents_reward, prey_reward, done = Env.step(agents_action, prey_action)
        time.sleep(0.2)

        agents_state = agents_next_state
        prey_state   = prey_next_state
        if done:
            print('Done!!!!!!!!!!!! at epoch{} , reward:{}'.format(epoch,agents_reward))
            print('prey reward: {}'.format(prey_reward))
            # add summary for each episode

            break
    if epoch ==max_epoch-1:
        print('Time up >>>>>>>>>>>>>>')

sess.close()
#Env.gen_gif()
