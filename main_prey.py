import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from env import Environ
from maddpg import MaDDPG
from preyddpg import PreyDDPG
import tensorflow as tf
from network_manual import NetworkManual


agent_state_dim = 5
agent_action_dim = 1
prey_action_dim =2
num_agents = 2
Save_Step = 4000
max_episode = 1000000
max_epoch = 1000

sess = tf.InteractiveSession()
#sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
maddpg = MaDDPG(sess,num_agents,agent_state_dim)
preyddpg = PreyDDPG(sess)
networks = NetworkManual(sess)

Env = Environ(num_agents, render=True, savefig=False)


for episode in range(1,max_episode):
    print('episode',episode)
    agents_state, prey_state = Env.reset()

    for epoch in range(max_epoch):

        agents_action = maddpg.noise_action(agents_state)
        #prey_action = np.random.uniform(-1, 1)
        prey_action = preyddpg.noise_action(prey_state)
        #print(preyddpg.action(prey_state))
        #print('state',prey_state)
        print('action',preyddpg.action(prey_state))

        agents_next_state, prey_next_state, agents_reward, prey_reward, done = Env.step(agents_action, prey_action)

        maddpg.perceive(agents_state, agents_action, agents_reward, agents_next_state,done)
        preyddpg.perceive(prey_state, prey_action, prey_reward, prey_next_state, done)
        agents_state = agents_next_state
        prey_state   = prey_next_state
        if done:
            print('Done!!!!!!!!!!!! at epoch{} , reward:{}'.format(epoch,agents_reward))
            print('prey reward: {}'.format(prey_reward))
            # add summary for each episode

            break
    if epoch ==max_epoch-1:
        print('Time up >>>>>>>>>>>>>>')

    if episode%Save_Step ==0:
        networks.save_network(episode)

sess.close()

