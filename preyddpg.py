import tensorflow as tf
import numpy as np
from prey_actor_network import ActorNetwork
from prey_critic_network import CriticNetwork
from ou_noise import OUNoise
from replay_buffer import ReplayBuffer

# dimension only for test 
STATE_DIM = 3
ACTION_DIM =2

# discout
GAMMA = 0.99
BATCH_SIZE = 64

REPLAY_BUFFER_SIZE = 1000000
REPLAY_START_SIZE = 10000
SAVE_STEPS = 10000
SUMMARY_BATCH_SIZE = 512

class PreyDDPG:
    def __init__(self,sess, num_agents=1,state_dim=8,action_dim=2):
        # track training times
        self.time_step = 0
        # use set session use GPU
        #self.sess = tf.InteractiveSession()
        #self.sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
        self.sess = sess
        self.num_agents = num_agents
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.agents = ActorNetwork(sess,state_dim, action_dim, agent_name = 'prey')
        # make sure create Criticnetwork later, summarise mean Q value inside
        self.critic = CriticNetwork(self.sess,state_dim,action_dim)
        self.exploration_noise = OUNoise((self.num_agents,action_dim), sigma=0.2)
        self.replay_buffer = ReplayBuffer(REPLAY_BUFFER_SIZE)
        # for store checkpoint
        #self.saver = tf.train.Saver()

    def train(self):
        minibatch = self.replay_buffer.get_batch(BATCH_SIZE)
        state_batch = np.zeros((BATCH_SIZE,self.state_dim))
        action_batch = np.zeros((BATCH_SIZE,self.action_dim))
        reward_batch = np.zeros((BATCH_SIZE))
        next_state_batch = np.zeros((BATCH_SIZE,self.state_dim))
        done_batch = np.zeros((BATCH_SIZE))
        for ii in range(BATCH_SIZE):
            state_batch[ii,:] = minibatch[ii][0]
            action_batch[ii,:] = minibatch[ii][1]
            reward_batch[ii]  = minibatch[ii][2]
            next_state_batch[ii,:] = minibatch[ii][3]
            done_batch[ii] = minibatch[ii][4]

        # calculate Gt batch
        next_action_batch = self.target_actions(next_state_batch)
        q_value_batch = self.critic.target_q(next_state_batch,next_action_batch)
        #print(q_value_batch)

        gt = np.zeros((BATCH_SIZE))
        for ii in range(BATCH_SIZE):
            if done_batch[ii]:
                gt[ii] = reward_batch[ii]
            else:
                gt[ii] = reward_batch[ii] + GAMMA*q_value_batch[ii,:]
        #update critic by minimizing the loss

        self.critic.train(gt,state_batch,action_batch)

        # update policy using the sampling gradients
        actions_for_grad = self.actions(state_batch)
        q_gradients_batch = self.critic.gradients(state_batch,actions_for_grad)
        self.train_agents(q_gradients_batch, state_batch)

        # update critic target network
        self.critic.update_target()

        # update actor target
        self.update_agents_target()

    def summary(self, record_num):
        if self.replay_buffer.count() > SUMMARY_BATCH_SIZE:
            mini_batch = self.replay_buffer.popn(SUMMARY_BATCH_SIZE)
            state_batch = np.zeros((SUMMARY_BATCH_SIZE, self.num_agents, self.state_dim))
            for ii in range(SUMMARY_BATCH_SIZE):
                state_batch[ii,:,:] = mini_batch[ii][0]

            actions_for_summary = self.actions(state_batch)
            self.critic.write_summaries(state_batch, actions_for_summary, record_num)



    def update_agents_target(self):
        self.agents.update_target()

    def train_agents(self,gradients_batch,state_batch):
        # gradients_batch = [batchsize* agents* action_dim]
        # state_batch = [batchsize* agents * state_dim ]
        self.agents.train(gradients_batch,state_batch)



    def create_multi_agents(self,sess, num_agents, state_dim, action_dim):
        agents =[]
        nets = None
        for ii in range(num_agents):
            agent_name = 'prey'+str(ii)
            agents.append(ActorNetwork(sess,state_dim, action_dim, agent_name,nets))
            nets = agents[-1].nets
        return agents

    def add_agents(self,add_num):
        for ii in range(add_num):
            #self.num_agents+=1

            agent_name = 'agent'+ str(self.num_agents)
            self.agents.append(ActorNetwork(self.sess,self.state_dim,self.action_dim,
                                            agent_name, self.agents[-1].nets))
            # the agents' name is from 0-num_agents-1 
            self.num_agents+=1

        # if add a new agent then reset the noise and replay buffer
        self.exploration_noise = OUNoise((self.num_agents, self.action_dim))
        #self.replay_buffer = ReplayBuffer(REPLAY_BUFFER_SIZE)
        self.replay_buffer.erase()
        # re-create a saver 
        # the new saver will contains all the savable variables.
        # otherwise only contains the initially created agents
        self.saver = tf.train.Saver()
        # reset the time step
        # self.time_step = 0


    def action(self,state): # here is action, for one state on agent, not batch_sized actions
        # state = [num_agents * state_dim]
        # actions = [num_agents *  action_dim]
        # action = np.zeros((self.num_agents,self.action_dim))
        # for ii in range(self.num_agents):
        #     action[ii,:] = self.agents[ii].action(state[ii,:])
        return self.agents.action(state)

    def actions(self,state_batch):
        #state = batch_size*numOfagents*state_dim
        #actions = batch_size*numOfagents*action_dim
        # batch_size = state_batch.shape[0]
        # actions = np.zeros((batch_size, self.num_agents, self.action_dim))
        # for ii in range(self.num_agents):
        #     actions[:,ii,:] = self.agents[ii].actions(state_batch[:,ii,:])
        return self.agents.actions(state_batch)


    def target_actions(self,state_batch):
        # the state size  is batch_size* num_agents * state_dimension
        # actions = np.zeros((state_batch.shape[0],self.num_agents,self.action_dim))
        # for ii in range(self.num_agents):
        #     actions[:,ii,:] = self.agents[ii].actions(state_batch[:,ii,:])
        return  self.agents.target_actions(state_batch)

    def noise_action(self,state):
        action = self.action(state)
        # clip the action, action \in [-1,+1]

        action = action + self.exploration_noise.noise()

        # circular angle help prevent overfitting to the -1 or +1.
        action[0,0] = (action[0,0]+1)%2 -1
        action[0,1] = np.clip(action[0,1],-1,1)
        return action

    def close_session(self):
        self.sess.close()

    def perceive(self,state,action,reward,next_state,done):
        # store {st,at,Rt+1,st+1}
        self.replay_buffer.add(state,action,reward,next_state,done)

        if self.replay_buffer.count() > REPLAY_START_SIZE:
            self.time_step += 1
            self.train()
            # if self.time_step % 10000 == 0:
            # self.actor_network.save_network(self.time_step)
            # self.critic_network.save_network(self.time_step)

            # Re-initialize the random process when an episode ends
        if done:
            self.exploration_noise.reset()


