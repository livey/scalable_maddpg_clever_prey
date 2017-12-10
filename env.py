import numpy as np

# the world dimension is fixed, from [-1,1]
agent_v = .1
prey_v = .1
agent_r = .1
prey_r = .1

class Environ:
    def __init__(self,num_agents):
        self.num_agents = num_agents

    def reset(self):
        prey_pos = np.random.uniform(-1,1,[1,2])
        agents_pos = np.zeros((self.num_agents,2))
        for ii in range(self.num_agents):
            pos = np.random.uniform(-1,1,[1,2])
            while(np.linalg.norm(pos-prey_pos, ord='fro')<agent_r+prey_r):
                pos = np.random.uniform(-1,1,[1,2])
            agents_pos[ii,:] = pos

        self.agents_pos = agents_pos
        self.prey_pos = prey_pos
        return agents_pos, prey_pos

    def step(self,agents_action, prey_action):
        agents_action *=np.pi
        prey_action   *=np.pi
        agents_next_pos = self.agents_pos + np.hstack((np.cos(agents_action),np.sin(agents_action)))*agent_v
        prey_next_pos = self.prey_pos + np.hstack((np.cos(prey_action),np.sin(prey_action)))*prey_v
        self.agents_pos = agents_next_pos
        self.prey_pos   = prey_next_pos
        agents_rewards,\
        prey_reward,\
        done = self.rewards(agents_next_pos, prey_next_pos)
        return agents_next_pos, prey_next_pos, agents_rewards, prey_reward,done

    def rewards(self,agents_pos, prey_pos):
        done = False
        # first consider if out of boundary
        agents_rewards = -0.1* np.ones(self.num_agents)
        prey_reward = 0.1
        for ii in range(self.num_agents):
            if np.abs(agents_pos[ii,0])>1 or np.abs(agents_pos[ii,1])>1:
                #agents_rewards[ii,0] = -10
                agents_rewards[:] = -10
                done = True
                prey_reward = 10

        if np.abs(prey_pos[0,0])>1 or np.abs(prey_pos[0,1])>1:
            prey_reward = 0.1
            done = True
        # whether collide
        for ii in range(self.num_agents):
            if np.linalg.norm(agents_pos[ii,:]-prey_pos[0,:],ord=2)< prey_r+ agent_r:
                prey_reward=0.1
                agents_rewards[:]=-.1
                done = True

        return agents_rewards, prey_reward ,done








