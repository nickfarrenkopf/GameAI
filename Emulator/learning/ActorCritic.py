import random
import numpy as np

import sys
sys.path.append('C:\\Users\\Nick\\Desktop\\Ava\\Programs')
from Library.General import DataThings as DT
from Library.General import FileThings as FT
from Library.NeuralNetworks.Actor import _ActorAPI as ACTOR
from Library.NeuralNetworks.Critic import _CriticAPI as CRITIC


### LEARNING METHOD ###

class ActorCriticMethod(object):
    
    def __init__(self, paths, agent, load_networks=True):
        """ """
        self.paths = paths
        self.agent = agent
        self.env = agent.env
        self.name = agent.name
        self.name_t = agent.name + '_target'

        # data
        self.n_actions = len(agent.actions)
        self.memory = []

        # training params
        self.alpha = 1e-3
        self.epsilon = 1.0
        self.epsilon_decay = .995
        self.gamma = .95
        self.batch_size = 32

        # networks
        if load_networks:
            self.load_actor_network()
            self.load_critic_network()
        

    ### API ###

    def get_action(self, S, with_random=True):
        """ """
        self.epsilon *= self.epsilon_decay
        if with_random and np.random.random() < self.epsilon:
                return self.agent.action_space.sample()
        return self.actor_network.get_preds(S)[0] ### what am returning?
    
    def remember(self, S, A, R, S_new, done):
        """ """
        self.memory.append([S, A, R, S_new, done])
   

    ### CREATE/LOAD ###

    def create_actor_network(self, hidden=[16,32]):
        """ """
        ACTOR.create(self.paths,self.name,self.S_size,hidden,self.n_actions)
        ACTOR.create(self.paths,self.name_t,self.S_size,hidden,self.n_actions)
        self.load_actor_network()

    def create_critic_network(self, S_hidden=[16,32], A_hidden=[32],
                              P_hidden=[32]):
        """ """
        CRITIC.create(self.paths,self.name,self.S_size,self.n_actions,S_hidden,
                      A_hidden,P_hidden)
        CRITIC.create(self.paths,self.name_t,self.S_size,self.n_actions,S_hidden,
                      A_hidden,P_hidden)
        self.load_critic_network()

    def load_actor_network(self):
        """ """
        self.actor_network = ACTOR.load(self.name, self.paths.load_json())
        self.actor_network_target = ACTOR.load(self.name_t,self.paths.load_json())

    def load_critic_network(self):
        """ """
        self.critic_network = CRITIC.load(self.name, self.paths.load_json())
        self.critic_network_target = CRITIC.load(self.name_t, self.paths.load_json())


    ### TRAIN ###

    def train(self):
        """ """
        memories = random.sample(self.memory, self.batch_size)
        self.train_critic(memories)
        self.train_actor(memories)

    def train_actor(self, memories):
        """ """
        for S, _, R, _, _ in memories:
            # predict new action
            A_new = self.actor_network.get_preds(S)[0]
            A_label = np.array([DT.new_label(A_new, self.n_actions)])
            # train actor based on critic gradients
            grads = self.critic_network.get_grads([S, A_label], np.array([R]))
            self.actor_network.train_network(S, grads[0], self.alpha)
            
    def train_critic(self, memories):
        """ """
        for S, A, R, S_new, done in memories:
            if not done:
                # predict new action
                A_target = self.actor_network_target.get_preds(S_new)
                A_label = np.array([DT.new_label(A_target, self.n_actions)])
                # calculate target reward
                R_new = self.critic_network_target.get_logits([S_new, A_label])
                R[0] += self.gamma * R_new[0]
            # train critic based on given action and reward
            A_label_old = np.array([DT.new_label(A, self.n_actions)])
            R = np.array([R])
            self.critic_network.train_network([S, A_label_old], R, self.alpha)


    ### TARGET ###
            
    def update_target(self):
        """ """
        self.update_actor_newtork_target()
        self.update_critic_network_target()

    def update_actor_network_target(self):
        """ """
        for i, W in enumerate(self.action_network.getWeights()):
            self.actor_network_target.layers[i].setWeights(W)

    def update_critic_network_target(self):
        """ """
        for i, W in enumerate(self.critic_network.getWeights()):
            self.critic_network_target.layers[i].setWeights(W)		


