import random
import numpy as np
import tensorflow as tf

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
        self.name = agent.name
        self.name_t = agent.name + '_target'

        # data params
        self.S_size = self.agent.env.S_size
        self.A_size = self.agent.A_size

        self.actions = agent.actions
        self.n_actions = len(agent.actions)
        self.memory = []

        # training params
        self.alpha = 1e-3
        self.epsilon = 1.0
        self.epsilon_decay = .995
        self.gamma = .95
        self.tau = .125
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
            R = np.array([[R]])
            A_new = self.actor_network.get_preds(S)[0]
            zeros = np.zeros((1, self.n_actions))
            zeros[0][A_new] = 1
            grads = self.critic_network.get_grads([S, zeros], R)[0]
            self.actor_network.train_network(S, grads, self.alpha)
            
    def train_critic(self, memories):
        """ """
        for S, A, R, S_new, done in memories:
            if not done:
                A_target = self.actor_network_target.get_preds(S_new)
                #idx = self.actions.index(A_target[0])
                zeros = np.zeros((1, self.n_actions))
                zeros[0][A_target[0]] = 1
                R_new = self.critic_network_target.get_logits([S_new, zeros])
                R += self.gamma * R_new[0]
            # what?
            R = np.array([R])
            self.critic_network.train_network([S, zeros], R, self.alpha)


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


