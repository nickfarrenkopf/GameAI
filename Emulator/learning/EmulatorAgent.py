import time
import random

import sys
sys.path.append('C:\\Users\\Nick\\Desktop\\Ava\\Programs')
from Library.Learning.Methods import Tabular as TL
from Library.Learning.Methods import ActorCritic as ACM


class EmulatorAgent(object):
    """ """

    def __init__(self, paths, env, key, actions, networks=True):
        """ """
        self.paths = paths
        self.env = env
        self.key = key
        self.actions = actions
        self.A_size = len(actions)

        # learning
        self.name = self.env.name + '_agent{}'.format(key)
        self.method = ACM.ActorCriticMethod(paths, self, load_networks=networks)


    ### API ###

    def save_nets(self):
        """ """
        self.method.actor_network.save_network()
        self.method.actor_TARGET.save_network()
        self.method.critic_network.save_network()
        self.method.critic_TARGET.save_network()

    def play(self, n_play=30):
        """ """
        for i in range(n_play):
            self.play_round()
            time.sleep(1)

    def play_round(self):
        """ """
        # game state
        S = self.env.get_state()
        # actions
        A = self.decide_action(S)
        EA.take_actions(A)
        print('Taking actions {}'.format(A))
        # AC3 - process reward
        R = self.read_reward(S)
        self.method.learn_from_reward(S, R)


    ### ACTION ###

    def decide_action(self, state, is_learned=True):
        """ """
        if is_learned:
            return self.method.choose_action(state)
        else:
            return [random.choice(self.actions)]


    ### REWARD ###

    def read_reward(self, state):
        """ """
        # classification of current game state
        pass


