import time
import random

from learning import ActorCritic as ACM

import sys
sys.path.append('C:\\Users\\Nick\\Desktop\\Ava\\Programs')
from Library.Learning.Methods import Tabular as TL


class EmulatorAgent(object):
    """ """

    def __init__(self, paths, env, key):
        """ """
        self.paths = paths
        self.env = env
        self.key = key
        self.name = self.env.name + '_agent0'

        # actions
        self.actions = EA.sets['emulator_2']

        # learning
        self.method = ACM.ActorCriticMethod(paths, self, load_networks=True)


    ### API ###

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
        #print('NEED TO DO - Get Reward')
        # classification of current game state
        pass


