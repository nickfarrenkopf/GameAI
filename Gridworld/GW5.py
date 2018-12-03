import random
import numpy as np

import Gridworld

import sys
sys.path.append('C:\\Users\\Nick\\Desktop\\Ava\\Programs')
from Library.General import Colors
from Library.Learning import SarsaTabular


class Gridworld_5(Gridworld.Gridworld):
    """ avoid the randomly moving enemy """

    def __init__(self, height, width):
        """ Gridworld size 5x5 """
        Gridworld.Gridworld.__init__(self, height, width)


    ### OVERRIDES ###

    def set_initial_state(self):
        """ """
        self.state[self.get_random_state_idx()] = 1
        self.terminal_states = [self.get_random_state_idx()]

    def set_terminal_states(self):
        """ """
        self.terminal_states = []

    def set_initial_method(self):
        """ base SARSA """
        self.method = SarsaTabular.SarsaTabular(self)
        self.method.set_parameters(lambdas=0.5, epsilon_decay=0.99)

    def set_color_grid(self):
        """ only terminal state  """
        self.reset_color_grid()
        self.draw_terminal_states()
        self.draw_agent()

    def take_action(self, action):
        """ return next state and reward given action """
        self.agent_take_action(self.AGENT_VAL, action)
        self.move_terminal_states(color=Colors.BLUE)
        self.reward = self.get_reward()
        self.set_color_grid()

    def get_reward(self):
        """ penalty if not in terminal state """
        if self.in_terminal_state():
            return -1
        return 0


    ### HELPER ###

    def move_terminal_states(self):
        """ """
        s = self.state_to_grid(self.terminal_states[0])
        a = random.choice(self.action_profile)
        s_n = np.array(s) + np.array(a)
        if self.is_valid_grid(*s_n):
            self.terminal_states = [self.grid_to_state(s_n)]


