import Gridworld

import sys
sys.path.append('C:\\Users\\Nick\\Desktop\\Ava\\Programs')
from Library.General import Colors
from Library.Learning import SarsaTabular


class Gridworld_2(Gridworld.Gridworld):
    """ Win if top row, 2nd from right or 2nd from left """

    def __init__(self, name, size, paths):
        """ Gridworld size 5x5 """
        Gridworld.Gridworld.__init__(self, size, size, paths)


    ### OVERRIDES ###

    def set_initial_state(self):
        """ random non-terminal start """
        self.state[self.get_random_state_idx(terminal=False)] = self.AGENT_VAL

    def set_terminal_states(self):
        """ top row, 2nd from left and 2nd from right """
        self.terminal_states = [1, self.width - 2]
    
    def set_initial_method(self):
        """ base SARSA  """
        self.method = SarsaTabular.SarsaTabular(self)
        self.method.set_parameters(lambdas=0.5, epsilon_decay=0.99)

    def set_color_grid(self):
        """ only terminal states """
        self.reset_color_grid()
        self.draw_terminal_states()
        self.draw_agent()

    def take_action(self, action):
        """ move agent, find reward, set new color grid """
        self.agent_take_action(self.AGENT_VAL, action)
        self.reward = self.get_reward()
        self.set_color_grid()

    def get_reward(self):
        """ left terminal state worth more than right terminal state """
        if self.state[self.terminal_states[0]] == 1:
            return 10
        if self.state[self.terminal_states[1]] == 1:
            return 0
        return -1


