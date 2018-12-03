import Gridworld

import sys
sys.path.append('C:\\Users\\Nick\\Desktop\\Ava\\Programs')
from Library.General import Colors
from Library.Learning import SarsaNetwork


class Gridworld_1(Gridworld.Gridworld):
    """ Win if top-left or bottom-right corner """

    def __init__(self, size, json_data):
        """ Gridworld size 5x5 """
        Gridworld.Gridworld.__init__(self, size, size, json_data)


    ### OVERRIDES ###

    def set_initial_state(self):
        """ random non-terminal start """
        self.state[self.get_random_state_idx(terminal=False)] = self.AGENT_VAL
        
    def set_terminal_states(self):
        """ top-left and bottom right corners """
        self.terminal_states = [0, self.state_size - 1]

    def set_initial_method(self):
        """ basic SARSA """
        self.method = SarsaNetwork.SarsaNetwork(self)

    def set_color_grid(self):
        """ default states  """
        self.reset_color_grid()
        self.draw_terminal_states()
        self.draw_agent()

    def take_action(self, action):
        """ move agent, find reward, set new color grid """
        self.agent_take_action(self.AGENT_VAL, action)
        self.reward = self.get_reward()
        self.set_color_grid()

    def get_reward(self):
        """ penalty if not in terminal state """
        if self.in_terminal_state():
            return 0
        return -1


