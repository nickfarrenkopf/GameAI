from gridworlds import Gridworld

import sys
sys.path.append('C:\\Users\\Nick\\Desktop\\Ava\\Programs')
from Library.General import Colors
from Library.Learning import SarsaTabular


class Gridworld_1(Gridworld.Gridworld):
    """ Win if top-left or bottom-right corner """

    def __init__(self, size):
        """ Gridworld size 5x5 """
        Gridworld.Gridworld.__init__(self, size, size)


    ### OVERRIDES ###

    def set_initial_state(self):
        """ random non-terminal start """
        self.state[self.random_nonterminal_state()] = 1
        
    def set_terminal_states(self):
        """ top-left and bottom right corners """
        self.terminal_states = [0, self.state_size - 1]

    def set_initial_method(self):
        """ basic SARSA """
        self.sarsa = SarsaTabular.SarsaTabular(self)

    def set_initial_color_grid(self):
        """ only terminal states  """
        self.color_grid = [Colors.WHITE for _ in self.state]
        for location in self.terminal_states:
            self.color_grid[location] = Colors.ORANGE

    def take_action(self, action):
        """ ? """
        self.move_piece_direction(1, action)
        reward = 0 if self.in_terminal_state() else -1
        return self.state, reward

    
