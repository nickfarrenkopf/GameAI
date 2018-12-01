from gridworlds import Gridworld

import sys
sys.path.append('C:\\Users\\Nick\\Desktop\\Ava\\Programs')
from Library.General import Colors
from Library.Learning import SarsaTabular


class Gridworld_2(Gridworld.Gridworld):
    """ Win if top row, 2nd from right or 2nd from left """

    def __init__(self, size):
        """ Gridworld size 5x5 """
        Gridworld.Gridworld.__init__(self, size, size)


    ### OVERRIDES ###

    def set_initial_state(self):
        """ random non-terminal start """
        self.state[self.random_nonterminal_state()] = 1

    def set_terminal_states(self):
        """ top row, 2nd from left and 2nd from right """
        self.terminal_states = [1, self.width - 2]
    
    def set_initial_method(self):
        """ base SARSA  """
        self.sarsa = SarsaTabular.SarsaTabular(self)

    def set_initial_color_grid(self):
        """ only terminal states """
        self.color_grid = [Colors.WHITE for _ in self.state]
        for location in self.terminal_states:
            self.color_grid[location] = Colors.ORANGE

    def take_action(self, action):
        """ ? """
        self.move_piece_direction(1, action)
        reward = 10 if self.state[self.terminal_states[0]] == 1 else -1
        reward = 1 if self.state[self.terminal_states[1]] == 1 else reward
        return self.state, reward


