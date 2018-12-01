from gridworlds import Gridworld

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
        """ randomizes agent and enemy """
        self.state[self.random_empty_state()] = 1
        self.terminal_states = [self.random_empty_state()]

    def set_terminal_states(self):
        """ no terminal states """
        self.terminal_states = None

    def set_initial_method(self):
        """ base SARSA """
        self.sarsa = SarsaTabular.SarsaTabular(self)

    def set_initial_color_grid(self):
        """ only terminal state  """
        self.color_grid = [Colors.WHITE for _ in self.state]
        for location in self.terminal_states:
            self.color_grid[location] = Colors.ORANGE

    def take_action(self, action):
        """ return next state and reward given action """
        self.move_piece_direction(1, action)
        action = 0
        if 2 in self.state:
            reward = -1
        else:
            reward = 1
            self.state[self.random_nonterminal_state()] = 2
        return self.state, reward


