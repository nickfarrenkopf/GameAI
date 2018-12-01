from gridworlds import Gridworld

import sys
sys.path.append('C:\\Users\\Nick\\Desktop\\Ava\\Programs')
from Library.General import Colors
from Library.Learning import SarsaTabular


class Gridworld_3(Gridworld.Gridworld):
    """  """

    def __init__(self, height, width):
        """ Gridworld size 4x12 """
        self.starting_place = width * (height - 1)
        Gridworld.Gridworld.__init__(self, height, width)


    ### OVERRIDES ###

    def set_initial_state(self):
        """ specific starting place """
        self.state[self.starting_place] = 1

    def set_terminal_states(self):
        """ bottom row except for left corner start and right corner points """
        self.terminal_states = list(range(self.starting_place + 1,
                                          self.state_size))

    def set_initial_method(self):
        """ with decay """
        self.sarsa = SarsaTabular.SarsaTabular(self)
        self.sarsa.set_parameters(lambdas=0.5, epsilon_decay=0.99)

    def set_initial_color_grid(self):
        """ bottom row death, left corner start, right corner end """
        self.color_grid = [Colors.WHITE for _ in self.state]
        for idx in self.terminal_states:
            self.color_grid[idx] = Colors.BLUE
        self.color_grid[-1] = Colors.ORANGE

    def take_action(self, action):
        """ ? """
        self.move_piece_direction(1, action)
        reward = -1
        if self.in_terminal_state():
            reward = 0 if self.state[-1] == 1 else -100
        return self.state, reward


