from gridworlds import Gridworld

import sys
sys.path.append('C:\\Users\\Nick\\Desktop\\Ava\\Programs')
from Library.General import Colors
from Library.Learning import SarsaTabular


class Gridworld_4(Gridworld.Gridworld):
    """ wind pushing to the left, with goal inside columns of wind """

    def __init__(self, height, width):
        """ Gridworld size 7x10 """
        self.starting_place = width * 3
        self.wind_1 = [3, 4, 5, 8]
        self.wind_2 = [6, 7]
        Gridworld.Gridworld.__init__(self, height, width)


    ### OVERRIDES ###

    def set_initial_state(self):
        """ specific starting place """
        self.state[self.starting_place] = 1

    def set_terminal_states(self):
        """ spot in middle to the right behind the wind """
        self.terminal_states = [self.width * 4 - 3]

    def set_initial_method(self):
        """ with decay """
        self.sarsa = SarsaTabular.SarsaTabular(self)
        self.sarsa.set_parameters(lambdas=0.5, epsilon_decay=0.99)

    def set_initial_color_grid(self):
        """ start to left, leftward windin middle, goal to the right """
        self.color_grid = [Colors.WHITE for _ in self.state]
        for i in range(self.height):
            for j in self.wind_1:
                self.color_grid[self.grid_to_state((i, j))] = Colors.LIGHTBLUE
            for j in self.wind_2:
                self.color_grid[self.grid_to_state((i, j))] = Colors.SKYBLUE
        for location in self.terminal_states:
            self.color_grid[location] = Colors.ORANGE

    def take_action(self, action):
        """ ? """
        self.move_piece_direction(1, action)
        reward = 0 if self.in_terminal_state() else -1
        # apply wind
        _, x = self.location_of(1, grid=True)
        if x in self.wind_1 + self.wind_2 and not self.in_terminal_state():
            for _ in range(2 if x in self.wind_2 else 1):
                self.move_piece_direction(1, (-1, 0))
        return self.state, reward


