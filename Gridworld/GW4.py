import Gridworld

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
        self.state[self.starting_place] = self.AGENT_VAL

    def set_terminal_states(self):
        """ spot in middle to the right behind the wind """
        self.terminal_states = [self.width * 4 - 3]

    def set_initial_method(self):
        """ with decay """
        self.method = SarsaTabular.SarsaTabular(self)
        self.method.set_parameters(lambdas=0.5, epsilon_decay=0.99)

    def set_color_grid(self):
        """ start to left, leftward windin middle, goal to the right """
        self.reset_color_grid()
        for i in range(self.height):
            for j in self.wind_1:
                self.color_grid[self.grid_to_state((i, j))] = Colors.LIGHTBLUE
            for j in self.wind_2:
                self.color_grid[self.grid_to_state((i, j))] = Colors.SKYBLUE
        self.draw_terminal_states()
        self.draw_agent()

    def take_action(self, action):
        """ move agent, apply wind, find reward, set new color grid """
        self.agent_take_action(self.AGENT_VAL, action)
        self.apply_wind()
        self.reward = self.get_reward()
        self.set_color_grid()

    def get_reward(self):
        """ penalty if not in terminal state """
        if self.in_terminal_state():
            return 0
        return -1


    ### HELPER ###

    def apply_wind(self):
        """ """
        _, x = self.location_of(self.AGENT_VAL, grid=True)
        if x in self.wind_1 + self.wind_2 and not self.in_terminal_state():
            for _ in range(2 if x in self.wind_2 else 1):
                self.agent_take_action(self.AGENT_VAL, (-1, 0))


