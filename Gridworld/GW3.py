import Gridworld

import sys
sys.path.append('C:\\Users\\Nick\\Desktop\\Ava\\Programs')
from Library.General import Colors
from Library.Learning import SarsaTabular


class Gridworld_3(Gridworld.Gridworld):
    """  """

    def __init__(self, name, height, width, paths):
        """ Gridworld size 4x12 """
        self.name = name
        self.starting_place = width * (height - 1)
        Gridworld.Gridworld.__init__(self, height, width, paths)


    ### OVERRIDES ###

    def set_initial_state(self):
        """ specific starting place """
        self.state[self.starting_place] = self.AGENT_VAL

    def set_terminal_states(self):
        """ bottom row except for left corner start and right corner points """
        self.terminal_states = list(range(self.starting_place + 1,
                                          self.state_size))

    def set_initial_method(self):
        """ with decay """
        self.method = SarsaTabular.SarsaTabular(self)
        self.method.set_parameters(lambdas=0.5, epsilon_decay=0.99)

    def set_color_grid(self):
        """ bottom row death, left corner start, right corner end """
        self.reset_color_grid()
        self.draw_terminal_states(color=Colors.BLUE)
        self.color_grid[-1] = Colors.ORANGE
        self.draw_agent()

    def take_action(self, action):
        """ move agent, find reward, set new color grid """
        self.agent_take_action(self.AGENT_VAL, action)
        self.reward = self.get_reward()
        self.set_color_grid()
    
    def get_reward(self):
        """ goal is right corner, fall of on bottom edge, penalty for slow """
        if self.in_terminal_state():
            if self.state[-1] == 1:
                return 0
            return -100
        return -1


