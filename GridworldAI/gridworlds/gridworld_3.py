import colors
from gridworlds import Gridworld
from sarsa_tabular import Sarsa_Tabular


class Gridworld_3(Gridworld.Gridworld):
    """ gridworld example 3 """

    def __init__(self, height, width):
        """ creates base gridworld and defines actions """
        self.starting_place = width * (height - 1)
        Gridworld.Gridworld.__init__(self, height, width)


    ### DEFAULTS ###

    def default_terminal_states(self):
        """ """
        return list(range(self.starting_place + 1, self.size))

    def define_method(self):
        """ """
        sarsa = Sarsa_Tabular(self)
        sarsa.set_parameters(lambdas=0.5, epsilon_decay=0.99)
        return sarsa

    def default_color_grid(self):
        """ """
        color_grid = [colors.WHITE for _ in self.state]
        for idx in self.terminal_states:
            color_grid[idx] = colors.BLUE
        color_grid[-1] = colors.ORANGE
        return color_grid

    def first_state(self):
        """ """
        self.state[self.starting_place] = 1

    def take_action(self, action):
        """ """
        self.move_piece_direction(1, action)
        reward = -1
        if self.in_terminal_state():
            reward = 0 if self.state[-1] == 1 else -100
        return self.state, reward


