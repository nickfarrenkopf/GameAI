import colors
from gridworlds import Gridworld
from sarsa_tabular import Sarsa_Tabular


class Gridworld_1(Gridworld.Gridworld):
    """ gridworld example 1 """

    def __init__(self, size):
        """ creates base gridworld and defines actions """
        Gridworld.Gridworld.__init__(self, size, size)


    ### DEFAULTS ###

    def set_terminal_states(self):
        """ """
        return [0, self.state_size - 1]

    def define_method(self):
        """ """
        sarsa = Sarsa_Tabular(self)
        return sarsa

    def default_color_grid(self):
        """ """
        color_grid = [colors.WHITE for _ in self.state]
        for location in self.terminal_states:
            color_grid[location] = colors.ORANGE
        return color_grid

    def set_initial_state(self):
        """ """
        self.state[self.random_nonterminal_state()] = 1

    def take_action(self, action):
        """ """
        self.move_piece_direction(1, action)
        reward = 0 if self.in_terminal_state() else -1
        return self.state, reward


