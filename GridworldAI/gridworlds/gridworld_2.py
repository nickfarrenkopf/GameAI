import colors
from gridworlds import Gridworld
from sarsa_tabular import Sarsa_Tabular


class Gridworld_2(Gridworld.Gridworld):
    """ """

    def __init__(self, size):
        """ """
        Gridworld.Gridworld.__init__(self, size, size)


    ### DEFAULTS ###

    def default_terminal_states(self):
        """ """
        return [1, self.width - 2]

    def define_method(self):
        """ """
        sarsa = Sarsa_Tabular(self)
        sarsa.set_parameters(decay_start=500)
        return sarsa

    def default_color_grid(self):
        """ """
        color_grid = [colors.WHITE for _ in self.state]
        for location in self.terminal_states:
            color_grid[location] = colors.ORANGE
        return color_grid
    
    def first_state(self):
        """ """
        self.state[self.random_nonterminal_state()] = 1

    def take_action(self, action):
        """ """
        self.move_piece_direction(1, action)
        reward = 10 if self.state[self.terminal_states[0]] == 1 else -1
        reward = 1 if self.state[self.terminal_states[1]] == 1 else reward
        return self.state, reward


