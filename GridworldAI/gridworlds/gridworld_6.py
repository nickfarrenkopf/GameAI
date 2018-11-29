import colors
from gridworlds import Gridworld
from sarsa_tabular import Sarsa_Tabular


class Gridworld_6(Gridworld.Gridworld):
    """ gridworld example 6 """

    def __init__(self, height, width):
        """ creates base gridworld and defines actions """
        Gridworld.Gridworld.__init__(self, height, width)


    ### PARENT OVERRIDE ###

    def default_terminal_states(self):
        """ """
        return []

    def define_method(self):
        """ """
        sarsa = Sarsa_Tabular(self)
        return sarsa

    def default_color_grid(self):
        """  """
        color_grid = [colors.WHITE for _ in self.state]
        for i in self.state:
            if i == 2:
                self.state[i] = colors.BLUE
        return color_grid

    def first_state(self):
        """ randomizes player location to non terminal empty state """
        self.state[self.random_nonterminal_state()] = 1
        self.state[self.random_nonterminal_state()] = 2

    def take_action(self, action):
        """ return next state and reward given action """
        self.move_piece_direction(1, action)
        if 2 in self.state:
            reward = -1
        else:
            reward = 1
            self.state[self.random_nonterminal_state()] = 2
        return self.state, reward


