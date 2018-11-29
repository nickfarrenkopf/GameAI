import colors
from gridworlds import Gridworld
from sarsa_tabular import Sarsa_Tabular


class Gridworld_4(Gridworld.Gridworld):
    """ gridworld example 4 """

    def __init__(self, height, width):
        """ creates base gridworld and defines actions """
        self.starting_place = width * 3
        self.wind_1 = [3,4,5,8]
        self.wind_2 = [6,7]
        Gridworld.Gridworld.__init__(self, height, width)


    ### DEFAULTS ###

    def default_terminal_states(self):
        """ """
        return [self.width * 4 - 3]

    def define_method(self):
        """ """
        sarsa = Sarsa_Tabular(self)
        sarsa.set_parameters(lambdas=0.5, epsilon_decay=0.99)
        return sarsa

    def default_color_grid(self):
        """ """
        color_grid = [colors.WHITE for _ in self.state]
        for i in range(self.height):
            for j in self.wind_1:
                color_grid[self.to_board((i, j))] = colors.LIGHTBLUE
            for j in self.wind_2:
                color_grid[self.to_board((i, j))] = colors.SKYBLUE
        for location in self.terminal_states:
            color_grid[location] = colors.ORANGE
        return color_grid

    def first_state(self):
        """ """
        self.state[self.starting_place] = 1

    def take_action(self, action):
        """ """
        self.move_piece_direction(1, action)
        reward = 0 if self.in_terminal_state() else -1
        # apply wind
        _, x = self.location_of(1, grid=True)
        if x in self.wind_1 + self.wind_2 and not self.in_terminal_state():
            for _ in range(2 if x in self.wind_2 else 1):
                self.move_piece_direction(1, (-1, 0))
        return self.state, reward


