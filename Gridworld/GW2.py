import Gridworld


class Gridworld_2(Gridworld.Gridworld):
    """ Win if top row, 2nd from right or 2nd from left """

    def __init__(self, paths):
        """ Gridworld size 5x5 """
        self.name = 'gridworld_2'
        Gridworld.Gridworld.__init__(self, 5, 5, paths)

        self.set_initial_state = self.set_default_initial_state
        self.set_color_grid = self.set_default_color_grid
        self.take_action = self.take_default_action
        self.initialize()
    

    ### OVERRIDES ###

    def set_terminal_states(self):
        """ top row, 2nd from left and 2nd from right """
        self.terminal_states = [1, self.width - 2]

    def get_reward(self):
        """ left terminal state worth more than right terminal state """
        if self.state[self.terminal_states[0]] == 1:
            return 10
        if self.state[self.terminal_states[1]] == 1:
            return 0
        return -1


