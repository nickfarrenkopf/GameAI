import Gridworld


class Gridworld_1(Gridworld.Gridworld):
    """ Win if top-left or bottom-right corner """

    def __init__(self, paths, run_training, run_pred):
        """ Gridworld size 5x5 """
        self.name = 'gridworld_1'
        Gridworld.Gridworld.__init__(self, 5, 5, paths, run_training, run_pred)

        self.set_initial_state = self.set_default_initial_state
        self.set_color_grid = self.draw_default_color_grid
        self.take_action = self.default_take_action
        self.initialize()


    ### OVERRIDES ###
  
    def set_terminal_states(self):
        """ top-left and bottom right corners """
        self.terminal_states = [0, self.state_size - 1]

    def get_reward(self):
        """ penalty if not in terminal state """
        if self.in_terminal_state():
            return 0
        return -1


