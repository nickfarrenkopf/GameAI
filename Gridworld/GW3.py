import Gridworld


class Gridworld_3(Gridworld.Gridworld):
    """  """

    def __init__(self, paths, run_training, run_pred):
        """ Gridworld size 4x12 """
        self.name = 'gridworld_3'
        self.height = 4
        self.width = 12
        self.start_idx = self.width * (self.height - 1)
        Gridworld.Gridworld.__init__(self, self.height, self.width, paths, run_training, run_pred)

        self.set_initial_state = self.set_starting_initial_state
        self.take_action = self.default_take_action
        self.initialize()


    ### OVERRIDES ###

    def set_terminal_states(self):
        """ bottom row except for left corner start and right corner points """
        self.terminal_states = list(range(self.start_idx + 1, self.state_size))

    def set_color_grid(self):
        """ bottom row death, left corner start, right corner end """
        self.draw_default_color_grid()
        self.draw_terminal_states(color=self.Colors.BLUE)
        self.color_grid[-1] = self.Colors.ORANGE
        self.draw_agent()
    
    def get_reward(self):
        """ goal is right corner, fall of on bottom edge, penalty for slow """
        if self.in_terminal_state():
            if self.state[-1] == 1:
                return 0
            return -10
        return -1


