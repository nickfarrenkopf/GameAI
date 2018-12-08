import random
import numpy as np

import Gridworld


class Gridworld_5(Gridworld.Gridworld):
    """ avoid the randomly moving enemy """

    def __init__(self, paths):
        """ Gridworld size 5x5 """
        self.name = 'gridworld5'
        self.height = 5
        self.width = 5
        Gridworld.Gridworld.__init__(self, self.height, self.width, paths)
        self.initialize()


    ### OVERRIDES ###

    def set_initial_state(self):
        """ """
        self.state[self.get_random_state_idx()] = self.AGENT_VAL
        self.terminal_states = [self.get_random_state_idx()]

    def set_terminal_states(self):
        """ """
        self.terminal_states = []
    
    def set_color_grid(self):
        """ only terminal state  """
        self.reset_color_grid()
        self.draw_terminal_states(color=self.Colors.BLUE)
        self.draw_agent()

    def take_action(self, action):
        """ return next state and reward given action """
        self.agent_take_action(self.AGENT_VAL, action)
        self.move_terminal_states()
        self.reward = self.get_reward()
        self.set_color_grid()

    def get_reward(self):
        """ penalty if not in terminal state """
        if self.in_terminal_state():
            return -100
        return 0


    ### HELPER ###

    def move_terminal_states(self):
        """ """
        state = self.state_to_grid(self.terminal_states[0])
        action = random.choice(self.action_profile)
        state_next = np.array(state) + np.array(action)
        if self.is_valid_grid(*state_next):
            self.terminal_states = [self.grid_to_state(state_next)]


