import itertools
import numpy as np

import sys
sys.path.append('C:\\Users\\Nick\\Desktop\\Ava\\Programs')
from Library.General import Colors
from Library.Learning import EnvironmentController as Environment




### GRIDWORLD ###

class Gridworld(object):
    """ """

    def __init__(self, height, width):
        """ creates gridworld with board and grid, then initializes """

        Environment.Environment.__init__(self)
        
        # gridworld environment params
        self.height = height
        self.width = width

        self.AGENT = 1

        # default env params
        self.state_size = height * width
        self.state = np.zeros(self.state_size, dtype=int)
        
        

        self.grid = np.reshape(self.state, (height, width))

        # action params
        self.max_move = self.min_move = 1
        
        # gridworld independent
        
        self.actions = self.define_actions()

        # gridworld dependent
        self.set_terminal_states()
        self.set_initial_method()
        self.set_initial_color_grid()


    ### STATE - BASE ###

    def set_initial_state(self):
        """ todo """
        pass

    def set_terminal_states(self):
        """ todo """
        pass

    def in_terminal_state(self):
        """ """
        return self.location_of(self.AGENT) in self.terminal_states

    def reset_environment(self):
        """ """
        for i in range(self.state_size):
            self.state[i] = 0
        self.set_initial_state()
        
    
    ### STATE - HELPER ###

    def random_terminal_state(self, piece):
        """ """
        idx = np.random.randint(self.state_size)
        while idx not in self.terminal_states or self.state[idx] != 0:
            idx = np.random.randint(self.state_size)
        return idx

    def random_nonterminal_state(self, terminal=False):
        """ """
        idx = np.random.randint(self.state_size)
        while terminal * (idx in self.terminal_states) or self.state[idx] != 0:
            idx = np.random.randint(self.state_size)
        return idx

    def random_empty_state(self):
        """ """
        idx = np.random.randint(self.state_size)
        while self.state[idx] != 0:
            idx = np.random.randint(self.state_size)
        return idx
        

    ### ACTION - BASE ###

    def define_actions(self):
        """ defines actions player can take """
        a1 = Environment.Action('y movement', [-1, 0, 1])
        a2 = Environment.Action('x movement', [-1, 0, 1])
        return [a1, a2]

    def take_action(self):
        """ todo """
        pass
    

    ### ACTION - HELPER ###

    def move_piece_direction(self, agent, grid_change):
        """ moves piece desired direction """
        start = self.location_of(agent, grid=True)
        end = np.array(start) + np.array(grid_change)
        if self.on_grid(end) and self.valid_move(grid_change):
            self.move_piece_to(start, end)

    def move_piece_to(self, start, end):
        """ moves piece from grid start to grid end """
        piece = self.grid[start[0], start[1]]
        self.grid[start[0], start[1]] = 0
        self.grid[end[0], end[1]] = piece

    def valid_move(self, grid_change):
        """ returns true if move is valid """
        movement = np.sum([np.abs(value) for value in grid_change])
        return movement <= self.max_move and movement >= self.min_move


    ### LEARNING ###

    def define_method(self):
        """ todo """
        pass
    

    ### GRID ###

    def set_initial_color_grid(self):
        """ """
        pass

    def state_to_grid(self, state_index):
        """ """
        return [state_index // self.width, state_index % self.width]

    def grid_to_state(self, grid_coord):
        """ """
        return grid_coord[0] * self.width + grid_coord[1]


    def on_grid(self, grid_coord):
        """ """
        row_check = grid_coord[0] >= 0 and grid_coord[0] < self.height
        col_check = grid_coord[1] >= 0 and grid_coord[1] < self.width
        return row_check and col_check

    def get_corners(self):
        """ """
        idx = [(i,j) for i in [0, self.height - 1] for j in [0, self.width - 1]]
        return [self.to_state(grid_coord) for grid_coord in idx]  


    def location_of(self, piece, grid=False):
        """ returns the first board position of piece """
        idx = list(self.state).index(piece)
        return idx if not grid else self.state_to_grid(idx)

    
    

    def get_action_profile(self):
        """ returns a list consisting of all possible action combos """
        action_values = [action.values for action in self.actions]
        return list(itertools.product(*action_values))  


