import itertools
import numpy as np

import colors


### ACTION CLASS ###

class Action(object):
    """ action player can take during turn """

    def __init__(self, name, values):
        """ defines action by name and range of possible values """
        self.name = name
        self.values = values


### GRIDWORLD ###

class Gridworld(object):
    """ """

    def __init__(self, height, width):
        """ creates gridworld with board and grid, then initializes """
        # gridworld variables
        self.height = height #
        self.width = width #
        self.size = height * width
        self.max_move = self.min_move = 1
        
        # gridworld independent
        self.state, self.grid = self.default_state()
        self.actions = self.default_actions()

        # gridworld dependent
        self.terminal_states = self.default_terminal_states()
        self.sarsa = self.define_method()
        self.color_grid = self.default_color_grid()


    ### DEFAULTS - IND ###

    def default_state(self):
        """ defines and returns state object that agent interacts with """
        zeros = np.zeros(self.size, dtype=int)
        grid = np.reshape(zeros, (self.height, self.width))
        return zeros, grid

    def default_actions(self):
        """ defines actions player can take """
        a1 = Action('y movement', [-1, 0, 1])
        a2 = Action('x movement', [-1, 0, 1])
        return [a1, a2]


    ### DEFAULTS - DEP ###

    def default_terminal_states(self):
        """ """
        pass

    def define_method(self):
        """ """
        pass

    def default_color_grid(self):
        """ """
        pass

    def first_state(self):
        """ """
        pass

    def take_action(self, action):
        """ """
        pass
 

    ### HELPER ###



    ### PLACE PIECE ###
    
    def random_nonterminal_state(self):
        """ initializes piece to random empty position, teminal specified """
        start = np.random.randint(self.size)
        while start in self.terminal_states or self.state[start] != 0:
            start = np.random.randint(self.size)
        return start



    def random_terminal_state(self, piece):
        """ returns random empty terminal state """
        states = [idx for idx in self.terminal_states if self.state[idx] == 0]
        self.state[np.random.choice(states)] = piece
   

    ### MOVE PIECE ###

    def move_piece_direction(self, piece, grid_change):
        """ moves piece desired direction """
        start = self.location_of(piece, grid=True)
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


    ### END ###

    def reset(self):
        """ resets board to all zeros then initializes board """
        for i in range(self.size):
            self.state[i] = 0
        self.first_state()


    ### HELPER ###

    def action_profile(self):
        """ returns a list consisting of all possible action combos """
        action_values = [action.values for action in self.actions]
        return list(itertools.product(*action_values))  

    def to_board(self, grid_coord):
        """ converts grid coordinates to board index """
        return grid_coord[0] * self.width + grid_coord[1]

    def to_grid(self, board_index):
        """ converts board index to grid coordinates """
        return [board_index // self.width, board_index % self.width]

    def on_grid(self, grid_coord):
        """ returns true if valid grid position """
        row_check = grid_coord[0] >= 0 and grid_coord[0] < self.height
        col_check = grid_coord[1] >= 0 and grid_coord[1] < self.width
        return row_check and col_check

    def location_of(self, piece, grid=False):
        """ returns the first board position of piece """
        idx = list(self.state).index(piece)
        return idx if not grid else self.to_grid(idx)
    
    def corners(self):
        """ returns corner board locations """
        idx = [(i,j) for i in [0, self.height - 1] for j in [0, self.width - 1]]
        return [self.to_board(grid_coord) for grid_coord in idx]  





    def in_terminal_state(self):
        """ returns true if grid in terminal state """
        return self.location_of(1) in self.terminal_states

