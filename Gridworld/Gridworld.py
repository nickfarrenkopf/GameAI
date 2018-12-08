import os
import random
import itertools
import numpy as np

import sys
sys.path.append('C:\\Users\\Nick\\Desktop\\Ava\\Programs')
from Library.General import Colors
from Library.Learning import EnvironmentController as Environment
from Library.Learning.Methods import SarsaTabular
from Library.Learning.Methods import SarsaNetwork
from Library.NeuralNetworks.Regression import RegressionAPI as REG


"""
TODO
 - manual movement

"""



### GRIDWORLD ###

class Gridworld(Environment.Environment):
    """ Base Gridworld object """

    def __init__(self, height, width, paths):
        """ """
        self.Colors = Colors
        Environment.Environment.__init__(self)

        # agent params
        self.AGENT_VAL = 1
        
        # state params
        self.height = height
        self.width = width
        self.state_size = height * width
        self.state = np.zeros(self.state_size, dtype=int)
        self.grid = np.reshape(self.state, (height, width))

        # action params
        self.max_move = 1
        self.actions = self.default_actions()
        self.action_profile = self.get_action_profile()

        # network params
        self.paths = paths
        self.json_data = paths.load_json()
        self.network = self.load_network()
        

    def initialize(self):
        """ """
        self.set_terminal_states()
        self.set_initial_state()
        self.set_initial_method()
        self.set_color_grid()


    ### TODO ###

    def set_initial_state(self):
        """ STATE """
        pass

    def set_terminal_states(self):
        """ STATE """
        pass

    def set_color_grid(self):
        """ GRID """
        pass

    def take_action(self):
        """ ACTION """
        pass

    def get_reward(self):
        """ REWARD """
        pass
    
    
    ### STATE ###

    def set_default_initial_state(self):
        """ random non-terminal start """
        self.state[self.get_random_state_idx(terminal=False)] = self.AGENT_VAL

    def set_starting_idx_state(self):
        """ specific starting place """
        self.state[self.start_idx] = self.AGENT_VAL
    
    def in_terminal_state(self):
        """ checks if agent is in terminal state """
        return self.location_of(self.AGENT_VAL) in self.terminal_states

    def reset_environment(self):
        """ clear current state object and set initial env state """
        for i in range(self.state_size):
            self.state[i] = 0
        self.set_initial_state()

    def get_random_state_idx(self, terminal=False):
        """ return index of random, empty, non/terminal state """
        idxs = [i for i, s in enumerate(self.state) if s == 0 and
                terminal != (i not in self.terminal_states)] 
        return random.choice(idxs)
        

    ### ACTION ###

    def default_actions(self):
        """ defines actions player can take """
        a1 = Environment.Action('y movement', [-1, 0, 1])
        a2 = Environment.Action('x movement', [-1, 0, 1])
        return [a1, a2]

    def take_default_action(self, action):
        """ """
        self.agent_take_action(self.AGENT_VAL, action)
        self.reward = self.get_reward()
        self.set_color_grid()

    def get_action_profile(self):
        """ returns a list consisting of all possible action combos """
        action_values = [action.values for action in self.actions]
        action_values = list(itertools.product(*action_values))
        action_values = [action for action in action_values if 0 in action]
        return sorted(action_values)

    def agent_take_action(self, agent, grid_change):
        """ moves piece desired direction """
        start = self.location_of(agent, grid=True)
        end = np.array(start) + np.array(grid_change)
        if self.is_valid_grid(*end):
            self.move_agent_to(start, end)

    def move_agent_to(self, start, end):
        """ moves piece from grid start to grid end """
        agent = self.grid[start[0], start[1]]
        self.grid[start[0], start[1]] = 0 
        self.grid[end[0], end[1]] = agent


   ### LEARNING ###
    
    def run_learning(self, listening_to_keys):
        """ """
        if self.method.end_of_episode:
            self.method.first_time_step()
            self.set_color_grid()
        elif not listening_to_keys:
            self.method.next_time_step()

    def run_episodes(self, n_episodes=10):
        """ """
        for _ in range(n_episodes):
            self.method.first_time_step()
            while not self.in_terminal_state():
                 self.method.next_time_step()


    ### NETWORK ###

    def set_initial_method(self):
        """ with decay """
        #self.method = SarsaTabular.SarsaTabular(self)
        self.method = SarsaNetwork.SarsaNetwork(self)
        self.method.set_parameters(epsilon_decay=0.9)
        self.method.set_parameters(lambdas=0.5, epsilon_decay=0.99)

    def create_network(self):
        """ """
        a_size = self.height * self.width + len(self.action_profile[0])
        REG.new_reg(self.paths, self.name, a_size, [64], 1, True)

    def load_network(self):
        """ """
        json_data = self.json_data['network']['reg']
        if self.name in json_data and not os.path.exists(json_data[self.name]['filepath']):
            self.create_network()
        if self.name not in json_data:
            self.create_network()
        self.json_data = self.paths.load_json()
        return REG.load_reg(self.name, self.json_data)


    ### DRAW SCREEN ###

    def reset_color_grid(self):
        """ reset color grid to all white """
        self.color_grid = [Colors.WHITE for _ in self.state]

    def draw_terminal_states(self, color=Colors.ORANGE):
        """ color terminal states """
        for location in self.terminal_states:
            self.color_grid[location] = color

    def draw_agent(self):
        """ color agent depending on current state """
        color = Colors.GREEN if self.in_terminal_state() else Colors.RED
        self.color_grid[self.location_of(self.AGENT_VAL)] = color
    
    def set_default_color_grid(self):
        """ default states  """
        self.reset_color_grid()
        self.draw_terminal_states()
        self.draw_agent()


    ### GRID ###

    def state_to_grid(self, state_index):
        """ convert state index to grid coordinate """
        return [state_index // self.width, state_index % self.width]

    def grid_to_state(self, grid_coord):
        """ convert grid coordinate into state index """
        return grid_coord[0] * self.width + grid_coord[1]

    def is_valid_grid(self, x, y):
        """ check if grid coordinate is valid """
        row_check = x >= 0 and x < self.height
        col_check = y >= 0 and y < self.width
        return row_check and col_check

    def location_of(self, agent, grid=False):
        """ returns the first board position of piece """
        idx = list(self.state).index(agent)
        return idx if not grid else self.state_to_grid(idx)


