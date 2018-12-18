import os
import random
import itertools
import numpy as np

import sys
sys.path.append('C:\\Users\\Nick\\Desktop\\Ava\\Programs')
from Library.General import Colors
from Library.Learning import EnvironmentController as Environment
from Library.Learning.Methods import SarsaTabular as ST
from Library.Learning.Methods import SarsaNetwork as SN
from Library.NeuralNetworks.Regression import RegressionAPI as REG


### GRIDWORLD ###

class Gridworld(Environment.Environment):
    """ Base Gridworld object """

    def __init__(self, paths, height, width):
        """ """
        Environment.Environment.__init__(self)
        self.paths = paths
        self.Colors = Colors

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
        self.actions = self.get_default_actions()
        self.action_profile = self.get_action_profile()
        self.action_size = len(self.action_profile[0])
        self.SA_size = self.state_size + self.action_size

        # network params
        self.hidden = [64]
        

    ### ENVIRONMENT ###

    """
    TODO
     - def set_initial_state(self):
     - def set_terminal_states(self):
     - def set_color_grid(self):
     - def take_action(self):
     - def get_reward(self):
    """
    
    def initialize(self):
        """ """
        self.set_terminal_states()
        self.set_initial_state()
        self.set_method()
        self.set_color_grid()

    def reset_environment(self):
        """ clear current state object and set initial env state """
        for i in range(self.state_size):
            self.state[i] = 0
        self.set_initial_state()
    

    ### STATE ###

    def set_default_initial_state(self):
        """ random non-terminal start """
        self.state[self.get_random_state_idx(False)] = self.AGENT_VAL

    def set_starting_initial_state(self):
        """ specific starting place """
        self.state[self.start_idx] = self.AGENT_VAL
    
    def in_terminal_state(self):
        """ checks if agent is in terminal state """
        return self.location_of(self.AGENT_VAL) in self.terminal_states

    def get_random_state_idx(self, allow_terminal=False):
        """ return index of random, empty, non/terminal state """
        idxs = [i for i, s in enumerate(self.state) if s == 0 and
                allow_terminal != (i not in self.terminal_states)] 
        return random.choice(idxs)
    

    ### ACTION ###

    def get_default_actions(self):
        """ defines actions player can take """
        a1 = Environment.Action('y movement', [-1, 0, 1])
        a2 = Environment.Action('x movement', [-1, 0, 1])
        return [a1, a2]

    def get_action_profile(self):
        """ returns a list consisting of all possible action combos """
        action_values = [action.values for action in self.actions]
        action_values = list(itertools.product(*action_values))
        action_values = [action for action in action_values
                         if np.sum(np.abs(action)) <= self.max_move]
        action_values = sorted(action_values)
        return action_values

    def take_action_default(self, action):
        """ """
        self.agent_take_action(self.AGENT_VAL, action)
        self.reward = self.get_reward()
        self.set_color_grid()

    def agent_take_action(self, agent, grid_change):
        """ moves piece desired direction """
        start = self.location_of(agent, grid=True)
        end = np.array(start) + np.array(grid_change)
        if self.is_valid_grid(*end):
            self.agent_move_to(start, end)

    def agent_move_to(self, start, end):
        """ moves piece from grid start to grid end """
        agent = self.grid[start[0], start[1]]
        self.grid[start[0], start[1]] = 0 
        self.grid[end[0], end[1]] = agent


    ### GRID ###

    def location_of(self, agent, grid=False):
        """ returns the first board position of piece """
        idx = list(self.state).index(agent)
        return idx if not grid else self.state_to_grid(idx)
    
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


    ### DRAW SCREEN ###

    def draw_blank_grid(self):
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
    
    def draw_default_color_grid(self):
        """ default states  """
        self.draw_blank_grid()
        self.draw_terminal_states()
        self.draw_agent()


    ### LEARNING ###

    def set_training_params(self, run_train, run_pred, train_start, with_decay):
        """ """
        self.run_train = run_train
        self.run_pred = run_pred
        self.train_start = train_start
        self.with_decay = with_decay

    def set_method(self):
        """ """
        if not self.run_train and not self.run_pred:
            print('Using SARSA tabular')
            self.method = ST.SarsaTabular(self)
        else:
            print('Using SARSA Network')
            self.method = SN.SarsaNetwork(self, self.run_train, self.run_pred,
                                          self.train_start)
        if self.with_decay:
            self.method.set_parameters(lambdas=0.5, epsilon_decay=0.99)

    def load_value_data(self):
        """ """
        data = self.paths.load_json()['learning']
        if self.name in data:
            self.method.json_to_value(data[self.name])

    def end_learning(self):
        """ """
        data = self.paths.load_json()
        data['learning'].update({self.name: {}})
        for k in self.method.Q.keys():
            SA = ','.join([str(i) for i in list(k)])
            data['learning'][self.name][SA] = self.method.Q[k].value
        self.paths.write_json(data)

    def iterate(self, action=None):
        """ """
        if self.method.end_of_episode:
            self.method.first_time_step()
            self.set_color_grid()
        else:
            self.method.next_time_step(action)

    def iterate_episodes(self, n_episodes=10):
        """ """
        for i in range(n_episodes):
            self.method.first_time_step()
            while not self.in_terminal_state():
                 self.method.next_time_step()
            if i % 10 == 0:
                print(' - iter {}'.format(i))

        
