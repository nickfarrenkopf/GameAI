import os
import time
import random
import itertools
import numpy as np

from learning import GridworldAgent as GA

import sys
sys.path.append('C:\\Users\\Nick\\Desktop\\Ava\\Programs')
from Library.General import Colors
from Library.Learning import Environment
from Library.Learning import AgentUtils as AU


### GRIDWORLD ###

class Gridworld(Environment.Environment):
    """ Base Gridworld object """

    def __init__(self, paths, height, width):
        """ """
        self.paths = paths
        Environment.Environment.__init__(self)
        
        # state params
        self.height = height
        self.width = width
        self.state_size = height * width
        self.state = np.zeros(self.state_size, dtype=int)
        self.grid = np.reshape(self.state, (height, width))

        # agent params
        self.default_agents()

    """
    TODO - child class
     - name
     - def set_terminal_states(self):
     - def set_initial_state(self):
     - def set_next_state(self):
     - def set_color_grid(self):
     
     - def get_reward(self):
    """


    ### ENVIRONMENT ###

    def reset(self):
        """ """
        self.reset_state()
        self.reset_agents()

    
    ### STATE ###

    def reset_state(self):
        """ """
        for i in range(self.state_size):
            self.state[i] = 0
        self.set_terminal_states()
        self.set_initial_state()
        self.set_color_grid()
    
    def in_terminal_state(self):
        """ checks if agent is in terminal state """
        return self.main_agent.in_terminal_state() # AG

    def random_empty_state(self):
        """ return index of random, empty, non/terminal state """
        idxs = [i for i, s in enumerate(self.state) if s == 0]
        return random.choice(idxs)


    ### AGENT ###





    def reset_agents(self):
        """ """
        for agent in self.agents.as_list():
            agent.method.first_time_step()

    def move_agent_to(self, agent, idx):
        """ """
        self.state[agent.location] = 0
        self.state[idx] = agent.KEY
        agent.location = idx

    def move_agent(self, agent, action):
        """ moves piece desired direction """
        grid = np.array(self.state_to_grid(agent.location)) + np.array(action)
        if self.is_valid_grid(*grid):
            self.move_agent_to(agent, self.grid_to_state(grid))
        
    
    ### DEFAULTS ###

    def default_agents(self): # AG ?
        """ """
        self.agents = AU.AgentGroup([GA.GridworldAgent(self, 1)])
        self.main_agent = self.agents.first()

    def default_initial_state(self):
        """ only agents in gridworld """
        for agent in self.agents.as_list():
            self.move_agent_to(agent, self.random_empty_state())

    def default_starting_state(self):
        """ only agents in gridworld """
        for agent in self.agents.as_list():
            self.move_agent_to(agent, agent.start_idx)

    def default_color_grid(self):
        """ default states  """
        self.draw_blank_grid()
        self.draw_terminal_states()
        self.draw_agents()




    ### OFFLINE ###

    def run_offline_episodes(self, n_episodes):
        """ """
        for _ in range(n_episodes):
            self.run_offline_episode()

    def run_offline_episode(self):
        """ """
        self.reset()
        while not self.in_terminal_state():
            A = self.main_agent.choose_action()
            self.move_agent(self.main_agent, A)
            self.main_agent.method.next_time_step(A, self.get_reward()) # when





    ### ONLINE ###

    def iterate(self):
        """ """
        action = self.main_agent.choose_action()
        self.run_step(action)

    def run_step(self, action):
        """ """
        if type(action) is str:
            action = self.main_agent.find_by_dict_key(action)
        # in terminal state
        if self.in_terminal_state():
            self.reset()
        # non terminal state
        else:   
            self.move_agent(self.main_agent, action)
            self.main_agent.method.next_time_step(action, self.get_reward())
            self.set_color_grid()






    
    ### GRID ###

    def state_to_grid(self, state_index):
        """ convert state index to grid coordinate """
        return [state_index // self.width, state_index % self.width]

    def grid_to_state(self, grid_coord):
        """ convert grid coordinate into state index """
        return grid_coord[0] * self.width + grid_coord[1]

    def location_of(self, key, grid=False):
        """ returns the first board position of piece """
        idx = list(self.state).index(key)
        return self.state_to_grid(idx) if grid else idx 
    
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
        for idx in self.terminal_states:
            self.color_grid[idx] = color

    def draw_agents(self):
        """ color agent depending on current state """
        for agent in self.agents.as_list(): # AG
            self.color_grid[agent.location] = agent.get_color()


