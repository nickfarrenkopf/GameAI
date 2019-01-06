import os
import time
import random
import itertools
import numpy as np

import GridworldUtils as GU
from learning import GridworldAgent as GA

import sys
sys.path.append('C:\\Users\\Nick\\Desktop\\Ava\\Programs')
from Library.General import Colors
from Library.Learning import EnvironmentUtils
from Library.Learning import AgentUtils


### GRIDWORLD ###

class Gridworld(EnvironmentUtils.Environment):
    """ Base Gridworld object """

    def __init__(self, paths, name, height, width):
        """ """
        EnvironmentUtils.Environment.__init__(self, paths, name)
        
        # state params
        self.height = height
        self.width = width
        self.state_size = height * width
        self.state = np.zeros(self.state_size, dtype=int)
        self.grid = np.reshape(self.state, (height, width))

        # agent params
        self.default_agents()

    """
    TODO
     - name
     - set_terminal_states
     - set_agent_states
     - set_next_state
     - set_color_grid
     - get_reward ?
    """






    ### ENVIRONMENT ###

    def reset(self):
        """ """
        for i in range(self.state_size):
            self.state[i] = 0
        self.set_terminal_states()
        self.set_agents() # reset
        self.set_color_grid()

    def in_terminal_state(self): # AG
        """ checks if agent is in terminal state """
        return self.main_agent.in_terminal_state()




    ### AGENT ###

    def default_agents(self): # AG ?
        """ """
        self.agents = AgentUtils.AgentGroup([GA.GridworldAgent(self, 1)])
        self.main_agent = self.agents.first()

    def default_initial_state(self):
        """ random agent starting place """
        for agent in self.agents.as_list():
            self.move_agent_to(agent, GU.random_empty_state(self.state))
            agent.reset()

    def default_starting_state(self):
        """ set agent starting place """
        for agent in self.agents.as_list():
            self.move_agent_to(agent, agent.start_idx)
            agent.reset()


    

    def move_agent_to(self, agent, idx):
        """ """
        self.state[agent.state_idx] = 0
        self.state[idx] = agent.KEY
        agent.state_idx = idx

    def move_agent(self, agent, action):
        """ moves piece desired direction """
        p1 = np.array(GU.state_to_grid(agent.location, self.width))
        p2 = p1 + np.array(action)
        if GU.is_valid_grid(p2[0], p2[1], self.height, self.width):
            self.move_agent_to(agent, GU.grid_to_state(p2))
        


    ### OFFLINE ###

    def run_offline_episodes(self, n_episodes):
        """ """
        for _ in range(n_episodes):
            self.run_offline_episode()

    def run_offline_episode(self):
        """ """
        self.reset()
        while not self.in_terminal_state():
            self.run_offlline_step()

    def run_offlline_step(self):
        """ """
        A = self.main_agent.choose_action()
        self.move_agent(self.main_agent, A)
        self.main_agent.learn()


    ### ONLINE ###

    def run_online_step(self, action_name):
        """ """
        # in terminal state
        if self.in_terminal_state():
            self.reset()
        # non terminal state
        else:
            A = self.main_agent.actions.find_by_name(action_name)
            self.move_agent(self.main_agent, A)
            self.main_agent.learn(A)
            self.draw_color_grid()


    ### DRAW SCREEN ###

    def draw_color_grid(self):
        """ default states  """
        self.draw_blank_grid()
        self.draw_color_grid_extra() # GW
        self.draw_terminal_states()
        self.draw_agents()

    def draw_blank_grid(self):
        """ reset color grid to all white """
        self.color_grid = [Colors.WHITE for _ in self.state]

    def draw_color_grid_extra(self):
        """ """
        pass

    def draw_terminal_states(self, color=Colors.ORANGE):
        """ color terminal states """
        for idx in self.terminal_states:
            self.color_grid[idx] = color

    def draw_agents(self):
        """ color agent depending on current state """
        for agent in self.agents.all():
            self.color_grid[agent.state_idx] = agent.get_color() # AG


