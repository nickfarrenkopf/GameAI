import GridworldUtils as GWU
from learning import GridworldAction
from learning import GridworldAgent as GWA
from learning import GridworldEnvironment as GWE
from learning import GridworldReward

import sys
sys.path.append('C:\\Users\\Nick\\Desktop\\Ava\\Programs')
from Library.Learning import AgentUtils


### GRIDWORLD ###

class Gridworld_1(GWE.Gridworld):
    """ Win if top-left or bottom-right corner """

    def __init__(self, paths):
        """ Gridworld size 5x5 """
        GWE.Gridworld.__init__(self, paths, 'gridworld_1', 5, 5)

        self.set_gridworld_agents()


    ### OVERRIDES ###

    def set_gridworld_agents(self):
        """ """
        self.agents = AgentUtils.AgentList([GW1_MainAgent(self, 1)])
        self.main_agent = self.agents.first()

    def set_terminal_states(self):
        """ top-left and bottom right corners """
        self.terminal_states = [0, self.state_size - 1]


### AGENTS ###

class GW1_MainAgent(GWA.GridworldAgent):
    """ """

    def __init__(self, environment, key):
        """ """
        self.actions = GridworldAction.actionset_1
        GWA.GridworldAgent.__init__(self, environment, key, self.actions)
        self.rewards = GridworldReward.rewardset_1


    ### REWARDS ###

    def reset(self):
        """ """
        self.state_idx = GWU.random_empty_state(self.env.state)
        self.method.first_time_step()

    def get_reward(self):
        """ """
        if self.in_terminal_state():
            return 1
        return 0


