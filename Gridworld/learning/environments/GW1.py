from learning import GridworldEnvironment as GWE
from learning import GridworldAgent as GWA


class Gridworld_1(GWE.Gridworld):
    """ Win if top-left or bottom-right corner """

    def __init__(self, paths):
        """ Gridworld size 5x5 """
        GWE.Gridworld.__init__(self, paths, 'gridworld_1', 5, 5)

        self.set_initial_state = self.default_initial_state


    ### OVERRIDES ###

    def set_terminal_states(self):
        """ top-left and bottom right corners """
        self.terminal_states = [0, self.state_size - 1]


### AGENTS ###

def GW1_MainAgent(GWA.GridworldAgent):
    """ """

    def __init__(self, environment, key):
        """ """
        GWA.GridworldAgent.__init__(self, environment, key)
        self.actions = GridworldAction.actionset_1
        self.rewards = GridworldReward.rewardset_1


    ### REWARDS ###

    def get_reward(self):
        """ """
        if self.in_terminal_state():
            return 1
        return -1





