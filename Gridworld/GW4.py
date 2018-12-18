import Gridworld


class Gridworld_4(Gridworld.Gridworld):
    """ wind pushing to the left, with goal inside columns of wind """

    def __init__(self, paths):
        """ Gridworld size 7x10 """
        self.name = 'gridworld_4'
        self.height = 7
        self.width = 10
        self.start_idx = self.width * 3
        self.wind_1 = [3, 4, 5, 8]
        self.wind_2 = [6, 7]
        Gridworld.Gridworld.__init__(self, self.height, self.width, paths)

        self.set_initial_state = self.set_starting_initial_state


    ### OVERRIDES ###

    def set_terminal_states(self):
        """ spot in middle to the right behind the wind """
        self.terminal_states = [self.width * 4 - 3]

    def set_color_grid(self):
        """ start to left, leftward windin middle, goal to the right """
        self.draw_blank_grid()
        g2s = self.grid_to_state
        for i in range(self.height):
            for j in self.wind_1:
                self.color_grid[g2s((i, j))] = self.Colors.LIGHTBLUE
            for j in self.wind_2:
                self.color_grid[g2s((i, j))] = self.Colors.SKYBLUE
        self.draw_terminal_states()
        self.draw_agent()

    def take_action(self, action):
        """ move agent, apply wind, find reward, set new color grid """
        self.apply_wind()
        self.default_take_action(action)

    def get_reward(self):
        """ penalty if not in terminal state """
        if self.in_terminal_state():
            return 1
        return 0


    ### HELPER ###

    def apply_wind(self):
        """ """
        _, x = self.location_of(self.AGENT_VAL, grid=True)
        if x in self.wind_1 + self.wind_2 and not self.in_terminal_state():
            for _ in range(2 if x in self.wind_2 else 1):
                self.agent_take_action(self.AGENT_VAL, (-1, 0))


