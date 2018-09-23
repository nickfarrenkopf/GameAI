import os
import itertools
import numpy as np
from pynput.keyboard import Key

from Library.General import Screen
from Library.General import DataThings as DT


class Test(object):
    """ """

    def __init__(self, test_path, auto_network):
        """ """

        # location
        self.base_path = test_path
        self.when_network_path = os.path.join(test_path, 'when')
        self.what_network_path = os.path.join(test_path, 'what')
        self.where_network_path = os.path.join(test_path, 'where')

        # network
        self.auto_network = auto_network
        self.network_inpupt_shape = 0

        # screen
        self.window = ((0, 0), (1080, 960))

        # actions
        self.actions = ['left_click', 'right_click', 'scroll', 'type']
        self.n_actions = len(self.actions)


    ### STATE ###

    def get_state(self):
        """ gets current gamestate from screen """
        wind = np.reshape(self.get_window(), (-1, self.height, self.width, 3))
        auto_mid = self.auto_network.get_flat(wind)
        return auto_mid


    ### WINDOW ###

    def get_window(self):
        """ """
        p1, p2 = self.window
        data = Screen.get_data_resized_xy(p1[0], p1[1], p2[0], p2[1],
                                          1024, 1024)
        return data

    def screencap_window(self, name='main_window'):
        """ """
        save_path = os.path.join(self.base_path, '{}.png'.format(name))
        Screen.save_image(self.get_window(), save_path)

    def set_window(self, radius=256):
        """ """
        print('Setting main window for {}...'.format(self.game_name))
        x0, y0 = Screen.get_click(' - click top left')
        x1, y1 = Screen.get_click(' - click bottom right')
        X = np.mean([x0, x1], dtype=np.int)
        Y = np.mean([y0, y1], dtype=np.int)
        x_rad = self.width // 2
        y_rad = self.height // 2
        self.window = ((Y - y_rad, X - x_rad), (Y + y_rad, X + x_rad))
        print(' - set window to {} {}'.format(*self.window))
        self.screencap_window(name='new_window')


