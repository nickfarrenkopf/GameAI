import os
import numpy as np

from Library.General import Screen
from Library.General import DataThings as DT



class Game(object):
    """ """

    def __init__(self, game_path, auto_network):
        """ """

        self.state_size = (256,)
        self.state_action_size = (257,)

        self.auto_network = auto_network


        self.environment_network_path = os.path.join(game_data,
                                                     'reward_network.h5')

        self.n_rewards = 3

        self.game_path = game_path
        self.image_path = os.path.join(game_path, 'gamedata')
        self.network_path = os.path.join(game_path, 'value_network.h5')
        
        # init THIGNS
        self.window = ((215, 1239), (727, 1751))
    


    def load_labels(self, label):
        """ """
        data = os.listdir(self.image_path)
        idxs = []
        for row in data:
            split = row.split('.')[0].split('_')
            if label == split[-1]:
                idxs.append(int(split[1]))
        return idxs, [label] * len(idxs)

    ### WINDOW ###

    def get_window(self):
        """ """
        w = self.window
        return Screen.get_data()[w[0][0]:w[1][0], w[0][1]:w[1][1], :]

    def set_window(self, radius=256):
        """ """
        print('Setting main window for {}...'.format(self.game_name))
        x0, y0 = Screen.get_click(' - click top left')
        x1, y1 = Screen.get_click(' - click bottom right')
        X = np.mean([x0, x1], dtype=np.int)
        Y = np.mean([y0, y1], dtype=np.int)
        self.window = ((Y - radius, X - radius), (Y + radius, X + radius))
        print(' - set window to {} {}'.format(*self.window))
        self.screencap_window()

    def screencap_window(self, name='main_window'):
        """ """
        new_path = os.path.join(self.image_path, '{}.png'.format(name))
        Screen.save_image(self.get_window(), new_path)


    ### GAMESTATE ###

    def get_gamestate(self):
        """ """
        data = self.get_window()
        flat = self.auto_network.get_flat(np.reshape(data, (1, 512, 512, 3)))
        return flat

    def get_all_gamedata_paths(self):
        """ """
        path = self.image_path
        return [os.path.join(path, file) for file in os.listdir(path)]



