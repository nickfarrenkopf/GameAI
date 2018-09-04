import os
import numpy as np
from Library import Screen

from keras.models import load_model


class Environment(object):
    """ """


    def __init__(self, game_path, game_name, auto_network):
        """ """
        self.game_path = game_path
        self.game_name = game_name

        self.auto_network = auto_network
    
        self.image_path = os.path.join(game_path, 'gamedata')
        self.network_path = os.path.join(game_path, 'value_network.h5')
        #self.auto_network = load_model(self.network_path)

        self.window = ((215, 1239), (727, 1751))
        self.state_size = (256,)



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

    def get_gamedata_paths(self):
        """ """
        path = self.image_path
        return [os.path.join(path, file) for file in os.listdir(path)]


    ### NETWORK ###

    def train_model_network(self, state_data, action_data):
        """ """
        n_actions = len(action_data) - 1
        for i in range(n_actions):
            data = state_data[i] + action_data[i]
            label = state_data[i + 1]
            network.train(data, labels) ####


    






