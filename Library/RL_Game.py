import os
import numpy as np
from keras.models import load_model


class RLComponent(object):
    """ """

    def __init__(self, network_path):
        """ """
        self.network_path = network_path
        self.network = None


    ### NETWORK ###

    def load_network(self):
        """ load keras reward network if exists, create otherwise """
        if not os.path.exists(self.network_path):
            self.create_network()
            self.save_network()
        else:
            self.network = load_model(self.network_path)
        
    def save_network(self, save_me):
        """ save reward network """
        if not save_me:
            print('{} network NOT saved'.format(self.name))
            return
        self.network.save(self.network_path)
        print('{} network saved to {}'.format(self.name, self.network_path))


    ### HELPER ###

    def shuffle_mes(self, array1, array2, shuffle_me):
        """ """
        if not shuffle_me:
            return array1, array2
        random_idxs = list(range(len(array1)))
        np.random.shuffle(random_idxs)
        array1 = np.array([array1[i] for i in random_idxs])
        array2 = np.array([array2[i] for i in random_idxs])
        return array1, array2

    
    def combine(self, state, action):
        """ """
        return np.array(list(state) + list(action))
        


class Game(object):
    """ """

    def __init__(self, game_path, auto_network):
        """ """

        self.state_size = (256,)
        self.state_action_size = (257,)

        self.auto_network = auto_network


        self.environment_network_path = 'Fix path in GAME'

        self.n_rewards = 3

        self.image_path = os.path.join(game_path, 'gamedata')
        self.network_path = os.path.join(game_path, 'value_network.h5')
        
        # init THIGNS
        self.window = ((215, 1239), (727, 1751))
    



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



