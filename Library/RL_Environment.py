import os
import numpy as np
from Library import Screen

from keras.models import load_model


class Environment(object):
    """ """

    def __init__(self, game_path, auto_network):
        """ """

        # input params
        self.base_path, self.game_name = os.path.split(game_path)
        self.auto_network = auto_network

        # extra files
        self.image_path = os.path.join(game_path, 'gamedata')
        self.network_path = os.path.join(game_path, 'value_network.h5')
        
        # init THIGNS
        self.window = ((215, 1239), (727, 1751))
        self.state_size = (256,)
        self.state_action_size = (257,)

        

        # load network
        self.load_network()



    def load_env_info(self):
        """ """
        path = os.path.join(self.game_path, 'environment_info.txt')
        dt.read_file(path)

        self.reward_n_classes = 0
        self.reward_label_to_int = 0



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


    ### NETWORK ###
    
    def load_network(self):
        """ load keras environment network if exists, create otherwise """
        if not os.path.exists(self.network_path):
            self.create_network()
            self.save_network()
        else:
            self.network = load_model(self.network_path)

    def create_network(self, n_hidden=64, n_layers=2):
        """ """
        # first layer
        network = Sequential()
        network.add(Dense(n_hidden, activation='relu',
                          input_shape=self.state_action_size))
        # hidden layers
        for _ in range(n_layers - 1):
            network.add(Dense(n_hidden, activation='relu'))            
        # last layer
        network.add(Dense(self.state_size[0], activation='relu'))
        network.compile(loss='sgd', optimizer='adam')
        self.network = network

    def save_network(self):
        """ save environment network """
        self.network.save(self.network_path)
        print('Environment network saved to {}'.format(self.network_path))
        

    ### TRAIN OFFLINE ###

    def train_newtork_offline(self, epochs=100):
        """ """
        n_actions = len(action_data) - 1
        for i in range(n_actions):
            data = state_data[i] + action_data[i]
            label = state_data[i + 1]
            network.train(data, labels) ####

    def test_network_offline(self):
        """ """
        pass


    ### TRAIN ONLINE ###
        
    def train_newtork_online(self, epochs=10):
        """ """
        pass

    def test_network_online(self):
        """ """
        pass   


