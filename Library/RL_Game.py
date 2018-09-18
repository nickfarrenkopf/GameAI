import os
from os.path import join
import itertools
import numpy as np

from pynput.keyboard import Key

from Library.General import Screen
from Library.General import DataThings as DT


class Game(object):
    """ """

    def __init__(self, game_path, auto_network):
        """ """

        # base params
        self.game_path = game_path
        self.game_name = os.path.basename(game_path)
        self.auto_network = auto_network
        
        # environment
        self.window = ((215, 1239), (727, 1751))
        self.state_size = (256,)
        self.state_action_size = (257,)


        # location
        self.images_path = join(game_path, 'images')
        self.state_path = join(game_path, 'gamedata')


        self.action_keys = [Key.left,Key.right,Key.up,Key.down,'x', 'z','w','q']
        self.label_keys = [Key.enter, Key.space]

        self.action_labels = ['left','right','down','up','x', 'z','w','q']

        # locations
        self.environment_network_path = join(game_path, 'reward_network.h5')
        self.value_network_path = join(game_path, 'value_network.h5')
        
        # rewards
        self.n_rewards = 3
        self.label_dict = {'enter':-10, 'space':1, 'left':0, 'right':0, 'up':0,
                      'down':0}
        
   

    ### FIELS ###

    """
    ?
    """

    def find_game_data(self, labels):
        """ """
        all_idxs = []
        all_labs = []
        for label in labels:
            idxs, labs = self.load_labels(label)
            all_idxs += idxs
            all_labs += (label * len(labs))
        return all_idxs, all_labs

    def load_labels(self, label):
        """ """
        print(label)
        data = os.listdir(self.state_path)
        idxs = []
        for row in data:
            split = row.split('_')
            if label == split[-2]:
                idxs.append(int(split[1]))
        return idxs, [label] * len(idxs)


    ### WINDOW ###

    """
    self.window
    self.game_name
    self.images_path
    Screen
    """

    def get_window(self):
        """ """
        return Screen.get_data_box(*itertools.chain.from_iterable(self.window))

    def set_window(self, radius=256):
        """ """
        print('Setting main window for {}...'.format(self.game_name))
        x0, y0 = Screen.get_click(' - click top left')
        x1, y1 = Screen.get_click(' - click bottom right')
        X = np.mean([x0, x1], dtype=np.int)
        Y = np.mean([y0, y1], dtype=np.int)
        self.window = ((Y - radius, X - radius), (Y + radius, X + radius))
        print(' - set window to {} {}'.format(*self.window))
        self.screencap_window(name='new_window')

    def screencap_window(self, name='main_window'):
        """ """
        new_path = os.path.join(self.state_path, '{}.png'.format(name))
        Screen.save_image(self.get_window(), new_path)


    ### GAMESTATE ###

    """
    self.auto_network
    self.state_path
    """

    def get_gamestate(self):
        """ gets current gamestate from screen """
        return self.auto_network.get_flat(self.get_window())

    def get_all_gamedata_paths(self):
        """ returns file locations for all gamestates """
        return [join(self.state_path, f) for f in os.listdir(self.state_path)]


    ### HELPER ###

    """
    NONE
    """

    def combine(self, state, action):
        """ combines gamestate and action into one array """
        return np.array(list(state) + list(action))

    def shuffle_mes(self, array1, array2, shuffle_me):
        """ shuffle two arrays (data and labels) together """
        if not shuffle_me:
            return array1, array2
        random_idxs = list(range(len(array1)))
        np.random.shuffle(random_idxs)
        array1 = np.array([array1[i] for i in random_idxs])
        array2 = np.array([array2[i] for i in random_idxs])
        return array1, array2
    

