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


        self.environment_network_path = os.path.join(game_path,
                                                     'reward_network.h5')

        self.n_rewards = 3

        self.game_path = game_path
        self.image_path = os.path.join(game_path, 'gamedata')
        
        self.network_path = os.path.join(game_path, 'value_network.h5')
        label_dict = {'enter':-10, 'space':1, 'left':0, 'right':0, 'up':0,
                      'down':0}
        
        # init THIGNS
        self.window = ((215, 1239), (727, 1751))


    ### FIELS ###

    def find_game_data(self, labels):
        """ """
        all_idxs = []
        all_labs = []
        for label in labels:
            idxs, labs = self.load_labels(label)
            all_idxs += idxs
            all_labs += (label_dict[label] * len(labs))
        return all_idxs, all_labs

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





    def combine(self, state, action):
        """ """
        return np.array(list(state) + list(action))


    # 

    ### REWARD ###

    def parse_reward_label_text(self, text_data):
        """ parse through labeled class data to return labels and indexes """
        all_labels = []
        all_idxs = []
        # loop over classes
        for class_data in text_data:
            label, text = class_data.split(':')
            # loop over indexes - range vs single
            for idx in text.split(' '):
                if '-' in idx:
                    ints = idx.split('-')
                    new_labels = list(range(int(ints[0]), int(ints[1])))
                else:
                    new_labels = int(idx)
                all_labels += new_labels
                all_idxs += [int(label)] * len(new_labels)
        return all_labels, all_idxs

    def load_reward_indexes_and_labels(self):
        """ FIX ME TO LOAD LABELS BASED ON KEY """
        text_info = dt.read_file(self.text_info_path)
        idxs, labels = self.parse_reward_label_text(text_info)
        

        
        #self.agent_0
        return idxs, labels


    ### FILE ###

    def load_text(self):
        """ FIX ME """
        with open(self.info_path, 'r') as file:
            data = file.read().split('\n')
        for row in data:
            if ':' not in row:
                agent_id = row
            else:
                key, value = row.split(':')
                if key == 'name':
                    name = value
                if key == 'actions':
                    actions = value
        return agent_id, name, actions.split(',')

    def create_text(self, name, actions):
        """ FIX ME """
        index = 1
        new_path = self.info_path.replace('.', '._1')
        data = '\n'.join([index, name, ','.join(actions)])
        with open(new_path, 'w') as file:
            file.write(data)
        print('Created new agent at {}'.format(new_path))




