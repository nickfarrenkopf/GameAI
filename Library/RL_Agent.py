from keras.models import Sequential
from keras.layers import Dense

from Library.General import Screen
from Library.General import DataThings as DT
from Library.RL_Component import RL_Component as RLC


import os

class Agent(RLC):
    """ """

    def __init__(self, game, environment):
        """ """ 

        # RL components
        self.name = 'agent'
        self.game = game
        self.env = environment
        
        # file location
        self.game_path = game.game_path
        self.

        self.reward_network_path = os.path.join(game.game_path, 'reward_network.h5')
        self.network_path = os.path.join(self.game_path, 'agent_network.h5')

        self.text_info_path = os.path.join(game.game_path, 'reward_labels.txt')
        
        #self.reward_labels = 0

        # load network
        #RLC.__init__(self, agent.reward_network_path)
        #self.load_network()


    ### NETWORK ###

    def create_network(self, n_hidden=64, n_layers=2):
        """ ??? """
        # first layer
        network = Sequential()
        network.add(Dense(n_hidden, activation='relu',
                          input_shape=self.game.state_action_size))
        # hidden layers
        for _ in range(n_layers - 1):
            network.add(Dense(n_hidden, activation='relu'))
        # last layer
        network.add(Dense(1, activation='relu'))
        network.compile(loss='mean_squared_error', optimizer='adam')
        self.network = network


    ### ACTION ###

    def choose_action(self):
        """ """
        pass

    def take_action(self, action):
        """ """
        pass


    ### TRAIN OFFLINE ###

    def train_newtork_offline(self, epochs=100, n_loop=10, save_me=False):
        """ """
        pass

    def test_network_offline(self):
        """ """
        pass

    def load_network_inputs(self, shuffle_me=True):
        """ """
        pass
    

    ### TRAIN ONLINE ###
        
    def train_newtork_online(self, epochs=10):
        """ """
        pass

    def test_network_online(self):
        """ """
        pass   




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







  
