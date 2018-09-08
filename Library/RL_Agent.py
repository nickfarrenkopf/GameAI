import os
from os.path import join
from keras.models import load_model

from Library import DataThings as dt


class Agent(object):
    """

    Agent is pacman player 1
    Agent is pacman player 2
    Agent is digdug player
    Agent is aw2 player

    """

    def __init__(self, game_path, environment):
        """ """

        # environment
        self.env = environment
        self.base_path, self.game_name = os.path.split(game_path)
        
        # file location
        self.info_path = os.path.join(game_path, 'agents.txt')
        self.network_path = os.path.join(game_path, 'value_network.h5')
        self.network = load_model(self.network_path)

        self.reward_network_path = join(game_path, 'reward_network.h5')

        # agent info
        #self.agent_id, self.agent_name, self.actions = self.load_text()

        self.text_info_path = join(game_path, 'reward_labels.txt')
        self.reward_labels = 0

        # load network
        self.load_network()


    ### NETWORK ###

    def load_network(self):
        """ load keras agent network if exists, create otherwise """
        if not os.path.exists(self.network_path):
            self.create_network()
            self.save_network()
        else:
            self.network = load_model(self.network_path)

    def create_network(self, n_hidden=64, n_layers=2):
        """ ??? """
        # first layer
        network = Sequential()
        network.add(Dense(n_hidden, activation='relu',
                          input_shape=self.env.state_action_size))
        # hidden layers
        for _ in range(n_layers - 1):
            network.add(Dense(n_hidden, activation='relu'))
        # last layer
        network.add(Dense(self.env.state_size[0], activation='relu'))
        network.compile(loss='sgd', optimizer='adam')
        self.network = network

    def save_network(self):
        """ save agent network """
        self.network.save(self.network_path)
        print('Agent network saved to {}'.format(self.network_path))
    

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
        """ """
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


    ### ACTION ###

    def choose_action(self, gamestate):
        """ """
        pass

    def take_action(self, action):
        """ """
        if action not in self.actions:
            print('Action not in list: {}'.format(self.actions))
            return False
        Screen.send_keys(a)



    ### TRAIN OFFLINE ###

    def train_newtork_offline(self, epochs=100):
        """ """
        pass

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

  
