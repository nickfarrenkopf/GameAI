import numpy as np
from keras.models import Sequential, load_model
from keras.layers import Dense

from Library.General import DataThings as DT
from Library.RL.Component import Component as RLC
import os

class Reward(object):
    """ """

    def __init__(self, game, agent):
        """ """
        
        # RL components
        self.name = 'reward'
        self.game = game
        self.agent = agent

        # load network
        #RLC.__init__(self, agent.reward_network_path)
        self.network_path = agent.reward_network_path
        self.load_network()

        
    def load_network(self):
        """ load keras reward network if exists, create otherwise """
        if not os.path.exists(self.network_path):
            self.create_network()
            self.save_network(True)
        else:
            self.network = load_model(self.network_path)
        
    def save_network(self, save_me):
        """ save reward network """
        if not save_me:
            print('{} network NOT saved'.format(self.name))
            return
        self.network.save(self.network_path)
        print('{} network saved to {}'.format(self.name, self.network_path))

    def print_metrics(self, data, labels):
        """ """
        metrics = self.network.evaluate(data, labels, verbose=0)
        print('Metrics: {}'.format(metrics))

    ### NETWORK ###

    def create_network(self, n_hidden=64, n_layers=2):
        """ create ANN reward nertwork with keras """
        # first layer
        network = Sequential()
        network.add(Dense(n_hidden, activation='relu',
                          input_shape=(256,)))
        # hidden layers
        for _ in range(n_layers - 1):
            network.add(Dense(n_hidden, activation='relu'))
        # last layer
        network.add(Dense(self.game.n_rewards, activation='softmax'))
        network.compile(loss='categorical_crossentropy', optimizer='adam',
                        metrics=['accuracy'])
        self.network = network


    ### RUN TIME ###

    def predict_reward(self):
        """ get prediction given environment gamestate """
        
        return self.network.predict(self.game.get_gamestate(), verbose=0)


    ### TRAIN - OFFLINE ###

    def train_network_offline(self, epochs=1000, n_loop=100, save_me=False):
        """ train keras network with saved gamestate data """
        _, data, _, labels = self.load_network_data()
        print(data.shape)
        print(labels.shape)
        print(self.network.input_shape)
        for _ in range(n_loop):
            self.network.fit(data, labels, epochs=epochs//n_loop, verbose=0)
            self.print_metrics(data, labels)            
        self.save_network(save_me)

    def test_network_offline(self):
        """ test network against all data """
        _, data, _, labels = self.load_network_data()
        self.print_metrics(data, labels)

    def load_network_data(self, shuffle_me=True):
        """ load gamestate images based on labeled class data """
        # find data based on agent rewards
        idxs, labels = self.game.find_game_data(self.game.reward_labels)
        _, labels, files = self.add_zero_reward_labels(idxs, labels)
        labels = np.array(labels)
        #files, labels = self.game.shuffle_mes(files, labels, shuffle_me)
        # load data and format to network
        datas = DT.load_datas(files)
        data = self.game.auto_network.get_flat(datas)
        one_hot = DT.to_one_hot(labels, n_classes=3)
        return datas, data, labels, one_hot

    def add_zero_reward_labels(self, idxs, labels):
        """ add zero reward labels given pre-labeled indexes """
        files = self.game.get_all_gamedata_paths()
        files_to_load = [files[i] for i in idxs]
        # loop over files and add every n in not in indexes
        for i in range(0, len(files), len(files) // len(labels) - 1):
            if i not in idxs:
                files_to_load.append(files[i])
                labels.append(0)
        return idxs, labels, files_to_load


    ### TRAIN - ONLINE ###  

    def train_network_online(self, epochs=10):
        """ """
        pass

    def test_network_online(self):
        """ """
        pass

