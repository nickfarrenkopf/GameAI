import os
import numpy as np
from keras.models import Sequential, load_model
from keras.layers import Dense

from Library import Screen
from Library import DataThings as dt


class Reward(object):
    """ """

    def __init__(self, environment, agent):
        """ """

        # agent and env
        self.env = environment
        self.agent = agent

        # load network
        self.network_path = agent.reward_network_path
        self.load_network()


    ### NETWORK ###

    def load_network(self):
        """ load keras reward network if exists, create otherwise """
        if not os.path.exists(self.network_path):
            self.create_network()
            self.save_network()
        else:
            self.network = load_model(self.network_path)

    def create_network(self, n_hidden=64, n_layers=2):
        """ create ANN reward nertwork with keras """
        # first layer
        network = Sequential()
        network.add(Dense(n_hidden, activation='relu',
                          input_shape=self.env.state_size))
        # hidden layers
        for _ in range(n_layers - 1):
            network.add(Dense(n_hidden, activation='relu'))
        # last layer
        network.add(Dense(self.n_classes, activation='softmax'))
        network.compile(loss='categorical_crossentropy', optimizer='adam',
                        metrics=['accuracy'])
        self.network = network

    def save_network(self):
        """ save reward network """
        self.network.save(self.network_path)
        print('Reward network saved to {}'.format(self.network_path))


    ### RUN TIME ###

    def get_reward(self):
        """ get prediction given environment gamestate """
        return self.network.predict(self.env.get_gamestate(), verbose=0)


    ### TRAIN - OFFLINE ###

    def train_network_offline(self, epochs=3000):
        """ train keras network with saved gamestate data """
        # get data
        data, _, labels = self.gamedata_files_to_network_inputs()
        flat = self.env.auto_network.get_flat(data)
        # loop and fit
        self.network.fit(flat, labels, epochs=epochs, verbose=2)

    def test_network_offline(self):
        """ ??? """
        # get data
        data, _, hot_labels = self.gamedata_files_to_network_inputs()
        flat = self.env.auto_network.get_flat(data)
        # reshape and fit
        flat = np.reshape(flat, (flat.shape[0], -1))
        labels = np.reshape(hot_labels, (hot_labels.shape[0], -1))
        thing = self.network.evaluate(flat, labels, verbose=0)
        print(thing)

    def gamedata_files_to_network_inputs(self, shuffle_me=True):
        """ load gamestate images based on labeled class data """
        # find gamestate indexes to load
        idxs, labels = self.agent.load_reward_indexes_and_labels()
        _, labels, files = self.add_zero_labels(idxs, labels)
        # shuffle order
        if shuffle_me:
            random_idxs = list(range(len(labels)))
            np.random.shuffle(random_idxs)
            files = np.array([files[i] for i in random_idxs])
            labels = np.array([labels[i] for i in random_idxs])
        # load data and format to network
        data = Screen.load_images(files)
        one_hot = dt.to_one_hot_labels(labels)
        return data, labels, one_hot

    def add_zero_labels(self, idxs, labels):
        """ """
        files = self.env.get_all_gamedata_paths()
        files_to_load = [file for i, file in enumerate(files) if i in idxs]
        for i in range(0, len(files), len(files) // len(labels) - 1):
            if i not in idxs:
                files_to_load.append(files[i])
                labels.append(0)
        return idxs, labels, files_to_load


    ### TRAIN - ONLINE ###  

    def train_network_online(self, epoches=10, alpha=0.000001):
        """ ??? """
        pass

    def test_network_online(self):
        """ """
        pass

    def get_reward_labels(self):
        """ ??? """
        done = False
        all_labels = []
        all_idxs = []
        while not done:
            # take input - Ex. 0 233-246 328-341
            text = input('Label:Indexes - ')
            if ':' in text:
                label, idxs = text.split(':')
                label = int(label)
                # split by space, add each base on type
                idxs = idxs.split(' ')
                change = []
                for idx_text in idxs:
                    # if raw number
                    if '-' in idx_text:
                        a, b = idx_text.split('-')
                        change = list(range(int(a), int(b) + 1))
                        all_idxs += change
                        all_labels += [label] * len(change)
                    else:
                        all_idxs += [int(idx_text)]
                        all_labels += [label]
            done = text == '0'
        return all_idxs, all_labels


