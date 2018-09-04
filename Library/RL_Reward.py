import os
import random
import numpy as np
from keras.models import Sequential, load_model
from keras.layers import Dense

import Screen
import data_things as dt


class Reward(object):
    """ """

    def __init__(self, environment, agent, n_classes=3):
        """ RL agents, RL dependent params, network """
        self.env = environment
        self.agent = agent

        self.n_classes = 3
        self.class_list = 0
        self.network_path = agent.reward_network_path
        
        self.load_network()


    ### NETWORK ###

    def load_network(self):
        """ load keras reward network if exists, create otherwise """
        if not os.path.exists(self.network_path):
            self.create_network()
        else:
            self.network = load_model(self.network_path)

    def create_network(self, n_hidden=32, n_n_hidden=2):
        """ create ANN reward nertwork with keras """
        # first layer
        network = Sequential()
        network.add(Dense(n_hidden, activation='relu',
                          input_shape=self.env.state_size))
        # hidden layers
        for _ in range(n_n_hidden - 1):
            network.add(Dense(n_hidden, activation='relu'))
        # last layer
        network.add(Dense(self.n_classes, activation='softmax'))
        network.compile(loss='categorical_crossentropy', optimizer='adam',
                        metrics=['accuracy'])
        # save network
        self.save_network()
        self.network = network

    def save_network(self):
        """ """
        self.network.save(self.network_path)
        print('Network saved to {}'.format(self.network_path))


    ### RUN ###

    def get_reward(self):
        """ get prediction for keras network """
        return self.network.predict(self.env.get_gamestate(), verbose=0)


    ### TRAIN - OFFLINE ###

    def train_network_offline(self, epochs=1000):
        """ train keras network """
        # get data
        data, cold_labels, _ = self.gamedata_files_to_network_inputs()
        labels = dt.to_one_hot(cold_labels, n_classes=self.n_classes)
        flat = self.env.auto_network.get_flat(data)
        # reshape and fit
        flat = np.reshape(flat, (flat.shape[0], -1))
        labels = np.reshape(labels, (labels.shape[0], -1))
        self.network.fit(flat, labels, epochs=epochs, verbose=2)

    def gamedata_files_to_network_inputs(self, shuffle_me=True):
        """ """
        # find files
        files = self.env.get_gamedata_filepaths()
        labels, idxs = self.load_reward_text_data()
        # add zero data
        files_to_load = [file for i, file in enumerate(files) if i in idxs]
        for i in range(0, len(files), len(files) // len(labels) - 1):
            if i not in idxs:
                files_to_load.append(files[i])
                labels.append(0)
        # load data
        data = np.array([Screen.load_image(file) for file in files_to_load])
        labels = np.array(labels)
        # shuffle order
        if shuffle_me:
            combined = list(zip(data, labels))
            random.shuffle(combined)
            data[:], labels[:] = zip(*combined)
        return data, labels, idxs




    def load_reward_text_data():
        """ """

        # load reward data from text file!!!
        
        labels = 0
        idxs = 0
        return labels, idxs


    ### TRAIN - ONLINE ###  

    def train_network_online(self, epoches=10, alpha=0.000001):
        """ """
        pass

    def get_reward_labels(self):
        """ WUT """
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



