import os
import random
import numpy as np
from keras.models import Sequential, load_model
from keras.layers import Dense

from Library import Screen
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
        self.text_info_path = os.path.join(self.env.game_path, 'rewards.txt')

        self.load_network()


    ### NETWORK ###

    def load_network(self):
        """ load keras reward network if exists, create otherwise """
        if not os.path.exists(self.network_path):
            self.create_network()
        else:
            self.network = load_model(self.network_path)

    def create_network(self, n_hidden=64, n_n_hidden=2):
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
        network.save(self.network_path)
        self.network = network

    def save_network(self):
        """ """
        self.network.save(self.network_path)
        print('Network saved to {}'.format(self.network_path))


    ### RUN ###

    def get_reward(self):
        """ get prediction for keras network """
        return self.network.predict(self.env.get_gamestate(), verbose=0)

    def test_network(self):
        """ """
        # get data
        data, _, hot_labels = self.gamedata_files_to_network_inputs()
        flat = self.env.auto_network.get_flat(data)
        # reshape and fit
        flat = np.reshape(flat, (flat.shape[0], -1))
        labels = np.reshape(hot_labels, (hot_labels.shape[0], -1))
        thing = self.network.evaluate(flat, labels, verbose=0)
        print(thing)


    ### TRAIN - OFFLINE ###

    def train_network_offline(self, epochs=3000):
        """ train keras network """
        # get data
        data, cold_labels, hot_labels = self.gamedata_files_to_network_inputs()
        flat = self.env.auto_network.get_flat(data)
        # reshape and fit
        flat = np.reshape(flat, (flat.shape[0], -1))
        labels = np.reshape(hot_labels, (hot_labels.shape[0], -1))
        
        self.network.fit(flat, labels, epochs=epochs, verbose=2)

    def gamedata_files_to_network_inputs(self, shuffle_me=True):
        """ """
        # load base data
        with open(self.text_info_path) as file:
            text_info = file.read().split('\n')
        idxs, labels = self.parse_reward_label_text(text_info)
        # which files to load then add zero data
        files = self.env.get_gamedata_paths()
        files_to_load = [file for i, file in enumerate(files) if i in idxs]
        for i in range(0, len(files), len(files) // len(labels) - 1):
            if i not in idxs:
                files_to_load.append(files[i])
                labels.append(0)
        # load data
        data = np.array([Screen.load_image(file) for file in files_to_load])
        labels = np.array(labels)
        label_set = list(sorted(set(labels)))
        one_hot = np.array([dt.new_label(label_set.index(label), len(label_set))
                            for label in labels])
        # shuffle order
        if shuffle_me:
            random_idxs = list(range(len(data)))
            np.random.shuffle(random_idxs)
            data = np.array([data[i] for i in random_idxs])
            labels = np.array([labels[i] for i in random_idxs])
            one_hot = np.array([one_hot[i] for i in random_idxs])
        return data, labels, one_hot

    def parse_reward_label_text(self, text_data):
        """ """
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


