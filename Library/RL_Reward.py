import os
import numpy as np
from PIL import Image
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout



class Reward(object):
    """ """

    def __init__(self, environment, agent):
        """ """
        self.env = environment
        self.agent = agent
        self.n_classes = 3

        self.network_path = agent.reward_network_path
        self.load_reward_network()


    ### NETWORK ###

    def load_reward_network(self):
        """ load keras reward network """
        if not os.path.exists(self.network_path):
            self.create_reward_network()
        else:
            self.network = load_model(self.network_path)

    def create_reward_network(self, n_hidden=32):
        """ create ANN reward nertwork with keras """
        network = Sequential()
        network.add(Dense(n_hidden, activation='relu',
                          input_shape=self.env.state_size))
        network.add(Dense(n_hidden, activation='relu'))
        network.add(Dense(self.n_classes, activation='softmax'))
        network.compile(loss='categorical_crossentropy', optimizer='adam',
                        metrics=['accuracy'])
        network.save(self.network_path)
        self.network = network

    def train_network(self):
        """ train keras network """
        # load files
        # label images with labels
        data, labels, idxs = self.files_to_labeled()
        lab_set = list(sorted(set(labels)))
        labels = [self.to_label(lab_set.index(l), 3) for l in labels]
        mids = self.env.auto_network.get_flat(data)
        #print(mids.shape)
        mids = np.reshape(mids, (mids.shape[0], mids.shape[1]))
        labels = np.reshape(labels, (mids.shape[0], -1))
        self.network.fit(mids, labels, epochs=1000, verbose=2)
        #self.network.save(self.network_path)
        print('Netwkr trained')

    def to_label(self, idx, lens):
        """ """
        label = np.zeros(lens)
        label[idx] = 1
        return label
                   

    ### RUN ###

    def get_reward(self):
        """ get prediction for keras network """
        pred = self.network.predict(self.env.get_gamestate(), verbose=0)
        return pred


    ### HELPER ###

    def files_to_labeled(self, shuffle_me=True):
        """ """
        files = self.env.get_gamedata_paths()
        imgs = [Image.open(file) for file in files]
        idxs, labels = self.get_reward_labels()
        data = [np.array(imgs[i]) / 255 for i in idxs]
        for i in range(0, len(imgs), len(imgs) // len(data)):
            if i not in idxs:
                data.append(np.array(imgs[i]) / 255)
                labels.append(0)
        # shuffle to array
        if shuffle_me:
            random_idxs = list(range(len(data)))
            np.random.shuffle(random_idxs)
            data = np.array([data[i] for i in random_idxs])
            labels = np.array([labels[i] for i in random_idxs])
        else:
            data = np.array(data)
            labels = np.array(labels)
        return data, labels, idxs
        

    def get_reward_labels(self):
        """ """
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


    
