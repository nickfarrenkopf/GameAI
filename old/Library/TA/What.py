import os
import numpy as np
from keras.models import Sequential
from keras.layers import Dense

from Library.General import DataThings as DT
from Library.TA import KerasNetwork as KN


class What(KN.KerasNetwork):
    """ WHAT NETWORK """

    def __init__(self, test, network_path):
        """ """

        # load network
        KN.KerasNetwork.__init__(self, test, test.what_network_path)
        self.load_network()


    ### NETWORK ###

    def create_network(self, n_hidden=64, n_layers=2):
        """ create ANN reward nertwork with keras """
        # first layer
        network = Sequential()
        network.add(Dense(n_hidden, activation='relu',
                          input_shape=self.test.network_input_shape))
        # hidden layers
        for _ in range(n_layers - 1):
            network.add(Dense(n_hidden, activation='relu'))
        # last layer
        network.add(Dense(self.env.n_actions, activation='softmax'))
        network.compile(loss='categorical_crossentropy', optimizer='adam',
                        metrics=['accuracy'])
        self.network = network


    ### TRAIN - ONLINE ###  

    def train_network_online(self, epochs=10):
        """ """
        pass


