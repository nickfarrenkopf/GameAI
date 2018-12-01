import os
import numpy as np
from keras.models import load_model


class Component(object):
    """ """

    def __init__(self, network_path):
        """ """
        self.network_path = network_path


    ### NETWORK ###

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


