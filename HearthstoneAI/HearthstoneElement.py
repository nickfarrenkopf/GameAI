import numpy as np
import tensorflow as tf
from os.path import join

import paths
import HearthstoneConstants as HC

import sys
sys.path.append('C:\\Users\\Nick\\Desktop\\Ava\\Programs')
from Library.General import DataThings as DT
from Library.General import FileThings as FT
from Library.Computer import Screen
from Library.NeuralNetworks.Autoencoder import _AutoencoderAPI as AUTO


### WATCHER ###

class HearthstoneElement(object):
    """ """

    def __init__(self, name):
        """ """

        # params
        self.name = name
        self.network_name = HC.json_data[name]['network_name']
        self.x, self.y = HC.json_data[name]['center']
        self.h, self.w = HC.json_data[name]['image_size']

        # network params
        self.w_n, self.h_n = HC.networks[self.network_name]['network_size']
        self.n_latent = HC.networks[self.network_name]['n_latent']
        self.every_n = HC.networks[self.network_name]['every_n']
        self.auto_network = None
        self.class_network = None

        # computer screen
        self.p1 = [self.x - self.w // 2, self.y - self.h // 2]
        self.p2 = [self.x + self.w // 2, self.y + self.h // 2]
        self.window = Screen.Window(self.p1, self.p2)


    ### WINDOW ###

    def get_window(self):
        """ """
        data = self.window.get_window()[0]
        data = DT.resize_image(data, self.w_n, self.h_n)
        return data

    def save_window(self, i=0, j=0):
        """ """
        filename = filepath = '{}_{}_{}.png'.format(self.name, i, j)
        if i != 0:
            filepath = join(paths.images_path, self.network_name, filename)
        FT.save_image_to_file(self.get_window(), filepath)

    def parse_window(self, data):
        """ """
        ds = DT.subdata_xy(data, self.h, self.w, self.x, self.y)
        ds = DT.resize_image(ds[0], self.w_n, self.h_n)
        return ds


    ### NETWORK ###

    def create_network(self):
        """ """
        # latent size < 0
        if self.n_latent == 400000:
            hidden_encode = [2, 3, 4]
            pools_encode = [2, 2, 2]
            hidden_decode = [5, 4, 3]
            pools_decode = [1, 1, 1]
            hidden_dense = [128]
            b = 1.1
        # latent size < 4
        if self.n_latent <= 4:
            hidden_encode = [2, 2, 4, 4]
            pools_encode = [1, 2, 1, 2]
            hidden_decode = []
            pools_decode = []
            hidden_dense = [512]
            b = 1.1
        # latent size 8
        elif self.n_latent == 8:
            hidden_encode = [3, 4, 5, 5, 6, 6]
            pools_encode = [2, 2, 1, 2, 1, 2]
            hidden_decode = [5, 4, 3, 2]
            pools_decode = [1, 1, 1, 2]
            hidden_dense = [512, 512]
            b = 1.1
        # latent size other
        elif self.n_latent == 16:
            hidden_encode = [4, 4, 5, 5, 6, 6]
            pools_encode = [1, 2, 1, 2, 1, 2]
            hidden_decode = [5, 4, 3, 2]
            pools_decode = [1, 1, 1]
            hidden_dense = [512]
            b = 1.1
        # latent size other
        elif self.n_latent == 32:
            hidden_encode = [3, 4, 4, 5, 5, 6, 6]
            pools_encode = [2, 2, 1, 2, 2, 1, 2]
            hidden_decode = [5, 4, 3, 2]
            pools_decode = [1, 1, 1, 1]
            hidden_dense = [512, 512]
            b = 1.1
        # power of 2
        hidden_encode = [2 ** i for i in hidden_encode]
        hidden_decode = [2 ** i for i in hidden_decode]
        AUTO.new(paths, self.network_name, self.w_n, self.h_n, length=3, patch=3,
                 hidden_encode=hidden_encode, pools_encode=pools_encode,
                 hidden_latent=hidden_dense, n_latent=self.n_latent, b=b,
                 pools_decode=pools_decode, hidden_decode=hidden_decode)

    def load_network(self, watchers):
        """ """
        # create if doesn't exist
        if self.network_name not in paths.load_json()['network']['auto']:
            self.create_network()
        # check if already loaded
        loaded = [w.network_name for w in watchers if w.auto_network != None]
        if self.network_name in loaded:
            idx = loaded.index(self.network_name)
            self.auto_network = watchers[idx].auto_network
        # load network if not loaded
        else:
            print('{} network loaded'.format(self.network_name))
            self.auto_network = AUTO.load(self.network_name, paths.load_json())


    ### ONLINE ###
        
    def train_auto(self, data, n_train=32, a=1e-4, b=1.1):
        """ """
        data = self.parse_window(data)
        for i in range(n_train):
            self.auto_network.train_network(data, a, b)
        loss = self.auto_network.get_loss(data)
        return loss


### HELPER ###

def load_elements(wanted, networks=False):
    """ """
    elements = [HearthstoneElement(n) for n in HC.elements if n in wanted]
    if networks:
        for e in elements:
            e.load_network(elements)
    return elements


### PROGRAM ###

if __name__ == '__main__':

    # load screen watchers
    #elements = load_elements(HC.elements, networks=False)
    elements = load_elements(['manaS', 'manaO'], networks=False)

    # screen cap
    if 0:
        for e in elements:
            e.save_window()


