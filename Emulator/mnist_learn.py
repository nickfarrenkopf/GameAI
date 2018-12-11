import os
import math
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data

import paths

import sys
sys.path.append('C:\\Users\\Nick\\Desktop\\Ava\\Programs')
from Library.General import DataThings as DT
from Library.General import FileThings as FT
from Library.NeuralNetworks import NetworkThings as NETS
from Library.NeuralNetworks.Autoencoder import AutoencoderAPI as AUTO
from Library.NeuralNetworks.Autoencoder import TrainAutoencoder as AUTO_T
from Library.NeuralNetworks.Autoencoder import LearnAutoencoder as LA
from Library.NeuralNetworks.Regression import RegressionAPI as REG


### ###

def load_data(one_hot=False, n=10000):
    """ """
    mnist = input_data.read_data_sets(mnist_path, one_hot=one_hot)
    train_pad = mnist.train.images[:n]
    return train_pad


### PROGRAM ###

if __name__ == '__main__':

    # set paths base
    name = 'mnist_test'
    paths.set_base(name)
    
    # load mnist data
    mnist_path = os.path.join(paths.base_path, 'data', 't10k-images-idx3-ubyte')
    data = load_data(n=100)
    data = np.reshape(data, (-1, 28, 28, 1))
    n, h, w, le = data.shape
    print('Data shape: {}'.format(data.shape))


    auto_hidden = [16, 8]


    """ AUTO """

    if 0: # LOAD
        auto_network = AUTO.load_auto(name, paths.load_json())

    if 0: # CREATE
        AUTO.new(paths, name, h, w, auto_hidden, length=1, with_binary=False)

    if 0: # TRAIN - DATA
        auto_network = AUTO.load(name, paths.load_json())
        AUTO.train_data(auto_network, data, h, w, n_train=1000,
                             kmax_img=100, kmax_cost=50, alpha=0.001)

    if 0: # TRAIN - PATHS
        data_path = paths.get_game_images()
        print('Number files: {}'.format(len(data_path)))
        auto_network = AUTO.load_auto(name, json_data)
        AUTO.train_auto_paths(auto_network, data_path,
                              h, w, h_d, w_d, n_train=500, alpha=0.001, n_plot=n,
                              kmax_img=50, kmax_cost=10)

    if 1: # LEARN
        LA.learn_by_data(paths, data)

    if 0: # TEST
        for i in range(10):
            fp = NETS.get_subset(paths.get_game_images(), 64, True)
            ds = FT.load_images_4d(fp, h_d, w_d, le)
            ds = DT.subdata(ds, h, w)
            AUTO_T.check_auto(auto_network, ds, h, w, 16, i, 1)
