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
    ds = load_data(n=100)
    ds = np.reshape(ds, (-1, 28, 28, 1))
    ds = np.pad(ds, ((0,0), (2,2), (2,2), (0,0)), mode='constant',
                  constant_values=0)
    n, h, w, le = ds.shape
    print('Data shape: {}'.format(ds.shape))


    auto_hidden = [16, 16, 16, 16]


    """ AUTO """

    if 0: # CREATE
        AUTO.new(paths, name, h, w, auto_hidden, length=le, patch=3, e=1e-8,
                 with_binary=True, reuse_weights=True, print_me=True,
                 hidden_feedforward=[])

    if 0: # LOAD - NETOWRK
        auto_network = AUTO.load(name, paths.load_json())

    if 0: # TRAIN - DATA
        AUTO.train_data(auto_network, ds, h, w, n_train=200, alpha=1e-3, n_plot=n//2,
                        kmax_img=20, kmax_cost=10, plot_r=True, plot_i=False,
                        do_subdata=True)
    
    if 1: # LEARN
        LA.learn_by_data(paths, ds)

    if 0: # TEST
        for i in range(10):
            pass
