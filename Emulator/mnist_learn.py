import os
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data

import tensorflow as tf
old_v = tf.logging.get_verbosity()
tf.logging.set_verbosity(tf.logging.ERROR)

import paths

import sys
sys.path.append('C:\\Users\\Nick\\Desktop\\Ava\\Programs')
from Library.General import DataThings as DT
from Library.General import FileThings as FT
from Library.NeuralNetworks import NetworkThings as NETS
from Library.NeuralNetworks.Autoencoder import AutoencoderAPI as AUTO
from Library.NeuralNetworks.Autoencoder import LearnAutoencoder as LA


### HELPER ###

def load_data(one_hot=False, n=100):
    """ """
    mnist = input_data.read_data_sets(mnist_path, one_hot=one_hot)
    train_pad = mnist.train.images[:n]
    return train_pad


### PROGRAM ###

if __name__ == '__main__':

    # set paths base
    name = 'mnist'
    paths.set_base(name)

    # auto network
    auto_hidden = [128, 64, 32, 16]
    
    # load mnist data
    mnist_path = os.path.join(paths.base_path, 'data', 't10k-images-idx3-ubyte')
    ds = DT.pad_me_4d(np.reshape(load_data(n=100), (-1, 28, 28, 1)), 4, 4)

    # auto data
    n, h, w, le = ds.shape
    h = w = 32
    print('\nData shape: {}'.format(ds.shape))

    
    """ AUTO """

    if 1: # CREATE
        AUTO.new(paths, name, h, w, auto_hidden, length=le, patch=3, e=1e-8,
                 with_binary=False, reuse_weights=True, print_me=True,
                 hidden_feedforward=[])

    if 1: # LOAD - NETOWRK
        auto_network = AUTO.load(name, paths.load_json())

    if 1: # TRAIN - DATA
        AUTO.train_data(auto_network, ds, h, w, n_train=200, alpha=1e-2,
                        n_plot=n//2, plot_r=True, plot_i=False, do_subdata=True,
                        kmax_img=20, kmax_cost=10)
    
    if 0: # LEARN
        LA.learn_by_data(paths, ds)

    if 0: # TEST
        for i in range(10):
            pass


