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
from Library.NeuralNetworks import TrainUtils as TU
from Library.NeuralNetworks.Autoencoder import _AutoencoderAPI as AUTO
from Library.NeuralNetworks.Classifier import _ClassifierAPI as CLASS


### HELPER ###

def load_data(one_hot=True, n=100):
    """ """
    mnist = input_data.read_data_sets(mnist_path, one_hot=one_hot)
    train_pad = mnist.train.images[:n]
    train_labels = mnist.train.labels[:n]
    return train_pad, train_labels


### PROGRAM ###

if __name__ == '__main__':

    # set paths base
    name = 'mnist'
    paths.set_base(name)

    # load mnist data
    mnist_path = os.path.join(paths.base_path, 'data', 't10k-images-idx3-ubyte')
    ds, ls = load_data(n=100)
    ds = DT.pad_me_4d(np.reshape(ds, (-1, 28, 28, 1)), 4, 4)
    print('\nData shape: {}'.format(ds.shape))
    print('Label shape: {}'.format(ls.shape))

    
    """ AUTO """

    auto_hidden = [64, 32, 16]
    n, h, w, le = ds.shape
    h = w = 32

    if 0: # CREATE
        AUTO.new(paths, name, h, w, auto_hidden, length=le, patch=3, e=1e-8,
                 with_binary=False, reuse_weights=True, print_me=True,
                 hidden_feedforward=[])

    if 1: # LOAD - NETOWRK
        auto_network = AUTO.load(name, paths.load_json())

    if 0: # TRAIN - DATA
        AUTO.train_data(auto_network, ds, h, w, n_train=100, alpha=1e-3,
                        n_plot=n//2, plot_r=True, plot_i=False, do_subdata=True,
                        kmax_img=20, kmax_cost=10)

    if 0: # TRAIN FULL - DATA
        AUTO.train_data_full(auto_network, ds, h, w, n_train=200, alpha=1e-3,
                             n_plot=n//2, do_subdata=True, kmax_cost=20)

    if 0: # TEST
        for i in range(10):
            pass


    """ CLASS """

    size = auto_network.flat_size
    class_hidden = [64, 64]
    n_classes = 10

    if 0: # CREATE
        CLASS.new(paths, name, size, class_hidden, n_classes, e=1e-8)

    if 1: # LOAD - NETOWRK
        class_network = CLASS.load(name, paths.load_json())

    if 1: # TRAIN - DATA
        CLASS.train_data_iter(auto_network, class_network, ds, ls, h, w,
                              n_train=1000, alpha=1e-3, kmax_cost=100,
                              do_subdata=True)


