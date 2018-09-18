import os
import time
import random
import itertools
import numpy as np
import matplotlib.pyplot as plt

from Library.General import DataThings as DT
from Library.Network import Autoencoder
from Library.Network import Networks


### HELPER ###

def get_cost_slope(network, data, labels, costs):
    """ returns array of costs and their ~average """
    costs.append(network.get_cost(data, labels)
                 if data.shape != labels.shape else network.get_cost(data))
    if len(costs) > 1:
        costs = costs[1:] if len(costs) > 30 else costs
        avg, _ = np.polyfit(range(len(costs)), costs, 1)
        return costs, avg
    return costs, 0

def check_save(network, k, k_max):
    """ check to save network """
    if k * k_max != 0 and k % k_max == 0:
        network.save_network(step=k)



"""
--- AUTOENCODER ---
"""

### FILE ###

new_auto = Autoencoder.create
load_auto = Networks.NetworkAuto


### TRAIN ###
import tensorflow as tf
import random

def train_auto(network, filepaths, h, w, n_train=1000, alpha=0.0001, n_plot=20,
               kmax_cost=10, kmax_img=100, kmax_save=0):
    """ iterate through data set to train autoencoder newtork """
    costs = []
    #network.sess.run(tf.initialize_all_variables())
    for k in range(n_train):
        idxs = random.sample(range(len(filepaths)), 100)
        data = DT.load_images([filepaths[i] for i in idxs])
        data = DT.pad_me(data, 4, 4)
        subdata = DT.subdata(data, 512, 512)
        #subdata = data
        network.train_network(subdata, alpha)
        costs = check_cost(network, subdata, subdata, costs, k, kmax_cost)
        check_auto(network, data, subdata, h, w, k, kmax_img, n_plot)
        check_save(network, k, kmax_save)

def check_cost(network, input_data, output_data, costs, k, k_max):
    """ check to print cost """
    if k % k_max == 0 and k != 0:
        costs, m = get_cost_slope(network, input_data, output_data, costs)
        print('Cost {} {:.7f} {:.7f}'.format(k, costs[-1], m))
    return costs


def check_auto(network, data, input_data, h, w, k, k_max, n_plot):
    """ """
    if k_max != 0 and k % k_max == 0:
        idxs = random.sample(range(len(input_data)), n_plot)
        ds = np.array([input_data[i] for i in idxs])
        plot_middle(network, ds, h, w, n_plot=n_plot, count=k)


### HELPER ###

def plot_before_after(network, data, h, w, n_plot=15, random=False, count=None):
    """ plot before and after for autoencoder network """
    start = 0 if not random else np.random.randint(len(data) - n_plot)
    #subdata = DT.subdata(data[start: start + n_plot], h, w)
    subdata = data
    outs = np.clip(network.get_outputs(subdata), 0, 1)
    outs = [[subdata[i], g] for i, g in enumerate(outs)]
    outs = list(itertools.chain.from_iterable(outs))
    save_path = 'plt_ba_{}'.format(count) if count or count == 0 else False
    DT.plot_data_multiple(outs, save_path=save_path)

def plot_middle(network, data, h, w, n_plot=15, random=False, count=None):
    """ plot middle layer for autoencoder network """
    MID = (16, 16) if h == 512 and w == 512 else (40, 30) ### FIX
    start = 0 #if not random else np.random.randint(len(data) - n_plot)
    #subdata = DT.subdata(data[start: start + n_plot], h, w)
    subdata = data[start: start + n_plot]
    #print(data.shape)
    mids = network.get_flat(subdata)
    #print('Max {}  Min {}'.format(mids.max(), mids.min()))
    outs = np.clip(network.get_outputs(subdata), 0, 1)
    both = [[subdata[i], outs[i], np.reshape(g, MID)]
            for i, g in enumerate(mids)]
    both = list(itertools.chain.from_iterable(both))
    save_path = 'plt_mid_{}'.format(count) if count or count == 0 else False
    DT.plot_data_multiple(both, save_path=save_path, n_x=4, n_y=6)

def to_middle_data(data, height, width):
    """ """
    return data[:, 40:40+height, 40:40+width, :]


