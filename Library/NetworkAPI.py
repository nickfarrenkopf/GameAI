import os
import time
import itertools
import numpy as np
import matplotlib.pyplot as plt

from Library import DataThings as DT
from Library.Network import Autoencoder
from Library.Network import Classifier
from Library.Network import Where
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

def train_auto(network, data, h, w, n_train=1000, alpha=0.0001, n_plot=40,
               kmax_cost=10, kmax_img=100, kmax_save=0):
    """ iterate through data set to train autoencoder newtork """
    costs = []
    for k in range(n_train):
        #subdata = DT.subdata(data, h, w)
        subdata = data
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
    if k_max != 0 and k % k_max == 0 and k != 0:
        subdata = input_data[:n_plot, :, :, :]
        #md = to_middle_data(data, h, w)
        plot_middle(network, subdata, h, w, n_plot=n_plot, count=k)


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
    print('Max {}  Min {}'.format(mids.max(), mids.min()))
    outs = np.clip(network.get_outputs(subdata), 0, 1)
    both = [[subdata[i], outs[i], np.reshape(g, MID)]
            for i, g in enumerate(mids)]
    both = list(itertools.chain.from_iterable(both))
    save_path = 'plt_mid_{}'.format(count) if count or count == 0 else False
    DT.plot_data_multiple(both, save_path=save_path, n_x=2, n_y=6)

def to_middle_data(data, height, width):
    """ """
    return data[:, 40:40+height, 40:40+width, :]



"""
--- CLASSIFIER ---
"""

### FILE ###

new_class = Classifier.create
load_class = Networks.NetworkClass


### TRAIN ###

def train_class(network, data, labels, vocab, h, w, n_train=10000, alpha=0.001,
                n_plot=30, kmax_cost=1000, kmax_img=1000):
    """ iterate through data set to train calssification newtork """
    costs = []
    for k in range(n_train):
        subdata = DT.subdata(data, h, w)
        network.train_network(subdata, labels, alpha)
        costs = check_accuracy(network, subdata, labels, costs, k, kmax_cost)
        check_class(network, data, labels, vocab, h, w, k, kmax_img, n_plot)

def check_accuracy(network, input_data, output_data, costs, k, kmax_cost):
    """ check to print accuracy """
    if k % kmax_cost == 0:
        costs, m = get_cost_slope(network, input_data, output_data, costs)
        acc = network.get_accuracy(input_data, output_data)
        print('Cost {} {:.7f} {:.7f} {:.7f}'.format(k, costs[-1], m, acc))
        return costs
    return costs

def check_class(network, subdata, labels, vocab, h, w, k, k_max, n_plot):
    """ """
    if k % k_max == 0 and k_max != 0:
        subdata = DT.subdata_xy(subdata, h, w, 50, 50)
        preds = network.get_preds(subdata)
        labs = [vocab[p] if p != list(labels[i]).index(1) else 1
                for i, p in enumerate(preds)]
        save_path = 'plt_class_{}.png'.format(k)
        DT.plot_data_multiple(subdata, labels=labs, save_path=save_path)


### HELPER ###

        

"""
--- WHERE ---
"""

### FILE ###

new_where = Where.create
load_where = Networks.NetworkClass


### TRAIN ###

def train_where(network, data, labels, h, w, n_train=1000, alpha=0.0001,
                n_plot=40, kmax_cost=10, kmax_img=100, kmax_save=0):
    """ iterate through data set to train autoencoder newtork """
    costs = []
    for k in range(n_train):
        subdata, sublabels = subdata_group(data, labels, h, w)
        print(subdata.shape)
        #print(sublabels.shape)
        #plt.imshow(subdata[0])
        #plt.show()
        sublabels = sublabels[:, ::8, ::8, :]
        fuck = to_plt_data(sublabels)
        sublabels = np.reshape(fuck, (4, -1))
        #print(fuck.min())
        #print(fuck.max())
        #print(fuck.mean())
        #print(fuck.shape)
        #plt.imshow(fuck)
        #plt.show()
        #a = 1 / 0
        network.train_network(subdata, sublabels, alpha)
        costs = check_cost(network, subdata, sublabels, costs, k, kmax_cost)
        #check_where(network, subdata, sublabels, h, w, k, kmax_img, n_plot)
        check_save(network, k, kmax_save)

def to_plt_data(labels):
    """ """
    return np.pad(labels[0] * 255, ((0, 0), (0, 0), (0, 2)),
                  mode='constant', constant_values=0)

def to_where_data(data, labels):
    """ """
    same = np.clip(np.sum((data != labels).astype(np.int), axis=2), 0, 1)
    return same

def subdata_group(data, labels, h, w):
    """ """
    idx1 = np.random.randint(data.shape[1] - h + 1)
    idx2 = np.random.randint(data.shape[2] - w + 1)
    ds = data[:,idx1:idx1+h,idx2:idx2+w,:]
    ls = labels[:,idx1:idx1+h,idx2:idx2+w,:]
    return ds, ls

def check_where(network, data, input_data, h, w, k, k_max, n_plot):
    """ """
    pass


### HELPER ###




"""
--- OLD ---
"""

def train_class_group(network, train_data, labels, size=128, alpha=0.0001,
                      n_train=10000, kmax_cost=200):
    """ trains newtorks given input data and labels """
    costs = []
    for k in range(n_train):
        for j in range(len(train_data)):
            labs = labels[j]
            subdata = DT.subdata(train_data[j], size=size)
            network.train_network(subdata, labs, alpha)
            # print cost
            costs = check_accuracy(network, subdata, labs, costs, k, kmax_cost)

def train_binary(network, data, vocab, word, size=128, n_train=1000,
                 a=0.0001, kmax_cost=100, kmax_nsub=100, kmax_nall=1000):
    """   WHAAAAAT   """
    small = DT.subdata(data, size=128)
    costs = []
    # word specific values
    new_labels = to_binary_labels(word, vocab)
    data_p, labels_p, data_n, labels_n = to_pos_neg_data(small, new_labels)
    # iterate
    for k in range(n_train):
        data_small = DT.subdata(small, size=size)
        data_p_small = DT.subdata(data_p, size=size)
        network.train_network(data_p_small, labels_p, a)
        # train others
        if k % kmax_nsub:
            data_n_small = DT.subdata(data_n, size=size)
            network.train_network(data_n_small, labels_n, a)
        if k % kmax_nall == 0:
            network.train_network(data_small, new_labels, a)
        # cost
        subdata = 0
        labels = 0
        costs = check_accuracy(network, subdata, labels, costs, k, kmax_cost)


