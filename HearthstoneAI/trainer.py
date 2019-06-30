import os
import time
import numpy as np

import paths
import HearthstoneElement as HE
import data_generation as DG

import sys
sys.path.append('C:\\Users\\Nick\\Desktop\\Ava\\Programs')
from Library.General import DataThings as DT
from Library.General import FileThings as FT
from Library.NeuralNetworks.Autoencoder import _AutoencoderAPI as AUTO
from Library.NeuralNetworks.Autoencoder import TestAutoencoder as TA
from Library.NeuralNetworks.Classifier import _ClassifierAPI as CLASS

from random import shuffle
### HELPER ###

def train_online():
    """ """
    pass

def train_special(loss_max=3):
    """ """
    padds = 15
    space = len(ds) // padds
    for i in range(padds):
        loss = 100
        ddd = ds[(i)*space:(i+1)*space]
        print('Training on {} {}'.format(i, ddd.shape))
        while loss > loss_max:
            a = 1e-3 if i <10 else 1e-4
            AUTO.train_data_iter(e.auto_network, ddd, e.w, e.h, n_train=1000,
                                 a=a, n_plot=20, kmax_cost=1001)
            loss = e.auto_network.get_io_loss(ddd)
            #print('Loss {} {}'.format(i, loss))
        if i % 2 == 0 and i != 0:
            AUTO.train_data_iter(e.auto_network, ddd[-40:], e.w, e.h,
                                 n_train=i+1,a=1e-4, n_plot=20, plot_r=True, 
                                 kmax_img=i, kmax_cost=2*i)
            
    
    

### PROGRAM ###

if __name__ == '__main__':


    """ GAME """

    # base folder
    name = 'mana'
    names = [name + 'S']
    filepath = paths.get_image_path(name)

    # file data
    files = FT.get_filepaths(filepath)
    shuffle(files)
    n = 5000
    

    """ DATA """    

    if 1: # IMAGE DATA
        ds = FT.load_images(files[:n])
        print('Data shape: {} {} {}'.format(ds.shape, ds.max(), ds.min()))

    if 1: # HEARTHSTONE ELEMENTS
        elements = HE.load_elements(names, networks=True)
        e = elements[0]

    if 0: # CLASS DATA
        
        labels = [lab for lab in paths.get_labels(name)
                  if '-2' not in lab[1:] and '-1' not in lab[1:]]
        labels = DG.load_some_labels(1)
        ds_small = FT.load_images([os.path.join(paths.images_path, name, 
                                    '{}'.format(lab[0]))
                               for lab in labels])
        
        labs1 = DT.to_one_hot([int(l[1]) for l in labels], n_classes=11)


    """ AUTO """

    if 0: # CREATE
        e.create_network()
        e.load_network([])

    if 1: # LOAD - NETWORK
        auto_network = e.auto_network
        auto_network.print_info()

    if 0: # TRAIN ITER - DATA
        print('Training on data with iters')
        AUTO.train_data_iter(e.auto_network, ds, e.w, e.h, n_train=1000000,
                             a=1e-3, n_plot=60, plot_r=True, plot_i=False,
                             kmax_img=10000, kmax_cost=5000)

    if 0: # TRAIN ITER - PATHS
        print('Training on data with paths')
        AUTO.train_path_iter(e.auto_network, files, e.w, e.h, n_train=100000,
                             a=1e-4, n_plot=16, plot_r=True, plot_i=False,
                             kmax_cost=1000, kmax_img=10000)


    if 1: # TEST
        print('Training on data with iters')
        AUTO.train_data_iter(e.auto_network, ds, e.w, e.h, n_train=5,
                             a=0, n_plot=60, plot_r=True, plot_i=False,
                             kmax_img=1, kmax_cost=9999)

    if 0:
        train_special()


    """ ONLINE """

    if 0: # TRAIN ONLINE
        train_online()


    """ CLASS """ 

    if 0: # CREATE
        size = 4
        n_classes=11
        class_hidden = [128, 128]
        CLASS.create(paths, name, size, class_hidden, n_classes)

    if 0: # LOAD CLASS
        class_network = CLASS.load(name, paths.load_json())
        class_network.print_info()

    

    if 0: # TRAIN
        CLASS.train_data_iter(auto_network, class_network, ds_small,
                              labs1, 0, 0, alpha=1e-3,
                              n_train=200000, kmax_cost=1000)


