import os
import time
import numpy as np
from os.path import join

import paths
import HearthstoneElement as HE

import sys
sys.path.append('C:\\Users\\Nick\\Desktop\\Ava\\Programs')
from Library.General import DataThings as DT
from Library.General import FileThings as FT
from Library.NeuralNetworks.Autoencoder import _AutoencoderAPI as AUTO
from Library.NeuralNetworks.Classifier import _ClassifierAPI as CLASS


### API ###


import data_generation as DG


def predict_mana():
    """ """
    # load labels
    labels = [lab for lab in paths.get_labels(name) if '-2' not in lab[1:]
              and '-1' not in lab[1:]]
    lab_names = [lab[0] for lab in labels]
    ds_small = FT.load_images([join(paths.images_path, name, 
                                    '{}'.format(lab[0]))
                               for lab in labels])
    print(ds_small.shape)
    # loop over mana elements
    for e in elements:
        if 'mana' in e.name:
            # get auto data
            e.save_window()
            data = np.concatenate([[e.get_window()], ds_small])
            x1 = e.auto_network.get_mu(data)
            #x2 = e.auto_network.get_sigma(data)
            #x3 = e.auto_network.get_latent(data)
            # mu
            print('mu')
            tops = DG.get_similar(x1[0:], x1[1:], top_n=7)
            print('Most similar ' + e.name)
            for p in tops:
                if p[2] in lab_names:
                    idx = lab_names.index(p[2])
                    print(p[:2], labels[idx][1:])
            # sigma
            print('sigma')
            #tops = DG.get_similar(x2[0:], x2[1:], top_n=7)
            #for p in tops:
            #    if p[2] in lab_names:
            #        idx = lab_names.index(p[2])
            #        print(p[:2], labels[idx][1:])
            # latenbt
            print('latent')
            #tops = DG.get_similar(x3[0:], x3[1:], top_n=7)
            #for p in tops:
            #    if p[2] in lab_names:
            #        idx = lab_names.index(p[2])
            #        print(p[:2], labels[idx][1:])
            # get most probable

def pred_man():
    """ """
    winds = []
    for i in range(20):
        for e in elements:
            e.save_window()
            winds.append(e.get_window())
            time.sleep(1)
    # network outs
    dats = np.array(winds) / 255
    print(dats.shape)
    print(dats.max(), dats.min(), np.mean(dats))
    outs = e.auto_network.get_outputs(dats)
    print(outs.shape)
    print(outs.max(), outs.min(), np.mean(outs))
    outs = np.clip(outs, 0, 1)
    dds = []
    for d, o in zip(dats, outs):
        dds += [d, o]
    DT.plot_data_multiple(dds)
    #return dats


### PROGRAM ###

if __name__ == '__main__':

    
    """ GAME """

    # base folder
    name = 'mana'
    names = [name + 'S', name + 'O']
    filepath = paths.get_image_path(name)

    # file data
    files = FT.get_filepaths(filepath)
    n = 5000
    

    """ DATA """    

    if 0: # IMAGE DATA
        ds = FT.load_images(files[:n])
        print('Data shape: {} {} {}'.format(ds.shape, ds.max(), ds.min()))

    if 1: # HEARTHSTONE ELEMENTS
        elements = HE.load_elements(names, networks=True)
        e = elements[0]


    """ NETWORK """

    if 1: # LOAD - NETWORK
        auto_network = e.auto_network
        auto_network.print_info()


    if 0: # LOAD CLASS
        class_network = CLASS.load(name, paths.load_json())
        class_network.print_info()


    pred_man()
    
