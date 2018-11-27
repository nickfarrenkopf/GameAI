import os
import numpy as np
from random import shuffle

import paths

import sys
sys.path.append('C:\\Users\\Nick\\Desktop\\Ava\\Programs')
from Library.General import DataThings as DT
from Library.NeuralNetworks.Autoencoder import AutoencoderAPI as AUTO
from Library.NeuralNetworks.Classifier import ClassifierAPI as CLASS


### PROGRAM ###

if __name__ == '__main__':


    """ APP """

    game = 'test'
    paths.set_base(game)
    json_data = paths.load_json()


    """ DATA GENERATION """
    
    if 0: # record audio
        #SR.record_single_timed(paths.audio_path, 'test')
        SR.record_until_done(paths.audio_path, 'test')


    """ AUTO """
 
    if 1: # PARAMS
        auto_path = paths.network_path
        h = w = 512
        h2 = w2 = 544

    if 0: # LOAD
        auto_network = AUTO.load_auto(game, json_data)

    if 0: # CREATE
        json_data = paths.load_json()
        hidden = [32, 32, 32, 32, 32, 32, 32, 64] # 1st iteration
        json_data = AUTO.new_auto(paths, game, h, w, hidden, length=3)
        auto_network = AUTO.load_auto(game, json_data)    

    if 0: # TRAIN
        ds = np.reshape(DT.load_datas(paths.get_game_images()[:32]), (-1, h2, w2, 3))
        print(ds.shape)
        AUTO.train_auto_data(paths, ds, h, w, n_train=10000,
                             kmax_img=10, kmax_cost=1)

    if 1: # LEARN
        ds = np.reshape(DT.load_datas(paths.get_game_images()[:16]), (-1, h2, w2, 3))
        print('Data shape: {}'.format(ds.shape))
        AUTO.learn_auto_data(paths, ds, h, w, kmax_cost=5, slope_min=1e-3,
                             slope_count_max=10)




    """ CLASS - AUTO """

    if 0: # PARAMS
        class_path = paths.network_path
        size = 64
        n_classes = 8

    if 0: # LOAD
        class_network = CLASS.class_auto(name, json_data)

    if 0: # CREATE
        name = 'test'
        hidden = [64, 64, 64]
        json_data = CLASS.new_class(paths, name, size, hidden, n_classes)

    if 0: # TRAIN
        class_network = CLASS.load_class(name, json_data)
        ds, ls = DT.load_data_labels(paths.get_audio_image_files(),
                                     randomize=False)
        ds = np.reshape(ds, (-1, h2, w2, 1))
        print('Data: {}'.format(ds.shape))
        print('Labels: {}'.format(ls.shape))
        CLASS.train_class_data(class_network, auto_network, ds, ls, h, w)




