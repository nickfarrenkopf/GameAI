import os
import numpy as np
from random import shuffle

import paths

import sys
sys.path.append('C:\\Users\\Nick\\Desktop\\Ava\\Programs')
from Library.General import DataThings as DT
from Library.General import FileThings as FT
from Library.General import Screen
from Library.NeuralNetworks.Autoencoder import AutoencoderAPI as AUTO
from Library.NeuralNetworks.Classifier import ClassifierAPI as CLASS


### PROGRAM ###



def record_game_data(path):
    """ """
    # starting file counter
    count = 0
    if len(os.listdir(path)) > 0:
        count = int(os.listdir(path)[-1].split('_')[1]) + 1
    print('Start count: {}'.format(count))
    # wait for key
    done = False
    while not done:
        key = Screen.get_key()
        print(key)
        # if in whitelist, take screencap
        if key in game.action_keys or key in game.label_keys:
            text = '0' * (4 - len(str(count))) + str(count)
            names = ['image_{}_{}_0'.format(text, values[keys.index(key)])]
            windows = [game.get_window()]
            if key in game.action_keys:         
                for i in range(1, 3):
                    text = '0' * (4 - len(str(count))) + str(count)
                    names.append('image_{}_{}_{}'.format(text,
                                                         values[keys.index(key)],
                                                         i))
                    windows.append(game.get_window())
                    time.sleep(0.1)
            for name, window in zip(names, windows):
                path = os.path.join(game.state_path, name + '.png')
                Screen.save_image(window, path)
            count += 1
        done = key == 'p'

if __name__ == '__main__':


    """ APP """

    game = 'kirby'
    paths.set_base(game)
    json_data = paths.load_json()


    """ AUTO """
 
    if 1: # PARAMS
        auto_path = paths.network_path
        h = w = 512
        h2 = w2 = 544

    if 1: # LOAD
        auto_network = AUTO.load_auto(game, json_data)

    if 1: # CREATE
        json_data = paths.load_json()
        hidden = [32, 32, 32, 32, 32, 32, 32, 64]
        json_data = AUTO.new_auto(paths, game, h, w, hidden, length=3)
        auto_network = AUTO.load_auto(game, json_data)    

    if 0: # TRAIN
        ds = FT.load_images_4d(paths.get_game_images()[:32], h2, w2, 3)
        print(ds.shape)
        AUTO.train_auto_data(auto_network, ds, h, w, n_train=300,
                             kmax_img=15, kmax_cost=5)

    if 0: # LEARN
        ds = FT.load_images_4d(paths.get_game_images(), h2, w2, 3)
        print('Data shape: {}'.format(ds.shape))
        AUTO.learn_auto_data(paths, ds, h, w, kmax_cost=2, slope_min=1e-3,
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




