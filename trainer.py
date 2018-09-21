import os
import numpy as np

import paths
from Library.General import DataThings as dt
from Library.Network import NetworkAPI as NETS


### PROGRAM ###

if __name__ == '__main__':

    s = 512
    path = paths.pacman_path
    net_path = os.path.join(path, 'networks')

    if 0:
        NETS.new_auto(paths.network_path, name, s, s, [64, 32, 16, 16, 8, 4])
    
    # load auto
    if 1:
        auto_network = dt.load_auto(net_path, 'AUTO_pacman_512_512_6_256')
        data = dt.load_images(os.path.join(path, 'gamedata'))
        print('Loaded network {}'.format(auto_network.name))
        print('Data shape: {}'.format(data.shape))

    # test auto
    if 1:
        #dt.plot_data_multiple(data)
        for i in range(10):
            NETS.check_auto(auto_network, data, s, s, i, 1, 30)

    # train auto
    if 0:
        print('Training...')
        auto_network = dt.load_auto(net_path, 'AUTO_pacman_512_512_6_256')
        data = dt.load_images(os.path.join(path, 'gamedata'))
        NETS.train_auto(auto_network, data, size, size,
                        n_train=10, kmax_img=1, kmax_cost=1)


