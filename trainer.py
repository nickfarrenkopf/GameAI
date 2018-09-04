import os
import numpy as np

import paths
import data_things as dt
from Library import NetworkAPI as NETS


### FILE ###

def load_auto(auto_name):
    """ """
    return NETS.load_auto(paths.network_path, auto_name)
  

### COMMANDS ###

"""

--- NEW AUTO ---
NETS.new_auto(paths.network_path, 'test', 512, 512, [64, 32, 32, 32, 16, 16])
NETS.new_auto(paths.network_path, 'test', 512, 512, [64, 32, 16, 8, 4, 4])

--- LOAD AUTO ---
auto_network = NETS.load_auto(paths.network_path, 'AUTO_test_512_512_5_1024')
auto_network = NETS.load_auto(paths.network_path, 'AUTO_test_512_512_6_256')

--- TRAIN AUTO ---
ds = dt.load_data()
NETS.train_auto(auto_network, ds, 512, 512, kmax_img=1, n_train=20, kmax_cost=1)

--- TEST AUTO ---
ds = dt.load_data()
dt.plot_data_multiple(ds)
dt.plot_midd???

"""


### PROGRAM ###

if __name__ == '__main__':

    path = paths.network_path
    
    # load auto
    if 0:
        auto_network = dt.load_auto(path, 'AUTO_test_512_512_6_256')
        ds = dt.load_data()
        print('Loaded network {}'.format(auto_network.name))
        print('Data shape: {}'.format(ds.shape))

    # test auto
    if 0:
        dt.plot_data_multiple(ds)

    # train auto
    if 0:
        NETS.train_auto(auto_network, ds[:len(ds) // 2], 512, 512, n_train=20,
                        kmax_img=1, kmax_cost=1)


