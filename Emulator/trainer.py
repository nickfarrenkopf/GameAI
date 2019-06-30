import os
import numpy as np


import paths
import data_things as DTTT
from learning import EmulatorEnvironment as EE
from learning import EmulatorAgent as EAg
from learning import EmulatorAction as EAc

import sys
sys.path.append('C:\\Users\\Nick\\Desktop\\Ava\\Programs')
from Library.General import DataThings as DT
from Library.General import FileThings as FT
from Library.NeuralNetworks.Autoencoder import _AutoencoderAPI as AUTO
from Library.NeuralNetworks.Classifier import _ClassifierAPI as CLASS


### HELPER ###

def load_memories():
    """ """
    memories = []
    auto_data = auto_network.get_latent(ds)
    # loop over files
    for i, f in enumerate(files[:n]):
        name, ep, idx, acts = os.path.basename(f).split('.')[0].split('_')
        # skip 
        if ep != '1' or '2' in acts:
            continue
        if '1' in acts and acts not in agent.actions:
            continue
        # initial values
        S, A, R, S_new = auto_data[i:i+1], 0, 0, auto_data[i+1:i+2]
        done = f == files[-1]
        # action
        if acts in agent.actions:
            A = agent.actions.index(acts)
        # if 0 - negative reward
        if '1' in acts:
            R = 1
        if '0' in acts:
            R = -1
        # save memory
        A, R = np.array([A]), np.array([R])
        memories.append((S, A, R, S_new, done))
    return memories


def generate_data(labels, with_random=True):
    """ """
    label_set = ['_NA'] + labels if with_random else labels
    files, labs = DTTT.get_filepaths_for_labels(set(labels), True, with_random)
    data = FT.load_images(files)
    labels = DT.to_one_hot(labs, label_set=label_set)
    return data[:100], labels[:100]


### PROGRAM ###

if __name__ == '__main__':

    """ GAME """
    # base folder
    name = 'pacman'
    paths.set_base(name)
    files = FT.get_filepaths(paths.image_path)

    # data params
    n = 20000
    h, w, le = (160, 160, 3)

    """ DATA """
    if 0: # AUTO DATA
        ds = FT.load_images(files[:n])
        print('Data shape: {} {} {}\n'.format(ds.shape, ds.max(), ds.min()))
    if 0: # CLASS DATA
        labels = set(('0', '1'))
        fs, ls = DTTT.get_filepaths_for_labels(labels, True, True)
        ds = FT.load_images(fs)
        labs = DT.to_one_hot(ls, label_set=['_NA','0','1'])
        n_classes = len(set(ls))
        print('Data n_files: {}  n_classes: {}\n'.format(len(ls), n_classes))

    """ AUTO """
    if 0: # CREATE
        paths.reset_json()
        hidden_encode = [2, 2, 2, 3, 3, 3]
        pools_encode  = [1, 2, 1, 2, 1, 2]
        hidden_decode = [4, 3, 2, 1]
        pools_decode  = [1, 1, 1, 1]
        hidden_dense  = [512]
        n_latent = 64
        b = 1.1
        AUTO.new(paths, name, h, w, length=le, patch=3, b=b, print_me=True,
                 hidden_encode=hidden_encode, pools_encode=pools_encode,
                 hidden_latent=hidden_dense, n_latent=n_latent, 
                 pools_decode=pools_decode, hidden_decode=hidden_decode)
    if 0: # LOAD - NETWORK
        auto_network = AUTO.load(name, paths.load_json())
    if 0: # TRAIN - DATA
        print('Training auto network on data')
        AUTO.train_data_iter(auto_network, ds, a=1e-3, n_train=500000, 
                             plot_r=True, kmax_img=5000, kmax_cost=1000)
    if 0: # TRAIN - PATHS
        print('Training auto network on files')
        AUTO.train_path_iter(auto_network, files, a=1e-4, n_train=50000000,
                             plot_r=True, kmax_cost=5000, kmax_img=10000)
    if 0: # TEST
        print('Testing auto network')
        AUTO.train_data_iter(auto_network, ds, a=0, n_train=5, n_plot=60,
                             plot_r=True, kmax_img=1, kmax_cost=9999)

    """ CLASS """ 
    if 0: # CREATE
        hidden_c = [16]
        CLASS.create(paths, name, auto_network.n_latent, hidden_c, n_classes)
    if 0: # LOAD CLASS
        class_network = CLASS.load(name, paths.load_json())
    if 0: # TRAIN
        CLASS.train_data_iter(auto_network, class_network, ds, labs, h, w,
                              alpha=1e-3, n_train=5000, kmax_cost=50, kp=0.5)
    if 1:
        n_classes = 3
        CLASS.optimize(paths, name, ['0','1'], generate_data)

    """ OTHER """

    if 0: # ENV and AGENT
        env = EE.Emulator(paths, 'pacman')
        #env.save_window()
        agent = EAg.EmulatorAgent(paths, env, 0, EAc.sets['emulator_2'], True)
        if 0:
            method.create_actor_network()
            method.create_critic_network()

    if 0: # MEMORY
        memories = load_memories()
        for i, (S, A, R, S_new, done) in enumerate(memories):
            agent.method.remember(S, A, R, S_new, done)
            agent.method.train()
            # print metrics
            if i % 32 == 0 and len(agent.method.memory) > 50:
                print('Iter {}'.format(i))
                agent.method.actor_action_metrics()
                agent.method.critic_value_metrics()
            



