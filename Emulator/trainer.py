import os
import time
import numpy as np

import paths
from learning import ActorCritic as MAT
from learning import EmulatorEnvironment as EE
from learning import EmulatorAgent as EAg
from learning import EmulatorAction as EAc

import sys
sys.path.append('C:\\Users\\Nick\\Desktop\\Ava\\Programs')
from Library.General import DataThings as DT
from Library.General import FileThings as FT
from Library.NeuralNetworks.Autoencoder import _AutoencoderAPI as AUTO
from Library.NeuralNetworks.Classifier import _ClassifierAPI as CLASS
from Library.NeuralNetworks.Embedding import _EmbeddingAPI as EMBED
from Library.NeuralNetworks.Actor import _ActorAPI as ACTOR
from Library.NeuralNetworks.Critic import _CriticAPI as CRITIC



### PROGRAM ###

if __name__ == '__main__':


    """ GAME """

    # base folder
    name = 'pacman'
    paths.set_base(name)
    files = FT.get_filepaths(paths.image_path)

    # data params
    n = 10000
    h = 160
    w = 160
    le = 3

    # class data
    class_path = paths.network_path
    size = 64
    n_classes = 4


    """ DATA """

    if 1: # AUTO DATA
        ds = FT.load_images(files[:n])
        #ds = DT.subdata(ds, h, w)
        print('Data shape: {} {} {}'.format(ds.shape, ds.max(), ds.min()))

    if 0: # CLASS DATA
        labels = set(('0','1','2'))
        #labels = set(('left','right','up','down'))
        fs, ls = paths.get_filepaths_for_labels(labels, True, True)
        ds = FT.load_images(fs)
        labs = DT.to_one_hot(ls)
        n_classes = len(set(ls))
        print('Data n_files: {}  n_classes: {}\n'.format(len(ls), n_classes))


    """ AUTO """

    if 0: # CREATE
        paths.reset_json()
        hidde_encode = [2, 2, 2, 3, 3, 3]
        pools_encode = [1, 2, 1, 2, 1, 2]
        hidde_decode = [4, 3, 2, 1]
        pools_decode = [1, 1, 1, 1]
        hidden_dense = [512]
        n_latent = 64
        hidden_encode = [2 ** i for i in hidde_encode]
        hidden_decode = [2 ** i for i in hidde_decode]
        b = 1.1
        AUTO.new(paths, name, h, w, length=le, patch=3, print_me=True,
                 hidden_encode=hidden_encode, pools_encode=pools_encode,
                 hidden_latent=hidden_dense, n_latent=n_latent, b=b,
                 pools_decode=pools_decode, hidden_decode=hidden_decode)

    if 1: # LOAD - NETWORK
        auto_network = AUTO.load(name, paths.load_json())
        #auto_network.print_info()

    if 0: # TRAIN ITER - DATA
        print('Training on data with iters')
        AUTO.train_data_iter(auto_network, ds, h, w, n_train=500000, a=1e-3,
                             n_plot=16, plot_r=True, plot_i=True,
                             do_subdata=False, kmax_img=5000, kmax_cost=1000)

    if 0: # TRAIN ITER - PATHS
        print('Training on data with paths')
        AUTO.train_path_iter(auto_network, files, h, w, n_train=50000000,
                             a=1e-4, n_plot=16, plot_r=True, plot_i=False,
                             kmax_cost=5000, kmax_img=10000, do_subdata=False)
    if 0: # TEST
        print('Training on data with iters')
        AUTO.train_data_iter(auto_network, ds, h, w, n_train=10,
                             a=0, n_plot=60, plot_r=True, plot_i=False,
                             kmax_img=1, kmax_cost=9999)


    """ CLASS """ 

    if 0: # CREATE
        class_hidden = [32]
        CLASS.create(paths, name, size, class_hidden, n_classes)

    if 0: # LOAD CLASS
        class_network = CLASS.load(name, paths.load_json())
        class_network.print_info()

    if 0: # TRAIN
        CLASS.train_data_iter(auto_network, class_network, ds, labs, h, w,
                              alpha=1e-3, n_train=5000, kmax_cost=100)



    if 1: # test image
        env = EE.Emulator(paths, 'pacman')
        env.save_window()
        #dd = env.get_window(network=True)
    if 1:
        key = 0
        actions = EAc.sets['emulator_2']
        ag = EAg.EmulatorAgent(env, key, actions)

    if 1: # MEMORY
        method = MAT.ActorCriticMethod(paths, ag, load_networks=True)
        #method.create_actor_network()
        #method.create_critic_network()
        # load memories

        memories = []
        previous_labels = ''

        auto_data = auto_network.get_latent(ds)

        # loo over files
        for i, f in enumerate(files):
            bn = os.path.basename(f)
            name, iters, idx, labels = bn.split('.')[0].split('_')
            
            # skip - only episode 1
            if int(iters) != 1:
                continue
            # skip - non in game
            if '2' in labels:
                continue
            # skip - non standard action
            if labels not in ag.actions and '1' in labels:
                continue
            
            # initial memory values
            S = auto_data[i:i+1]
            A = 0
            R = 0
            S_new = auto_data[i+1:i+2]
            done = f == files[-1]

            # action
            if labels in ag.actions:
                A = ag.actions.index(labels)
            
            # if 0 - negative reward
            if '1' in labels:
                R = 1
            if '0' in labels:
                R = -1
            A = np.array([A])
            
            # save memory
            memories.append((S, A, R, S_new, done))
            previous_labels = labels
            

        # loop through memories
        for i, (S, A, R, S_new, done) in enumerate(memories):

            if i % 10 == 0:
                print('Iter {} {} {}'.format(i, A, R))
            method.remember(S, A, R, S_new, done)
            if len(method.memory) < method.batch_size:
                continue
            method.train()
            



