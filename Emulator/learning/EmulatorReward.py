import os
import itertools
import numpy as np

import sys
sys.path.append('C:\\Users\\Nick\\Desktop\\Ava\\Programs')
from Library.General import DataThings as DT
from Library.General import FileThings as FT



### REWARD ###

class EmulatorReward(object):
    """ """

    def __init__(self, paths, env, name):
        """ """
        self.paths = paths
        self.env = env
        self.name = name


    def load_rewards(self):
        """ """
        # load text file of rewards
        f = os.path.join(self.paths.game_path, 'custom_labels.txt')
        text = FT.read_file_csv(f)
        files = [os.path.join(self.paths.image_path, d[0]) for d in text]
        labs = [int(d[1]) for d in text]
        imgs = FT.load_images(files)
        #
        data = self.env.auto_network.get_latent(imgs)
        d1d = [d for i, d in enumerate(data) if labs[i] == 1]
        d1 = np.reshape(d1d, (-1, 8, 8))
        means = np.mean(d1, axis=0)
        d2 = np.array([d - means for d in d1]) * 2
        #
        combed = [[a, b, c] for a, b, c in zip(imgs, d1, d2)]
        combed = list(itertools.chain.from_iterable(combed))
        #print(np.array(combed).shape)
        #DT.plot_data_multiple(combed)
        self.ds = np.array(d1d)
        self.ls = labs


    def get_reward(self):
        """ """
        # read screen
        data = self.env.get_state()
        # find if reward similar
        #sims = get_similar(data[0], self.ds)
        sims = get_similar(data[0], self.ds)
        print(sims)
        
        


### REWARDS ###




