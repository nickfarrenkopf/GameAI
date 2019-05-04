import time

import sys
sys.path.append('C:\\Users\\Nick\\Desktop\\Ava\\Programs')
from Library.Computer import Screen
from Library.Computer import Mouse
from Library.General import FileThings as FT
from Library.NeuralNetworks.Autoencoder import _AutoencoderAPI as AUTO
import tensorflow as tf

import PIL
from PIL import Image
import numpy as np
import paths


### WATCHER ###

class ScreenWatcher(object):
    """ """

    def __init__(self, name, network_name, x, y, w, h, w_n, h_n, n_latent,
                 every_n):
        """ """

        # params
        self.name = name
        self.network_name = network_name
        self.x = int(x)
        self.y = int(y)
        self.w = int(w)
        self.h = int(h)
        self.w_n = int(w_n)
        self.h_n = int(h_n)
        self.n_latent = int(n_latent)
        self.every_n = int(every_n)

        # screen window
        self.p1 = [self.x - self.w // 2, self.y - self.h // 2]
        self.p2 = [self.x + self.w // 2, self.y + self.h // 2]
        self.window = Screen.Window(self.p1, self.p2)

        self.auto_network = None
        # network
        #self.create_network()
        #self.load_network()
        

    ### FILE ###

    def get_window(self):
        """ """
        ds = self.window.get_window()[0]
        dds = resize_image(ds, self.w_n, self.h_n)
        return dds

    def save_window(self, i, j):
        """ """
        import os
        ds = self.get_window()
        nam = '{}_{}_{}.png'.format(self.name, i, j)
        ps = os.path.join(paths.images_path, self.network_name, nam)
        if i == 'test':
            ps = os.path.join(nam)
        FT.save_image_to_file(ds, ps)

    def create_network(self):
        """ """
        tf.reset_default_graph()
        if self.n_latent == 8:
            hidden_encode = [3, 4, 5, 5, 6, 6]
            pools_encode = [2, 2, 2, 1, 2 if self.n_latent > 4 else 1, 1]
            hidden_decode = [5, 4, 3, 2]
            pools_decode = [1, 1, 1, 1 if self.n_latent > 4 else 2]
            hidden_dense = [512, 512]
        else:
            hidden_encode = [3, 4, 5, 5, 6, 6]
            pools_encode = [2, 2, 2, 1, 2 if self.n_latent > 4 else 1, 1]
            hidden_decode = [5, 4, 3, 2]
            pools_decode = [1, 1, 1, 1 if self.n_latent > 4 else 2]
            hidden_dense = [256, 256, 256]
        hidden_encode = [2 ** i for i in hidden_encode]
        hidden_decode = [2 ** i for i in hidden_decode]
        AUTO.new(paths, self.network_name, self.w_n, self.h_n, length=3, patch=4,
                 hidden_encode=hidden_encode,
                 pools_encode=pools_encode,
                 hidden_latent=hidden_dense, n_latent=self.n_latent,
                 pools_decode=pools_decode, hidden_decode=hidden_decode)

    def load_network(self, watchers):
        """ """
        if self.network_name not in paths.load_json()['network']['auto']:
            self.create_network()
        names = [w.network_name for w in watchers if w.auto_network != None]
        if self.network_name not in names:
            print('{} loaded'.format(self.network_name))
            self.auto_network = AUTO.load(self.network_name, paths.load_json())
        else:
            idx = names.index(self.network_name)
            self.auto_network = watchers[idx].auto_network


    ### TRAINING ###

    def listen(self):
        """ """
        pass
        
    def train_auto(self, n_train=64, a=1e-4):
        """ """
        ds = np.array([self.get_window()]) / 255
        #ds = np.reshape(ds, (1, self.w, self.h, 3))
        for i in range(n_train):
            self.auto_network.train_network(ds, a, 1.1)
        cost = self.auto_network.get_loss(ds)
        return cost


### HELPER ###

def record_data():
    """ """
    print('Recording data')
    for i in range(5000):
        for w in watchers:
            if i % w.every_n == 0:
                w.save_window(3, i)
        time.sleep(1)

def load_watchers(i=0,j=100, nets=False):
    """ """
    watchers = []
    params = paths.load_params()
    for k in params:
        watchers.append(ScreenWatcher(params[k]))
        if nets:
            watchers[-1].load_network(watchers)
    return watchers

def resize_image(data, w2, h2):
    """ """
    img = Image.fromarray(np.uint8(data*255))
    img = img.resize((h2, w2), PIL.Image.ANTIALIAS)
    return np.array(img)


def train_all():
    """ """
    for _ in range(50):
        costs = []
        for w in watchers:
            costs.append(w.train_auto())
        total = ''
        for i, w in enumerate(watchers[:7]):
            total += '{:.3f} '.format(costs[i])
        print(total)
        #print(costs)


### PROGRAM ###

if __name__ == '__main__':



    ts = load_watchers(nets=False)
    watchers = ts

    if 0:
        for t in ts:
            t.save_window('test','')

    if 0:
        train_all()


    if 1:
        record_data()



