import numpy as np

import sys
sys.path.append('C:\\Users\\Nick\\Desktop\\Ava\\Programs')
from Library.General import DataThings as DT
from Library.General import FileThings as FT
from Library.Computer import Screen
from Library.NeuralNetworks.Autoencoder import _AutoencoderAPI as AUTO


### EMULATOR ###

class Emulator(object):
    """ Base Emulator object """


    def __init__(self, paths, name):
        """ """
        self.paths = paths
        self.name = name
        
        # screen window
        self.p1 = (70, 1341)
        self.p2 = (579, 1850)
        self.window = Screen.Window(self.p1, self.p2)

        # auto network
        self.w_n = 160
        self.h_n = 160
        self.network_shape = (-1, self.w_n, self.h_n, 3)
        self.auto_network = AUTO.load(name, paths.load_json())

        # state
        self.S_size = self.auto_network.latent_shape[-1]


    ### WINDOW ###

    def get_window(self, resized=True, network=False):
        """ """
        data = self.window.get_window()[0]
        if resized:
            data = DT.resize_image(data, self.w_n, self.h_n)
        if network:
            data = np.reshape(data, self.network_shape)
        return data

    def save_window(self):
        """ """
        FT.save_image_to_file(self.get_window(), self.name + '.png')


    ### STATE ###

    def get_state(self):
        """ """
        data = self.get_window(resized=True, network=True)
        latent = self.auto_network.get_latent(data)
        return latent


