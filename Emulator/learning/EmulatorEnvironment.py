import numpy as np

import paths

import sys
sys.path.append('C:\\Users\\Nick\\Desktop\\Ava\\Programs')
from Library.Computer import Window
from Library.General import FileThings as FT
from Library.Learning import EnvironmentUtils
from Library.NeuralNetworks.Autoencoder import _AutoencoderAPI as AUTO


### EMULATOR ###

class Emulator(EnvironmentUtils.Environment):
    """ Base Gridworld object """


    def __init__(self, paths, name):
        """ """
        EnvironmentUtils.Environment.__init__(self, paths, name)
        
        # window
        self.p1 = (70, 1341)
        self.p2 = (579, 1850)
        self.window = Window.Window(self.p1, self.p2)

        # image
        self.test_filepath = 'state.png'

        # auto network
        self.auto_network = AUTO.load(name, paths.load_json())


    ### STATE ###

    def get_window(self):
        """ """
        return self.window.get_window()

    def get_state(self):
        """ """
        data = np.reshape(self.window.get_window(), (1,512,512,3))
        auto_data = self.auto_network.get_flat(data)
        #print('State shape {}'.format(auto_data.shape))
        return auto_data

    def save_state_image(self):
        """ """
        FT.save_image_to_file(self.window.get_window(), self.test_filepath)
        print('Saved state to {}'.format(self.test_filepath))


