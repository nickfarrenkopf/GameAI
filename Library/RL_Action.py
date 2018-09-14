from keras.models import Sequential
from keras.layers import Dense

from Library.General import DataThings as DT
from Library.RL_Component import RL_Component as RLC


class Action(RLC):
    """ """

    def __init__(self, game, agent):
        """ """

        # RL components
        self.name = 'action'
        self.game = game
        self.agent = agent

        # load network
        #RLC.__init__(self, 'WHERE MY NETWORK AT')
        #self.load_network()


    ### NETWORK ###

    def create_network(self, h_hidden=64, n_layers=2):
        """ """
        pass


    ### RUN TIME ###

    def WHAT_DO(self):
        """ """
        pass


    ### TRAIN - OFFLINE ###

    def train_network_offline(self, epochs=100, n_loop=10, save_me=False):
        """ """
        pass

    def test_network_offline(self):
        """ """
        pass


    ### TRAIN - ONLINE ###  

    def train_network_online(self, epochs=10, save_me=False):
        """ """
        pass

    def test_network_online(self):
        """ """
        pass


