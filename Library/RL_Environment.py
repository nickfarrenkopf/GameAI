from keras.models import Sequential
from keras.layers import Dense

from Library.General import DataThings as DT
from Library.RL_Component import RL_Component as RLC


class Environment(RLC):
    """ """

    def __init__(self, game):
        """ """
        
        # RL components
        self.name = 'environment'
        self.game = game
        
        # load network
        RLC.__init__(self, game.environment_network_path)
        self.load_network()


    ### NETWORK ###

    def create_network(self, n_hidden=64, n_layers=2):
        """ """
        # first layer
        network = Sequential()
        network.add(Dense(n_hidden, activation='relu',
                          input_shape=self.game.state_action_size))
        # hidden layers
        for _ in range(n_layers - 1):
            network.add(Dense(n_hidden, activation='relu'))            
        # last layer
        network.add(Dense(self.game.state_size[0], activation='relu'))
        network.compile(loss='mean_squared_error', optimizer='adam')
        self.network = network


    ### RUN TIME ###

    def predict_next_state(self, action):
        """ """
        network_inputs = self.game.combine(self.game.get_game_state(), action)
        return self.newtork.predict(network_inputs, verbose=0)[0]
    

    ### TRAIN OFFLINE ###

    def train_newtork_offline(self, epochs=100, n_loop=10, save_me=False):
        """ """
        pass

    def test_network_offline(self):
        """ """
        pass

    def load_network_data(self, shuffle_me=True):
        """ """
        pass
    

    ### TRAIN ONLINE ###
        
    def train_newtork_online(self, epochs=10):
        """ """
        pass

    def test_network_online(self):
        """ """
        pass   


