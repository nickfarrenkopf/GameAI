from keras.models import Sequential
from keras.layers import Dense

from Library.General import Screen
from Library.General import DataThings as DT
from Library import RL_Component


class Environment(RL_Component):
    """ """

    def __init__(self, game):
        """ """
        
        # RL components
        self.name = 'environment'
        self.game = game
        
        # load network
        RL_Component.__init__(self, game.environment_network_path)
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

    def get_next_state(self, action):
        """ """
        network_inputs = self.combine(self.game.get_game_state(), action)
        return self.newtork.predict(network_inputs, verbose=0)
    

    ### TRAIN OFFLINE ###

    def train_newtork_offline(self, epochs=100, n_loop=10, save_me=False):
        """ """
        n_actions = len(action_data) - 1
        for i in range(n_actions):
            data = state_data[i] + action_data[i]
            label = state_data[i + 1]
            network.train(data, labels) ####
        self.save_network(save_me)

    def test_network_offline(self):
        """ """
        pass


    ### TRAIN ONLINE ###
        
    def train_newtork_online(self, epochs=10):
        """ """
        pass

    def test_network_online(self):
        """ """
        pass   


