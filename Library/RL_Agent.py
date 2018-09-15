from keras.models import Sequential
from keras.layers import Dense

from Library.General import Screen
from Library.General import DataThings as DT
from Library.RL_Component import RL_Component as RLC


class Agent(RLC):
    """ """

    def __init__(self, game, environment):
        """ """ 

        # RL components
        self.name = 'agent'
        self.game = game
        self.env = environment
        
        # file location
        from os.path import join
        self.value_network_path = join(game.game_path, 'value_network.h5')
        self.reward_network_path = join(game.game_path, 'reward_network.h5')
        
        # load network
        RLC.__init__(self, self.value_network_path)
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
        network.add(Dense(1, activation='relu'))
        network.compile(loss='mean_squared_error', optimizer='adam')
        self.network = network


    ### RUN TIME ###

    def predict_action(self):
        """ """
        for action in self.actions:
            data = self.game.combine(self.game.get_gamestate(), action)
            values.append(self.network.predict(data))
        return self.actions[values.index(values.max())]

    def take_action(self, action):
        """ """
        Screen.send_keys(action)


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


