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
        self.n_listen = 3
        
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

    def train_newtork_offline(self, epochs=1000, n_loop=100, save_me=False):
        """ """
        data, acts, labels = self.load_network_data()
        for _ in range(n_loop):
            ds = np.array([self.game.combine(d, a) for d, a in zip(data, acts)])
            for i in range(1, self.n_listen):                
                ls = labels[i::self.n_listen]
                self.network.fit(ds, ls, epochs=epochs//n_loop, verbose=0)
                self.print_metrics(data, labels)            
        self.save_network(save_me)
        
    def test_network_offline(self):
        """ """
        data, _, labels = self.load_network_inputs()
        self.print_metrics(data, labels)

    def load_network_data(self, shuffle_me=True):
        """ """
        import os
        files = self.game.get_all_gamedata_paths()
        basenames = [os.path.basename(file) for file in files]
        data = DT.load_images(files[::self.n_listen])
        actions = [bn.split('_')[2] for bn in basenames[::self.n_listen]]
        label_idxs = [i for i in range(len(files)) if i % self.n_listen != 0]
        labels = DT.load_images([files[i] for i in label_idxs])
        return data, actions, labels
    

    ### TRAIN ONLINE ###
        
    def train_newtork_online(self, epochs=10):
        """ """
        pass

    def test_network_online(self):
        """ """
        pass   


