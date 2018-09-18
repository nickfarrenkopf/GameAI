from keras.models import Sequential
from keras.layers import Dense

from Library.General import DataThings as DT
from Library.RL_Component import RL_Component as RLC


class Reward(RLC):
    """ """

    def __init__(self, game, agent):
        """ """
        
        # RL components
        self.name = 'reward'
        self.game = game
        self.agent = agent

        # load network
        RLC.__init__(self, agent.reward_network_path)
        self.load_network()


    ### NETWORK ###

    def create_network(self, n_hidden=64, n_layers=2):
        """ create ANN reward nertwork with keras """
        # first layer
        network = Sequential()
        network.add(Dense(n_hidden, activation='relu',
                          input_shape=self.game.state_size))
        # hidden layers
        for _ in range(n_layers - 1):
            network.add(Dense(n_hidden, activation='relu'))
        # last layer
        network.add(Dense(self.game.n_rewards, activation='softmax'))
        network.compile(loss='categorical_crossentropy', optimizer='adam',
                        metrics=['accuracy'])
        self.network = network


    ### RUN TIME ###

    def predict_reward(self):
        """ get prediction given environment gamestate """
        return self.network.predict(self.game.get_gamestate(), verbose=0)


    ### TRAIN - OFFLINE ###

    def train_network_offline(self, epochs=1000, n_loop=100, save_me=False):
        """ train keras network with saved gamestate data """
        data, _, labels = self.load_network_data()
        for _ in range(n_loop):
            self.network.fit(data, labels, epochs=epochs//n_loop, verbose=0)
            self.print_metrics(data, labels)            
        self.save_network(save_me)

    def test_network_offline(self):
        """ test network against all data """
        data, _, labels = self.load_network_inputs()
        self.print_metrics(data, labels)

    def load_network_data(self, shuffle_me=True):
        """ load gamestate images based on labeled class data """
        # find data based on agent rewards
        idxs, labels = self.game.find_game_data(self.agent.reward_labels)
        _, labels, files = self.add_zero_reward_labels(idxs, labels)
        files, labels = self.shuffle_mes(files, labels, shuffle_me)
        # load data and format to network
        data = self.game.auto_network.get_flat(DT.load_images(files))
        one_hot = DT.to_one_hot_labels(labels)
        return data, labels, one_hot

    def add_zero_reward_labels(self, idxs, labels):
        """ add zero reward labels given pre-labeled indexes """
        files = self.game.get_all_gamedata_paths()
        files_to_load = [files[i] for i in idxs]
        # loop over files and add every n in not in indexes
        for i in range(0, len(files), len(files) // len(labels) - 1):
            if i not in idxs:
                files_to_load.append(files[i])
                labels.append(0)
        return idxs, labels, files_to_load


    ### TRAIN - ONLINE ###  

    def train_network_online(self, epochs=10):
        """ """
        pass

    def test_network_online(self):
        """ """
        pass


